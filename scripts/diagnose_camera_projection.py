#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

try:
    import pyarrow.parquet as pq
except Exception as exc:  # pragma: no cover
    raise ModuleNotFoundError("pyarrow is required. Install with `python -m pip install pyarrow`.") from exc

try:
    import pinocchio as pin
except Exception as exc:  # pragma: no cover
    raise ModuleNotFoundError("pinocchio is required. Install with conda-forge pinocchio.") from exc

import yaml
from scipy.spatial.transform import Rotation as R

from utils.image_utils import build_intrinsics, infer_image_size
from utils.ik_utils import IKConfig, PinocchioIKSolver
from utils.oss_utils import configure_path_mapping, oss_to_local
from utils.retarget_utils import (
    align_poses_to_workstation,
    axis_angle_to_rotation_matrix,
    build_pose_matrices,
    build_transform_matrix,
    compute_eef_poses,
    ensure_bool,
    express_poses_in_frame,
    get_default_camera_matrix,
    load_mano_layer,
    load_pose_archive,
    mano_forward,
    pose_matrices_to_vectors,
    resolve_torch_device,
    scale_traj_translation,
    smooth_pose_matrices,
    to_numpy,
    to_torch,
    world_to_camera,
)


def _safe_stats(values: np.ndarray) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"n": 0, "mean": None, "p95": None, "max": None}
    return {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "p95": float(np.percentile(array, 95.0)),
        "max": float(np.max(array)),
    }


def _rotation_error_deg(rot_a: np.ndarray, rot_b: np.ndarray) -> float:
    delta = rot_a.T @ rot_b
    trace = float(np.trace(delta))
    cosine = float(np.clip((trace - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _pose_vector_to_matrix(pose: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = axis_angle_to_rotation_matrix(np.asarray(pose[3:6], dtype=np.float32)).astype(np.float64)
    transform[:3, 3] = np.asarray(pose[:3], dtype=np.float64)
    return transform


def _project_point(point_world: np.ndarray, world_from_camera: np.ndarray, intrinsics: np.ndarray) -> np.ndarray | None:
    camera_from_world = np.linalg.inv(world_from_camera)
    point_cam = (camera_from_world @ np.r_[np.asarray(point_world, dtype=np.float64), 1.0])[:3]
    z_value = float(point_cam[2])
    if z_value <= 1.0e-6:
        return None
    u_value = float(intrinsics[0, 0] * point_cam[0] / z_value + intrinsics[0, 2])
    v_value = float(intrinsics[1, 1] * point_cam[1] / z_value + intrinsics[1, 2])
    return np.asarray([u_value, v_value], dtype=np.float64)


def _load_process_config(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file_obj:
        payload = yaml.safe_load(file_obj)
    if not isinstance(payload, list):
        raise ValueError(f"process config must be a list: {path}")
    by_name = {}
    for item in payload:
        if not isinstance(item, dict):
            continue
        process_type = item.get("type")
        if isinstance(process_type, str):
            by_name[process_type] = item.get("params", {})
    return by_name


def _extract_hand_keypoints(
    archive: dict[str, Any],
    *,
    traj: Any,
    frame_count: int,
    side: str,
    mano_model_dir: str,
    device_name: str,
    use_pca: bool,
    flat_hand_mean: bool,
    fix_shapedirs: bool,
) -> tuple[np.ndarray, np.ndarray]:
    keypoints = np.full((frame_count, 21, 3), np.nan, dtype=np.float32)
    valid = np.zeros((frame_count,), dtype=np.bool_)
    section = archive.get(f"{side}_hand")
    if section is None:
        return keypoints, valid

    mano_params = section.get("mano_params")
    if not isinstance(mano_params, dict):
        return keypoints, valid

    device = resolve_torch_device(device_name)
    global_orient = to_torch(mano_params["global_orient"], device).reshape(-1, 3)
    hand_pose = to_torch(mano_params["hand_pose"], device).reshape(-1, 45)
    betas = to_torch(mano_params["betas"], device).reshape(-1, 10)
    transl = to_torch(mano_params["transl"], device).reshape(-1, 3)

    local_count = min(frame_count, len(global_orient), len(hand_pose), len(betas), len(transl), len(traj))
    if local_count <= 0:
        return keypoints, valid

    valid[:local_count] = True
    pred_valid = section.get("pred_valid")
    if pred_valid is not None:
        valid_mask = ensure_bool(pred_valid).reshape(-1)
        valid[:local_count] &= valid_mask[:local_count]

    mano_layer = load_mano_layer(
        mano_model_dir,
        side,
        use_pca,
        flat_hand_mean,
        fix_shapedirs,
        device_name,
    )
    _, joints = mano_forward(
        mano_layer,
        global_orient=global_orient[:local_count],
        hand_pose=hand_pose[:local_count],
        betas=betas[:local_count],
        transl=transl[:local_count],
    )
    joints = world_to_camera(joints, traj[:local_count])
    joints_np = to_numpy(joints).astype(np.float32, copy=False)

    keypoints[:local_count] = joints_np
    keypoints[~valid] = np.nan
    return keypoints, valid


def _recompute_retarget(
    *,
    source_data_path: Path,
    retarget_params: dict[str, Any],
    load_data_params: dict[str, Any],
) -> dict[str, Any]:
    archive = load_pose_archive(source_data_path)
    slam_data = archive["slam_data"]

    fps_value = archive.get("fps", slam_data.get("fps", 30.0))
    fps = float(np.asarray(to_numpy(fps_value)).reshape(-1)[0])
    frame_count = int(len(to_numpy(slam_data["traj"])))

    device_name = str(load_data_params.get("device", "cpu"))
    device = resolve_torch_device(device_name)
    traj = to_torch(slam_data["traj"], device).reshape(-1, 7)
    scale = to_torch(slam_data["scale"], device).reshape(1)
    traj = scale_traj_translation(traj[:frame_count], scale)

    mano_model_dir = str(load_data_params.get("mano_model_dir", "./assets/mano_v1_2/models"))
    use_pca = bool(load_data_params.get("use_pca", False))
    flat_hand_mean = bool(load_data_params.get("flat_hand_mean", True))
    fix_shapedirs = bool(load_data_params.get("fix_shapedirs", True))

    left_keypoints, left_valid = _extract_hand_keypoints(
        archive,
        traj=traj,
        frame_count=frame_count,
        side="left",
        mano_model_dir=mano_model_dir,
        device_name=device_name,
        use_pca=use_pca,
        flat_hand_mean=flat_hand_mean,
        fix_shapedirs=fix_shapedirs,
    )
    right_keypoints, right_valid = _extract_hand_keypoints(
        archive,
        traj=traj,
        frame_count=frame_count,
        side="right",
        mano_model_dir=mano_model_dir,
        device_name=device_name,
        use_pca=use_pca,
        flat_hand_mean=flat_hand_mean,
        fix_shapedirs=fix_shapedirs,
    )

    retarget_scheme = str(retarget_params.get("retarget_scheme", "legacy")).strip().lower() or "legacy"
    left_camera_pose = compute_eef_poses(left_keypoints, left_valid, side="left", scheme=retarget_scheme)
    right_camera_pose = compute_eef_poses(right_keypoints, right_valid, side="right", scheme=retarget_scheme)

    camera_matrix = get_default_camera_matrix(
        camera_elevation_deg=float(retarget_params.get("camera_elevation_deg", 45.0))
    )
    left_world = build_pose_matrices(left_camera_pose, camera_matrix)
    right_world = build_pose_matrices(right_camera_pose, camera_matrix)
    left_world, right_world, camera_matrix = align_poses_to_workstation(
        left_world,
        right_world,
        workstation_center=retarget_params.get("workspace_center", [0.25, 0.0, 0.4]),
        camera_matrix=camera_matrix,
    )

    if bool(retarget_params.get("use_ekf", True)):
        ekf = dict(retarget_params.get("ekf", {}))
        dt = 1.0 / max(fps, 1.0e-6)
        edge_pad = int(ekf.get("edge_pad", 24))
        left_world = smooth_pose_matrices(
            left_world,
            dt=dt,
            edge_pad=edge_pad,
            q_pos=float(ekf.get("q_pos", 120.0)),
            q_rot=float(ekf.get("q_rot", 120.0)),
            r_pos=float(ekf.get("r_pos", 2.0e-3)),
            r_rot=float(ekf.get("r_rot", 4.0e-3)),
            innovation_gate_pos=ekf.get("innovation_gate_pos"),
            innovation_gate_rot=ekf.get("innovation_gate_rot", 1.2),
            outlier_noise_scale=float(ekf.get("outlier_noise_scale", 1.0e6)),
            rotvec_wrap_count=int(ekf.get("rotvec_wrap_count", 2)),
        )
        right_world = smooth_pose_matrices(
            right_world,
            dt=dt,
            edge_pad=edge_pad,
            q_pos=float(ekf.get("q_pos", 120.0)),
            q_rot=float(ekf.get("q_rot", 120.0)),
            r_pos=float(ekf.get("r_pos", 2.0e-3)),
            r_rot=float(ekf.get("r_rot", 4.0e-3)),
            innovation_gate_pos=ekf.get("innovation_gate_pos"),
            innovation_gate_rot=ekf.get("innovation_gate_rot", 1.2),
            outlier_noise_scale=float(ekf.get("outlier_noise_scale", 1.0e6)),
            rotvec_wrap_count=int(ekf.get("rotvec_wrap_count", 2)),
        )

    left_pinch = np.nan_to_num(np.asarray(left_camera_pose[:, 6], dtype=np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    right_pinch = np.nan_to_num(np.asarray(right_camera_pose[:, 6], dtype=np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    left_pinch = np.clip(left_pinch, 0.0, 1.0)
    right_pinch = np.clip(right_pinch, 0.0, 1.0)

    left_base_transform = build_transform_matrix(
        translation=retarget_params.get("left_base_translation", [0.0, 0.3, 0.0])
    )
    right_base_transform = build_transform_matrix(
        translation=retarget_params.get("right_base_translation", [0.0, -0.3, 0.0])
    )
    left_base_pose = pose_matrices_to_vectors(express_poses_in_frame(left_world, left_base_transform), pinch=left_pinch)
    right_base_pose = pose_matrices_to_vectors(express_poses_in_frame(right_world, right_base_transform), pinch=right_pinch)

    intrinsics = build_intrinsics(slam_data["img_focal"], slam_data["img_center"])
    image_h, image_w = infer_image_size(slam_data["img_center"])

    return {
        "fps": float(fps),
        "intrinsics": np.asarray(intrinsics, dtype=np.float64).reshape(3, 3),
        "image_size_hw": (int(image_h), int(image_w)),
        "camera_extrinsics": np.asarray(camera_matrix, dtype=np.float64).reshape(4, 4),
        "retarget_scheme": retarget_scheme,
        "left_base_pose": np.asarray(left_base_pose, dtype=np.float64),
        "right_base_pose": np.asarray(right_base_pose, dtype=np.float64),
        "left_world_pose": np.asarray(left_world, dtype=np.float64),
        "right_world_pose": np.asarray(right_world, dtype=np.float64),
    }


def _diagnose_storage_consistency(
    *,
    parquet_payload: dict[str, Any],
    recomputed: dict[str, Any],
    left_base_translation: list[float],
    right_base_translation: list[float],
) -> dict[str, Any]:
    left_stored = np.asarray(parquet_payload["left_pose"], dtype=np.float64)
    right_stored = np.asarray(parquet_payload["right_pose"], dtype=np.float64)
    left_ref = np.asarray(recomputed["left_base_pose"], dtype=np.float64)
    right_ref = np.asarray(recomputed["right_base_pose"], dtype=np.float64)

    n = min(len(left_stored), len(right_stored), len(left_ref), len(right_ref))
    left_stored = left_stored[:n]
    right_stored = right_stored[:n]
    left_ref = left_ref[:n]
    right_ref = right_ref[:n]

    left_mask = np.isfinite(left_stored[:, :6]).all(axis=1) & np.isfinite(left_ref[:, :6]).all(axis=1)
    right_mask = np.isfinite(right_stored[:, :6]).all(axis=1) & np.isfinite(right_ref[:, :6]).all(axis=1)

    left_trans = np.linalg.norm(left_stored[left_mask, :3] - left_ref[left_mask, :3], axis=1) if left_mask.any() else np.array([])
    right_trans = np.linalg.norm(right_stored[right_mask, :3] - right_ref[right_mask, :3], axis=1) if right_mask.any() else np.array([])

    left_rot = []
    for pose_a, pose_b in zip(left_stored[left_mask], left_ref[left_mask]):
        rot_a = axis_angle_to_rotation_matrix(np.asarray(pose_a[3:6], dtype=np.float32))
        rot_b = axis_angle_to_rotation_matrix(np.asarray(pose_b[3:6], dtype=np.float32))
        left_rot.append(_rotation_error_deg(rot_a, rot_b))
    right_rot = []
    for pose_a, pose_b in zip(right_stored[right_mask], right_ref[right_mask]):
        rot_a = axis_angle_to_rotation_matrix(np.asarray(pose_a[3:6], dtype=np.float32))
        rot_b = axis_angle_to_rotation_matrix(np.asarray(pose_b[3:6], dtype=np.float32))
        right_rot.append(_rotation_error_deg(rot_a, rot_b))

    world_from_camera = np.asarray(parquet_payload["camera_extrinsics"][0], dtype=np.float64).reshape(4, 4)
    camera_from_world = np.linalg.inv(world_from_camera)
    left_base_tf = build_transform_matrix(translation=left_base_translation).astype(np.float64)
    right_base_tf = build_transform_matrix(translation=right_base_translation).astype(np.float64)
    left_world_ref = np.asarray(recomputed["left_world_pose"], dtype=np.float64)[:n]
    right_world_ref = np.asarray(recomputed["right_world_pose"], dtype=np.float64)[:n]

    left_cam_trans = []
    right_cam_trans = []
    left_cam_rot = []
    right_cam_rot = []
    for idx in range(n):
        if np.isfinite(left_stored[idx, :6]).all() and np.isfinite(left_world_ref[idx, :3, :3]).all():
            world_from_left = left_base_tf @ _pose_vector_to_matrix(left_stored[idx])
            camera_left = camera_from_world @ world_from_left
            camera_left_ref = camera_from_world @ left_world_ref[idx]
            left_cam_trans.append(float(np.linalg.norm(camera_left[:3, 3] - camera_left_ref[:3, 3])))
            left_cam_rot.append(_rotation_error_deg(camera_left[:3, :3], camera_left_ref[:3, :3]))
        if np.isfinite(right_stored[idx, :6]).all() and np.isfinite(right_world_ref[idx, :3, :3]).all():
            world_from_right = right_base_tf @ _pose_vector_to_matrix(right_stored[idx])
            camera_right = camera_from_world @ world_from_right
            camera_right_ref = camera_from_world @ right_world_ref[idx]
            right_cam_trans.append(float(np.linalg.norm(camera_right[:3, 3] - camera_right_ref[:3, 3])))
            right_cam_rot.append(_rotation_error_deg(camera_right[:3, :3], camera_right_ref[:3, :3]))

    return {
        "left_base_translation_error_m": _safe_stats(left_trans),
        "right_base_translation_error_m": _safe_stats(right_trans),
        "left_base_rotation_error_deg": _safe_stats(np.asarray(left_rot, dtype=np.float64)),
        "right_base_rotation_error_deg": _safe_stats(np.asarray(right_rot, dtype=np.float64)),
        "left_recovered_camera_translation_error_m": _safe_stats(np.asarray(left_cam_trans, dtype=np.float64)),
        "right_recovered_camera_translation_error_m": _safe_stats(np.asarray(right_cam_trans, dtype=np.float64)),
        "left_recovered_camera_rotation_error_deg": _safe_stats(np.asarray(left_cam_rot, dtype=np.float64)),
        "right_recovered_camera_rotation_error_deg": _safe_stats(np.asarray(right_cam_rot, dtype=np.float64)),
    }


def _diagnose_ik_projection(
    *,
    parquet_payload: dict[str, Any],
    meta_data: dict[str, Any],
    ee_frames: dict[str, str],
) -> dict[str, Any]:
    intrinsics = np.asarray(parquet_payload["intrinsics"][0], dtype=np.float64).reshape(3, 3)
    world_from_camera = np.asarray(parquet_payload["camera_extrinsics"][0], dtype=np.float64).reshape(4, 4)

    side_stats: dict[str, Any] = {}
    for side in ("left", "right"):
        pose_key = f"{side}_pose"
        joint_key = f"{side}_joint_position"
        reachable_key = f"{side}_reachable"
        base_frame = str(meta_data.get(f"{side}_frame") or f"{side}_base_link")
        ee_frame = str(ee_frames.get(side, f"{side}_tcp"))
        joint_names = meta_data.get(f"{side}_joint_names")
        if not isinstance(joint_names, list) or not joint_names:
            side_stats[side] = {"error": f"missing {side}_joint_names in meta_data"}
            continue

        solver = PinocchioIKSolver(
            urdf_path=str(meta_data.get("ik_urdf_path", "")),
            package_dirs=["./assets"],
            ee_frame=ee_frame,
            base_frame=base_frame,
            joint_names=[str(name) for name in joint_names],
            config=IKConfig(),
        )

        metric_position = []
        metric_orientation = []
        true_position = []
        true_orientation = []
        pixel_error = []
        used_frames = 0

        for pose, joints, reachable in zip(
            parquet_payload.get(pose_key, []),
            parquet_payload.get(joint_key, []),
            parquet_payload.get(reachable_key, []),
        ):
            if pose is None or joints is None or not bool(reachable):
                continue
            pose_array = np.asarray(pose, dtype=np.float64).reshape(-1)
            joint_array = np.asarray(joints, dtype=np.float64).reshape(-1)
            if pose_array.size < 6 or joint_array.size != len(solver.q_indices_arr):
                continue
            if not np.isfinite(pose_array[:6]).all() or not np.isfinite(joint_array).all():
                continue

            q_full = solver.neutral.copy()
            q_full[solver.q_indices_arr] = joint_array
            target = solver._target_from_pose(pose_array[:6])
            metrics = solver._frame_metrics(
                q=q_full,
                target=target,
                q_prev=q_full[solver.q_indices_arr],
                iteration=0,
            )

            pin.forwardKinematics(solver.model, solver.data, q_full)
            pin.updateFramePlacements(solver.model, solver.data)
            current = solver.data.oMf[solver.ee_frame_id]

            target_pos = np.asarray(target.translation, dtype=np.float64)
            current_pos = np.asarray(current.translation, dtype=np.float64)
            target_rot = np.asarray(target.rotation, dtype=np.float64)
            current_rot = np.asarray(current.rotation, dtype=np.float64)

            true_position.append(float(np.linalg.norm(current_pos - target_pos)))
            true_orientation.append(_rotation_error_deg(current_rot, target_rot))
            metric_position.append(float(metrics["position_error"]))
            metric_orientation.append(float(metrics["orientation_error"]))

            uv_target = _project_point(target_pos, world_from_camera, intrinsics)
            uv_current = _project_point(current_pos, world_from_camera, intrinsics)
            if uv_target is not None and uv_current is not None:
                pixel_error.append(float(np.linalg.norm(uv_current - uv_target)))
            used_frames += 1

        metric_position_arr = np.asarray(metric_position, dtype=np.float64)
        metric_orientation_arr = np.asarray(metric_orientation, dtype=np.float64)
        true_position_arr = np.asarray(true_position, dtype=np.float64)
        true_orientation_deg_arr = np.asarray(true_orientation, dtype=np.float64)
        true_orientation_rad_arr = np.deg2rad(true_orientation_deg_arr)

        swapped_hint = None
        if metric_position_arr.size > 0 and metric_orientation_arr.size > 0:
            direct_pos_gap = np.nanmean(np.abs(metric_position_arr - true_position_arr))
            swapped_pos_gap = np.nanmean(np.abs(metric_orientation_arr - true_position_arr))
            direct_rot_gap = np.nanmean(np.abs(metric_orientation_arr - true_orientation_rad_arr))
            swapped_rot_gap = np.nanmean(np.abs(metric_position_arr - true_orientation_rad_arr))
            swapped_hint = {
                "direct_position_gap": float(direct_pos_gap),
                "swapped_position_gap": float(swapped_pos_gap),
                "direct_rotation_gap": float(direct_rot_gap),
                "swapped_rotation_gap": float(swapped_rot_gap),
                "likely_swapped": bool(swapped_pos_gap < direct_pos_gap and swapped_rot_gap < direct_rot_gap),
            }

        side_stats[side] = {
            "used_frames": int(used_frames),
            "ik_metric_position_error_m_like": _safe_stats(metric_position_arr),
            "ik_metric_orientation_error_rad_like": _safe_stats(metric_orientation_arr),
            "true_tcp_translation_error_m": _safe_stats(true_position_arr),
            "true_tcp_rotation_error_deg": _safe_stats(true_orientation_deg_arr),
            "tcp_projection_pixel_error": _safe_stats(np.asarray(pixel_error, dtype=np.float64)),
            "metric_component_swap_hint": swapped_hint,
        }

    return side_stats


def _parse_ee_frames(process_params: dict[str, Any]) -> dict[str, str]:
    output = {"left": "left_tcp", "right": "right_tcp"}
    ik_params = process_params.get("inverse_kinematics")
    if not isinstance(ik_params, dict):
        return output
    sides = ik_params.get("sides")
    if not isinstance(sides, dict):
        return output
    for side in ("left", "right"):
        side_cfg = sides.get(side)
        if not isinstance(side_cfg, dict):
            continue
        value = side_cfg.get("ee_frame")
        if isinstance(value, str) and value:
            output[side] = value
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose camera/intrinsics/extrinsics and projection consistency.")
    parser.add_argument("--sample-id", type=str, required=True, help="Sample id (without extension).")
    parser.add_argument("--output-dir", type=Path, default=Path("./outputs"), help="Processed output root.")
    parser.add_argument(
        "--process-config",
        type=Path,
        default=Path("./config/processes/process.yaml"),
        help="Pipeline process config.",
    )
    parser.add_argument("--path-local-mount", type=Path, default=Path("."), help="OSS local mount.")
    parser.add_argument("--path-oss-prefix", type=str, default="oss://", help="OSS prefix.")
    args = parser.parse_args()

    configure_path_mapping(str(args.path_local_mount.expanduser().resolve()), args.path_oss_prefix)
    output_root = args.output_dir.expanduser().resolve()
    sample_id = str(args.sample_id)

    meta_path = (output_root / "meta_data" / f"{sample_id}.json").resolve()
    data_path = (output_root / "data" / f"{sample_id}.parquet").resolve()
    if not meta_path.is_file():
        raise FileNotFoundError(f"meta_data not found: {meta_path}")
    if not data_path.is_file():
        raise FileNotFoundError(f"parquet not found: {data_path}")

    with open(meta_path, "r", encoding="utf-8") as file_obj:
        meta_data = json.load(file_obj)
    process_params = _load_process_config(args.process_config.expanduser().resolve())
    retarget_params = process_params.get("retarget", {}) if isinstance(process_params, dict) else {}
    load_data_params = process_params.get("load_data", {}) if isinstance(process_params, dict) else {}

    source_data_value = meta_data.get("source_data_path")
    if not isinstance(source_data_value, str) or not source_data_value:
        raise ValueError("meta_data missing source_data_path")
    source_data_path = Path(oss_to_local(source_data_value)).expanduser().resolve()
    if not source_data_path.is_file():
        raise FileNotFoundError(f"source_data_path not found: {source_data_path}")

    table = pq.read_table(
        str(data_path),
        columns=[
            "left_pose",
            "right_pose",
            "left_joint_position",
            "right_joint_position",
            "left_reachable",
            "right_reachable",
            "intrinsics",
            "camera_extrinsics",
            "image_height",
            "image_width",
        ],
    )
    parquet_payload = table.to_pydict()

    recomputed = _recompute_retarget(
        source_data_path=source_data_path,
        retarget_params=retarget_params if isinstance(retarget_params, dict) else {},
        load_data_params=load_data_params if isinstance(load_data_params, dict) else {},
    )

    meta_intrinsics = np.asarray(meta_data.get("intrinsics"), dtype=np.float64).reshape(3, 3)
    meta_extrinsics = np.asarray(meta_data.get("camera_extrinsics"), dtype=np.float64).reshape(4, 4)
    parquet_intrinsics = np.asarray(parquet_payload["intrinsics"][0], dtype=np.float64).reshape(3, 3)
    parquet_extrinsics = np.asarray(parquet_payload["camera_extrinsics"][0], dtype=np.float64).reshape(4, 4)

    camera_input_consistency = {
        "meta_vs_parquet_intrinsics_max_abs": float(np.max(np.abs(meta_intrinsics - parquet_intrinsics))),
        "meta_vs_parquet_extrinsics_max_abs": float(np.max(np.abs(meta_extrinsics - parquet_extrinsics))),
        "recomputed_vs_parquet_intrinsics_max_abs": float(np.max(np.abs(recomputed["intrinsics"] - parquet_intrinsics))),
        "recomputed_vs_parquet_extrinsics_max_abs": float(
            np.max(np.abs(recomputed["camera_extrinsics"] - parquet_extrinsics))
        ),
        "meta_image_size_hw": [int(meta_data.get("image_height")), int(meta_data.get("image_width"))],
        "recomputed_image_size_hw": [int(recomputed["image_size_hw"][0]), int(recomputed["image_size_hw"][1])],
        "parquet_image_size_hw_first_row": [
            int(parquet_payload["image_height"][0]),
            int(parquet_payload["image_width"][0]),
        ],
    }

    storage_consistency = _diagnose_storage_consistency(
        parquet_payload=parquet_payload,
        recomputed=recomputed,
        left_base_translation=list(retarget_params.get("left_base_translation", [0.0, 0.3, 0.0])),
        right_base_translation=list(retarget_params.get("right_base_translation", [0.0, -0.3, 0.0])),
    )
    ee_frames = _parse_ee_frames(process_params if isinstance(process_params, dict) else {})
    ik_projection_consistency = _diagnose_ik_projection(
        parquet_payload=parquet_payload,
        meta_data=meta_data,
        ee_frames=ee_frames,
    )

    report = {
        "sample_id": sample_id,
        "paths": {
            "meta_data_path": str(meta_path),
            "parquet_path": str(data_path),
            "source_data_path": str(source_data_path),
        },
        "camera_input_consistency": camera_input_consistency,
        "storage_consistency": storage_consistency,
        "ik_projection_consistency": ik_projection_consistency,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

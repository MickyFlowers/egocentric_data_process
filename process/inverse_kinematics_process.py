from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np

from utils.image_utils import draw_axes_2d, draw_line
from utils.ik_utils import (
    IKConfig,
    PinocchioIKSolver,
    ensure_pose_array,
    ensure_sample_array,
    map_gripper_samples_to_joint_targets,
)
from utils.retarget_utils import axis_angle_to_rotation_matrix

from .core import BaseProcess, register_process

try:
    import pinocchio as pin
except Exception:  # pragma: no cover - optional runtime dependency for IK visualization
    pin = None

try:
    import pyrender
    import trimesh
except Exception:  # pragma: no cover - optional runtime dependency for mesh rendering
    pyrender = None
    trimesh = None


@register_process("inverse_kinematics")
class InverseKinematicsProcess(BaseProcess):
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._solver_cache: dict[str, PinocchioIKSolver] = {}

    def run(self, sample: dict[str, Any], context: Any) -> dict[str, Any]:
        eef_payload = sample.get("eef")
        if not isinstance(eef_payload, dict):
            eef_path_value = sample.get("eef_path")
            if not isinstance(eef_path_value, str) or not eef_path_value:
                raise KeyError("missing fixed field 'eef' (and optional fallback field 'eef_path')")
            eef_path_local = self.resolve_sample_path(eef_path_value)
            with open(eef_path_local, "r", encoding="utf-8") as file_obj:
                eef_payload = json.load(file_obj)

        if "poses" not in eef_payload or not isinstance(eef_payload["poses"], dict):
            raise ValueError("invalid retarget payload (missing poses)")

        side_configs = self._resolve_side_configs()
        results: dict[str, Any] = {}
        collision_inputs: dict[str, dict[str, Any]] = {}
        collision_solvers: dict[str, PinocchioIKSolver] = {}
        for side_name, side_cfg in side_configs.items():
            side_poses_raw = eef_payload["poses"].get(side_name)
            if side_poses_raw is None:
                continue
            side_poses = ensure_pose_array(side_poses_raw)
            frame_count = int(side_poses.shape[0])
            gripper_joint_names = list(side_cfg.get("gripper_joint_names", []))
            side_sample = None
            if gripper_joint_names:
                side_sample = self._extract_side_sample(eef_payload=eef_payload, side_name=side_name)
                side_sample = ensure_sample_array(side_sample, count=frame_count)

            solver = self._get_solver(side_name=side_name, side_cfg=side_cfg)
            initial_q = side_cfg.get("initial_joint_positions")
            solved = solver.solve_trajectory(
                pose_vectors=side_poses[:, :6],
                initial_joint_positions=initial_q,
                compute_collision=False,
            )
            collision_solvers[side_name] = solver

            arm_positions = np.asarray(solved["joint_positions"], dtype=np.float64)
            arm_joint_names = list(solved["joint_names"])
            ekf_applied = bool(solved.get("ekf_applied", False))
            gripper_positions = None
            gripper_ekf_applied = False
            if gripper_joint_names:
                gripper_positions = self._build_gripper_targets(
                    side_sample=np.asarray(side_sample, dtype=np.float64),
                    side_cfg=side_cfg,
                    expected_joint_count=len(gripper_joint_names),
                )
                if bool(solver.config.use_joint_ekf_smoothing) and gripper_positions.shape[0] > 1:
                    gripper_positions = self._smooth_gripper_targets_with_ekf(
                        solver=solver,
                        gripper_positions=gripper_positions,
                        side_cfg=side_cfg,
                    )
                    gripper_ekf_applied = True
                full_joint_names = arm_joint_names + gripper_joint_names
                full_joint_positions = np.concatenate([arm_positions, gripper_positions], axis=1)
            else:
                full_joint_names = arm_joint_names
                full_joint_positions = arm_positions

            limit_violation_count = self._count_limit_violations(
                positions=arm_positions,
                lower=solver.lower,
                upper=solver.upper,
            )
            continuity = self._continuity_statistics(arm_positions)
            position_error = np.asarray(solved["position_error"], dtype=np.float64)
            orientation_error = np.asarray(solved["orientation_error"], dtype=np.float64)
            reachable = np.asarray(solved["reachable"], dtype=np.bool_)
            collision = None
            valid_error_mask = np.isfinite(position_error) & np.isfinite(orientation_error)
            mean_position_error = self._finite_mean(position_error)
            mean_orientation_error = self._finite_mean(orientation_error)
            reachable_ratio = (
                float(np.mean(reachable[valid_error_mask]))
                if valid_error_mask.any()
                else None
            )
            collision_count = None
            collision_ratio = None

            results[side_name] = {
                "joint_names": full_joint_names,
                "arm_joint_names": arm_joint_names,
                "arm_joint_positions": arm_positions.tolist(),
                "gripper_joint_names": gripper_joint_names,
                "gripper_joint_positions": gripper_positions.tolist() if gripper_positions is not None else [],
                "joint_positions": full_joint_positions.tolist(),
                "ik_error": position_error.tolist(),
                "position_error": position_error.tolist(),
                "orientation_error": orientation_error.tolist(),
                "reachable": reachable.tolist(),
                "collision": collision.tolist() if collision is not None else None,
                "ik_error_mean": mean_position_error,
                "mean_position_error": mean_position_error,
                "mean_orientation_error": mean_orientation_error,
                "reachable_ratio": reachable_ratio,
                "collision_count": collision_count,
                "collision_ratio": collision_ratio,
                "limit_violation_count": int(limit_violation_count),
                "continuity": continuity,
                "valid_error_frame_count": int(np.count_nonzero(valid_error_mask)),
                "invalid_error_frame_count": int(valid_error_mask.size - np.count_nonzero(valid_error_mask)),
                "solver": {
                    "position_tolerance": float(solver.config.position_tolerance),
                    "orientation_tolerance": float(solver.config.orientation_tolerance),
                    "early_stop_position_tolerance": float(solver.early_stop_position_tolerance),
                    "early_stop_orientation_tolerance": float(solver.early_stop_orientation_tolerance),
                    "position_weight": float(solver.config.position_weight),
                    "orientation_weight": float(solver.config.orientation_weight),
                    "continuity_weight": float(solver.config.continuity_weight),
                    "collision_available": bool(solver.collision_available),
                    "error_computed_after_ekf": ekf_applied,
                    "gripper_ekf_smoothing": gripper_ekf_applied,
                    "collision_enabled": bool(solver.collision_enabled),
                    "collision_filter_adjacent_pairs": bool(solver.config.collision_filter_adjacent_pairs),
                    "collision_filter_neutral_touching_pairs": bool(
                        solver.config.collision_filter_neutral_touching_pairs
                    ),
                    "collision_neutral_touching_tolerance": float(
                        solver.config.collision_neutral_touching_tolerance
                    ),
                    "collision_ignore_links": sorted(solver.ignored_collision_links),
                    "collision_force_include_link_pairs": [
                        list(pair) for pair in solver.config.collision_force_include_link_pairs
                    ],
                    "collision_pair_count_total": int(solver.collision_pair_count_total),
                    "collision_pair_count_active": int(solver.collision_pair_count_active),
                    "collision_pair_count_filtered": int(solver.collision_pair_count_filtered),
                },
            }
            if side_sample is not None:
                results[side_name]["sample"] = np.asarray(side_sample, dtype=np.float64).tolist()
            collision_inputs[side_name] = {
                "joint_names": list(full_joint_names),
                "joint_positions": np.asarray(full_joint_positions, dtype=np.float64),
            }

        self._apply_combined_collision_metrics(
            results=results,
            collision_inputs=collision_inputs,
            collision_solvers=collision_solvers,
        )

        payload = {
            "video_path": eef_payload.get("video_path", sample.get("video_path")),
            "data_path": eef_payload.get("data_path", sample.get("data_path")),
            "fps": float(eef_payload.get("fps", sample.get("fps", 30.0))),
            "frames": eef_payload.get("frames", {}),
            "ik": results,
            "meta": {
                "urdf_path": str(self.resolve_path(self.params["urdf_path"])),
                "collision_mode": "combined_bimanual_state",
                "collision_pair_filter_adjacent_pairs": bool(
                    next(iter(collision_solvers.values())).config.collision_filter_adjacent_pairs
                ) if collision_solvers else None,
                "collision_pair_filter_neutral_touching_pairs": bool(
                    next(iter(collision_solvers.values())).config.collision_filter_neutral_touching_pairs
                ) if collision_solvers else None,
                "collision_ignore_links": sorted(next(iter(collision_solvers.values())).ignored_collision_links)
                if collision_solvers
                else [],
                "collision_force_include_link_pairs": [
                    list(pair) for pair in next(iter(collision_solvers.values())).config.collision_force_include_link_pairs
                ] if collision_solvers else [],
            },
        }
        if isinstance(sample.get("eef_path"), str):
            payload["eef_path"] = sample.get("eef_path")
        output_payload = self._format_output_payload(payload)
        sample["ik"] = output_payload

        sample["last_process"] = self.name
        return sample

    def _write_visualization(
        self,
        sample: dict[str, Any],
        context: Any,
        *,
        ik_payload: dict[str, Any],
        side_configs: dict[str, dict[str, Any]],
        target_pose_map: dict[str, Any],
    ) -> None:
        output_root = self.extend_output_dir(self.params["output_dir"], os.path.join("samples", "ik"))
        output_path, remote_output_path = self.build_output_paths(
            sample,
            output_dir=output_root,
            extension=".mp4",
        )
        fps = float(ik_payload.get("fps", sample.get("fps", 30.0)))
        width = int(self.params.get("ik_visualization_width", 960))
        height = int(self.params.get("ik_visualization_height", 540))
        mesh_enabled = bool(self.params.get("ik_mesh_render", True))

        with context.staged_output(str(output_path)) as temp_output_path:
            mesh_error = None
            if mesh_enabled:
                try:
                    rendered = self._render_ik_mesh_video(
                        temp_output_path=temp_output_path,
                        ik_payload=ik_payload,
                        side_configs=side_configs,
                        target_pose_map=target_pose_map,
                        fps=fps,
                        width=width,
                        height=height,
                    )
                    if rendered:
                        sample["ik_video_path"] = remote_output_path
                        return
                except Exception as exc:  # pragma: no cover - visualization fallback path
                    mesh_error = exc

            self._render_ik_skeleton_video(
                temp_output_path=temp_output_path,
                ik_payload=ik_payload,
                side_configs=side_configs,
                target_pose_map=target_pose_map,
                fps=fps,
                width=width,
                height=height,
            )
            if mesh_error is not None:
                print(f"[inverse_kinematics] mesh render failed, fallback to skeleton render: {mesh_error}")

        sample["ik_video_path"] = remote_output_path

    def _render_ik_mesh_video(
        self,
        *,
        temp_output_path: str,
        ik_payload: dict[str, Any],
        side_configs: dict[str, dict[str, Any]],
        target_pose_map: dict[str, Any],
        fps: float,
        width: int,
        height: int,
    ) -> bool:
        if pin is None or pyrender is None or trimesh is None:
            return False

        urdf_path = str(self.resolve_path(self.params["urdf_path"]))
        package_dirs = self._resolve_package_dirs({})
        model = pin.buildModelFromUrdf(urdf_path)
        data = model.createData()
        visual_model = pin.buildGeomFromUrdf(
            model,
            urdf_path,
            pin.GeometryType.VISUAL,
            package_dirs,
        )
        if len(visual_model.geometryObjects) == 0:
            return False
        visual_data = pin.GeometryData(visual_model)

        streams, frame_count = self._build_ik_render_stream(
            model=model,
            ik_payload=ik_payload,
            side_configs=side_configs,
        )
        if frame_count <= 0:
            return False
        target_tcp_world = self._build_target_tcp_world_poses(
            model=model,
            data=data,
            side_configs=side_configs,
            target_pose_map=target_pose_map,
            frame_count=frame_count,
        )

        scene = pyrender.Scene(
            bg_color=np.array([246, 246, 246, 255], dtype=np.uint8),
            ambient_light=np.array([0.35, 0.35, 0.35], dtype=np.float32),
        )
        fov_deg = float(self.params.get("ik_visualization_fov_deg", 42.0))
        camera = pyrender.PerspectiveCamera(
            yfov=np.deg2rad(fov_deg),
            aspectRatio=float(width) / max(float(height), 1.0),
        )
        eye = np.asarray(self.params.get("ik_visualization_camera_eye", [1.1, -0.05, 0.8]), dtype=np.float64)
        target = np.asarray(self.params.get("ik_visualization_camera_target", [0.0, 0.0, 0.15]), dtype=np.float64)
        up = np.asarray(self.params.get("ik_visualization_camera_up", [0.0, 0.0, 1.0]), dtype=np.float64)
        view_matrix = self._look_at_view_matrix(eye=eye, target=target, up=up)
        scene.add(camera, pose=np.linalg.inv(view_matrix))

        light = pyrender.DirectionalLight(color=np.ones(3, dtype=np.float32), intensity=2.2)
        scene.add(light, pose=np.linalg.inv(self._look_at_view_matrix(eye=[1.2, 0.8, 1.0], target=target, up=up)))
        fill_light = pyrender.DirectionalLight(color=np.ones(3, dtype=np.float32), intensity=1.1)
        scene.add(fill_light, pose=np.linalg.inv(self._look_at_view_matrix(eye=[-1.0, -0.6, 0.8], target=target, up=up)))

        mesh_nodes: list[Any | None] = []
        for geometry_object in visual_model.geometryObjects:
            mesh_path = str(getattr(geometry_object, "meshPath", "") or "")
            if not mesh_path:
                mesh_nodes.append(None)
                continue
            mesh_file = Path(mesh_path)
            if not mesh_file.exists():
                mesh_nodes.append(None)
                continue
            loaded = trimesh.load(str(mesh_file), force="mesh", process=False)
            if isinstance(loaded, trimesh.Scene):
                if len(loaded.geometry) == 0:
                    mesh_nodes.append(None)
                    continue
                mesh = loaded.dump(concatenate=True)
            else:
                mesh = loaded
            if mesh is None or mesh.vertices is None or len(mesh.vertices) == 0:
                mesh_nodes.append(None)
                continue
            mesh = mesh.copy()
            scale = np.asarray(getattr(geometry_object, "meshScale", [1.0, 1.0, 1.0]), dtype=np.float64).reshape(3)
            if not np.allclose(scale, 1.0):
                mesh.apply_transform(np.diag([scale[0], scale[1], scale[2], 1.0]))
            mesh_node = scene.add(
                pyrender.Mesh.from_trimesh(mesh, smooth=False),
                pose=np.eye(4, dtype=np.float64),
            )
            mesh_nodes.append(mesh_node)

        if all(node is None for node in mesh_nodes):
            return False

        marker_colors = {
            "left": np.array([1.0, 0.85, 0.2, 0.9], dtype=np.float32),
            "right": np.array([0.2, 0.95, 0.95, 0.9], dtype=np.float32),
        }
        marker_radius = float(self.params.get("ik_target_marker_radius", 0.015))
        marker_nodes: dict[str, Any] = {}
        for side_name, transforms in target_tcp_world.items():
            if len(transforms) == 0:
                continue
            sphere = trimesh.creation.icosphere(subdivisions=2, radius=marker_radius)
            material = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=marker_colors.get(side_name, np.array([0.95, 0.6, 0.1, 0.9], dtype=np.float32)),
                metallicFactor=0.0,
                roughnessFactor=0.6,
                alphaMode="BLEND",
            )
            marker_mesh = pyrender.Mesh.from_trimesh(sphere, material=material, smooth=False)
            marker_nodes[side_name] = scene.add(marker_mesh, pose=np.eye(4, dtype=np.float64))

        renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
        neutral = pin.neutral(model).astype(np.float64, copy=True)
        try:
            writer = imageio.get_writer(
                temp_output_path,
                format="FFMPEG",
                fps=fps,
                codec="libx264",
                macro_block_size=1,
                quality=None,
                ffmpeg_log_level="error",
                output_params=["-preset", "ultrafast", "-crf", "20", "-pix_fmt", "yuv420p"],
            )
            try:
                for frame_index in range(frame_count):
                    q = self._compose_joint_state(
                        model=model,
                        neutral=neutral,
                        streams=streams,
                        frame_index=frame_index,
                    )
                    pin.forwardKinematics(model, data, q)
                    pin.updateGeometryPlacements(model, data, visual_model, visual_data, q)

                    for geom_index, node in enumerate(mesh_nodes):
                        if node is None:
                            continue
                        placement = visual_data.oMg[geom_index]
                        pose = np.eye(4, dtype=np.float64)
                        pose[:3, :3] = placement.rotation
                        pose[:3, 3] = placement.translation
                        scene.set_pose(node, pose=pose)

                    for side_name, node in marker_nodes.items():
                        transforms = target_tcp_world.get(side_name, [])
                        if frame_index >= len(transforms):
                            continue
                        target_tf = transforms[frame_index]
                        if target_tf is None:
                            continue
                        scene.set_pose(node, pose=target_tf)

                    color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
                    writer.append_data(color[:, :, :3])
            finally:
                writer.close()
        finally:
            renderer.delete()
        return True

    def _render_ik_skeleton_video(
        self,
        *,
        temp_output_path: str,
        ik_payload: dict[str, Any],
        side_configs: dict[str, dict[str, Any]],
        target_pose_map: dict[str, Any],
        fps: float,
        width: int,
        height: int,
    ) -> None:
        if pin is None:
            raise ModuleNotFoundError("pinocchio is required for IK visualization")

        urdf_path = str(self.resolve_path(self.params["urdf_path"]))
        model = pin.buildModelFromUrdf(urdf_path)
        data = model.createData()
        streams, frame_count = self._build_ik_render_stream(
            model=model,
            ik_payload=ik_payload,
            side_configs=side_configs,
        )
        if frame_count <= 0:
            raise ValueError("cannot build IK visualization: no valid IK trajectory")
        target_tcp_world = self._build_target_tcp_world_poses(
            model=model,
            data=data,
            side_configs=side_configs,
            target_pose_map=target_pose_map,
            frame_count=frame_count,
        )

        fov_deg = float(self.params.get("ik_visualization_fov_deg", 42.0))
        eye = np.asarray(self.params.get("ik_visualization_camera_eye", [1.1, -0.05, 0.8]), dtype=np.float64)
        target = np.asarray(self.params.get("ik_visualization_camera_target", [0.0, 0.0, 0.15]), dtype=np.float64)
        up = np.asarray(self.params.get("ik_visualization_camera_up", [0.0, 0.0, 1.0]), dtype=np.float64)
        view_matrix = self._look_at_view_matrix(eye=eye, target=target, up=up)

        colors = {
            "left": np.array([230, 80, 80], dtype=np.uint8),
            "right": np.array([80, 120, 230], dtype=np.uint8),
        }
        target_origin_colors = {
            "left": np.array([255, 215, 0], dtype=np.uint8),
            "right": np.array([0, 235, 235], dtype=np.uint8),
        }
        neutral = pin.neutral(model).astype(np.float64, copy=True)
        writer = imageio.get_writer(
            temp_output_path,
            format="FFMPEG",
            fps=fps,
            codec="libx264",
            macro_block_size=1,
            quality=None,
            ffmpeg_log_level="error",
            output_params=["-preset", "ultrafast", "-crf", "22", "-pix_fmt", "yuv420p"],
        )
        try:
            for frame_index in range(frame_count):
                frame = np.full((height, width, 3), 246, dtype=np.uint8)
                q = self._compose_joint_state(
                    model=model,
                    neutral=neutral,
                    streams=streams,
                    frame_index=frame_index,
                )
                pin.forwardKinematics(model, data, q)
                pin.updateFramePlacements(model, data)

                for side_name, stream in streams.items():
                    world_points = []
                    for joint_id in stream["joint_ids"]:
                        world_points.append(data.oMi[joint_id].translation.copy())
                    world_points.append(data.oMf[stream["ee_frame_id"]].translation.copy())
                    points_world = np.asarray(world_points, dtype=np.float64)
                    uv, valid = self._project_world_points(
                        points_world=points_world,
                        view_matrix=view_matrix,
                        width=width,
                        height=height,
                        fov_deg=fov_deg,
                    )
                    color = colors.get(side_name, np.array([120, 120, 120], dtype=np.uint8))
                    for point_index in range(1, len(uv)):
                        if not bool(valid[point_index - 1] and valid[point_index]):
                            continue
                        draw_line(frame, uv[point_index - 1], uv[point_index], color, thickness=2)

                axis_length = float(self.params.get("ik_target_axis_length", 0.04))
                for side_name, transforms in target_tcp_world.items():
                    if frame_index >= len(transforms):
                        continue
                    target_tf = transforms[frame_index]
                    if target_tf is None:
                        continue
                    origin = target_tf[:3, 3]
                    rotation = target_tf[:3, :3]
                    axis_points_world = np.stack(
                        [
                            origin,
                            origin + axis_length * rotation[:, 0],
                            origin + axis_length * rotation[:, 1],
                            origin + axis_length * rotation[:, 2],
                        ],
                        axis=0,
                    )
                    axis_uv, axis_valid = self._project_world_points(
                        points_world=axis_points_world,
                        view_matrix=view_matrix,
                        width=width,
                        height=height,
                        fov_deg=fov_deg,
                    )
                    if not bool(axis_valid.all()):
                        continue
                    draw_axes_2d(
                        frame,
                        axis_uv,
                        origin_color=target_origin_colors.get(side_name, np.array([255, 200, 0], dtype=np.uint8)),
                        axis_colors=(
                            np.array([255, 0, 0], dtype=np.uint8),
                            np.array([0, 255, 0], dtype=np.uint8),
                            np.array([0, 0, 255], dtype=np.uint8),
                        ),
                        origin_radius=3,
                        axis_radius=2,
                        thickness=2,
                    )
                writer.append_data(frame)
        finally:
            writer.close()

    def _build_target_tcp_world_poses(
        self,
        *,
        model: Any,
        data: Any,
        side_configs: dict[str, dict[str, Any]],
        target_pose_map: dict[str, Any],
        frame_count: int,
    ) -> dict[str, list[np.ndarray | None]]:
        targets: dict[str, list[np.ndarray | None]] = {}
        if not isinstance(target_pose_map, dict):
            return targets

        pin.forwardKinematics(model, data, pin.neutral(model))
        pin.updateFramePlacements(model, data)

        for side_name, side_cfg in side_configs.items():
            side_poses = target_pose_map.get(side_name)
            if side_poses is None:
                continue
            poses = np.asarray(side_poses, dtype=np.float64)
            if poses.ndim != 2 or poses.shape[1] < 6:
                continue

            world_from_base = np.eye(4, dtype=np.float64)
            base_frame_name = side_cfg.get("base_frame")
            if isinstance(base_frame_name, str) and base_frame_name:
                base_frame_id = int(model.getFrameId(base_frame_name))
                if base_frame_id < len(model.frames):
                    base_pose = data.oMf[base_frame_id]
                    world_from_base[:3, :3] = base_pose.rotation
                    world_from_base[:3, 3] = base_pose.translation

            side_targets: list[np.ndarray | None] = [None] * frame_count
            for frame_index in range(min(frame_count, int(poses.shape[0]))):
                pose_vec = poses[frame_index, :6]
                if not np.isfinite(pose_vec).all():
                    continue
                local_tf = np.eye(4, dtype=np.float64)
                local_tf[:3, :3] = axis_angle_to_rotation_matrix(pose_vec[3:6]).astype(np.float64)
                local_tf[:3, 3] = pose_vec[:3]
                side_targets[frame_index] = world_from_base @ local_tf
            targets[side_name] = side_targets
        return targets

    def _build_ik_render_stream(
        self,
        *,
        model: Any,
        ik_payload: dict[str, Any],
        side_configs: dict[str, dict[str, Any]],
    ) -> tuple[dict[str, dict[str, Any]], int]:
        streams: dict[str, dict[str, Any]] = {}
        frame_count = 0
        ik_section = ik_payload.get("ik", {})
        if not isinstance(ik_section, dict):
            return streams, frame_count

        for side_name, side_result in ik_section.items():
            if not isinstance(side_result, dict):
                continue
            side_cfg = side_configs.get(side_name)
            if side_cfg is None:
                continue

            arm_joint_positions = np.asarray(side_result.get("arm_joint_positions", []), dtype=np.float64)
            if arm_joint_positions.ndim != 2 or arm_joint_positions.shape[0] == 0:
                continue
            arm_joint_names = list(side_result.get("arm_joint_names", side_cfg.get("joint_names", [])))
            if len(arm_joint_names) != arm_joint_positions.shape[1]:
                continue

            joint_ids: list[int] = []
            q_indices: list[int] = []
            valid_joint_stream = True
            for joint_name in arm_joint_names:
                joint_id = int(model.getJointId(str(joint_name)))
                if joint_id == 0:
                    valid_joint_stream = False
                    break
                if int(model.nqs[joint_id]) != 1:
                    valid_joint_stream = False
                    break
                joint_ids.append(joint_id)
                q_indices.append(int(model.idx_qs[joint_id]))
            if not valid_joint_stream:
                continue

            ee_frame_name = str(side_cfg.get("ee_frame", ""))
            ee_frame_id = int(model.getFrameId(ee_frame_name))
            if ee_frame_id >= len(model.frames):
                continue

            streams[side_name] = {
                "positions": arm_joint_positions,
                "q_indices": np.asarray(q_indices, dtype=np.int64),
                "joint_ids": joint_ids,
                "ee_frame_id": ee_frame_id,
            }
            frame_count = max(frame_count, int(arm_joint_positions.shape[0]))

        return streams, frame_count

    @staticmethod
    def _compose_joint_state(
        *,
        model: Any,
        neutral: np.ndarray,
        streams: dict[str, dict[str, Any]],
        frame_index: int,
    ) -> np.ndarray:
        q = neutral.copy()
        for stream in streams.values():
            positions = np.asarray(stream["positions"], dtype=np.float64)
            if frame_index >= positions.shape[0]:
                continue
            q_indices = np.asarray(stream["q_indices"], dtype=np.int64)
            q[q_indices] = positions[frame_index]
        return q

    @staticmethod
    def _look_at_view_matrix(eye: Any, target: Any, up: Any) -> np.ndarray:
        eye_vec = np.asarray(eye, dtype=np.float64).reshape(3)
        target_vec = np.asarray(target, dtype=np.float64).reshape(3)
        up_vec = np.asarray(up, dtype=np.float64).reshape(3)
        forward = target_vec - eye_vec
        forward = forward / max(float(np.linalg.norm(forward)), 1.0e-8)
        right = np.cross(forward, up_vec)
        right = right / max(float(np.linalg.norm(right)), 1.0e-8)
        true_up = np.cross(right, forward)
        true_up = true_up / max(float(np.linalg.norm(true_up)), 1.0e-8)

        view = np.eye(4, dtype=np.float64)
        view[:3, 0] = right
        view[:3, 1] = true_up
        view[:3, 2] = forward
        view[:3, 3] = eye_vec
        world_to_camera = np.linalg.inv(view)
        return world_to_camera

    @staticmethod
    def _project_world_points(
        *,
        points_world: np.ndarray,
        view_matrix: np.ndarray,
        width: int,
        height: int,
        fov_deg: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
        homogeneous = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float64)], axis=1)
        points_cam = (view_matrix @ homogeneous.T).T[:, :3]
        z = points_cam[:, 2]
        half_fov = np.deg2rad(max(float(fov_deg), 1.0e-3) * 0.5)
        focal = float(width) * 0.5 / max(np.tan(half_fov), 1.0e-6)
        safe_z = np.where(np.abs(z) > 1.0e-6, z, 1.0e-6)
        u = focal * (points_cam[:, 0] / safe_z) + float(width) * 0.5
        v = -focal * (points_cam[:, 1] / safe_z) + float(height) * 0.5
        valid = (z > 1.0e-6) & np.isfinite(u) & np.isfinite(v)
        uv = np.stack([u, v], axis=1).astype(np.float32, copy=False)
        return uv, valid

    def _extract_side_sample(self, *, eef_payload: dict[str, Any], side_name: str) -> np.ndarray:
        gripper_section = eef_payload.get("gripper_signal")
        if not isinstance(gripper_section, dict):
            raise ValueError("invalid retarget payload: missing 'gripper_signal' dictionary")
        if side_name not in gripper_section:
            raise ValueError(f"invalid retarget payload: missing gripper_signal['{side_name}']")
        return np.asarray(gripper_section[side_name], dtype=np.float64)

    def _format_output_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        decimals = self.params.get("float_decimals")
        if decimals is None:
            return payload
        try:
            digits = int(decimals)
        except (TypeError, ValueError) as exc:
            raise ValueError("float_decimals must be an integer") from exc
        if digits < 0:
            return payload
        formatted = self._round_nested_floats(payload, digits=digits)
        if not isinstance(formatted, dict):
            raise TypeError("formatted output payload must remain a dictionary")
        return formatted

    @classmethod
    def _round_nested_floats(cls, value: Any, *, digits: int) -> Any:
        if isinstance(value, dict):
            return {key: cls._round_nested_floats(item, digits=digits) for key, item in value.items()}
        if isinstance(value, list):
            return [cls._round_nested_floats(item, digits=digits) for item in value]
        if isinstance(value, (float, np.floating)):
            scalar = float(value)
            if not np.isfinite(scalar):
                return None
            return round(scalar, digits)
        return value

    def _build_gripper_targets(
        self,
        *,
        side_sample: np.ndarray,
        side_cfg: dict[str, Any],
        expected_joint_count: int,
    ) -> np.ndarray:
        closed_positions = side_cfg.get("gripper_closed_positions")
        open_positions = side_cfg.get("gripper_open_positions")
        if closed_positions is None or open_positions is None:
            raise ValueError(
                "gripper_joint_names is configured but gripper_closed_positions/gripper_open_positions are missing"
            )
        closed_list = list(closed_positions)
        open_list = list(open_positions)
        if len(closed_list) != expected_joint_count or len(open_list) != expected_joint_count:
            raise ValueError(
                "gripper_joint_names size does not match gripper_closed_positions/gripper_open_positions sizes"
            )
        return map_gripper_samples_to_joint_targets(
            side_sample,
            closed_positions=closed_list,
            open_positions=open_list,
        )

    def _smooth_gripper_targets_with_ekf(
        self,
        *,
        solver: PinocchioIKSolver,
        gripper_positions: np.ndarray,
        side_cfg: dict[str, Any],
    ) -> np.ndarray:
        closed_positions = np.asarray(side_cfg.get("gripper_closed_positions"), dtype=np.float64).reshape(-1)
        open_positions = np.asarray(side_cfg.get("gripper_open_positions"), dtype=np.float64).reshape(-1)
        if closed_positions.shape[0] != gripper_positions.shape[1] or open_positions.shape[0] != gripper_positions.shape[1]:
            raise ValueError(
                "gripper_closed_positions/gripper_open_positions size mismatch with gripper target dimensions"
            )
        lower = np.minimum(closed_positions, open_positions)
        upper = np.maximum(closed_positions, open_positions)
        return solver.smooth_trajectory_with_ekf(
            gripper_positions,
            lower=lower,
            upper=upper,
        )

    def _count_limit_violations(self, *, positions: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> int:
        tol = 1.0e-8
        below = positions < (lower.reshape(1, -1) - tol)
        above = positions > (upper.reshape(1, -1) + tol)
        return int(np.count_nonzero(below | above))

    def _apply_combined_collision_metrics(
        self,
        *,
        results: dict[str, Any],
        collision_inputs: dict[str, dict[str, Any]],
        collision_solvers: dict[str, PinocchioIKSolver],
    ) -> None:
        if not results:
            return

        collision_solver = next(
            (solver for solver in collision_solvers.values() if solver.collision_available),
            None,
        )
        if collision_solver is None:
            for side_payload in results.values():
                solver_map = side_payload.get("solver")
                if isinstance(solver_map, dict):
                    solver_map["collision_mode"] = "combined_bimanual_state"
            return

        combined_joint_names: list[str] = []
        combined_joint_positions: list[np.ndarray] = []
        side_frame_counts: dict[str, int] = {}
        frame_count = 0
        for side_name, side_payload in results.items():
            collision_input = collision_inputs.get(side_name)
            if not isinstance(collision_input, dict):
                continue

            joint_names = collision_input.get("joint_names")
            if not isinstance(joint_names, list):
                continue

            positions = np.asarray(collision_input.get("joint_positions"), dtype=np.float64)
            if positions.ndim != 2:
                continue
            if positions.shape[1] != len(joint_names):
                raise ValueError(
                    f"combined collision joint dimension mismatch for side '{side_name}': "
                    f"expected {len(joint_names)}, got {positions.shape[1]}"
                )
            side_frame_counts[side_name] = int(positions.shape[0])
            frame_count = max(frame_count, int(positions.shape[0]))
            combined_joint_names.extend(str(name) for name in joint_names)
            combined_joint_positions.append(positions)

            solver_map = side_payload.get("solver")
            if isinstance(solver_map, dict):
                solver_map["collision_mode"] = "combined_bimanual_state"

        if frame_count <= 0 or not combined_joint_positions:
            return

        aligned_positions = [
            self._align_joint_trajectory_for_collision(positions, frame_count)
            for positions in combined_joint_positions
        ]
        combined_collision = collision_solver.compute_collision_flags_from_joint_state_trajectory(
            joint_names=combined_joint_names,
            joint_positions=np.concatenate(aligned_positions, axis=1),
        )
        for side_name, side_payload in results.items():
            side_frame_count = side_frame_counts.get(side_name, 0)
            side_collision = None
            if combined_collision is not None and side_frame_count > 0:
                side_collision = combined_collision[:side_frame_count]
            collision_count, collision_ratio = self._summarize_collision(side_collision)
            side_payload["collision"] = side_collision.tolist() if side_collision is not None else None
            side_payload["collision_count"] = collision_count
            side_payload["collision_ratio"] = collision_ratio

    @staticmethod
    def _align_joint_trajectory_for_collision(positions: np.ndarray, frame_count: int) -> np.ndarray:
        trajectory = np.asarray(positions, dtype=np.float64)
        if trajectory.ndim != 2:
            raise ValueError(f"trajectory shape mismatch: expected [N, D], got {trajectory.shape}")
        if frame_count <= trajectory.shape[0]:
            return trajectory[:frame_count].copy()
        if trajectory.shape[0] == 0:
            return np.zeros((frame_count, trajectory.shape[1]), dtype=np.float64)
        pad_count = frame_count - trajectory.shape[0]
        tail = np.repeat(trajectory[-1:, :], pad_count, axis=0)
        return np.concatenate([trajectory, tail], axis=0)

    def _continuity_statistics(self, positions: np.ndarray) -> dict[str, float | None]:
        if positions.shape[0] <= 1:
            return {"max_abs_delta": 0.0, "mean_abs_delta": 0.0}
        delta = np.diff(positions, axis=0)
        abs_delta = np.abs(delta)
        finite = abs_delta[np.isfinite(abs_delta)]
        if finite.size == 0:
            return {"max_abs_delta": None, "mean_abs_delta": None}
        return {
            "max_abs_delta": float(np.max(finite)),
            "mean_abs_delta": float(np.mean(finite)),
        }

    @staticmethod
    def _finite_mean(values: np.ndarray) -> float | None:
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return None
        return float(np.mean(finite))

    @staticmethod
    def _summarize_collision(collision: np.ndarray | None) -> tuple[int | None, float | None]:
        if collision is None:
            return None, None
        frame_count = int(collision.shape[0])
        collision_count = int(np.count_nonzero(collision))
        collision_ratio = float(collision_count / frame_count) if frame_count > 0 else None
        return collision_count, collision_ratio

    def _resolve_side_configs(self) -> dict[str, dict[str, Any]]:
        configured = self.params.get("sides")
        if isinstance(configured, dict) and configured:
            return {key: dict(value) for key, value in configured.items()}
        return {
            "left": {
                "base_frame": "left_base_link",
                "ee_frame": "left_tcp",
                "joint_names": [f"left_joint{i}" for i in range(1, 7)],
            },
            "right": {
                "base_frame": "right_base_link",
                "ee_frame": "right_tcp",
                "joint_names": [f"right_joint{i}" for i in range(1, 7)],
            },
        }

    def _get_solver(self, *, side_name: str, side_cfg: dict[str, Any]) -> PinocchioIKSolver:
        solver_cfg = self._build_solver_config(side_cfg)
        cache_payload = {
            "side": side_name,
            "urdf": str(self.resolve_path(self.params["urdf_path"])),
            "package_dirs": self._resolve_package_dirs(side_cfg),
            "ee_frame": side_cfg["ee_frame"],
            "base_frame": side_cfg.get("base_frame"),
            "joint_names": list(side_cfg["joint_names"]),
            "solver": solver_cfg.__dict__,
        }
        cache_key = json.dumps(cache_payload, sort_keys=True)
        cached = self._solver_cache.get(cache_key)
        if cached is not None:
            return cached

        solver = PinocchioIKSolver(
            urdf_path=self.resolve_path(self.params["urdf_path"]),
            package_dirs=self._resolve_package_dirs(side_cfg),
            ee_frame=str(side_cfg["ee_frame"]),
            joint_names=list(side_cfg["joint_names"]),
            base_frame=side_cfg.get("base_frame"),
            config=solver_cfg,
        )
        self._solver_cache[cache_key] = solver
        return solver

    def _build_solver_config(self, side_cfg: dict[str, Any]) -> IKConfig:
        solver_params = dict(self.params.get("solver", {}))
        solver_params.update(dict(side_cfg.get("solver", {})))
        collision_params = dict(solver_params.get("collision", {}))
        ignore_links = collision_params.get("ignore_links", [])
        if isinstance(ignore_links, str):
            ignore_links = [ignore_links]
        if not isinstance(ignore_links, list):
            raise TypeError("collision.ignore_links must be a string or list of strings")
        force_include_link_pairs = collision_params.get("force_include_link_pairs", [])
        if not isinstance(force_include_link_pairs, list):
            raise TypeError("collision.force_include_link_pairs must be a list of link pairs")
        normalized_force_pairs: list[tuple[str, str]] = []
        for pair_index, pair in enumerate(force_include_link_pairs):
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise TypeError(
                    "collision.force_include_link_pairs items must be length-2 lists/tuples "
                    f"(invalid index {pair_index})"
                )
            first = str(pair[0]).strip()
            second = str(pair[1]).strip()
            if not first or not second or first == second:
                raise ValueError(
                    "collision.force_include_link_pairs items must contain two distinct non-empty link names "
                    f"(invalid index {pair_index})"
                )
            normalized_force_pairs.append((first, second))
        early_stop_position_tolerance = solver_params.get("early_stop_position_tolerance")
        early_stop_orientation_tolerance = solver_params.get("early_stop_orientation_tolerance")
        return IKConfig(
            position_weight=float(solver_params.get("position_weight", 8.0)),
            orientation_weight=float(solver_params.get("orientation_weight", 1.0)),
            continuity_weight=float(solver_params.get("continuity_weight", 0.05)),
            damping=float(solver_params.get("damping", 1.0e-4)),
            max_iterations=int(solver_params.get("max_iterations", 64)),
            max_joint_step=float(solver_params.get("max_joint_step", 0.15)),
            max_frame_delta=float(solver_params.get("max_frame_delta", 0.35)),
            position_tolerance=float(solver_params.get("position_tolerance", 0.02)),
            orientation_tolerance=float(solver_params.get("orientation_tolerance", 0.25)),
            early_stop_position_tolerance=(
                float(early_stop_position_tolerance) if early_stop_position_tolerance is not None else None
            ),
            early_stop_orientation_tolerance=(
                float(early_stop_orientation_tolerance) if early_stop_orientation_tolerance is not None else None
            ),
            collision_weight=float(collision_params.get("weight", 5.0)),
            collision_safe_distance=float(collision_params.get("safe_distance", 0.01)),
            collision_check_interval=int(collision_params.get("check_interval", 2)),
            collision_filter_adjacent_pairs=bool(collision_params.get("filter_adjacent_pairs", True)),
            collision_filter_neutral_touching_pairs=bool(
                collision_params.get("filter_neutral_touching_pairs", True)
            ),
            collision_neutral_touching_tolerance=float(
                collision_params.get("neutral_touching_tolerance", 1.0e-9)
            ),
            collision_ignore_links=tuple(str(name) for name in ignore_links),
            collision_force_include_link_pairs=tuple(normalized_force_pairs),
            use_collision=bool(collision_params.get("enabled", False)),
            require_collision=bool(collision_params.get("required", False)),
            use_joint_ekf_smoothing=bool(solver_params.get("use_joint_ekf_smoothing", True)),
            ekf_dt=float(solver_params.get("ekf_dt", 1.0)),
            ekf_process_noise_position=float(solver_params.get("ekf_process_noise_position", 2.0e-4)),
            ekf_process_noise_velocity=float(solver_params.get("ekf_process_noise_velocity", 8.0e-3)),
            ekf_measurement_noise=float(solver_params.get("ekf_measurement_noise", 2.0e-3)),
            ekf_initial_position_var=float(solver_params.get("ekf_initial_position_var", 2.0e-4)),
            ekf_initial_velocity_var=float(solver_params.get("ekf_initial_velocity_var", 8.0e-2)),
        )

    def _resolve_package_dirs(self, side_cfg: dict[str, Any]) -> list[str]:
        dirs = side_cfg.get("package_dirs", self.params.get("package_dirs", ["./assets"]))
        if isinstance(dirs, str):
            dirs = [dirs]
        if not isinstance(dirs, list):
            raise TypeError("package_dirs must be a string or list of strings")
        return [str(self.resolve_path(path_value)) for path_value in dirs]

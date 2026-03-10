from __future__ import annotations

import os
from typing import Any

import imageio.v2 as imageio
import numpy as np

from utils.image_utils import draw_axes_2d, project_points, scale_points_2d
from utils.retarget_utils import (
    axis_angle_to_rotation_matrix,
    align_poses_to_workstation,
    build_pose_matrices,
    build_transform_matrix,
    compute_eef_poses,
    express_poses_in_frame,
    get_default_camera_matrix,
    pose_matrices_to_vectors,
    smooth_pose_matrices,
)

from .core import BaseProcess, register_process


@register_process("retarget")
class RetargetProcess(BaseProcess):
    def run(self, sample: dict[str, Any], context: Any) -> dict[str, Any]:
        retarget_scheme = str(self.params.get("retarget_scheme", "legacy")).strip().lower() or "legacy"
        left_camera_poses = compute_eef_poses(
            sample["left_hand"]["keypoints"],
            sample["left_hand"]["valid"],
            side="left",
            scheme=retarget_scheme,
        )
        right_camera_poses = compute_eef_poses(
            sample["right_hand"]["keypoints"],
            sample["right_hand"]["valid"],
            side="right",
            scheme=retarget_scheme,
        )
        fps = float(sample["fps"])

        camera_matrix = get_default_camera_matrix(
            camera_elevation_deg=self.params.get("camera_elevation_deg", 45.0)
        )
        left_world = build_pose_matrices(left_camera_poses, camera_matrix)
        right_world = build_pose_matrices(right_camera_poses, camera_matrix)
        left_world, right_world, camera_matrix = align_poses_to_workstation(
            left_world,
            right_world,
            workstation_center=self.params.get("workspace_center", [0.35, 0.0, 0.1]),
            camera_matrix=camera_matrix,
        )

        if self.params.get("use_ekf", True):
            ekf_params = dict(self.params.get("ekf", {}))
            dt = 1.0 / max(fps, 1e-6)
            edge_pad = int(ekf_params.get("edge_pad", 24))
            left_world = smooth_pose_matrices(
                left_world,
                dt=dt,
                edge_pad=edge_pad,
                q_pos=float(ekf_params.get("q_pos", 120.0)),
                q_rot=float(ekf_params.get("q_rot", 120.0)),
                r_pos=float(ekf_params.get("r_pos", 2e-3)),
                r_rot=float(ekf_params.get("r_rot", 4e-3)),
                innovation_gate_pos=ekf_params.get("innovation_gate_pos"),
                innovation_gate_rot=ekf_params.get("innovation_gate_rot", 1.2),
                outlier_noise_scale=float(ekf_params.get("outlier_noise_scale", 1.0e6)),
                rotvec_wrap_count=int(ekf_params.get("rotvec_wrap_count", 2)),
            )
            right_world = smooth_pose_matrices(
                right_world,
                dt=dt,
                edge_pad=edge_pad,
                q_pos=float(ekf_params.get("q_pos", 120.0)),
                q_rot=float(ekf_params.get("q_rot", 120.0)),
                r_pos=float(ekf_params.get("r_pos", 2e-3)),
                r_rot=float(ekf_params.get("r_rot", 4e-3)),
                innovation_gate_pos=ekf_params.get("innovation_gate_pos"),
                innovation_gate_rot=ekf_params.get("innovation_gate_rot", 1.2),
                outlier_noise_scale=float(ekf_params.get("outlier_noise_scale", 1.0e6)),
                rotvec_wrap_count=int(ekf_params.get("rotvec_wrap_count", 2)),
            )

        left_pinch = self._extract_gripper_sample(left_camera_poses)
        right_pinch = self._extract_gripper_sample(right_camera_poses)
        left_pinch = self._smooth_gripper_signal(
            left_pinch,
            valid_mask=self._extract_gripper_valid_mask(left_camera_poses),
        )
        right_pinch = self._smooth_gripper_signal(
            right_pinch,
            valid_mask=self._extract_gripper_valid_mask(right_camera_poses),
        )
        left_base_transform = build_transform_matrix(
            translation=self.params.get("left_base_translation", [0.0, 0.3, 0.0])
        )
        right_base_transform = build_transform_matrix(
            translation=self.params.get("right_base_translation", [0.0, -0.3, 0.0])
        )
        left_base_poses = pose_matrices_to_vectors(
            express_poses_in_frame(left_world, left_base_transform),
            pinch=left_pinch,
        )
        right_base_poses = pose_matrices_to_vectors(
            express_poses_in_frame(right_world, right_base_transform),
            pinch=right_pinch,
        )

        payload = {
            "video_path": sample["video_path"],
            "data_path": sample.get("data_path"),
            "fps": fps,
            "intrinsics": np.asarray(sample["intrinsics"], dtype=np.float32).tolist(),
            "camera_extrinsics": np.asarray(camera_matrix, dtype=np.float32).tolist(),
            "image_size": [int(sample["image_size"][0]), int(sample["image_size"][1])],
            "frames": {
                "left": "left_base_link",
                "right": "right_base_link",
            },
            "retarget_scheme": retarget_scheme,
            "poses": {
                "left": left_base_poses[:, :6].tolist(),
                "right": right_base_poses[:, :6].tolist(),
            },
            "gripper_signal": {
                "left": left_pinch.tolist(),
                "right": right_pinch.tolist(),
            },
        }
        sample["eef"] = payload
        sample["gripper_signal"] = payload["gripper_signal"]

        sample["last_process"] = self.name
        return sample

    @staticmethod
    def _extract_gripper_sample(camera_poses: np.ndarray) -> np.ndarray:
        if camera_poses.shape[1] <= 6:
            return np.zeros((camera_poses.shape[0],), dtype=np.float32)
        values = np.asarray(camera_poses[:, 6], dtype=np.float32).reshape(-1)
        values = np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0)
        return np.clip(values, 0.0, 1.0).astype(np.float32, copy=False)

    @staticmethod
    def _extract_gripper_valid_mask(camera_poses: np.ndarray) -> np.ndarray:
        if camera_poses.ndim != 2 or camera_poses.shape[1] <= 6:
            return np.zeros((camera_poses.shape[0],), dtype=np.bool_)
        return np.isfinite(np.asarray(camera_poses[:, 6], dtype=np.float32)).reshape(-1)

    def _smooth_gripper_signal(
        self,
        values: np.ndarray,
        *,
        valid_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        signal = np.asarray(values, dtype=np.float32).reshape(-1)
        if signal.size <= 1:
            return np.clip(signal, 0.0, 1.0).astype(np.float32, copy=False)

        filter_cfg = self.params.get("gripper_filter", {})
        if not isinstance(filter_cfg, dict):
            filter_cfg = {}
        if not bool(filter_cfg.get("enabled", True)):
            return np.clip(signal, 0.0, 1.0).astype(np.float32, copy=False)

        valid = None
        if valid_mask is not None:
            valid = np.asarray(valid_mask, dtype=np.bool_).reshape(-1)
            if valid.shape[0] != signal.shape[0]:
                valid = None
        if valid is None:
            valid = np.ones((signal.shape[0],), dtype=np.bool_)

        working = self._fill_invalid_gripper_values(
            signal,
            valid_mask=valid,
            hold_invalid=bool(filter_cfg.get("hold_invalid", True)),
        )

        median_window = int(filter_cfg.get("median_window", 3))
        if median_window > 1:
            working = self._median_filter_1d(working, window_size=median_window)

        ema_alpha = float(filter_cfg.get("ema_alpha", 0.35))
        ema_alpha = float(np.clip(ema_alpha, 1.0e-3, 1.0))
        ema_passes = max(1, int(filter_cfg.get("ema_passes", 1)))
        working = self._bidirectional_ema_filter(working, alpha=ema_alpha, passes=ema_passes)
        return np.clip(working, 0.0, 1.0).astype(np.float32, copy=False)

    @staticmethod
    def _fill_invalid_gripper_values(
        values: np.ndarray,
        *,
        valid_mask: np.ndarray,
        hold_invalid: bool,
    ) -> np.ndarray:
        signal = np.asarray(values, dtype=np.float32).reshape(-1)
        valid = np.asarray(valid_mask, dtype=np.bool_).reshape(-1)
        if signal.shape[0] != valid.shape[0]:
            raise ValueError("gripper signal and valid mask length mismatch")
        if not hold_invalid or valid.all():
            return signal.astype(np.float32, copy=True)

        valid_indices = np.flatnonzero(valid)
        if valid_indices.size <= 0:
            return np.zeros_like(signal, dtype=np.float32)

        filled = signal.astype(np.float32, copy=True)
        invalid_indices = np.flatnonzero(~valid)
        if invalid_indices.size > 0:
            filled[invalid_indices] = np.interp(
                invalid_indices.astype(np.float32),
                valid_indices.astype(np.float32),
                filled[valid_indices].astype(np.float32),
            ).astype(np.float32, copy=False)
        return filled

    @staticmethod
    def _median_filter_1d(values: np.ndarray, *, window_size: int) -> np.ndarray:
        signal = np.asarray(values, dtype=np.float32).reshape(-1)
        if signal.size <= 1:
            return signal.astype(np.float32, copy=True)
        size = max(1, int(window_size))
        if size <= 1:
            return signal.astype(np.float32, copy=True)
        if (size % 2) == 0:
            size += 1

        pad = size // 2
        padded = np.pad(signal, (pad, pad), mode="edge")
        filtered = np.empty_like(signal, dtype=np.float32)
        for frame_index in range(signal.shape[0]):
            filtered[frame_index] = float(np.median(padded[frame_index : frame_index + size]))
        return filtered

    @classmethod
    def _bidirectional_ema_filter(cls, values: np.ndarray, *, alpha: float, passes: int) -> np.ndarray:
        filtered = np.asarray(values, dtype=np.float32).reshape(-1).astype(np.float64, copy=True)
        for _ in range(max(1, int(passes))):
            filtered = cls._ema_filter_1d(filtered, alpha=alpha)
            filtered = cls._ema_filter_1d(filtered[::-1], alpha=alpha)[::-1]
        return filtered.astype(np.float32, copy=False)

    @staticmethod
    def _ema_filter_1d(values: np.ndarray, *, alpha: float) -> np.ndarray:
        signal = np.asarray(values, dtype=np.float64).reshape(-1)
        if signal.size <= 1:
            return signal.astype(np.float64, copy=True)

        out = np.empty_like(signal, dtype=np.float64)
        out[0] = signal[0]
        decay = 1.0 - float(alpha)
        for frame_index in range(1, signal.shape[0]):
            out[frame_index] = float(alpha) * signal[frame_index] + decay * out[frame_index - 1]
        return out

    def _write_visualization(
        self,
        sample: dict[str, Any],
        context: Any,
        *,
        left_poses: np.ndarray,
        right_poses: np.ndarray,
    ) -> None:
        output_root = self.extend_output_dir(self.params["output_dir"], os.path.join("samples", "eef"))
        output_path, remote_output_path = self.build_output_paths(
            sample,
            output_dir=output_root,
            extension=".mp4",
        )

        reader = imageio.get_reader(self.resolve_sample_path(sample["video_path"]))
        reader_meta = reader.get_meta_data()
        video_fps = float(reader_meta.get("fps") or sample["fps"] or 30.0)
        axis_length = float(self.params.get("axis_length", 0.03))
        image_size = sample["image_size"]
        intrinsics = sample["intrinsics"]

        try:
            with context.staged_output(str(output_path)) as temp_output_path:
                writer = imageio.get_writer(
                    temp_output_path,
                    format="FFMPEG",
                    fps=video_fps,
                    codec="libx264",
                    macro_block_size=1,
                    quality=None,
                    ffmpeg_log_level="error",
                    output_params=["-preset", "ultrafast", "-crf", "35", "-pix_fmt", "yuv420p"],
                )
                try:
                    for frame_index, frame in enumerate(reader):
                        if frame_index >= len(left_poses) and frame_index >= len(right_poses):
                            break

                        rendered = np.asarray(frame).copy()
                        frame_size = rendered.shape[:2]
                        for side, poses, origin_color in (
                            ("left", left_poses, np.array([255, 255, 0], dtype=np.uint8)),
                            ("right", right_poses, np.array([0, 255, 255], dtype=np.uint8)),
                        ):
                            if frame_index >= len(poses):
                                continue
                            pose_vec = np.asarray(poses[frame_index], dtype=np.float32)
                            if not np.isfinite(pose_vec[:6]).all():
                                continue

                            origin = pose_vec[:3]
                            rotation = axis_angle_to_rotation_matrix(pose_vec[3:6])
                            axis_points = np.stack(
                                [
                                    origin,
                                    origin + axis_length * rotation[:, 0],
                                    origin + axis_length * rotation[:, 1],
                                    origin + axis_length * rotation[:, 2],
                                ],
                                axis=0,
                            )
                            axis_uv, axis_depth_valid = project_points(axis_points, intrinsics)
                            if not axis_depth_valid.all():
                                continue
                            axis_uv = scale_points_2d(axis_uv, image_size, frame_size)
                            draw_axes_2d(
                                rendered,
                                axis_uv,
                                origin_color=origin_color,
                                axis_colors=(
                                    np.array([255, 0, 0], dtype=np.uint8),
                                    np.array([0, 255, 0], dtype=np.uint8),
                                    np.array([0, 0, 255], dtype=np.uint8),
                                ),
                            )

                        writer.append_data(rendered)
                finally:
                    writer.close()
            sample["eef_video_path"] = remote_output_path
        finally:
            reader.close()

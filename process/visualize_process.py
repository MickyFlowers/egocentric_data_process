from __future__ import annotations

import json
from typing import Any

import numpy as np

from utils.retarget_utils import (
    build_pose_matrices,
    build_transform_matrix,
    pose_matrices_to_vectors,
    transform_pose_matrices,
)

from .core import BaseProcess, register_process
from .inverse_kinematics_process import InverseKinematicsProcess
from .load_data_process import LoadDataProcess
from .retarget_process import RetargetProcess


@register_process("visualize")
class VisualizeProcess(BaseProcess):
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.strict = bool(self.params.get("strict", False))

        hand_params = self._build_hand_params()
        eef_params = self._build_eef_params()
        ik_params = self._build_ik_params()

        self._hand_visualizer = LoadDataProcess(self._build_helper_config("hand_visualize_helper", hand_params))
        self._eef_visualizer = RetargetProcess(self._build_helper_config("eef_visualize_helper", eef_params))
        self._ik_visualizer = InverseKinematicsProcess(self._build_helper_config("ik_visualize_helper", ik_params))

    def run(self, sample: dict[str, Any], context: Any) -> dict[str, Any]:
        if not bool(sample.get("visualize", False)):
            sample["last_process"] = self.name
            return sample
        if context is None:
            raise ValueError("visualize process requires pipeline context")

        errors: list[str] = []
        for step_name, step_fn in (
            ("hand", self._render_hand),
            ("eef", self._render_eef),
            ("ik", self._render_ik),
        ):
            try:
                step_fn(sample=sample, context=context)
            except Exception as exc:
                message = f"{step_name} visualization failed: {exc}"
                errors.append(message)
                print(f"[visualize] {message}")
                if self.strict:
                    raise

        if errors:
            sample["visualization_errors"] = errors
        sample["last_process"] = self.name
        return sample

    def _render_hand(self, *, sample: dict[str, Any], context: Any) -> None:
        required = ("left_hand", "right_hand", "intrinsics", "image_size", "video_path")
        for key in required:
            if key not in sample:
                raise KeyError(f"missing '{key}' in sample for hand visualization")
        self._hand_visualizer._write_visualization(sample, context)

    def _render_eef(self, *, sample: dict[str, Any], context: Any) -> None:
        eef_payload = self._resolve_eef_payload(sample)
        sample.setdefault("fps", float(eef_payload.get("fps", sample.get("fps", 30.0))))
        if "intrinsics" not in sample and eef_payload.get("intrinsics") is not None:
            sample["intrinsics"] = np.asarray(eef_payload["intrinsics"], dtype=np.float32)
        if "image_size" not in sample and eef_payload.get("image_size") is not None:
            sample["image_size"] = np.asarray(eef_payload["image_size"], dtype=np.int32)

        left_poses, right_poses = self._build_camera_pose_vectors(eef_payload)
        self._eef_visualizer._write_visualization(
            sample,
            context,
            left_poses=left_poses,
            right_poses=right_poses,
        )

    def _render_ik(self, *, sample: dict[str, Any], context: Any) -> None:
        ik_payload = self._resolve_ik_payload(sample)
        eef_payload = self._resolve_eef_payload(sample)
        side_configs = self._ik_visualizer._resolve_side_configs()
        self._ik_visualizer._write_visualization(
            sample,
            context,
            ik_payload=ik_payload,
            side_configs=side_configs,
            target_pose_map=eef_payload.get("poses", {}),
        )

    def _resolve_eef_payload(self, sample: dict[str, Any]) -> dict[str, Any]:
        eef_payload = sample.get("eef")
        if isinstance(eef_payload, dict):
            return eef_payload

        eef_path = sample.get("eef_path")
        if not isinstance(eef_path, str) or not eef_path:
            raise KeyError("missing 'eef' payload and 'eef_path' in sample")
        with open(self.resolve_sample_path(eef_path), "r", encoding="utf-8") as file_obj:
            payload = json.load(file_obj)
        if not isinstance(payload, dict):
            raise ValueError("invalid eef payload format")
        sample["eef"] = payload
        return payload

    def _resolve_ik_payload(self, sample: dict[str, Any]) -> dict[str, Any]:
        ik_payload = sample.get("ik")
        if isinstance(ik_payload, dict):
            return ik_payload

        ik_path = sample.get("ik_path")
        if not isinstance(ik_path, str) or not ik_path:
            raise KeyError("missing 'ik' payload and 'ik_path' in sample")
        with open(self.resolve_sample_path(ik_path), "r", encoding="utf-8") as file_obj:
            payload = json.load(file_obj)
        if not isinstance(payload, dict):
            raise ValueError("invalid ik payload format")
        sample["ik"] = payload
        return payload

    def _build_camera_pose_vectors(self, eef_payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        pose_map = eef_payload.get("poses")
        if not isinstance(pose_map, dict):
            raise ValueError("invalid eef payload: missing 'poses'")

        camera_extrinsics = self._parse_camera_extrinsics(eef_payload.get("camera_extrinsics"))
        camera_inverse = np.linalg.inv(camera_extrinsics).astype(np.float32)

        left_base = self._normalize_pose_sequence(pose_map.get("left"))
        right_base = self._normalize_pose_sequence(pose_map.get("right"))

        left_transform = build_transform_matrix(
            translation=self._eef_visualizer.params.get("left_base_translation", [0.0, 0.3, 0.0])
        )
        right_transform = build_transform_matrix(
            translation=self._eef_visualizer.params.get("right_base_translation", [0.0, -0.3, 0.0])
        )

        left_camera = self._convert_base_to_camera(left_base, left_transform, camera_inverse)
        right_camera = self._convert_base_to_camera(right_base, right_transform, camera_inverse)
        return left_camera, right_camera

    @staticmethod
    def _convert_base_to_camera(
        poses_base: np.ndarray,
        base_transform: np.ndarray,
        camera_inverse: np.ndarray,
    ) -> np.ndarray:
        if poses_base.shape[0] == 0:
            return np.zeros((0, 6), dtype=np.float32)

        local_mats = build_pose_matrices(poses_base, np.eye(4, dtype=np.float32))
        world_mats = transform_pose_matrices(local_mats, base_transform)
        camera_mats = transform_pose_matrices(world_mats, camera_inverse)
        vectors = pose_matrices_to_vectors(camera_mats)
        return np.asarray(vectors[:, :6], dtype=np.float32)

    @staticmethod
    def _normalize_pose_sequence(values: Any) -> np.ndarray:
        if values is None:
            return np.zeros((0, 6), dtype=np.float32)
        array = np.asarray(values, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] < 6:
            raise ValueError("eef poses must be shaped as [N, >=6]")
        return np.asarray(array[:, :6], dtype=np.float32)

    @staticmethod
    def _parse_camera_extrinsics(value: Any) -> np.ndarray:
        if value is None:
            return np.eye(4, dtype=np.float32)
        matrix = np.asarray(value, dtype=np.float32)
        if matrix.shape != (4, 4):
            raise ValueError("camera_extrinsics must be a 4x4 matrix")
        return matrix

    @staticmethod
    def _optional_params_dict(value: Any, *, field_name: str) -> dict[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise TypeError(f"visualize.params.{field_name} must be a mapping when provided")
        return dict(value)

    def _build_hand_params(self) -> dict[str, Any]:
        params = self._optional_params_dict(self.params.get("hand"), field_name="hand")
        common_output_dir = self.params.get("output_dir")
        if common_output_dir is not None:
            params.setdefault("output_dir", common_output_dir)
        params.setdefault("device", self.params.get("device", "cpu"))
        return params

    def _build_eef_params(self) -> dict[str, Any]:
        params = self._optional_params_dict(self.params.get("eef"), field_name="eef")
        common_output_dir = self.params.get("output_dir")
        if common_output_dir is not None:
            params.setdefault("output_dir", common_output_dir)
        params.setdefault("axis_length", float(self.params.get("axis_length", 0.03)))
        if "left_base_translation" in self.params:
            params.setdefault("left_base_translation", self.params["left_base_translation"])
        if "right_base_translation" in self.params:
            params.setdefault("right_base_translation", self.params["right_base_translation"])
        return params

    def _build_ik_params(self) -> dict[str, Any]:
        params = self._optional_params_dict(self.params.get("ik"), field_name="ik")
        common_output_dir = self.params.get("output_dir")
        if common_output_dir is not None:
            params.setdefault("output_dir", common_output_dir)

        for key in (
            "urdf_path",
            "package_dirs",
            "sides",
            "ik_mesh_render",
            "ik_visualization_width",
            "ik_visualization_height",
            "ik_visualization_fov_deg",
            "ik_visualization_camera_eye",
            "ik_visualization_camera_target",
            "ik_visualization_camera_up",
            "ik_target_axis_length",
            "ik_target_marker_radius",
        ):
            if key in self.params:
                params.setdefault(key, self.params[key])
        params.setdefault("urdf_path", "./assets/aloha_new_description/urdf/dual_piper.urdf")
        return params

    def _build_helper_config(self, name: str, params: dict[str, Any]) -> dict[str, Any]:
        return {
            "name": f"{self.name}_{name}",
            "_config_root": str(self._config_root),
            "params": params,
        }

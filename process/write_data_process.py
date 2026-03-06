from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .core import BaseProcess, register_process


@register_process("write_data")
class WriteDataProcess(BaseProcess):
    TRAJECTORY_DIR = "data"
    SAMPLE_METADATA_DIR = "meta_data"

    def run(self, sample: dict[str, Any], context: Any) -> dict[str, Any]:
        if context is None:
            raise ValueError("write_data process requires a pipeline context")

        eef_payload = sample.get("eef")
        if not isinstance(eef_payload, dict):
            raise KeyError("missing 'eef' in sample for write_data process")

        ik_payload = sample.get("ik")
        if not isinstance(ik_payload, dict):
            raise KeyError("missing 'ik' in sample for write_data process")

        columns, frame_count = self._build_parquet_columns(
            sample=sample,
            eef_payload=eef_payload,
            ik_payload=ik_payload,
        )
        output_dir = self.params["output_dir"]
        data_root = self.extend_output_dir(output_dir, self.TRAJECTORY_DIR)
        parquet_local_path, parquet_remote_path = self.build_output_paths(
            sample,
            output_dir=data_root,
            extension=".parquet",
        )
        self._write_parquet(
            context=context,
            local_path=parquet_local_path,
            columns=columns,
        )

        sample_metadata_root = self.extend_output_dir(output_dir, self.SAMPLE_METADATA_DIR)
        sample_metadata_local_path, sample_metadata_remote_path = self.build_output_paths(
            sample,
            output_dir=sample_metadata_root,
            extension=".json",
        )
        sample_metadata_entry = self._build_sample_metadata_entry(
            sample=sample,
            eef_payload=eef_payload,
            ik_payload=ik_payload,
            frame_count=frame_count,
            data_remote_path=parquet_remote_path,
            sample_metadata_remote_path=sample_metadata_remote_path,
        )
        context.write_json(
            str(sample_metadata_local_path),
            sample_metadata_entry,
            indent=int(self.params.get("metadata_indent", self.params.get("stats_indent", 2))),
            sort_keys=bool(self.params.get("metadata_sort_keys", self.params.get("stats_sort_keys", True))),
        )

        sample["trajectory_path"] = parquet_remote_path
        sample["meta_data_path"] = sample_metadata_remote_path
        sample["last_process"] = self.name
        return sample

    def _write_parquet(
        self,
        *,
        context: Any,
        local_path: Path,
        columns: dict[str, Any],
    ) -> None:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "pyarrow is required by write_data process. Install it with `python -m pip install pyarrow`."
            ) from exc

        table = pa.table(
            {
                "sample_id": pa.array(columns["sample_id"], type=pa.string()),
                "frame_index": pa.array(columns["frame_index"], type=pa.int32()),
                "fps": pa.array(columns["fps"], type=pa.float32()),
                "image_height": pa.array(columns["image_height"], type=pa.int32()),
                "image_width": pa.array(columns["image_width"], type=pa.int32()),
                "intrinsics": pa.array(columns["intrinsics"], type=pa.list_(pa.float32())),
                "camera_extrinsics": pa.array(columns["camera_extrinsics"], type=pa.list_(pa.float32())),
                "left_pose": pa.array(columns["left_pose"], type=pa.list_(pa.float32())),
                "right_pose": pa.array(columns["right_pose"], type=pa.list_(pa.float32())),
                "left_joint_position": pa.array(columns["left_joint_position"], type=pa.list_(pa.float32())),
                "right_joint_position": pa.array(columns["right_joint_position"], type=pa.list_(pa.float32())),
                "left_reachable": pa.array(columns["left_reachable"], type=pa.bool_()),
                "right_reachable": pa.array(columns["right_reachable"], type=pa.bool_()),
                "left_collision": pa.array(columns["left_collision"], type=pa.bool_()),
                "right_collision": pa.array(columns["right_collision"], type=pa.bool_()),
                "left_position_error": pa.array(columns["left_position_error"], type=pa.float32()),
                "right_position_error": pa.array(columns["right_position_error"], type=pa.float32()),
                "left_orientation_error": pa.array(columns["left_orientation_error"], type=pa.float32()),
                "right_orientation_error": pa.array(columns["right_orientation_error"], type=pa.float32()),
                "left_gripper_signal": pa.array(columns["left_gripper_signal"], type=pa.float32()),
                "right_gripper_signal": pa.array(columns["right_gripper_signal"], type=pa.float32()),
            }
        )
        with context.staged_output(str(local_path)) as temp_output_path:
            pq.write_table(
                table,
                temp_output_path,
                compression=str(self.params.get("compression", "snappy")),
            )

    def _build_parquet_columns(
        self,
        *,
        sample: dict[str, Any],
        eef_payload: dict[str, Any],
        ik_payload: dict[str, Any],
    ) -> tuple[dict[str, Any], int]:
        left_pose = self._extract_eef_vector_sequence(eef_payload, side="left", key="poses")
        right_pose = self._extract_eef_vector_sequence(eef_payload, side="right", key="poses")
        left_gripper = self._extract_eef_scalar_sequence(eef_payload, side="left", key="gripper_signal")
        right_gripper = self._extract_eef_scalar_sequence(eef_payload, side="right", key="gripper_signal")

        left_joint_position = self._extract_ik_vector_sequence(ik_payload, side="left", key="joint_positions")
        right_joint_position = self._extract_ik_vector_sequence(ik_payload, side="right", key="joint_positions")
        left_reachable = self._extract_ik_bool_sequence(ik_payload, side="left", key="reachable")
        right_reachable = self._extract_ik_bool_sequence(ik_payload, side="right", key="reachable")
        left_collision = self._extract_ik_bool_sequence(ik_payload, side="left", key="collision")
        right_collision = self._extract_ik_bool_sequence(ik_payload, side="right", key="collision")
        left_position_error = self._extract_ik_scalar_sequence(ik_payload, side="left", key="position_error")
        right_position_error = self._extract_ik_scalar_sequence(ik_payload, side="right", key="position_error")
        left_orientation_error = self._extract_ik_scalar_sequence(ik_payload, side="left", key="orientation_error")
        right_orientation_error = self._extract_ik_scalar_sequence(ik_payload, side="right", key="orientation_error")

        frame_count = max(
            len(left_pose),
            len(right_pose),
            len(left_gripper),
            len(right_gripper),
            len(left_joint_position),
            len(right_joint_position),
            len(left_reachable),
            len(right_reachable),
            len(left_collision),
            len(right_collision),
            len(left_position_error),
            len(right_position_error),
            len(left_orientation_error),
            len(right_orientation_error),
        )
        fps = self._safe_float(eef_payload.get("fps", sample.get("fps")))
        image_height, image_width = self._normalize_image_size(
            eef_payload.get("image_size", sample.get("image_size"))
        )
        intrinsics = self._flatten_numeric_vector(
            eef_payload.get("intrinsics", sample.get("intrinsics")),
            expected_dim=9,
        )
        camera_extrinsics = self._flatten_numeric_vector(
            eef_payload.get("camera_extrinsics"),
            expected_dim=16,
        )

        columns = {
            "sample_id": [str(sample["sample_id"])] * frame_count,
            "frame_index": list(range(frame_count)),
            "fps": [fps] * frame_count,
            "image_height": [image_height] * frame_count,
            "image_width": [image_width] * frame_count,
            "intrinsics": [intrinsics] * frame_count,
            "camera_extrinsics": [camera_extrinsics] * frame_count,
            "left_pose": self._pad_sequence(left_pose, frame_count),
            "right_pose": self._pad_sequence(right_pose, frame_count),
            "left_joint_position": self._pad_sequence(left_joint_position, frame_count),
            "right_joint_position": self._pad_sequence(right_joint_position, frame_count),
            "left_reachable": self._pad_sequence(left_reachable, frame_count),
            "right_reachable": self._pad_sequence(right_reachable, frame_count),
            "left_collision": self._pad_sequence(left_collision, frame_count),
            "right_collision": self._pad_sequence(right_collision, frame_count),
            "left_position_error": self._pad_sequence(left_position_error, frame_count),
            "right_position_error": self._pad_sequence(right_position_error, frame_count),
            "left_orientation_error": self._pad_sequence(left_orientation_error, frame_count),
            "right_orientation_error": self._pad_sequence(right_orientation_error, frame_count),
            "left_gripper_signal": self._pad_sequence(left_gripper, frame_count),
            "right_gripper_signal": self._pad_sequence(right_gripper, frame_count),
        }
        return columns, frame_count

    def _build_sample_metadata_entry(
        self,
        *,
        sample: dict[str, Any],
        eef_payload: dict[str, Any],
        ik_payload: dict[str, Any],
        frame_count: int,
        data_remote_path: str,
        sample_metadata_remote_path: str,
    ) -> dict[str, Any]:
        frames = eef_payload.get("frames", {})
        frame_map = frames if isinstance(frames, dict) else {}
        image_height, image_width = self._normalize_image_size(
            eef_payload.get("image_size", sample.get("image_size"))
        )
        intrinsics = self._flatten_numeric_vector(
            eef_payload.get("intrinsics", sample.get("intrinsics")),
            expected_dim=9,
        )
        camera_extrinsics = self._flatten_numeric_vector(
            eef_payload.get("camera_extrinsics"),
            expected_dim=16,
        )

        ik_meta = ik_payload.get("meta") if isinstance(ik_payload.get("meta"), dict) else {}
        ik_section = ik_payload.get("ik") if isinstance(ik_payload.get("ik"), dict) else {}

        output: dict[str, Any] = {
            "sample_id": str(sample["sample_id"]),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "frame_count": int(frame_count),
            "data_path": data_remote_path,
            "meta_data_path": sample_metadata_remote_path,
            "video_path": eef_payload.get("video_path", sample.get("video_path")),
            "source_data_path": eef_payload.get("data_path", sample.get("data_path")),
            "fps": self._safe_float(eef_payload.get("fps", sample.get("fps"))),
            "visualize": bool(sample.get("visualize", False)),
            "last_process": sample.get("last_process"),
            "left_frame": frame_map.get("left"),
            "right_frame": frame_map.get("right"),
            "image_height": image_height,
            "image_width": image_width,
            "intrinsics": intrinsics,
            "camera_extrinsics": camera_extrinsics,
            "ik_collision_mode": ik_meta.get("collision_mode"),
            "ik_collision_pair_filter_adjacent_pairs": ik_meta.get("collision_pair_filter_adjacent_pairs"),
            "ik_collision_pair_filter_neutral_touching_pairs": ik_meta.get(
                "collision_pair_filter_neutral_touching_pairs"
            ),
            "ik_collision_ignore_links": ik_meta.get("collision_ignore_links"),
            # "ik_urdf_path": ik_meta.get("urdf_path"),
        }

        output.update(self._build_ik_side_summary("left", ik_section.get("left")))
        output.update(self._build_ik_side_summary("right", ik_section.get("right")))
        return self._to_json_compatible(output)

    def _build_ik_side_summary(self, side: str, side_payload: Any) -> dict[str, Any]:
        if not isinstance(side_payload, dict):
            return {
                f"{side}_joint_names": [],
                f"{side}_reachable_ratio": None,
                f"{side}_collision_count": None,
                f"{side}_collision_ratio": None,
                f"{side}_mean_position_error": None,
                f"{side}_mean_orientation_error": None,
                f"{side}_valid_error_frame_count": None,
                f"{side}_invalid_error_frame_count": None,
                f"{side}_limit_violation_count": None,
                f"{side}_continuity_max_abs_delta": None,
                f"{side}_continuity_mean_abs_delta": None,
                f"{side}_solver_collision_available": None,
                f"{side}_solver_collision_enabled": None,
                f"{side}_solver_collision_mode": None,
                f"{side}_solver_collision_filter_adjacent_pairs": None,
                f"{side}_solver_collision_filter_neutral_touching_pairs": None,
                f"{side}_solver_collision_ignore_links": [],
                f"{side}_solver_collision_pair_count_total": None,
                f"{side}_solver_collision_pair_count_active": None,
                f"{side}_solver_collision_pair_count_filtered": None,
            }

        continuity = side_payload.get("continuity")
        continuity_map = continuity if isinstance(continuity, dict) else {}
        solver = side_payload.get("solver")
        solver_map = solver if isinstance(solver, dict) else {}
        joint_names = side_payload.get("joint_names")
        if not isinstance(joint_names, list):
            joint_names = []

        return {
            f"{side}_joint_names": joint_names,
            f"{side}_reachable_ratio": self._safe_float(side_payload.get("reachable_ratio")),
            f"{side}_collision_count": self._safe_int(side_payload.get("collision_count")),
            f"{side}_collision_ratio": self._safe_float(side_payload.get("collision_ratio")),
            f"{side}_mean_position_error": self._safe_float(side_payload.get("mean_position_error")),
            f"{side}_mean_orientation_error": self._safe_float(side_payload.get("mean_orientation_error")),
            f"{side}_valid_error_frame_count": self._safe_int(side_payload.get("valid_error_frame_count")),
            f"{side}_invalid_error_frame_count": self._safe_int(side_payload.get("invalid_error_frame_count")),
            f"{side}_limit_violation_count": self._safe_int(side_payload.get("limit_violation_count")),
            f"{side}_continuity_max_abs_delta": self._safe_float(continuity_map.get("max_abs_delta")),
            f"{side}_continuity_mean_abs_delta": self._safe_float(continuity_map.get("mean_abs_delta")),
            f"{side}_solver_collision_available": (
                bool(solver_map.get("collision_available")) if "collision_available" in solver_map else None
            ),
            f"{side}_solver_collision_enabled": (
                bool(solver_map.get("collision_enabled")) if "collision_enabled" in solver_map else None
            ),
            f"{side}_solver_collision_mode": (
                str(solver_map.get("collision_mode")) if solver_map.get("collision_mode") is not None else None
            ),
            f"{side}_solver_collision_filter_adjacent_pairs": (
                bool(solver_map.get("collision_filter_adjacent_pairs"))
                if "collision_filter_adjacent_pairs" in solver_map
                else None
            ),
            f"{side}_solver_collision_filter_neutral_touching_pairs": (
                bool(solver_map.get("collision_filter_neutral_touching_pairs"))
                if "collision_filter_neutral_touching_pairs" in solver_map
                else None
            ),
            f"{side}_solver_collision_ignore_links": (
                list(solver_map.get("collision_ignore_links"))
                if isinstance(solver_map.get("collision_ignore_links"), list)
                else []
            ),
            f"{side}_solver_collision_pair_count_total": self._safe_int(
                solver_map.get("collision_pair_count_total")
            ),
            f"{side}_solver_collision_pair_count_active": self._safe_int(
                solver_map.get("collision_pair_count_active")
            ),
            f"{side}_solver_collision_pair_count_filtered": self._safe_int(
                solver_map.get("collision_pair_count_filtered")
            ),
        }

    def _extract_ik_side_payload(self, ik_payload: dict[str, Any], *, side: str) -> dict[str, Any]:
        ik_section = ik_payload.get("ik")
        if not isinstance(ik_section, dict):
            return {}
        side_payload = ik_section.get(side)
        if not isinstance(side_payload, dict):
            return {}
        return side_payload

    def _extract_ik_vector_sequence(
        self,
        ik_payload: dict[str, Any],
        *,
        side: str,
        key: str,
    ) -> list[list[float] | None]:
        side_payload = self._extract_ik_side_payload(ik_payload, side=side)
        return self._normalize_vector_sequence(side_payload.get(key))

    def _extract_ik_bool_sequence(
        self,
        ik_payload: dict[str, Any],
        *,
        side: str,
        key: str,
    ) -> list[bool | None]:
        side_payload = self._extract_ik_side_payload(ik_payload, side=side)
        return self._normalize_bool_sequence(side_payload.get(key))

    def _extract_ik_scalar_sequence(
        self,
        ik_payload: dict[str, Any],
        *,
        side: str,
        key: str,
    ) -> list[float | None]:
        side_payload = self._extract_ik_side_payload(ik_payload, side=side)
        return self._normalize_scalar_sequence(side_payload.get(key))

    def _extract_eef_vector_sequence(
        self,
        eef_payload: dict[str, Any],
        *,
        side: str,
        key: str,
    ) -> list[list[float] | None]:
        section = eef_payload.get(key)
        if not isinstance(section, dict):
            return []
        return self._normalize_vector_sequence(section.get(side))

    def _extract_eef_scalar_sequence(
        self,
        eef_payload: dict[str, Any],
        *,
        side: str,
        key: str,
    ) -> list[float | None]:
        section = eef_payload.get(key)
        if not isinstance(section, dict):
            return []
        return self._normalize_scalar_sequence(section.get(side))

    @staticmethod
    def _normalize_image_size(value: Any) -> tuple[int | None, int | None]:
        if value is None:
            return None, None
        array = np.asarray(value).reshape(-1)
        if array.size < 2:
            return None, None
        height = WriteDataProcess._safe_int(array[0])
        width = WriteDataProcess._safe_int(array[1])
        return height, width

    @staticmethod
    def _flatten_numeric_vector(value: Any, *, expected_dim: int) -> list[float | None] | None:
        if value is None:
            return None
        try:
            array = np.asarray(value, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            return None
        if array.size == 0:
            return None
        if expected_dim > 0 and array.size != expected_dim:
            return None
        output: list[float | None] = []
        for item in array:
            if not np.isfinite(item):
                output.append(None)
            else:
                output.append(float(item))
        return output

    @staticmethod
    def _normalize_vector_sequence(value: Any) -> list[list[float] | None]:
        if value is None:
            return []
        try:
            array = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError):
            return []
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        if array.ndim != 2:
            return []

        sequence: list[list[float] | None] = []
        for row in array:
            if not np.isfinite(row).any():
                sequence.append(None)
                continue
            sequence.append(
                [
                    None if not np.isfinite(item) else float(item)
                    for item in row
                ]
            )
        return sequence

    @staticmethod
    def _normalize_scalar_sequence(value: Any) -> list[float | None]:
        if value is None:
            return []
        try:
            array = np.asarray(value, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            return []
        output: list[float | None] = []
        for item in array:
            output.append(None if not np.isfinite(item) else float(item))
        return output

    @staticmethod
    def _normalize_bool_sequence(value: Any) -> list[bool | None]:
        if value is None:
            return []
        try:
            array = np.asarray(value).reshape(-1)
        except (TypeError, ValueError):
            return []
        output: list[bool | None] = []
        for item in array:
            if item is None:
                output.append(None)
                continue
            if isinstance(item, (float, np.floating)) and not np.isfinite(float(item)):
                output.append(None)
                continue
            output.append(bool(item))
        return output

    @staticmethod
    def _pad_sequence(values: list[Any], frame_count: int) -> list[Any]:
        if frame_count <= len(values):
            return list(values[:frame_count])
        return list(values) + [None] * (frame_count - len(values))

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            scalar = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(scalar):
            return None
        return scalar

    @staticmethod
    def _safe_int(value: Any) -> int | None:
        if value is None:
            return None
        try:
            scalar = int(value)
        except (TypeError, ValueError):
            return None
        return scalar

    @classmethod
    def _to_json_compatible(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): cls._to_json_compatible(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cls._to_json_compatible(item) for item in value]
        if isinstance(value, tuple):
            return [cls._to_json_compatible(item) for item in value]
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.ndarray):
            return cls._to_json_compatible(value.tolist())
        if isinstance(value, (np.bool_, bool)):
            return bool(value)
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating, float)):
            scalar = float(value)
            if not np.isfinite(scalar):
                return None
            return scalar
        return value

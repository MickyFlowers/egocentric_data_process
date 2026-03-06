from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import imageio.v2 as imageio
import numpy as np

from .core import BaseProcess, register_process


@register_process("render_filter")
class RenderFilterProcess(BaseProcess):
    def run(self, sample: dict[str, Any], context: Any) -> dict[str, Any]:
        reachable_threshold = float(self.params.get("reachable_ratio_threshold", 0.9))
        collision_threshold = float(self.params.get("collision_ratio_threshold", 0.05))
        meta_data = self._resolve_meta_data(sample)

        left_ratio = self._safe_float(meta_data.get("left_reachable_ratio"))
        right_ratio = self._safe_float(meta_data.get("right_reachable_ratio"))
        left_collision_ratio = self._safe_float(meta_data.get("left_collision_ratio"))
        right_collision_ratio = self._safe_float(meta_data.get("right_collision_ratio"))
        passed = (
            left_ratio is not None
            and right_ratio is not None
            and left_collision_ratio is not None
            and right_collision_ratio is not None
            and left_ratio > reachable_threshold
            and right_ratio > reachable_threshold
            and left_collision_ratio < collision_threshold
            and right_collision_ratio < collision_threshold
        )

        sample["render_filter_passed"] = passed
        sample["render_filter_threshold"] = reachable_threshold
        sample["render_filter_collision_threshold"] = collision_threshold
        sample["render_filter_left_ratio"] = left_ratio
        sample["render_filter_right_ratio"] = right_ratio
        sample["render_filter_left_collision_ratio"] = left_collision_ratio
        sample["render_filter_right_collision_ratio"] = right_collision_ratio
        sample["last_process"] = self.name
        return sample

    def _resolve_meta_data(self, sample: dict[str, Any]) -> dict[str, Any]:
        cached = sample.get("meta_data")
        if isinstance(cached, dict):
            return cached

        meta_data_path = sample.get("meta_data_path")
        if not isinstance(meta_data_path, str) or not meta_data_path:
            raise KeyError("missing 'meta_data' and 'meta_data_path' for render_filter process")
        with open(self.resolve_sample_path(meta_data_path), "r", encoding="utf-8") as file_obj:
            payload = json.load(file_obj)
        if not isinstance(payload, dict):
            raise ValueError("meta_data payload must be a dictionary")
        sample["meta_data"] = payload
        return payload

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(number):
            return None
        return number


@dataclass
class TrajectoryPayload:
    joint_names: list[str]
    joint_positions: np.ndarray  # [T, J]
    left_gripper_signal: np.ndarray | None  # [T]
    right_gripper_signal: np.ndarray | None  # [T]
    intrinsics: np.ndarray  # [3, 3]
    camera_extrinsics: np.ndarray  # [4, 4]
    source_image_height: int | None
    source_image_width: int | None

    @property
    def frame_count(self) -> int:
        return int(self.joint_positions.shape[0])


@register_process("render")
class RenderProcess(BaseProcess):
    RENDER_DIR = "render"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._renderer: _GenesisSegmentationRenderer | None = None
        self._renderer_spec: dict[str, Any] | None = None

    def run(self, sample: dict[str, Any], context: Any) -> dict[str, Any]:
        if not bool(sample.get("render_filter_passed", True)):
            self._remove_stale_render_output(sample)
            sample["render_skipped"] = True
            sample["last_process"] = self.name
            return sample
        if context is None:
            raise ValueError("render process requires pipeline context")

        meta_data = self._resolve_meta_data(sample)
        trajectory_path = sample.get("trajectory_path")
        if not isinstance(trajectory_path, str) or not trajectory_path:
            raise KeyError("missing 'trajectory_path' in sample for render process")
        trajectory = self._load_trajectory(
            parquet_path=self.resolve_sample_path(trajectory_path),
            meta_data=meta_data,
        )
        if trajectory.frame_count <= 0:
            raise ValueError("empty trajectory in parquet data")

        video_path = self._resolve_video_path(sample, meta_data)
        urdf_path = self._resolve_urdf_path(meta_data)
        output_dir = self.params.get("output_dir")
        if output_dir is None:
            raise ValueError("render process requires params.output_dir")

        reader = imageio.get_reader(video_path)
        first_frame = None
        try:
            first_frame = np.asarray(reader.get_data(0))
        except Exception as exc:  # pragma: no cover - video backend dependent
            reader.close()
            raise RuntimeError(f"failed to read first frame from video: {video_path}") from exc

        source_h, source_w = int(first_frame.shape[0]), int(first_frame.shape[1])
        default_fov_deg = float(self.params.get("default_fov_deg", 55.0))
        backend = str(self.params.get("genesis_backend", "cpu"))
        renderer, target_h, target_w = self._get_or_create_renderer(
            urdf_path=urdf_path,
            joint_names=trajectory.joint_names,
            source_h=source_h,
            source_w=source_w,
            default_fov_deg=default_fov_deg,
            backend=backend,
        )
        intrinsics_source_h = trajectory.source_image_height if trajectory.source_image_height else source_h
        intrinsics_source_w = trajectory.source_image_width if trajectory.source_image_width else source_w
        intrinsics_scaled = self._scale_intrinsics(
            trajectory.intrinsics,
            source_h=intrinsics_source_h,
            source_w=intrinsics_source_w,
            target_h=target_h,
            target_w=target_w,
        )
        fps = self._safe_float(meta_data.get("fps")) or 30.0

        render_root = self.extend_output_dir(output_dir, self.RENDER_DIR)
        render_local_path, render_remote_path = self.build_output_paths(
            sample,
            output_dir=render_root,
            extension=".mp4",
        )

        with context.staged_output(str(render_local_path)) as temp_output_path:
            writer = imageio.get_writer(
                temp_output_path,
                format="FFMPEG",
                fps=fps,
                codec="libx264",
                macro_block_size=1,
                quality=None,
                ffmpeg_log_level="error",
                output_params=["-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p"],
            )
            last_mask = np.zeros((target_h, target_w), dtype=bool)
            last_render_rgb = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            try:
                for frame_index, frame in enumerate(reader):
                    if frame_index >= trajectory.frame_count:
                        break
                    frame_bgr = cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR)

                    render_rgb, segmentation = renderer.render_rgb_and_mask(
                        joint_positions=trajectory.joint_positions[frame_index],
                        left_gripper_signal=(
                            float(trajectory.left_gripper_signal[frame_index])
                            if trajectory.left_gripper_signal is not None
                            else None
                        ),
                        right_gripper_signal=(
                            float(trajectory.right_gripper_signal[frame_index])
                            if trajectory.right_gripper_signal is not None
                            else None
                        ),
                        camera_extrinsics=trajectory.camera_extrinsics,
                        intrinsics=intrinsics_scaled,
                    )
                    if render_rgb is None:
                        render_rgb = last_render_rgb
                    else:
                        last_render_rgb = render_rgb
                    if segmentation is None:
                        segmentation = last_mask
                    else:
                        last_mask = segmentation

                    blended = self._overlay_robot_rgb(
                        frame_bgr=frame_bgr,
                        robot_rgb=render_rgb,
                        mask=segmentation,
                        output_h=target_h,
                        output_w=target_w,
                    )
                    writer.append_data(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
            finally:
                writer.close()
                reader.close()

        sample["render_video_path"] = render_remote_path
        sample["render_success"] = True
        sample["last_process"] = self.name
        return sample

    def _get_or_create_renderer(
        self,
        *,
        urdf_path: str,
        joint_names: list[str],
        source_h: int,
        source_w: int,
        default_fov_deg: float,
        backend: str,
    ) -> tuple["_GenesisSegmentationRenderer", int, int]:
        target_h, target_w = self._resolve_target_size(source_h=source_h, source_w=source_w)
        current_spec = {
            "urdf_path": str(Path(urdf_path).resolve()),
            "joint_names": tuple(joint_names),
            "default_fov_deg": float(default_fov_deg),
            "backend": str(backend).lower(),
            "height": int(target_h),
            "width": int(target_w),
        }

        if self._renderer is not None and self._renderer_spec == current_spec:
            return self._renderer, target_h, target_w

        self._destroy_renderer()
        self._renderer = _GenesisSegmentationRenderer(
            urdf_path=urdf_path,
            joint_names=joint_names,
            width=target_w,
            height=target_h,
            default_fov_deg=default_fov_deg,
            init_backend=backend,
        )
        self._renderer_spec = dict(current_spec)
        return self._renderer, target_h, target_w

    def _resolve_target_size(self, *, source_h: int, source_w: int) -> tuple[int, int]:
        configured_height = self.params.get("render_height")
        configured_width = self.params.get("render_width")
        if configured_height is not None and configured_width is not None:
            try:
                fixed_h = int(configured_height)
                fixed_w = int(configured_width)
            except (TypeError, ValueError) as exc:
                raise ValueError("render_height/render_width must be integers") from exc
            if fixed_h <= 0 or fixed_w <= 0:
                raise ValueError("render_height/render_width must be positive")
            even_h, even_w = self._ensure_even_frame_size(height=fixed_h, width=fixed_w)
            if (even_h, even_w) != (fixed_h, fixed_w):
                print(
                    "[render] adjusted render_height/render_width to even values for ffmpeg yuv420p: "
                    f"{fixed_h}x{fixed_w} -> {even_h}x{even_w}"
                )
            return even_h, even_w

        return self._compute_target_size(
            source_h=source_h,
            source_w=source_w,
            target_short_side=int(self.params.get("target_short_side", 512)),
        )

    def _destroy_renderer(self) -> None:
        if self._renderer is None:
            self._renderer_spec = None
            return
        try:
            self._renderer.close()
        finally:
            self._renderer = None
            self._renderer_spec = None

    def _remove_stale_render_output(self, sample: dict[str, Any]) -> None:
        output_dir = self.params.get("output_dir")
        video_path = sample.get("video_path")
        sample_id = sample.get("sample_id")
        if output_dir is None or not isinstance(video_path, str) or not video_path or not isinstance(sample_id, str):
            return
        try:
            render_root = self.extend_output_dir(output_dir, self.RENDER_DIR)
            render_local_path, _ = self.build_output_paths(
                sample,
                output_dir=render_root,
                extension=".mp4",
            )
        except Exception:
            return
        if render_local_path.exists():
            render_local_path.unlink()

    def _resolve_meta_data(self, sample: dict[str, Any]) -> dict[str, Any]:
        cached = sample.get("meta_data")
        if isinstance(cached, dict):
            return cached

        meta_data_path = sample.get("meta_data_path")
        if not isinstance(meta_data_path, str) or not meta_data_path:
            raise KeyError("missing 'meta_data' and 'meta_data_path' for render process")
        with open(self.resolve_sample_path(meta_data_path), "r", encoding="utf-8") as file_obj:
            payload = json.load(file_obj)
        if not isinstance(payload, dict):
            raise ValueError("meta_data payload must be a dictionary")
        sample["meta_data"] = payload
        return payload

    def _resolve_video_path(self, sample: dict[str, Any], meta_data: dict[str, Any]) -> str:
        video_value = sample.get("video_path", meta_data.get("video_path"))
        if not isinstance(video_value, str) or not video_value:
            raise KeyError("missing video_path in sample/meta_data")
        sample["video_path"] = video_value
        return self.resolve_sample_path(video_value)

    def _resolve_urdf_path(self, meta_data: dict[str, Any]) -> str:
        urdf_value = self.params.get("urdf_path") or meta_data.get("ik_urdf_path")
        if not isinstance(urdf_value, str) or not urdf_value:
            raise ValueError("render process requires params.urdf_path or meta_data.ik_urdf_path")
        return str(self.resolve_path(urdf_value))

    def _load_trajectory(self, *, parquet_path: str, meta_data: dict[str, Any]) -> TrajectoryPayload:
        try:
            import pyarrow.parquet as pq
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "pyarrow is required for render process. Install it with `python -m pip install pyarrow`."
            ) from exc

        schema_names = set(pq.read_schema(parquet_path).names)
        if "left_joint_position" not in schema_names and "right_joint_position" not in schema_names:
            raise ValueError("render process requires at least one joint position column in parquet")
        table = pq.read_table(
            parquet_path,
            columns=[
                name
                for name in (
                    "left_joint_position",
                    "right_joint_position",
                    "left_gripper_signal",
                    "right_gripper_signal",
                    "intrinsics",
                    "camera_extrinsics",
                    "image_height",
                    "image_width",
                )
                if name in schema_names
            ],
        )
        payload = table.to_pydict()
        left_rows = list(payload.get("left_joint_position", []))
        right_rows = list(payload.get("right_joint_position", []))

        left_joint_names = self._normalize_joint_names(meta_data.get("left_joint_names"), prefix="left_joint")
        right_joint_names = self._normalize_joint_names(meta_data.get("right_joint_names"), prefix="right_joint")
        left_dim = len(left_joint_names)
        right_dim = len(right_joint_names)
        if left_dim <= 0 and right_dim <= 0:
            raise ValueError("render process requires left/right joint names in meta_data")

        frame_count = max(len(left_rows), len(right_rows))
        positions = np.zeros((frame_count, left_dim + right_dim), dtype=np.float32)
        for frame_index in range(frame_count):
            if left_dim > 0:
                left_vector = self._normalize_joint_vector(
                    left_rows[frame_index] if frame_index < len(left_rows) else None,
                    dim=left_dim,
                )
                positions[frame_index, :left_dim] = left_vector
            if right_dim > 0:
                right_vector = self._normalize_joint_vector(
                    right_rows[frame_index] if frame_index < len(right_rows) else None,
                    dim=right_dim,
                )
                positions[frame_index, left_dim : left_dim + right_dim] = right_vector

        intrinsics = self._resolve_intrinsics(payload=payload, meta_data=meta_data)
        extrinsics = self._resolve_camera_extrinsics(payload=payload, meta_data=meta_data)
        source_h, source_w = self._resolve_source_image_size(payload=payload, meta_data=meta_data)
        left_gripper_signal = self._resolve_gripper_signal(
            values=payload.get("left_gripper_signal", []),
            frame_count=frame_count,
        )
        right_gripper_signal = self._resolve_gripper_signal(
            values=payload.get("right_gripper_signal", []),
            frame_count=frame_count,
        )

        return TrajectoryPayload(
            joint_names=left_joint_names + right_joint_names,
            joint_positions=positions,
            left_gripper_signal=left_gripper_signal,
            right_gripper_signal=right_gripper_signal,
            intrinsics=intrinsics,
            camera_extrinsics=extrinsics,
            source_image_height=source_h,
            source_image_width=source_w,
        )

    @staticmethod
    def _normalize_joint_names(value: Any, *, prefix: str) -> list[str]:
        if not isinstance(value, list):
            return []
        names: list[str] = []
        for index, item in enumerate(value):
            if isinstance(item, str) and item:
                names.append(item)
            else:
                names.append(f"{prefix}{index + 1}")
        return names

    @staticmethod
    def _normalize_joint_vector(row: Any, *, dim: int) -> np.ndarray:
        if row is None:
            return np.zeros((dim,), dtype=np.float32)
        array = np.asarray(row, dtype=np.float32).reshape(-1)
        if array.size == dim:
            return array
        if array.size > dim:
            return np.asarray(array[:dim], dtype=np.float32)
        output = np.zeros((dim,), dtype=np.float32)
        output[: array.size] = array
        return output

    @staticmethod
    def _first_valid_entry(values: Any) -> Any:
        if not isinstance(values, list):
            return None
        for item in values:
            if item is None:
                continue
            return item
        return None

    def _resolve_intrinsics(self, *, payload: dict[str, Any], meta_data: dict[str, Any]) -> np.ndarray:
        camera_meta = meta_data.get("camera")
        camera_map = camera_meta if isinstance(camera_meta, dict) else {}
        candidates = [
            self._first_valid_entry(payload.get("intrinsics", [])),
            meta_data.get("intrinsics"),
            meta_data.get("camera_intrinsics"),
            camera_map.get("intrinsics"),
            camera_map.get("intrinsic"),
            camera_map.get("K"),
        ]
        for candidate in candidates:
            if candidate is None:
                continue
            try:
                return self._reshape_intrinsics(candidate)
            except ValueError:
                continue
        raise ValueError("render process failed to resolve camera intrinsics from parquet/meta_data")

    def _resolve_camera_extrinsics(self, *, payload: dict[str, Any], meta_data: dict[str, Any]) -> np.ndarray:
        camera_meta = meta_data.get("camera")
        camera_map = camera_meta if isinstance(camera_meta, dict) else {}
        candidates = [
            self._first_valid_entry(payload.get("camera_extrinsics", [])),
            meta_data.get("camera_extrinsics"),
            meta_data.get("camera_extrinsic"),
            camera_map.get("camera_extrinsics"),
            camera_map.get("extrinsics"),
            camera_map.get("extrinsic"),
            camera_map.get("pose"),
            camera_map.get("matrix"),
        ]
        for candidate in candidates:
            if candidate is None:
                continue
            try:
                return self._reshape_extrinsics(candidate)
            except ValueError:
                continue
        raise ValueError("render process failed to resolve camera_extrinsics from parquet/meta_data")

    def _resolve_source_image_size(
        self,
        *,
        payload: dict[str, Any],
        meta_data: dict[str, Any],
    ) -> tuple[int | None, int | None]:
        image_h = self._safe_int(self._first_valid_entry(payload.get("image_height", [])))
        image_w = self._safe_int(self._first_valid_entry(payload.get("image_width", [])))
        if image_h and image_w:
            return image_h, image_w

        image_h = self._safe_int(meta_data.get("image_height"))
        image_w = self._safe_int(meta_data.get("image_width"))
        if image_h and image_w:
            return image_h, image_w

        camera_meta = meta_data.get("camera")
        if isinstance(camera_meta, dict):
            img_size = camera_meta.get("img_size")
            if isinstance(img_size, list) and len(img_size) >= 2:
                width = self._safe_int(img_size[0])
                height = self._safe_int(img_size[1])
                if height and width:
                    return height, width
        return None, None

    @staticmethod
    def _resolve_gripper_signal(*, values: Any, frame_count: int) -> np.ndarray | None:
        if frame_count <= 0 or not isinstance(values, list):
            return None
        if len(values) <= 0:
            return None

        out = np.zeros((frame_count,), dtype=np.float32)
        has_value = False
        last_value = 0.0
        for idx in range(frame_count):
            raw = values[idx] if idx < len(values) else None
            if raw is None:
                out[idx] = last_value
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError):
                out[idx] = last_value
                continue
            if not np.isfinite(value):
                out[idx] = last_value
                continue
            has_value = True
            last_value = float(np.clip(value, 0.0, 1.0))
            out[idx] = last_value
        return out if has_value else None

    @staticmethod
    def _reshape_intrinsics(value: Any) -> np.ndarray:
        if isinstance(value, dict):
            nested = value.get("intrinsics") or value.get("intrinsic") or value.get("K")
            if nested is not None:
                value = nested
            elif {"fx", "fy", "cx", "cy"}.issubset(value.keys()):
                fx = float(value["fx"])
                fy = float(value["fy"])
                cx = float(value["cx"])
                cy = float(value["cy"])
                return np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
        matrix = np.asarray(value, dtype=np.float32).reshape(-1)
        if matrix.size != 9:
            raise ValueError("invalid intrinsics, expected 9 values")
        intrinsics = matrix.reshape(3, 3)
        if float(intrinsics[2, 2]) == 0.0:
            intrinsics[2, 2] = 1.0
        return intrinsics

    @staticmethod
    def _reshape_extrinsics(value: Any) -> np.ndarray:
        if isinstance(value, dict):
            nested = (
                value.get("camera_extrinsics")
                or value.get("extrinsics")
                or value.get("matrix")
                or value.get("pose")
            )
            if nested is not None:
                value = nested
            elif {"rotation", "translation"}.issubset(value.keys()):
                rotation = np.asarray(value["rotation"], dtype=np.float32).reshape(3, 3)
                translation = np.asarray(value["translation"], dtype=np.float32).reshape(3)
                matrix = np.eye(4, dtype=np.float32)
                matrix[:3, :3] = rotation
                matrix[:3, 3] = translation
                return matrix
        matrix = np.asarray(value, dtype=np.float32).reshape(-1)
        if matrix.size != 16:
            raise ValueError("invalid camera_extrinsics, expected 16 values")
        return matrix.reshape(4, 4)

    @staticmethod
    def _compute_target_size(*, source_h: int, source_w: int, target_short_side: int) -> tuple[int, int]:
        if source_h <= 0 or source_w <= 0:
            raise ValueError("invalid source frame size")
        if target_short_side <= 0:
            raise ValueError("target_short_side must be positive")
        scale = float(target_short_side) / float(min(source_h, source_w))
        target_h = max(1, int(round(source_h * scale)))
        target_w = max(1, int(round(source_w * scale)))
        return RenderProcess._ensure_even_frame_size(height=target_h, width=target_w)

    @staticmethod
    def _ensure_even_frame_size(*, height: int, width: int) -> tuple[int, int]:
        h = max(2, int(height))
        w = max(2, int(width))
        if (h % 2) != 0:
            h += 1
        if (w % 2) != 0:
            w += 1
        return h, w

    @staticmethod
    def _scale_intrinsics(
        intrinsics: np.ndarray,
        *,
        source_h: int,
        source_w: int,
        target_h: int,
        target_w: int,
    ) -> np.ndarray:
        if source_h <= 0 or source_w <= 0:
            raise ValueError("invalid source image size for intrinsics scaling")
        scaled = np.asarray(intrinsics, dtype=np.float32).copy()
        sx = float(target_w) / float(source_w)
        sy = float(target_h) / float(source_h)
        scaled[0, 0] *= sx
        scaled[1, 1] *= sy
        scaled[0, 2] *= sx
        scaled[1, 2] *= sy
        return scaled

    @staticmethod
    def _overlay_robot_rgb(
        *,
        frame_bgr: np.ndarray,
        robot_rgb: np.ndarray,
        mask: np.ndarray,
        output_h: int,
        output_w: int,
    ) -> np.ndarray:
        background = np.asarray(frame_bgr, dtype=np.uint8)
        robot_rgb_local = np.asarray(robot_rgb, dtype=np.uint8)
        if robot_rgb_local.ndim != 3 or robot_rgb_local.shape[2] < 3:
            raise ValueError("invalid robot rgb frame from renderer")
        robot_bgr = cv2.cvtColor(robot_rgb_local[:, :, :3], cv2.COLOR_RGB2BGR)
        mask_bool = np.asarray(mask, dtype=bool)

        if robot_bgr.shape[:2] != background.shape[:2]:
            robot_bgr = cv2.resize(robot_bgr, (background.shape[1], background.shape[0]), interpolation=cv2.INTER_LINEAR)
        if mask_bool.shape != background.shape[:2]:
            mask_bool = cv2.resize(
                mask_bool.astype(np.uint8),
                (background.shape[1], background.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        output = background.copy()
        if bool(mask_bool.any()):
            output[mask_bool] = robot_bgr[mask_bool]

        if output.shape[0] != output_h or output.shape[1] != output_w:
            interpolation = (
                cv2.INTER_AREA
                if output.shape[0] >= output_h and output.shape[1] >= output_w
                else cv2.INTER_LINEAR
            )
            output = cv2.resize(output, (output_w, output_h), interpolation=interpolation)
        return output

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(number):
            return None
        return number

    @staticmethod
    def _safe_int(value: Any) -> int | None:
        try:
            number = int(value)
        except (TypeError, ValueError):
            return None
        if number <= 0:
            return None
        return number


class _GenesisSegmentationRenderer:
    _initialized = False
    _GRIPPER_OPEN_MAX = 0.04

    def __init__(
        self,
        *,
        urdf_path: str,
        joint_names: list[str],
        width: int,
        height: int,
        default_fov_deg: float,
        init_backend: str,
    ) -> None:
        try:
            import genesis as gs
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "genesis is required for render process. Install it with `python -m pip install genesis-world`."
            ) from exc

        self._gs = gs
        self._joint_names = list(joint_names)
        self._width = int(width)
        self._height = int(height)
        self._default_fov_deg = float(default_fov_deg)
        self._init_backend = str(init_backend).lower()
        self._ensure_initialized()
        self._scene = self._create_scene()
        self._robot = self._scene.add_entity(
            self._gs.morphs.URDF(file=str(Path(urdf_path).resolve()), fixed=True)
        )
        self._controlled_dofs_idx_local = self._resolve_control_indices(index_type="dofs")
        self._controlled_qs_idx_local = self._resolve_control_indices(index_type="qs")
        self._left_gripper_dofs_idx_local = self._resolve_named_control_indices(
            joint_names=["left_joint7", "left_joint8"],
            index_type="dofs",
        )
        self._right_gripper_dofs_idx_local = self._resolve_named_control_indices(
            joint_names=["right_joint7", "right_joint8"],
            index_type="dofs",
        )
        self._left_gripper_qs_idx_local = self._resolve_named_control_indices(
            joint_names=["left_joint7", "left_joint8"],
            index_type="qs",
        )
        self._right_gripper_qs_idx_local = self._resolve_named_control_indices(
            joint_names=["right_joint7", "right_joint8"],
            index_type="qs",
        )
        self._camera = self._scene.add_camera(
            res=(self._width, self._height),
            pos=(1.2, 0.0, 0.8),
            lookat=(0.0, 0.0, 0.2),
            up=(0.0, 0.0, 1.0),
            fov=self._default_fov_deg,
            GUI=False,
        )
        self._scene.build(n_envs=1)

    def render_rgb_and_mask(
        self,
        *,
        joint_positions: np.ndarray,
        left_gripper_signal: float | None,
        right_gripper_signal: float | None,
        camera_extrinsics: np.ndarray,
        intrinsics: np.ndarray,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        self._set_joint_positions(np.asarray(joint_positions, dtype=np.float32).reshape(-1))
        self._set_gripper_positions(
            left_gripper_signal=left_gripper_signal,
            right_gripper_signal=right_gripper_signal,
        )
        self._set_camera_pose(camera_extrinsics=np.asarray(camera_extrinsics, dtype=np.float32))
        self._set_camera_intrinsics(np.asarray(intrinsics, dtype=np.float32))
        rgb_raw, seg_raw = self._render_rgb_and_segmentation_raw()
        rgb = self._normalize_render_rgb(rgb_raw)
        mask = self._normalize_segmentation_mask(seg_raw)
        return rgb, mask

    def close(self) -> None:
        if self._scene is None:
            return
        self._scene.destroy()
        self._scene = None
        self._camera = None
        self._robot = None

    def _ensure_initialized(self) -> None:
        if _GenesisSegmentationRenderer._initialized:
            return
        backend = self._gs.gpu if "gpu" in self._init_backend else self._gs.cpu
        self._gs.init(backend=backend)
        _GenesisSegmentationRenderer._initialized = True

    def _create_scene(self) -> Any:
        return self._gs.Scene(
            sim_options=self._gs.options.SimOptions(gravity=(0.0, 0.0, 0.0)),
            rigid_options=self._gs.options.RigidOptions(
                gravity=(0.0, 0.0, 0.0),
                enable_collision=False,
                enable_joint_limit=False,
                enable_self_collision=False,
                enable_neutral_collision=False,
                enable_adjacent_collision=False,
                disable_constraint=True,
            ),
            vis_options=self._gs.options.VisOptions(segmentation_level="entity"),
            renderer=self._gs.options.renderers.Rasterizer(),
            show_viewer=False,
        )

    def _set_joint_positions(self, values: np.ndarray) -> None:
        vector = np.asarray(values, dtype=np.float32).reshape(-1)

        controlled_dofs = list(self._controlled_dofs_idx_local)
        if controlled_dofs and vector.size == len(controlled_dofs):
            self._robot.set_dofs_position(vector, dofs_idx_local=controlled_dofs, zero_velocity=False)
            return

        controlled_qs = list(self._controlled_qs_idx_local)
        if controlled_qs and vector.size == len(controlled_qs):
            self._robot.set_qpos(vector, qs_idx_local=controlled_qs, zero_velocity=False, skip_forward=False)
            return

        n_dofs = int(self._robot.n_dofs)
        if n_dofs > 0 and vector.size == n_dofs:
            self._robot.set_dofs_position(vector, zero_velocity=False)
            return

        n_qs = int(self._robot.n_qs)
        if n_qs > 0 and vector.size == n_qs:
            self._robot.set_qpos(vector, zero_velocity=False, skip_forward=False)
            return

        raise RuntimeError(
            "failed to set robot joint positions: "
            f"input_dim={vector.size}, controlled_dofs={len(controlled_dofs)}, "
            f"controlled_qs={len(controlled_qs)}, robot_n_dofs={n_dofs}, robot_n_qs={n_qs}"
        )

    def _resolve_control_indices(self, *, index_type: str) -> list[int]:
        if index_type not in {"dofs", "qs"}:
            return []

        joint_name_map: dict[str, Any] = {}
        for joint in self._robot.joints:
            name = joint.name
            normalized = self._normalize_joint_name(name)
            if normalized and normalized not in joint_name_map:
                joint_name_map[normalized] = joint

        collected: list[int] = []
        unresolved: list[str] = []
        for joint_name in self._joint_names:
            if not isinstance(joint_name, str) or not joint_name:
                continue
            try:
                joint = self._robot.get_joint(name=joint_name)
            except Exception:
                normalized = self._normalize_joint_name(joint_name)
                joint = joint_name_map.get(normalized)
                if joint is None:
                    unresolved.append(joint_name)
                    continue
            indices = joint.dofs_idx_local if index_type == "dofs" else joint.qs_idx_local
            for index in indices:
                try:
                    value = int(index)
                except (TypeError, ValueError):
                    continue
                collected.append(value)

        if not collected:
            if unresolved:
                print(
                    "[render] warning: unresolved joint names for control indices "
                    f"({index_type}): {unresolved}"
                )
            return []
        if unresolved:
            print(
                "[render] warning: partially unresolved joint names for control indices "
                f"({index_type}): {unresolved}"
            )
        # Keep order, remove duplicates.
        return list(dict.fromkeys(collected))

    def _resolve_named_control_indices(
        self,
        *,
        joint_names: list[str],
        index_type: str,
    ) -> list[int]:
        if index_type not in {"dofs", "qs"}:
            return []
        collected: list[int] = []
        for joint_name in joint_names:
            if not isinstance(joint_name, str) or not joint_name:
                continue
            try:
                joint = self._robot.get_joint(name=joint_name)
            except Exception:
                continue
            indices = joint.dofs_idx_local if index_type == "dofs" else joint.qs_idx_local
            for index in indices:
                try:
                    value = int(index)
                except (TypeError, ValueError):
                    continue
                collected.append(value)
        return list(dict.fromkeys(collected))

    def _set_gripper_positions(
        self,
        *,
        left_gripper_signal: float | None,
        right_gripper_signal: float | None,
    ) -> None:
        left_pair = self._build_gripper_pair(left_gripper_signal)
        right_pair = self._build_gripper_pair(right_gripper_signal)

        if left_pair is not None:
            self._set_joint_pair(
                values=left_pair,
                dofs_idx_local=self._left_gripper_dofs_idx_local,
                qs_idx_local=self._left_gripper_qs_idx_local,
            )
        if right_pair is not None:
            self._set_joint_pair(
                values=right_pair,
                dofs_idx_local=self._right_gripper_dofs_idx_local,
                qs_idx_local=self._right_gripper_qs_idx_local,
            )

    @classmethod
    def _build_gripper_pair(cls, signal: float | None) -> np.ndarray | None:
        if signal is None:
            return None
        try:
            value = float(signal)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(value):
            return None
        opening = float(np.clip(value, 0.0, 1.0)) * float(cls._GRIPPER_OPEN_MAX)
        return np.asarray([opening, -opening], dtype=np.float32)

    def _set_joint_pair(
        self,
        *,
        values: np.ndarray,
        dofs_idx_local: list[int],
        qs_idx_local: list[int],
    ) -> None:
        vector = np.asarray(values, dtype=np.float32).reshape(-1)
        if len(dofs_idx_local) == vector.size:
            self._robot.set_dofs_position(vector, dofs_idx_local=dofs_idx_local, zero_velocity=False)
            return
        if len(qs_idx_local) == vector.size:
            self._robot.set_qpos(vector, qs_idx_local=qs_idx_local, zero_velocity=False, skip_forward=False)
            return

    @staticmethod
    def _normalize_joint_name(name: str) -> str:
        return "".join(ch for ch in str(name).lower() if ch.isalnum())

    def _set_camera_pose(self, *, camera_extrinsics: np.ndarray) -> None:
        world_from_camera_cv = np.asarray(camera_extrinsics, dtype=np.float32).reshape(4, 4)
        cv_to_gl = np.asarray(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        world_from_camera_gl = world_from_camera_cv @ cv_to_gl
        self._camera.set_pose(transform=world_from_camera_gl)

    def _set_camera_intrinsics(self, intrinsics: np.ndarray) -> None:
        intr = np.asarray(intrinsics, dtype=np.float32).reshape(3, 3)
        fy = float(intr[1, 1])
        fov_y = float(np.degrees(2.0 * np.arctan((float(self._height) / 2.0) / max(fy, 1e-6))))
        self._camera._fov = fov_y
        for cache_key in ("projection_matrix", "intrinsics", "f", "cx", "cy"):
            self._camera.__dict__.pop(cache_key, None)

    def _render_rgb_and_segmentation_raw(self) -> tuple[Any | None, Any | None]:
        rgb, _, segmentation, _ = self._camera.render(
            rgb=True,
            depth=False,
            segmentation=True,
            colorize_seg=False,
            normal=False,
            force_render=True,
        )
        return rgb, segmentation

    @staticmethod
    def _normalize_render_rgb(value: Any) -> np.ndarray | None:
        if value is None:
            return None
        rgb = np.asarray(value)
        if rgb.ndim == 4:
            rgb = rgb[0]
        if rgb.ndim != 3 or rgb.shape[2] < 3:
            return None
        rgb = rgb[:, :, :3]
        if rgb.dtype == np.uint8:
            return rgb
        rgb_float = np.asarray(rgb, dtype=np.float32)
        max_value = float(np.nanmax(rgb_float)) if rgb_float.size > 0 else 0.0
        if max_value <= 1.0 + 1e-6:
            rgb_float = np.clip(rgb_float, 0.0, 1.0) * 255.0
        else:
            rgb_float = np.clip(rgb_float, 0.0, 255.0)
        return rgb_float.astype(np.uint8)

    @staticmethod
    def _normalize_segmentation_mask(value: Any) -> np.ndarray | None:
        if value is None:
            return None
        mask = np.asarray(value)
        if mask.ndim == 4:
            mask = mask[0]
        if mask.ndim == 3:
            if mask.shape[2] == 1:
                mask = mask[:, :, 0]
            else:
                mask = np.any(mask != 0, axis=2)
        if mask.ndim != 2:
            return None
        return np.asarray(mask != 0, dtype=bool)

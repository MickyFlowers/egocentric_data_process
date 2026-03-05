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
        threshold = float(self.params.get("reachable_ratio_threshold", 0.9))
        meta_data = self._resolve_meta_data(sample)

        left_ratio = self._safe_float(meta_data.get("left_reachable_ratio"))
        right_ratio = self._safe_float(meta_data.get("right_reachable_ratio"))
        passed = (
            left_ratio is not None
            and right_ratio is not None
            and left_ratio > threshold
            and right_ratio > threshold
        )

        sample["render_filter_passed"] = passed
        sample["render_filter_threshold"] = threshold
        sample["render_filter_left_ratio"] = left_ratio
        sample["render_filter_right_ratio"] = right_ratio
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
    intrinsics: np.ndarray  # [3, 3]
    camera_extrinsics: np.ndarray  # [4, 4]

    @property
    def frame_count(self) -> int:
        return int(self.joint_positions.shape[0])


@register_process("render")
class RenderProcess(BaseProcess):
    RENDER_DIR = "render"

    def run(self, sample: dict[str, Any], context: Any) -> dict[str, Any]:
        if not bool(sample.get("render_filter_passed", True)):
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
        target_h, target_w = self._compute_target_size(
            source_h=source_h,
            source_w=source_w,
            target_short_side=int(self.params.get("target_short_side", 512)),
        )
        intrinsics_scaled = self._scale_intrinsics(
            trajectory.intrinsics,
            source_h=source_h,
            source_w=source_w,
            target_h=target_h,
            target_w=target_w,
        )
        fps = self._safe_float(meta_data.get("fps")) or 30.0

        renderer = _GenesisSegmentationRenderer(
            urdf_path=urdf_path,
            joint_names=trajectory.joint_names,
            width=target_w,
            height=target_h,
            default_fov_deg=float(self.params.get("default_fov_deg", 55.0)),
            init_backend=str(self.params.get("genesis_backend", "cpu")),
        )

        render_root = self.extend_output_dir(output_dir, self.RENDER_DIR)
        render_local_path, render_remote_path = self.build_output_paths(
            sample,
            output_dir=render_root,
            extension=".mp4",
        )
        blend_alpha = float(self.params.get("overlay_alpha", 0.4))
        overlay_color = np.asarray(self.params.get("overlay_color_bgr", [0, 200, 255]), dtype=np.uint8).reshape(3)

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
            try:
                for frame_index, frame in enumerate(reader):
                    if frame_index >= trajectory.frame_count:
                        break
                    frame_bgr = cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR)
                    frame_resized = cv2.resize(frame_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)

                    segmentation = renderer.render_segmentation(
                        joint_positions=trajectory.joint_positions[frame_index],
                        camera_extrinsics=trajectory.camera_extrinsics,
                        intrinsics=intrinsics_scaled,
                    )
                    if segmentation is None:
                        segmentation = last_mask
                    else:
                        last_mask = segmentation

                    blended = self._overlay_mask(
                        frame_bgr=frame_resized,
                        mask=segmentation,
                        color_bgr=overlay_color,
                        alpha=blend_alpha,
                    )
                    writer.append_data(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
            finally:
                writer.close()
                reader.close()

        sample["render_video_path"] = render_remote_path
        sample["render_success"] = True
        sample["last_process"] = self.name
        return sample

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

        table = pq.read_table(
            parquet_path,
            columns=[
                "left_joint_position",
                "right_joint_position",
                "intrinsics",
                "camera_extrinsics",
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

        intrinsics_values = self._first_valid_entry(payload.get("intrinsics", []))
        if intrinsics_values is None:
            intrinsics_values = meta_data.get("intrinsics")
        intrinsics = self._reshape_intrinsics(intrinsics_values)

        extrinsics_values = self._first_valid_entry(payload.get("camera_extrinsics", []))
        if extrinsics_values is None:
            extrinsics_values = meta_data.get("camera_extrinsics")
        extrinsics = self._reshape_extrinsics(extrinsics_values)

        return TrajectoryPayload(
            joint_names=left_joint_names + right_joint_names,
            joint_positions=positions,
            intrinsics=intrinsics,
            camera_extrinsics=extrinsics,
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

    @staticmethod
    def _reshape_intrinsics(value: Any) -> np.ndarray:
        matrix = np.asarray(value, dtype=np.float32).reshape(-1)
        if matrix.size != 9:
            raise ValueError("invalid intrinsics, expected 9 values")
        intrinsics = matrix.reshape(3, 3)
        if float(intrinsics[2, 2]) == 0.0:
            intrinsics[2, 2] = 1.0
        return intrinsics

    @staticmethod
    def _reshape_extrinsics(value: Any) -> np.ndarray:
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
        return target_h, target_w

    @staticmethod
    def _scale_intrinsics(
        intrinsics: np.ndarray,
        *,
        source_h: int,
        source_w: int,
        target_h: int,
        target_w: int,
    ) -> np.ndarray:
        scaled = np.asarray(intrinsics, dtype=np.float32).copy()
        sx = float(target_w) / max(float(source_w), 1.0)
        sy = float(target_h) / max(float(source_h), 1.0)
        scaled[0, 0] *= sx
        scaled[0, 2] *= sx
        scaled[1, 1] *= sy
        scaled[1, 2] *= sy
        return scaled

    @staticmethod
    def _overlay_mask(
        *,
        frame_bgr: np.ndarray,
        mask: np.ndarray,
        color_bgr: np.ndarray,
        alpha: float,
    ) -> np.ndarray:
        output = np.asarray(frame_bgr, dtype=np.uint8).copy()
        mask_bool = np.asarray(mask, dtype=bool)
        if mask_bool.shape != output.shape[:2]:
            mask_bool = cv2.resize(
                mask_bool.astype(np.uint8),
                (output.shape[1], output.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        if not bool(mask_bool.any()):
            return output
        clamped_alpha = float(np.clip(alpha, 0.0, 1.0))
        base = output[mask_bool].astype(np.float32)
        tint = np.asarray(color_bgr, dtype=np.float32).reshape(1, 3)
        output[mask_bool] = np.clip(base * (1.0 - clamped_alpha) + tint * clamped_alpha, 0.0, 255.0).astype(
            np.uint8
        )
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


class _GenesisSegmentationRenderer:
    _initialized = False

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
        self._robot = self._add_urdf_entity(Path(urdf_path).resolve())
        self._build_scene()
        self._camera = self._add_camera()

    def render_segmentation(
        self,
        *,
        joint_positions: np.ndarray,
        camera_extrinsics: np.ndarray,
        intrinsics: np.ndarray,
    ) -> np.ndarray | None:
        self._set_joint_positions(np.asarray(joint_positions, dtype=np.float32).reshape(-1))
        self._set_camera_pose(camera_extrinsics=np.asarray(camera_extrinsics, dtype=np.float32))
        self._set_camera_intrinsics(np.asarray(intrinsics, dtype=np.float32))
        step_fn = getattr(self._scene, "step", None)
        if callable(step_fn):
            step_fn()
        raw = self._render_segmentation_raw()
        if raw is None:
            return None
        mask = np.asarray(raw)
        if mask.ndim == 3:
            if mask.shape[2] == 1:
                mask = mask[:, :, 0]
            else:
                mask = np.any(mask != 0, axis=2)
        return np.asarray(mask != 0, dtype=bool)

    def _ensure_initialized(self) -> None:
        if _GenesisSegmentationRenderer._initialized:
            return
        init_fn = getattr(self._gs, "init", None)
        if not callable(init_fn):
            raise RuntimeError("genesis.init is not available")
        backend = None
        if "gpu" in self._init_backend and hasattr(self._gs, "gpu"):
            backend = getattr(self._gs, "gpu")
        elif hasattr(self._gs, "cpu"):
            backend = getattr(self._gs, "cpu")

        if backend is None:
            init_fn()
        else:
            try:
                init_fn(backend=backend)
            except TypeError:
                init_fn()
        _GenesisSegmentationRenderer._initialized = True

    def _create_scene(self) -> Any:
        scene_cls = getattr(self._gs, "Scene", None)
        if scene_cls is None:
            raise RuntimeError("genesis.Scene is not available")
        for kwargs in (
            {"show_viewer": False},
            {"headless": True},
            {},
        ):
            try:
                return scene_cls(**kwargs)
            except TypeError:
                continue
        return scene_cls()

    def _add_urdf_entity(self, urdf_path: Path) -> Any:
        morphs = getattr(self._gs, "morphs", None)
        urdf_ctor = getattr(morphs, "URDF", None) if morphs is not None else None
        if urdf_ctor is None:
            raise RuntimeError("genesis.morphs.URDF is not available")

        urdf_morph = None
        for kwargs in (
            {"file": str(urdf_path)},
            {"path": str(urdf_path)},
            {"filename": str(urdf_path)},
        ):
            try:
                urdf_morph = urdf_ctor(**kwargs)
                break
            except TypeError:
                continue
        if urdf_morph is None:
            urdf_morph = urdf_ctor(str(urdf_path))
        return self._scene.add_entity(urdf_morph)

    def _build_scene(self) -> None:
        build_fn = getattr(self._scene, "build", None)
        if callable(build_fn):
            try:
                build_fn()
            except TypeError:
                build_fn(n_envs=1)

    def _add_camera(self) -> Any:
        add_camera = getattr(self._scene, "add_camera", None) or getattr(self._scene, "create_camera", None)
        if not callable(add_camera):
            raise RuntimeError("genesis scene has no camera creation method")

        base_kwargs = {
            "pos": [1.2, 0.0, 0.8],
            "lookat": [0.0, 0.0, 0.2],
            "up": [0.0, 0.0, 1.0],
            "fov": self._default_fov_deg,
        }
        attempts = (
            {"res": (self._width, self._height), **base_kwargs},
            {"width": self._width, "height": self._height, **base_kwargs},
            {"res": (self._width, self._height)},
            {"width": self._width, "height": self._height},
        )
        for kwargs in attempts:
            try:
                return add_camera(**kwargs)
            except TypeError:
                continue
        return add_camera()

    def _set_joint_positions(self, values: np.ndarray) -> None:
        for method_name, kwargs in (
            ("set_qpos", {"joint_names": self._joint_names}),
            ("set_qpos", {}),
            ("set_dofs_position", {"joint_names": self._joint_names}),
            ("set_dofs_position", {}),
            ("control_dofs_position", {"joint_names": self._joint_names}),
            ("control_dofs_position", {}),
        ):
            method = getattr(self._robot, method_name, None)
            if not callable(method):
                continue
            try:
                method(values.tolist(), **kwargs)
                return
            except TypeError:
                try:
                    method(values, **kwargs)
                    return
                except TypeError:
                    continue
        raise RuntimeError("failed to set robot joint positions in genesis renderer")

    def _set_camera_pose(self, *, camera_extrinsics: np.ndarray) -> None:
        world_from_camera = np.asarray(camera_extrinsics, dtype=np.float32).reshape(4, 4)
        rotation = world_from_camera[:3, :3]
        eye = world_from_camera[:3, 3]
        forward = rotation[:, 2]
        up = -rotation[:, 1]
        lookat = eye + forward

        camera = self._camera
        for method_name, kwargs in (
            ("set_pose", {"pos": eye.tolist(), "lookat": lookat.tolist(), "up": up.tolist()}),
            ("set_pose", {"position": eye.tolist(), "lookat": lookat.tolist(), "up": up.tolist()}),
            ("set_extrinsics", {"extrinsics": world_from_camera}),
            ("set_transform", {"matrix": world_from_camera}),
        ):
            method = getattr(camera, method_name, None)
            if not callable(method):
                continue
            try:
                method(**kwargs)
                return
            except TypeError:
                continue

    def _set_camera_intrinsics(self, intrinsics: np.ndarray) -> None:
        intr = np.asarray(intrinsics, dtype=np.float32).reshape(3, 3)
        fx = float(intr[0, 0])
        fy = float(intr[1, 1])
        cx = float(intr[0, 2])
        cy = float(intr[1, 2])

        camera = self._camera
        for method_name, kwargs in (
            (
                "set_intrinsics",
                {
                    "fx": fx,
                    "fy": fy,
                    "cx": cx,
                    "cy": cy,
                    "width": self._width,
                    "height": self._height,
                },
            ),
            ("set_intrinsics", {"K": intr, "width": self._width, "height": self._height}),
            ("set_K", {"K": intr}),
            ("set_projection_from_intrinsics", {"K": intr, "width": self._width, "height": self._height}),
        ):
            method = getattr(camera, method_name, None)
            if not callable(method):
                continue
            try:
                method(**kwargs)
                return
            except TypeError:
                continue

    def _render_segmentation_raw(self) -> Any:
        camera = self._camera
        render_fn = getattr(camera, "render", None)
        if callable(render_fn):
            for kwargs in (
                {"segmentation": True},
                {"seg": True},
                {"render_segmentation": True},
                {},
            ):
                try:
                    output = render_fn(**kwargs)
                except TypeError:
                    continue
                parsed = self._parse_segmentation_output(output)
                if parsed is not None:
                    return parsed

        render_fn = getattr(self._scene, "render", None)
        if callable(render_fn):
            for kwargs in (
                {"segmentation": True},
                {"seg": True},
                {},
            ):
                try:
                    output = render_fn(**kwargs)
                except TypeError:
                    continue
                parsed = self._parse_segmentation_output(output)
                if parsed is not None:
                    return parsed
        return None

    @staticmethod
    def _parse_segmentation_output(output: Any) -> Any:
        if output is None:
            return None
        if isinstance(output, dict):
            for key in ("segmentation", "seg", "mask", "id"):
                if key in output:
                    return output[key]
            return None
        if isinstance(output, tuple) or isinstance(output, list):
            # Heuristic: segmentation is usually 2D integer map.
            for item in reversed(output):
                array = np.asarray(item)
                if array.ndim == 2:
                    return array
                if array.ndim == 3 and array.shape[2] == 1:
                    return array[:, :, 0]
            return None
        array = np.asarray(output)
        if array.ndim in {2, 3}:
            return array
        return None

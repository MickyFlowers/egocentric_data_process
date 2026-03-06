#!/usr/bin/env python3
"""单条轨迹 Genesis GUI 调试脚本（含相机位姿）。"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import imageio.v2 as imageio
import numpy as np

if __package__ in {None, ""}:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.oss_utils import configure_path_mapping, oss_to_local


@dataclass
class DebugSample:
    sample_id: str
    meta_data_path: Path
    parquet_path: Path
    meta_data: dict[str, Any]


@dataclass
class DebugTrajectory:
    joint_names: list[str]
    joint_positions: np.ndarray  # [T, J]
    left_gripper_signal: np.ndarray | None  # [T]
    right_gripper_signal: np.ndarray | None  # [T]
    intrinsics: np.ndarray  # [3, 3]
    camera_extrinsics: np.ndarray  # [4, 4]
    source_image_height: int | None
    source_image_width: int | None
    fps: float

    @property
    def frame_count(self) -> int:
        return int(self.joint_positions.shape[0])


def _first_valid_entry(values: Any) -> Any:
    if not isinstance(values, list):
        return None
    for item in values:
        if item is None:
            continue
        return item
    return None


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def _safe_int(value: Any) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    if number <= 0:
        return None
    return number


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


def _normalize_joint_names(value: Any, *, prefix: str, fallback_dim: int | None = None) -> list[str]:
    if isinstance(value, list) and value:
        names: list[str] = []
        for idx, item in enumerate(value):
            if isinstance(item, str) and item:
                names.append(item)
            else:
                names.append(f"{prefix}{idx + 1}")
        return names
    if fallback_dim is None:
        return []
    return [f"{prefix}{idx + 1}" for idx in range(fallback_dim)]


def _normalize_joint_vector(row: Any, *, dim: int) -> np.ndarray:
    if row is None:
        return np.zeros((dim,), dtype=np.float32)
    array = np.asarray(row, dtype=np.float32).reshape(-1)
    if array.size == dim:
        return array
    if array.size > dim:
        return np.asarray(array[:dim], dtype=np.float32)
    out = np.zeros((dim,), dtype=np.float32)
    out[: array.size] = array
    return out


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


def _ensure_even_size(height: int, width: int) -> tuple[int, int]:
    h = max(2, int(height))
    w = max(2, int(width))
    if (h % 2) != 0:
        h += 1
    if (w % 2) != 0:
        w += 1
    return h, w


def _compute_short_side_size(source_h: int, source_w: int, target_short_side: int) -> tuple[int, int]:
    if source_h <= 0 or source_w <= 0:
        raise ValueError("invalid source image size")
    if target_short_side <= 0:
        raise ValueError("target_short_side must be positive")
    scale = float(target_short_side) / float(min(source_h, source_w))
    target_h = int(round(float(source_h) * scale))
    target_w = int(round(float(source_w) * scale))
    return _ensure_even_size(target_h, target_w)


def _resolve_camera_size(
    *,
    source_h: int | None,
    source_w: int | None,
    camera_height: int | None,
    camera_width: int | None,
    target_short_side: int,
) -> tuple[int, int]:
    src_h = int(source_h) if source_h is not None else None
    src_w = int(source_w) if source_w is not None else None
    source_valid = bool(src_h and src_h > 0 and src_w and src_w > 0)

    if camera_height is not None and camera_width is not None:
        return _ensure_even_size(int(camera_height), int(camera_width))

    if source_valid and camera_height is not None:
        h = int(camera_height)
        w = int(round(float(h) * float(src_w) / float(src_h)))
        return _ensure_even_size(h, w)
    if source_valid and camera_width is not None:
        w = int(camera_width)
        h = int(round(float(w) * float(src_h) / float(src_w)))
        return _ensure_even_size(h, w)

    if source_valid:
        return _compute_short_side_size(int(src_h), int(src_w), int(target_short_side))

    if camera_height is not None and camera_width is None:
        return _ensure_even_size(int(camera_height), int(round(int(camera_height) * 16.0 / 9.0)))
    if camera_width is not None and camera_height is None:
        return _ensure_even_size(int(round(int(camera_width) * 9.0 / 16.0)), int(camera_width))
    return _ensure_even_size(540, 960)


def _resolve_sample(args: argparse.Namespace) -> DebugSample:
    if args.meta_data_path and args.parquet_path:
        meta_data_path = Path(args.meta_data_path).expanduser().resolve()
        parquet_path = Path(args.parquet_path).expanduser().resolve()
        if not meta_data_path.is_file():
            raise FileNotFoundError(f"meta_data_path not found: {meta_data_path}")
        if not parquet_path.is_file():
            raise FileNotFoundError(f"parquet_path not found: {parquet_path}")
        with open(meta_data_path, "r", encoding="utf-8") as file_obj:
            meta_data = json.load(file_obj)
        if not isinstance(meta_data, dict):
            raise ValueError(f"invalid meta_data payload: {meta_data_path}")
        sample_id = str(meta_data.get("sample_id") or meta_data_path.stem)
        return DebugSample(
            sample_id=sample_id,
            meta_data_path=meta_data_path,
            parquet_path=parquet_path,
            meta_data=meta_data,
        )

    if not args.sample_id:
        raise ValueError("provide either (--meta-data-path and --parquet-path) or --sample-id")

    input_dir = Path(args.input_dir).expanduser().resolve()
    sample_relative = Path(str(args.sample_id))
    meta_data_path = (input_dir / "meta_data" / sample_relative.parent / f"{sample_relative.stem}.json").resolve()
    parquet_path = (input_dir / "data" / sample_relative.parent / f"{sample_relative.stem}.parquet").resolve()

    if not meta_data_path.is_file():
        raise FileNotFoundError(f"meta_data file not found: {meta_data_path}")
    if not parquet_path.is_file():
        raise FileNotFoundError(f"parquet file not found: {parquet_path}")

    with open(meta_data_path, "r", encoding="utf-8") as file_obj:
        meta_data = json.load(file_obj)
    if not isinstance(meta_data, dict):
        raise ValueError(f"invalid meta_data payload: {meta_data_path}")

    return DebugSample(
        sample_id=str(args.sample_id),
        meta_data_path=meta_data_path,
        parquet_path=parquet_path,
        meta_data=meta_data,
    )


def _resolve_intrinsics(payload: dict[str, Any], meta_data: dict[str, Any]) -> np.ndarray:
    camera_meta = meta_data.get("camera")
    camera_map = camera_meta if isinstance(camera_meta, dict) else {}
    candidates = [
        _first_valid_entry(payload.get("intrinsics", [])),
        meta_data.get("intrinsics"),
        meta_data.get("camera_intrinsics"),
        camera_map.get("intrinsics"),
        camera_map.get("intrinsic"),
        camera_map.get("K"),
    ]
    for value in candidates:
        if value is None:
            continue
        try:
            return _reshape_intrinsics(value)
        except ValueError:
            continue
    raise ValueError("failed to resolve camera intrinsics from parquet/meta_data")


def _resolve_extrinsics(payload: dict[str, Any], meta_data: dict[str, Any]) -> np.ndarray:
    camera_meta = meta_data.get("camera")
    camera_map = camera_meta if isinstance(camera_meta, dict) else {}
    candidates = [
        _first_valid_entry(payload.get("camera_extrinsics", [])),
        meta_data.get("camera_extrinsics"),
        meta_data.get("camera_extrinsic"),
        camera_map.get("camera_extrinsics"),
        camera_map.get("extrinsics"),
        camera_map.get("extrinsic"),
        camera_map.get("pose"),
        camera_map.get("matrix"),
    ]
    for value in candidates:
        if value is None:
            continue
        try:
            return _reshape_extrinsics(value)
        except ValueError:
            continue
    raise ValueError("failed to resolve camera extrinsics from parquet/meta_data")


def _resolve_image_size(payload: dict[str, Any], meta_data: dict[str, Any]) -> tuple[int | None, int | None]:
    h = _safe_int(_first_valid_entry(payload.get("image_height", [])))
    w = _safe_int(_first_valid_entry(payload.get("image_width", [])))
    if h and w:
        return h, w

    h = _safe_int(meta_data.get("image_height"))
    w = _safe_int(meta_data.get("image_width"))
    if h and w:
        return h, w

    camera_meta = meta_data.get("camera")
    if isinstance(camera_meta, dict):
        img_size = camera_meta.get("img_size")
        if isinstance(img_size, list) and len(img_size) >= 2:
            w = _safe_int(img_size[0])
            h = _safe_int(img_size[1])
            if h and w:
                return h, w
    return None, None


def _resolve_gripper_signal(values: Any, *, frame_count: int) -> np.ndarray | None:
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


def _load_trajectory(sample: DebugSample) -> DebugTrajectory:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("pyarrow is required. Install with `python -m pip install pyarrow`.") from exc

    schema_names = set(pq.read_schema(sample.parquet_path).names)
    if "left_joint_position" not in schema_names and "right_joint_position" not in schema_names:
        raise ValueError("parquet must contain left_joint_position or right_joint_position")

    table = pq.read_table(
        sample.parquet_path,
        columns=[
            name
            for name in (
                "left_joint_position",
                "right_joint_position",
                "intrinsics",
                "camera_extrinsics",
                "image_height",
                "image_width",
                "fps",
                "left_gripper_signal",
                "right_gripper_signal",
            )
            if name in schema_names
        ],
    )
    payload = table.to_pydict()
    left_rows = list(payload.get("left_joint_position", []))
    right_rows = list(payload.get("right_joint_position", []))

    left_dim_from_data = len(np.asarray(left_rows[0]).reshape(-1)) if left_rows else 0
    right_dim_from_data = len(np.asarray(right_rows[0]).reshape(-1)) if right_rows else 0
    left_joint_names = _normalize_joint_names(
        sample.meta_data.get("left_joint_names"),
        prefix="left_joint",
        fallback_dim=left_dim_from_data if left_dim_from_data > 0 else None,
    )
    right_joint_names = _normalize_joint_names(
        sample.meta_data.get("right_joint_names"),
        prefix="right_joint",
        fallback_dim=right_dim_from_data if right_dim_from_data > 0 else None,
    )
    left_dim = len(left_joint_names)
    right_dim = len(right_joint_names)
    if left_dim <= 0 and right_dim <= 0:
        raise ValueError("failed to resolve joint names/dims from meta_data and parquet")

    frame_count = max(len(left_rows), len(right_rows))
    positions = np.zeros((frame_count, left_dim + right_dim), dtype=np.float32)
    for frame_idx in range(frame_count):
        if left_dim > 0:
            left_vec = _normalize_joint_vector(
                left_rows[frame_idx] if frame_idx < len(left_rows) else None,
                dim=left_dim,
            )
            positions[frame_idx, :left_dim] = left_vec
        if right_dim > 0:
            right_vec = _normalize_joint_vector(
                right_rows[frame_idx] if frame_idx < len(right_rows) else None,
                dim=right_dim,
            )
            positions[frame_idx, left_dim : left_dim + right_dim] = right_vec

    intrinsics = _resolve_intrinsics(payload=payload, meta_data=sample.meta_data)
    extrinsics = _resolve_extrinsics(payload=payload, meta_data=sample.meta_data)
    image_h, image_w = _resolve_image_size(payload=payload, meta_data=sample.meta_data)
    left_gripper_signal = _resolve_gripper_signal(payload.get("left_gripper_signal", []), frame_count=frame_count)
    right_gripper_signal = _resolve_gripper_signal(payload.get("right_gripper_signal", []), frame_count=frame_count)

    fps = _safe_float(sample.meta_data.get("fps"))
    if fps is None:
        fps = _safe_float(_first_valid_entry(payload.get("fps", [])))
    if fps is None or fps <= 0.0:
        fps = 30.0

    return DebugTrajectory(
        joint_names=left_joint_names + right_joint_names,
        joint_positions=positions,
        left_gripper_signal=left_gripper_signal,
        right_gripper_signal=right_gripper_signal,
        intrinsics=intrinsics,
        camera_extrinsics=extrinsics,
        source_image_height=image_h,
        source_image_width=image_w,
        fps=float(fps),
    )


class GenesisGuiDebugger:
    _initialized = False
    _GRIPPER_OPEN_MAX = 0.04

    def __init__(
        self,
        *,
        urdf_path: Path,
        joint_names: list[str],
        camera_width: int,
        camera_height: int,
        backend: str,
        default_fov_deg: float,
    ) -> None:
        try:
            import genesis as gs
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "genesis is required. Install with `python -m pip install genesis-world`."
            ) from exc

        self._gs = gs
        self._joint_names = list(joint_names)
        self._camera_width = int(camera_width)
        self._camera_height = int(camera_height)
        self._default_fov_deg = float(default_fov_deg)
        self._backend = str(backend).lower()
        self._init_once()

        self._scene = self._create_scene()
        self._robot = self._add_urdf_entity(urdf_path.resolve())
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
        self._camera = None

        add_camera_error: Exception | None = None
        try:
            self._camera = self._add_camera()
        except Exception as exc:
            add_camera_error = exc

        self._build_scene()

        if self._camera is None:
            self._camera = self._get_existing_camera()
        if self._camera is None:
            try:
                self._camera = self._add_camera()
            except Exception as exc:
                if add_camera_error is not None:
                    raise RuntimeError(
                        "failed to create debug camera before/after scene.build; "
                        f"before_build_error={add_camera_error}, after_build_error={exc}"
                    ) from exc
                raise
        if self._camera is None:
            raise RuntimeError("failed to create/find camera in GUI scene")

    def step_frame(
        self,
        *,
        joint_positions: np.ndarray,
        left_gripper_signal: float | None,
        right_gripper_signal: float | None,
        camera_extrinsics: np.ndarray,
        intrinsics: np.ndarray,
        viewer_follow_camera: bool,
    ) -> None:
        self._set_joint_positions(np.asarray(joint_positions, dtype=np.float32).reshape(-1))
        self._set_gripper_positions(
            left_gripper_signal=left_gripper_signal,
            right_gripper_signal=right_gripper_signal,
        )
        world_from_camera_gl = self._set_camera_pose(np.asarray(camera_extrinsics, dtype=np.float32))
        self._set_camera_intrinsics(np.asarray(intrinsics, dtype=np.float32))
        if viewer_follow_camera:
            self._set_viewer_pose(world_from_camera_gl)
        self._refresh_visualizer_only()

    def idle(self) -> None:
        self._refresh_visualizer_only()

    def render_rgb_and_mask(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        rgb_raw, seg_raw = self._render_rgb_and_segmentation_raw()
        rgb = self._normalize_rgb(rgb_raw)
        mask = self._normalize_segmentation_mask(seg_raw)
        return rgb, mask

    def render_rgb(self) -> np.ndarray | None:
        rgb, _ = self.render_rgb_and_mask()
        return rgb

    def _render_rgb_and_segmentation_raw(self) -> tuple[Any | None, Any | None]:
        fallback_rgb = None

        camera = self._camera
        render_fn = getattr(camera, "render", None)
        if callable(render_fn):
            for kwargs in (
                {"rgb": True, "depth": False, "segmentation": True, "normal": False},
                {"rgb": True, "segmentation": True},
                {"rgb": True, "depth": False, "segmentation": False, "normal": False},
                {"rgb": True},
                {},
            ):
                try:
                    output = render_fn(**kwargs)
                except TypeError:
                    continue
                rgb, segmentation = self._parse_render_output(output)
                if rgb is not None and fallback_rgb is None:
                    fallback_rgb = rgb
                if segmentation is not None:
                    return (rgb if rgb is not None else fallback_rgb), segmentation

        render_scene = getattr(self._scene, "render", None)
        if callable(render_scene):
            for kwargs in (
                {"rgb": True, "depth": False, "segmentation": True, "normal": False},
                {"rgb": True, "segmentation": True},
                {"rgb": True, "depth": False, "segmentation": False, "normal": False},
                {"rgb": True},
                {},
            ):
                try:
                    output = render_scene(**kwargs)
                except TypeError:
                    continue
                rgb, segmentation = self._parse_render_output(output)
                if rgb is not None and fallback_rgb is None:
                    fallback_rgb = rgb
                if segmentation is not None:
                    return (rgb if rgb is not None else fallback_rgb), segmentation
        return fallback_rgb, None

    def _init_once(self) -> None:
        if GenesisGuiDebugger._initialized:
            return
        init_fn = getattr(self._gs, "init", None)
        if not callable(init_fn):
            raise RuntimeError("genesis.init is not available")
        backend = None
        if "gpu" in self._backend and hasattr(self._gs, "gpu"):
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
        GenesisGuiDebugger._initialized = True

    def _create_scene(self) -> Any:
        scene_cls = getattr(self._gs, "Scene", None)
        if scene_cls is None:
            raise RuntimeError("genesis.Scene is not available")

        option_kwargs: dict[str, Any] = {}
        options = getattr(self._gs, "options", None)
        if options is not None:
            sim_options_cls = getattr(options, "SimOptions", None)
            rigid_options_cls = getattr(options, "RigidOptions", None)
            vis_options_cls = getattr(options, "VisOptions", None)
            viewer_options_cls = getattr(options, "ViewerOptions", None)
            if sim_options_cls is not None:
                try:
                    option_kwargs["sim_options"] = sim_options_cls(gravity=(0.0, 0.0, 0.0))
                except Exception:
                    pass
            if rigid_options_cls is not None:
                try:
                    option_kwargs["rigid_options"] = rigid_options_cls(
                        gravity=(0.0, 0.0, 0.0),
                        enable_collision=False,
                        enable_joint_limit=False,
                        enable_self_collision=False,
                        enable_neutral_collision=False,
                        enable_adjacent_collision=False,
                        disable_constraint=True,
                    )
                except Exception:
                    pass
            if vis_options_cls is not None:
                try:
                    option_kwargs["vis_options"] = vis_options_cls(segmentation_level="entity")
                except Exception:
                    pass
            if viewer_options_cls is not None:
                try:
                    option_kwargs["viewer_options"] = viewer_options_cls()
                except Exception:
                    pass

        attempts = (
            {"show_viewer": True, **option_kwargs},
            {"headless": False, **option_kwargs},
            option_kwargs,
            {"show_viewer": True},
            {"headless": False},
            {},
        )
        for kwargs in attempts:
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
            {"file": str(urdf_path), "fixed": True},
            {"path": str(urdf_path), "fixed": True},
            {"filename": str(urdf_path), "fixed": True},
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
        if not callable(build_fn):
            return
        for kwargs in (
            {"show_viewer": True},
            {},
        ):
            try:
                build_fn(**kwargs)
                return
            except TypeError:
                continue
        try:
            build_fn(n_envs=1)
        except TypeError:
            build_fn()

    def _get_existing_camera(self) -> Any:
        cameras = getattr(self._scene, "cameras", None)
        if isinstance(cameras, list) and cameras:
            return cameras[0]
        if isinstance(cameras, tuple) and cameras:
            return cameras[0]
        get_camera = getattr(self._scene, "get_camera", None)
        if callable(get_camera):
            for camera_index in (0, "0"):
                try:
                    camera = get_camera(camera_index)
                except Exception:
                    continue
                if camera is not None:
                    return camera
        return None

    def _add_camera(self) -> Any:
        add_camera = getattr(self._scene, "add_camera", None) or getattr(self._scene, "create_camera", None)
        if not callable(add_camera):
            raise RuntimeError("scene has no add_camera/create_camera")

        base_kwargs = {
            "pos": [1.2, 0.0, 0.8],
            "lookat": [0.0, 0.0, 0.2],
            "up": [0.0, 0.0, 1.0],
            "fov": self._default_fov_deg,
        }
        attempts = (
            {"res": (self._camera_width, self._camera_height), "GUI": True, **base_kwargs},
            {"res": (self._camera_width, self._camera_height), "gui": True, **base_kwargs},
            {"width": self._camera_width, "height": self._camera_height, "GUI": True, **base_kwargs},
            {"width": self._camera_width, "height": self._camera_height, **base_kwargs},
            {"res": (self._camera_width, self._camera_height), **base_kwargs},
            {"width": self._camera_width, "height": self._camera_height},
            {"res": (self._camera_width, self._camera_height)},
            {},
        )
        for kwargs in attempts:
            try:
                return add_camera(**kwargs)
            except TypeError:
                continue
        return add_camera()

    def _set_joint_positions(self, values: np.ndarray) -> None:
        vector = np.asarray(values, dtype=np.float32).reshape(-1)
        controlled_dofs = list(self._controlled_dofs_idx_local)
        if controlled_dofs and vector.size == len(controlled_dofs):
            method = getattr(self._robot, "set_dofs_position", None)
            if callable(method):
                method(vector, dofs_idx_local=controlled_dofs, zero_velocity=False)
                return

        controlled_qs = list(self._controlled_qs_idx_local)
        if controlled_qs and vector.size == len(controlled_qs):
            method = getattr(self._robot, "set_qpos", None)
            if callable(method):
                method(vector, qs_idx_local=controlled_qs, zero_velocity=False, skip_forward=False)
                return

        n_dofs = int(getattr(self._robot, "n_dofs", -1))
        if n_dofs > 0 and vector.size == n_dofs:
            method = getattr(self._robot, "set_dofs_position", None)
            if callable(method):
                method(vector, zero_velocity=False)
                return

        n_qs = int(getattr(self._robot, "n_qs", -1))
        if n_qs > 0 and vector.size == n_qs:
            method = getattr(self._robot, "set_qpos", None)
            if callable(method):
                method(vector, zero_velocity=False, skip_forward=False)
                return

        raise RuntimeError(
            "failed to set robot joint positions: "
            f"input_dim={vector.size}, controlled_dofs={len(controlled_dofs)}, "
            f"controlled_qs={len(controlled_qs)}, robot_n_dofs={n_dofs}, robot_n_qs={n_qs}"
        )

    def get_joint_feedback(self, *, expected_dim: int | None = None) -> np.ndarray | None:
        controlled_dofs = list(self._controlled_dofs_idx_local)
        if controlled_dofs:
            feedback = self._read_dofs_feedback(controlled_dofs)
            feedback = self._fit_feedback_dim(feedback, expected_dim=expected_dim)
            if feedback is not None:
                return feedback

        controlled_qs = list(self._controlled_qs_idx_local)
        if controlled_qs:
            feedback = self._read_qs_feedback(controlled_qs)
            feedback = self._fit_feedback_dim(feedback, expected_dim=expected_dim)
            if feedback is not None:
                return feedback

        feedback = self._fit_feedback_dim(self._read_dofs_feedback(None), expected_dim=expected_dim)
        if feedback is not None:
            return feedback
        return self._fit_feedback_dim(self._read_qs_feedback(None), expected_dim=expected_dim)

    def _read_dofs_feedback(self, indices: list[int] | None) -> np.ndarray | None:
        method = getattr(self._robot, "get_dofs_position", None)
        if not callable(method):
            return None
        attempts: list[dict[str, Any]] = []
        if indices:
            attempts.extend(
                [
                    {"dofs_idx_local": indices},
                    {"dofs_idx": indices},
                ]
            )
        attempts.append({})
        for kwargs in attempts:
            try:
                raw = method(**kwargs) if kwargs else method()
            except TypeError:
                continue
            vector = self._to_vector(raw)
            if vector is None:
                continue
            if indices:
                if vector.size == len(indices):
                    return vector
                max_index = max(indices)
                if vector.size > max_index:
                    return np.asarray(vector[np.asarray(indices, dtype=np.int64)], dtype=np.float32)
            else:
                return vector
        return None

    def _read_qs_feedback(self, indices: list[int] | None) -> np.ndarray | None:
        method = getattr(self._robot, "get_qpos", None)
        if not callable(method):
            return None
        attempts: list[dict[str, Any]] = []
        if indices:
            attempts.extend(
                [
                    {"qs_idx_local": indices},
                    {"qs_idx": indices},
                ]
            )
        attempts.append({})
        for kwargs in attempts:
            try:
                raw = method(**kwargs) if kwargs else method()
            except TypeError:
                continue
            vector = self._to_vector(raw)
            if vector is None:
                continue
            if indices:
                if vector.size == len(indices):
                    return vector
                max_index = max(indices)
                if vector.size > max_index:
                    return np.asarray(vector[np.asarray(indices, dtype=np.int64)], dtype=np.float32)
            else:
                return vector
        return None

    @staticmethod
    def _fit_feedback_dim(vector: np.ndarray | None, *, expected_dim: int | None) -> np.ndarray | None:
        if vector is None:
            return None
        if expected_dim is None:
            return vector
        if int(expected_dim) <= 0:
            return None
        if vector.size == int(expected_dim):
            return vector
        return None

    @staticmethod
    def _to_vector(value: Any) -> np.ndarray | None:
        if value is None:
            return None
        data = value
        if hasattr(data, "detach"):
            data = data.detach()
        if hasattr(data, "cpu"):
            data = data.cpu()
        try:
            vector = np.asarray(data, dtype=np.float32).reshape(-1)
        except Exception:
            return None
        if vector.size <= 0:
            return None
        return vector

    def _resolve_control_indices(self, *, index_type: str) -> list[int]:
        if index_type not in {"dofs", "qs"}:
            return []
        attr_name = "dofs_idx_local" if index_type == "dofs" else "qs_idx_local"

        joint_name_map: dict[str, Any] = {}
        for joint in list(getattr(self._robot, "joints", [])):
            name = getattr(joint, "name", None)
            if not isinstance(name, str) or not name:
                continue
            normalized = self._normalize_joint_name(name)
            if normalized and normalized not in joint_name_map:
                joint_name_map[normalized] = joint

        collected: list[int] = []
        for joint_name in self._joint_names:
            if not isinstance(joint_name, str) or not joint_name:
                continue
            try:
                joint = self._robot.get_joint(name=joint_name)
            except Exception:
                normalized = self._normalize_joint_name(joint_name)
                joint = joint_name_map.get(normalized)
                if joint is None:
                    continue
            indices = getattr(joint, attr_name, None)
            if isinstance(indices, list):
                for index in indices:
                    try:
                        value = int(index)
                    except (TypeError, ValueError):
                        continue
                    collected.append(value)
        return list(dict.fromkeys(collected))

    def _resolve_named_control_indices(
        self,
        *,
        joint_names: list[str],
        index_type: str,
    ) -> list[int]:
        if index_type not in {"dofs", "qs"}:
            return []
        attr_name = "dofs_idx_local" if index_type == "dofs" else "qs_idx_local"
        collected: list[int] = []
        for joint_name in joint_names:
            if not isinstance(joint_name, str) or not joint_name:
                continue
            try:
                joint = self._robot.get_joint(name=joint_name)
            except Exception:
                continue
            indices = getattr(joint, attr_name, None)
            if isinstance(indices, list):
                for index in indices:
                    try:
                        value = int(index)
                    except (TypeError, ValueError):
                        continue
                    collected.append(value)
        return list(dict.fromkeys(collected))

    @staticmethod
    def _normalize_joint_name(name: str) -> str:
        return "".join(ch for ch in str(name).lower() if ch.isalnum())

    def _set_camera_pose(self, camera_extrinsics: np.ndarray) -> np.ndarray:
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
        rotation = world_from_camera_gl[:3, :3]
        eye = world_from_camera_gl[:3, 3]
        forward = -rotation[:, 2]
        up = rotation[:, 1]
        lookat = eye + forward

        camera = self._camera
        set_pose = getattr(camera, "set_pose", None)
        if callable(set_pose):
            for kwargs in (
                {"transform": world_from_camera_gl},
                {"matrix": world_from_camera_gl},
                {"pose": world_from_camera_gl},
                {"T": world_from_camera_gl},
                {"pos": eye.tolist(), "lookat": lookat.tolist(), "up": up.tolist()},
                {"position": eye.tolist(), "lookat": lookat.tolist(), "up": up.tolist()},
            ):
                try:
                    set_pose(**kwargs)
                    return world_from_camera_gl
                except TypeError:
                    continue
            try:
                set_pose(world_from_camera_gl)
                return world_from_camera_gl
            except TypeError:
                pass

        for method_name, kwargs in (
            ("set_transform", {"matrix": world_from_camera_gl}),
            ("set_extrinsics", {"extrinsics": world_from_camera_gl}),
            ("set_extrinsics", {"matrix": world_from_camera_gl}),
        ):
            method = getattr(camera, method_name, None)
            if not callable(method):
                continue
            try:
                method(**kwargs)
                return world_from_camera_gl
            except TypeError:
                continue
        return world_from_camera_gl

    def _set_camera_intrinsics(self, intrinsics: np.ndarray) -> None:
        intr = np.asarray(intrinsics, dtype=np.float32).reshape(3, 3)
        fx = float(intr[0, 0])
        fy = float(intr[1, 1])
        cx = float(intr[0, 2])
        cy = float(intr[1, 2])
        fov_y = float(np.degrees(2.0 * np.arctan((float(self._camera_height) / 2.0) / max(fy, 1e-6))))
        
        camera = self._camera
        if hasattr(camera, "set_params"):
            try:
                camera.set_params(fov=fov_y, intrinsics=intr)
            except TypeError:
                try:
                    camera.set_params(intrinsics=intr)
                except TypeError:
                    camera.set_params(fov=fov_y)
        

    def _set_viewer_pose(self, world_from_camera_gl: np.ndarray) -> None:
        viewer = getattr(self._scene, "viewer", None)
        if viewer is None:
            return
        rotation = world_from_camera_gl[:3, :3]
        eye = world_from_camera_gl[:3, 3]
        forward = -rotation[:, 2]
        up = rotation[:, 1]
        lookat = eye + forward

        for method_name, kwargs in (
            ("set_camera_pose", {"pos": eye.tolist(), "lookat": lookat.tolist(), "up": up.tolist()}),
            ("set_camera_pose", {"position": eye.tolist(), "lookat": lookat.tolist(), "up": up.tolist()}),
            ("set_pose", {"pos": eye.tolist(), "lookat": lookat.tolist(), "up": up.tolist()}),
            ("look_at", {"eye": eye.tolist(), "target": lookat.tolist(), "up": up.tolist()}),
        ):
            method = getattr(viewer, method_name, None)
            if not callable(method):
                continue
            try:
                method(**kwargs)
                return
            except TypeError:
                continue

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
            method = getattr(self._robot, "set_dofs_position", None)
            if callable(method):
                method(vector, dofs_idx_local=dofs_idx_local, zero_velocity=False)
                return
        if len(qs_idx_local) == vector.size:
            method = getattr(self._robot, "set_qpos", None)
            if callable(method):
                method(vector, qs_idx_local=qs_idx_local, zero_velocity=False, skip_forward=False)
                return

    def _refresh_visualizer_only(self) -> None:
        visualizer = getattr(self._scene, "visualizer", None)
        if visualizer is None:
            return

        update_visual_states = getattr(visualizer, "update_visual_states", None)
        if callable(update_visual_states):
            try:
                update_visual_states(force_render=True)
            except TypeError:
                update_visual_states()

        viewer = getattr(self._scene, "viewer", None)
        if viewer is not None:
            update = getattr(viewer, "update", None)
            if callable(update):
                for kwargs in (
                    {"auto_refresh": True, "force": True},
                    {"auto_refresh": True},
                    {"force": True},
                    {},
                ):
                    try:
                        update(**kwargs)
                        break
                    except TypeError:
                        continue

    @staticmethod
    def _parse_render_output(output: Any) -> tuple[Any | None, Any | None]:
        if output is None:
            return None, None
        if isinstance(output, dict):
            rgb = None
            segmentation = None
            for key in ("rgb", "color", "image"):
                if key in output and output[key] is not None:
                    rgb = output[key]
                    break
            for key in ("segmentation", "seg", "mask", "id"):
                if key in output and output[key] is not None:
                    segmentation = output[key]
                    break
            return rgb, segmentation
        if isinstance(output, tuple) or isinstance(output, list):
            rgb = None
            segmentation = None
            for item in output:
                array = np.asarray(item)
                if rgb is None and array.ndim >= 3 and array.shape[-1] in {3, 4}:
                    rgb = item
                    continue
                if segmentation is None and (array.ndim == 2 or (array.ndim == 3 and array.shape[-1] == 1)):
                    segmentation = item
            return rgb, segmentation
        array = np.asarray(output)
        if array.ndim >= 3 and array.shape[-1] in {3, 4}:
            return output, None
        if array.ndim == 2:
            return None, output
        return None, None

    @staticmethod
    def _normalize_rgb(value: Any) -> np.ndarray | None:
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
        if mask.ndim == 3:
            if mask.shape[0] == 1 and mask.shape[-1] != 1:
                mask = mask[0]
            elif mask.shape[-1] == 1:
                mask = mask[:, :, 0]
            else:
                mask = mask[:, :, 0]
        if mask.ndim != 2:
            return None
        if mask.dtype == np.bool_:
            return mask
        if np.issubdtype(mask.dtype, np.integer):
            return mask > 0
        mask_float = np.asarray(mask, dtype=np.float32)
        if mask_float.size <= 0:
            return None
        max_value = float(np.nanmax(mask_float))
        if max_value <= 1.0 + 1e-6:
            return mask_float > 0.5
        return mask_float > 0.0


def _resolve_output_video_path(
    *,
    output_video: Path | None,
    output_video_dir: Path,
    sample_id: str,
) -> Path:
    if output_video is not None:
        target = output_video.expanduser()
        if target.suffix.lower() != ".mp4":
            target = target.with_suffix(".mp4")
        return target.resolve()
    return (output_video_dir.expanduser().resolve() / f"{sample_id}.mp4").resolve()


def _resolve_source_video_path(sample: DebugSample) -> Path:
    candidates = (
        sample.meta_data.get("video_path"),
        sample.meta_data.get("source_video_path"),
        sample.meta_data.get("video"),
    )
    for value in candidates:
        if not isinstance(value, str) or not value:
            continue
        path = Path(oss_to_local(value)).expanduser().resolve()
        if path.is_file():
            return path
    raise FileNotFoundError("failed to resolve source video path from meta_data")


def _read_video_frame_rgb(reader: Any, frame_idx: int) -> np.ndarray | None:
    try:
        frame = np.asarray(reader.get_data(int(frame_idx)))
    except Exception:
        return None
    if frame.ndim == 2:
        frame = np.stack([frame] * 3, axis=-1)
    if frame.ndim != 3 or frame.shape[2] < 3:
        return None
    return np.asarray(frame[:, :, :3], dtype=np.uint8)


def _overlay_robot_rgb(
    *,
    frame_rgb: np.ndarray,
    robot_rgb: np.ndarray,
    mask: np.ndarray,
    output_h: int,
    output_w: int,
) -> np.ndarray:
    background = np.asarray(frame_rgb, dtype=np.uint8)
    robot = np.asarray(robot_rgb, dtype=np.uint8)
    if robot.ndim != 3 or robot.shape[2] < 3:
        raise ValueError("invalid robot rgb frame")
    mask_bool = np.asarray(mask, dtype=bool)

    if robot.shape[:2] != background.shape[:2]:
        robot = cv2.resize(robot[:, :, :3], (background.shape[1], background.shape[0]), interpolation=cv2.INTER_LINEAR)
    else:
        robot = robot[:, :, :3]
    if mask_bool.shape != background.shape[:2]:
        mask_bool = cv2.resize(
            mask_bool.astype(np.uint8),
            (background.shape[1], background.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

    output = background.copy()
    if bool(mask_bool.any()):
        output[mask_bool] = robot[mask_bool]

    if output.shape[0] != output_h or output.shape[1] != output_w:
        interpolation = (
            cv2.INTER_AREA
            if output.shape[0] >= output_h and output.shape[1] >= output_w
            else cv2.INTER_LINEAR
        )
        output = cv2.resize(output, (int(output_w), int(output_h)), interpolation=interpolation)
    return np.asarray(output, dtype=np.uint8)


def _ensure_video_frame_size(frame_rgb: np.ndarray, *, target_h: int, target_w: int) -> np.ndarray:
    frame = np.asarray(frame_rgb)
    if frame.ndim != 3 or frame.shape[2] < 3:
        raise ValueError(f"invalid render frame shape: {frame.shape}")
    if frame.shape[0] == target_h and frame.shape[1] == target_w:
        return frame[:, :, :3]
    resized = cv2.resize(frame[:, :, :3], (int(target_w), int(target_h)), interpolation=cv2.INTER_LINEAR)
    return np.asarray(resized, dtype=np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Debug one processed trajectory in Genesis GUI with camera pose.",
    )
    parser.add_argument("--input-dir", type=Path, default=Path("./outputs"), help="Processed output root.")
    parser.add_argument("--sample-id", type=str, default=None, help="Sample id under input-dir.")
    parser.add_argument("--meta-data-path", type=Path, default=None, help="Explicit meta_data json path.")
    parser.add_argument("--parquet-path", type=Path, default=None, help="Explicit trajectory parquet path.")
    parser.add_argument("--urdf-path", type=Path, default=None, help="Override URDF path.")
    parser.add_argument("--backend", type=str, default="cpu", choices=["cpu", "gpu"], help="Genesis backend.")
    parser.add_argument("--camera-width", type=int, default=None, help="GUI debug camera width.")
    parser.add_argument("--camera-height", type=int, default=None, help="GUI debug camera height.")
    parser.add_argument(
        "--target-short-side",
        type=int,
        default=512,
        help="When camera size is not fully specified, keep source aspect ratio and scale short side to this value.",
    )
    parser.add_argument("--default-fov-deg", type=float, default=55.0, help="Fallback camera fov.")
    parser.add_argument("--start-frame", type=int, default=0, help="Start frame index.")
    parser.add_argument("--max-frames", type=int, default=0, help="Maximum frames to play, 0 means all.")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride.")
    parser.add_argument("--fps", type=float, default=0.0, help="Playback fps override; <=0 uses sample fps.")
    parser.add_argument(
        "--joint-error-every",
        type=int,
        default=0,
        help="Print per-frame joint feedback absolute error every N frames; 0 disables per-frame logs.",
    )
    parser.add_argument("--loop", action="store_true", help="Loop playback.")
    parser.add_argument("--viewer-follow-camera", action="store_true", help="Move viewer to sample camera pose.")
    parser.add_argument("--print-camera-pose", action="store_true", help="Print camera pose matrix each frame.")
    parser.add_argument("--write-video", action="store_true", help="Write rendered debug video.")
    parser.add_argument("--output-video", type=Path, default=None, help="Explicit output video path (.mp4).")
    parser.add_argument(
        "--output-video-dir",
        type=Path,
        default=Path("./outputs/debug_render"),
        help="Output video dir when --output-video is not provided.",
    )
    parser.add_argument("--video-crf", type=int, default=26, help="Video CRF for libx264.")
    parser.add_argument("--hold", action="store_true", help="Keep GUI alive after playback.")
    parser.add_argument(
        "--path-local-mount",
        type=Path,
        default=None,
        help="Path-mapping local_mount for oss:// paths (optional).",
    )
    parser.add_argument(
        "--path-oss-prefix",
        type=str,
        default="oss://",
        help="Path-mapping oss_prefix for oss:// paths.",
    )
    args = parser.parse_args()

    local_mount = args.path_local_mount.expanduser().resolve() if args.path_local_mount else Path(".").resolve()
    configure_path_mapping(str(local_mount), args.path_oss_prefix)

    sample = _resolve_sample(args)
    trajectory = _load_trajectory(sample)
    if trajectory.frame_count <= 0:
        raise ValueError("trajectory is empty")

    urdf_value = str(args.urdf_path) if args.urdf_path else sample.meta_data.get("ik_urdf_path")
    if not isinstance(urdf_value, str) or not urdf_value:
        raise ValueError("URDF path is required (pass --urdf-path or provide ik_urdf_path in meta_data)")
    urdf_path = Path(oss_to_local(urdf_value)).expanduser().resolve()
    if not urdf_path.is_file():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    source_h = trajectory.source_image_height
    source_w = trajectory.source_image_width
    camera_h, camera_w = _resolve_camera_size(
        source_h=source_h,
        source_w=source_w,
        camera_height=args.camera_height,
        camera_width=args.camera_width,
        target_short_side=int(args.target_short_side),
    )
    intrinsics_source_h = source_h if source_h else camera_h
    intrinsics_source_w = source_w if source_w else camera_w
    intrinsics_scaled = _scale_intrinsics(
        trajectory.intrinsics,
        source_h=intrinsics_source_h,
        source_w=intrinsics_source_w,
        target_h=camera_h,
        target_w=camera_w,
    )

    debugger = GenesisGuiDebugger(
        urdf_path=urdf_path,
        joint_names=trajectory.joint_names,
        camera_width=camera_w,
        camera_height=camera_h,
        backend=str(args.backend),
        default_fov_deg=float(args.default_fov_deg),
    )

    start = max(0, int(args.start_frame))
    stride = max(1, int(args.stride))
    last_frame = trajectory.frame_count
    if int(args.max_frames) > 0:
        last_frame = min(last_frame, start + int(args.max_frames) * stride)
    frame_indices = list(range(start, last_frame, stride))
    if not frame_indices:
        raise ValueError("no frame to play, check start-frame/max-frames/stride")

    play_fps = float(args.fps) if float(args.fps) > 0.0 else float(trajectory.fps)
    play_fps = max(play_fps, 1e-3)
    frame_interval = 1.0 / play_fps
    should_write_video = bool(args.write_video or args.output_video is not None)
    if should_write_video and bool(args.loop):
        raise ValueError("video writing does not support --loop. Disable --loop or disable video writing.")

    video_writer = None
    source_video_reader = None
    source_video_path: Path | None = None
    output_video_path: Path | None = None
    logged_render_frame_size = False
    logged_feedback_unavailable = False
    last_render_rgb = np.zeros((camera_h, camera_w, 3), dtype=np.uint8)
    last_mask = np.zeros((camera_h, camera_w), dtype=bool)
    last_source_frame_rgb = np.zeros((camera_h, camera_w, 3), dtype=np.uint8)
    joint_dim = int(trajectory.joint_positions.shape[1])
    joint_error_abs_sum = np.zeros((joint_dim,), dtype=np.float64)
    joint_error_abs_max = np.zeros((joint_dim,), dtype=np.float64)
    frame_mean_abs_errors: list[float] = []
    frame_max_abs_errors: list[float] = []
    compared_frames = 0
    if should_write_video:
        output_video_path = _resolve_output_video_path(
            output_video=args.output_video,
            output_video_dir=args.output_video_dir,
            sample_id=sample.sample_id,
        )
        source_video_path = _resolve_source_video_path(sample)
        source_video_reader = imageio.get_reader(str(source_video_path))
        output_video_path.parent.mkdir(parents=True, exist_ok=True)
        output_params = ["-preset", "veryfast", "-crf", str(int(np.clip(args.video_crf, 0, 51))), "-pix_fmt", "yuv420p"]
        video_writer = imageio.get_writer(
            str(output_video_path),
            format="FFMPEG",
            fps=play_fps,
            codec="libx264",
            macro_block_size=1,
            quality=None,
            ffmpeg_log_level="error",
            output_params=output_params,
        )

    print(f"sample_id: {sample.sample_id}")
    print(f"meta_data_path: {sample.meta_data_path}")
    print(f"parquet_path: {sample.parquet_path}")
    print(f"urdf_path: {urdf_path}")
    print(f"render_size: {camera_w}x{camera_h}")
    if source_video_path is not None:
        print(f"source_video_path: {source_video_path}")
    print(f"frame_count: {trajectory.frame_count}, play_frames: {len(frame_indices)}, fps: {play_fps:.3f}")
    print("camera_extrinsics (world_from_camera_cv):")
    print(np.asarray(trajectory.camera_extrinsics, dtype=np.float32))

    try:
        while True:
            for frame_idx in frame_indices:
                tic = time.perf_counter()
                debugger.step_frame(
                    joint_positions=trajectory.joint_positions[frame_idx],
                    left_gripper_signal=(
                        float(trajectory.left_gripper_signal[frame_idx])
                        if trajectory.left_gripper_signal is not None
                        else None
                    ),
                    right_gripper_signal=(
                        float(trajectory.right_gripper_signal[frame_idx])
                        if trajectory.right_gripper_signal is not None
                        else None
                    ),
                    camera_extrinsics=trajectory.camera_extrinsics,
                    intrinsics=intrinsics_scaled,
                    viewer_follow_camera=bool(args.viewer_follow_camera),
                )
                if args.print_camera_pose:
                    print(f"[frame={frame_idx}] camera_extrinsics=")
                    print(np.asarray(trajectory.camera_extrinsics, dtype=np.float32))

                commanded = np.asarray(trajectory.joint_positions[frame_idx], dtype=np.float32).reshape(-1)
                feedback = debugger.get_joint_feedback(expected_dim=commanded.size)
                if feedback is None:
                    if not logged_feedback_unavailable:
                        print("joint_feedback: unavailable for current robot backend API")
                        logged_feedback_unavailable = True
                else:
                    abs_error = np.abs(np.asarray(feedback, dtype=np.float32) - commanded)
                    joint_error_abs_sum += np.asarray(abs_error, dtype=np.float64)
                    joint_error_abs_max = np.maximum(joint_error_abs_max, np.asarray(abs_error, dtype=np.float64))
                    frame_mean_abs_errors.append(float(np.mean(abs_error)))
                    frame_max_abs_errors.append(float(np.max(abs_error)))
                    compared_frames += 1
                    report_every = max(0, int(args.joint_error_every))
                    if report_every > 0 and (compared_frames % report_every) == 0:
                        print(
                            f"[frame={frame_idx}] joint_abs_error_mean={float(np.mean(abs_error)):.6e}, "
                            f"joint_abs_error_max={float(np.max(abs_error)):.6e}"
                        )

                if video_writer is not None:
                    render_rgb, segmentation = debugger.render_rgb_and_mask()
                    if render_rgb is None:
                        render_rgb = last_render_rgb
                    else:
                        last_render_rgb = render_rgb
                    if segmentation is None:
                        segmentation = last_mask
                    else:
                        last_mask = segmentation

                    if source_video_reader is None:
                        raise RuntimeError("source video reader is not initialized")
                    source_frame = _read_video_frame_rgb(source_video_reader, frame_idx)
                    if source_frame is None:
                        source_frame = last_source_frame_rgb
                    else:
                        last_source_frame_rgb = source_frame

                    blended = _overlay_robot_rgb(
                        frame_rgb=source_frame,
                        robot_rgb=render_rgb,
                        mask=segmentation,
                        output_h=camera_h,
                        output_w=camera_w,
                    )
                    if not logged_render_frame_size:
                        print(
                            "captured_render_frame_size: "
                            f"{int(render_rgb.shape[1])}x{int(render_rgb.shape[0])}, "
                            f"captured_mask_size: {int(segmentation.shape[1])}x{int(segmentation.shape[0])}"
                        )
                        logged_render_frame_size = True
                    video_frame = _ensure_video_frame_size(
                        np.asarray(blended, dtype=np.uint8),
                        target_h=camera_h,
                        target_w=camera_w,
                    )
                    video_writer.append_data(video_frame)

                elapsed = time.perf_counter() - tic
                sleep_sec = frame_interval - elapsed
                if sleep_sec > 0:
                    time.sleep(sleep_sec)
            if not args.loop:
                break
    finally:
        if video_writer is not None:
            video_writer.close()
        if source_video_reader is not None:
            source_video_reader.close()

    if output_video_path is not None:
        print(f"render_video_path: {output_video_path}")

    if compared_frames > 0:
        joint_mean_abs = joint_error_abs_sum / float(compared_frames)
        global_mean_abs = float(np.mean(joint_mean_abs))
        global_max_abs = float(np.max(joint_error_abs_max))
        frame_p95_abs = float(np.percentile(np.asarray(frame_max_abs_errors, dtype=np.float64), 95.0))
        print("joint_feedback_error_summary:")
        print(
            f"  compared_frames={compared_frames}, "
            f"global_mean_abs={global_mean_abs:.6e}, frame_p95_max_abs={frame_p95_abs:.6e}, global_max_abs={global_max_abs:.6e}"
        )
        top_k = min(6, len(trajectory.joint_names), joint_mean_abs.size)
        if top_k > 0:
            order = np.argsort(-joint_mean_abs)[:top_k]
            print("  top_joint_mean_abs_errors:")
            for index in order.tolist():
                joint_name = (
                    trajectory.joint_names[index]
                    if index < len(trajectory.joint_names)
                    else f"joint_{index}"
                )
                print(
                    f"    {joint_name}: mean_abs={float(joint_mean_abs[index]):.6e}, "
                    f"max_abs={float(joint_error_abs_max[index]):.6e}"
                )

    if args.hold:
        print("Playback finished. Holding GUI... Press Ctrl+C to exit.")
        while True:
            debugger.idle()
            time.sleep(1.0 / max(play_fps, 15.0))


if __name__ == "__main__":
    main()

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import imageio.v2 as imageio
import numpy as np
from scipy.spatial.transform import Rotation as Rscipy

from utils.retarget_utils import axis_angle_to_rotation_matrix, build_transform_matrix

from .core import register_process
from .render_process import ClipLightConfig, RenderProcess


@dataclass
class GripperOnlyTrajectoryPayload:
    left_pose: np.ndarray  # [T, 7]
    right_pose: np.ndarray  # [T, 7]
    left_gripper_signal: np.ndarray | None  # [T]
    right_gripper_signal: np.ndarray | None  # [T]
    intrinsics: np.ndarray  # [3, 3]
    camera_extrinsics: np.ndarray  # [4, 4]
    source_image_height: int | None
    source_image_width: int | None

    @property
    def frame_count(self) -> int:
        return int(max(self.left_pose.shape[0], self.right_pose.shape[0]))


@dataclass(frozen=True)
class RandomizedJointConfig:
    enabled: bool
    joint_names: tuple[str, ...]
    joint_limits: tuple[tuple[float, float], ...]
    max_step: tuple[float, ...]
    acceleration_std: tuple[float, ...]
    randomize_initial_joint_positions: bool
    seed: int


@dataclass
class _ArmEntityState:
    entity: Any
    root_link: Any
    tcp_link: Any
    non_gripper_dofs_idx_local: list[int]
    non_gripper_qs_idx_local: list[int]
    gripper_dofs_idx_local: list[int]
    gripper_qs_idx_local: list[int]


@register_process("render_gripper_only")
class RenderGripperOnlyProcess(RenderProcess):
    RENDER_DIR = "render_gripper_only"

    def run(self, sample: dict[str, Any], context: Any) -> dict[str, Any]:
        if context is None:
            raise ValueError("render_gripper_only process requires pipeline context")

        meta_data = self._resolve_meta_data(sample)
        trajectory_path = sample.get("trajectory_path")
        if not isinstance(trajectory_path, str) or not trajectory_path:
            raise KeyError("missing 'trajectory_path' in sample for render_gripper_only process")
        trajectory = self._load_gripper_trajectory(
            parquet_path=self.resolve_sample_path(trajectory_path),
            meta_data=meta_data,
        )
        if trajectory.frame_count <= 0:
            raise ValueError("empty trajectory in parquet data")

        video_path = self._resolve_video_path(sample, meta_data)
        urdf_path = self._resolve_urdf_path(meta_data)
        output_dir = self.params.get("output_dir")
        if output_dir is None:
            raise ValueError("render_gripper_only process requires params.output_dir")

        reader = imageio.get_reader(video_path)
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
            source_h=source_h,
            source_w=source_w,
            default_fov_deg=default_fov_deg,
            backend=backend,
        )
        target_tcp_world = self._build_target_tcp_world_poses(trajectory=trajectory)
        renderer.prepare_clip(
            clip_id=self._resolve_clip_id(
                sample=sample,
                trajectory_path=trajectory_path,
                video_path=video_path,
            ),
            frame_count=trajectory.frame_count,
            left_target_tcp_world=target_tcp_world["left"][0] if target_tcp_world["left"] else None,
            right_target_tcp_world=target_tcp_world["right"][0] if target_tcp_world["right"] else None,
            left_gripper_signal=(
                float(trajectory.left_gripper_signal[0]) if trajectory.left_gripper_signal is not None else None
            ),
            right_gripper_signal=(
                float(trajectory.right_gripper_signal[0]) if trajectory.right_gripper_signal is not None else None
            ),
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

                    left_target_tcp_world = (
                        target_tcp_world["left"][frame_index] if frame_index < len(target_tcp_world["left"]) else None
                    )
                    right_target_tcp_world = (
                        target_tcp_world["right"][frame_index]
                        if frame_index < len(target_tcp_world["right"])
                        else None
                    )
                    render_rgb, segmentation = renderer.render_rgb_and_mask(
                        frame_index=frame_index,
                        left_target_tcp_world=left_target_tcp_world,
                        right_target_tcp_world=right_target_tcp_world,
                        left_gripper_signal=(
                            float(trajectory.left_gripper_signal[frame_index])
                            if trajectory.left_gripper_signal is not None
                            and frame_index < trajectory.left_gripper_signal.shape[0]
                            else None
                        ),
                        right_gripper_signal=(
                            float(trajectory.right_gripper_signal[frame_index])
                            if trajectory.right_gripper_signal is not None
                            and frame_index < trajectory.right_gripper_signal.shape[0]
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
        source_h: int,
        source_w: int,
        default_fov_deg: float,
        backend: str,
    ) -> tuple["_GenesisGripperOnlyRenderer", int, int]:
        target_h, target_w = self._resolve_target_size(source_h=source_h, source_w=source_w)
        camera_near, camera_far = self._resolve_camera_clip_planes()
        clip_light_config = self._resolve_clip_light_config()
        random_joint_config = self._resolve_randomized_joint_config()
        use_gripper_signal = bool(self.params.get("use_gripper_signal", True))
        gripper_open_max = float(self.params.get("gripper_open_max", 0.04))
        current_spec = {
            "urdf_path": str(Path(urdf_path).resolve()),
            "default_fov_deg": float(default_fov_deg),
            "backend": str(backend).lower(),
            "height": int(target_h),
            "width": int(target_w),
            "camera_near": float(camera_near),
            "camera_far": float(camera_far),
            "clip_light_enabled": bool(clip_light_config.enabled),
            "clip_light_ambient": tuple(float(value) for value in clip_light_config.ambient_light),
            "clip_light_xy_range": tuple(float(value) for value in clip_light_config.xy_range),
            "clip_light_z_range": tuple(float(value) for value in clip_light_config.z_range),
            "clip_light_intensity_range": tuple(float(value) for value in clip_light_config.intensity_range),
            "clip_light_seed": int(clip_light_config.seed),
            "random_joint_config": random_joint_config,
            "use_gripper_signal": use_gripper_signal,
            "gripper_open_max": float(gripper_open_max),
        }

        if self._renderer is not None and self._renderer_spec == current_spec:
            return self._renderer, target_h, target_w

        self._destroy_renderer()
        self._renderer = _GenesisGripperOnlyRenderer(
            urdf_path=urdf_path,
            width=target_w,
            height=target_h,
            default_fov_deg=default_fov_deg,
            init_backend=backend,
            camera_near=camera_near,
            camera_far=camera_far,
            clip_light_config=clip_light_config,
            random_joint_config=random_joint_config,
            use_gripper_signal=use_gripper_signal,
            gripper_open_max=gripper_open_max,
        )
        self._renderer_spec = dict(current_spec)
        return self._renderer, target_h, target_w

    def _load_gripper_trajectory(self, *, parquet_path: str, meta_data: dict[str, Any]) -> GripperOnlyTrajectoryPayload:
        try:
            import pyarrow.parquet as pq
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "pyarrow is required for render_gripper_only process. Install it with `python -m pip install pyarrow`."
            ) from exc

        schema_names = set(pq.read_schema(parquet_path).names)
        if "left_pose" not in schema_names and "right_pose" not in schema_names:
            raise ValueError("render_gripper_only process requires at least one pose column in parquet")

        table = pq.read_table(
            parquet_path,
            columns=[
                name
                for name in (
                    "left_pose",
                    "right_pose",
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
        left_rows = list(payload.get("left_pose", []))
        right_rows = list(payload.get("right_pose", []))
        left_signal_values = list(payload.get("left_gripper_signal", []))
        right_signal_values = list(payload.get("right_gripper_signal", []))
        frame_count = max(len(left_rows), len(right_rows), len(left_signal_values), len(right_signal_values))
        if frame_count <= 0:
            raise ValueError("render_gripper_only process found no frames in parquet pose trajectory")

        intrinsics = self._resolve_intrinsics(payload=payload, meta_data=meta_data)
        extrinsics = self._resolve_camera_extrinsics(payload=payload, meta_data=meta_data)
        source_h, source_w = self._resolve_source_image_size(payload=payload, meta_data=meta_data)
        left_gripper_signal = self._resolve_gripper_signal(values=left_signal_values, frame_count=frame_count)
        right_gripper_signal = self._resolve_gripper_signal(values=right_signal_values, frame_count=frame_count)

        return GripperOnlyTrajectoryPayload(
            left_pose=self._normalize_pose_rows(left_rows, frame_count=frame_count),
            right_pose=self._normalize_pose_rows(right_rows, frame_count=frame_count),
            left_gripper_signal=left_gripper_signal,
            right_gripper_signal=right_gripper_signal,
            intrinsics=intrinsics,
            camera_extrinsics=extrinsics,
            source_image_height=source_h,
            source_image_width=source_w,
        )

    def _build_target_tcp_world_poses(
        self,
        *,
        trajectory: GripperOnlyTrajectoryPayload,
    ) -> dict[str, list[np.ndarray | None]]:
        left_base_transform = build_transform_matrix(
            translation=self.params.get("left_base_translation", [0.0, 0.3, 0.0])
        ).astype(np.float64)
        right_base_transform = build_transform_matrix(
            translation=self.params.get("right_base_translation", [0.0, -0.3, 0.0])
        ).astype(np.float64)
        hold_invalid = bool(self.params.get("hold_invalid_tcp_pose", True))
        return {
            "left": self._pose_rows_to_world_transforms(
                pose_rows=trajectory.left_pose,
                base_transform=left_base_transform,
                hold_invalid=hold_invalid,
            ),
            "right": self._pose_rows_to_world_transforms(
                pose_rows=trajectory.right_pose,
                base_transform=right_base_transform,
                hold_invalid=hold_invalid,
            ),
        }

    def _resolve_randomized_joint_config(self) -> RandomizedJointConfig:
        enabled = bool(self.params.get("randomize_non_gripper_joints", True))
        raw_names = self.params.get("non_gripper_joint_names", ["joint4", "joint5", "joint6"])
        if not isinstance(raw_names, (list, tuple)):
            raise ValueError("render_gripper_only requires params.non_gripper_joint_names as a sequence")
        joint_names = tuple(str(name) for name in raw_names if isinstance(name, str) and str(name))
        if not enabled or not joint_names:
            return RandomizedJointConfig(
                enabled=False,
                joint_names=tuple(),
                joint_limits=tuple(),
                max_step=tuple(),
                acceleration_std=tuple(),
                randomize_initial_joint_positions=False,
                seed=int(self.params.get("random_joint_seed", 0)),
            )

        default_ranges = {
            "joint4": (-0.65, 0.65),
            "joint5": (-0.65, 0.65),
            "joint6": (-1.0, 1.0),
        }
        raw_ranges = self.params.get("non_gripper_joint_ranges", {})
        joint_limits: list[tuple[float, float]] = []
        if isinstance(raw_ranges, dict):
            for joint_name in joint_names:
                range_value = raw_ranges.get(joint_name, default_ranges.get(joint_name, (-0.5, 0.5)))
                joint_limits.append(self._resolve_signed_range(range_value, name=f"non_gripper_joint_ranges.{joint_name}"))
        elif isinstance(raw_ranges, (list, tuple)) and len(raw_ranges) == len(joint_names):
            for joint_name, range_value in zip(joint_names, raw_ranges):
                joint_limits.append(self._resolve_signed_range(range_value, name=f"non_gripper_joint_ranges.{joint_name}"))
        else:
            raise ValueError(
                "render_gripper_only requires params.non_gripper_joint_ranges as a mapping or sequence matching "
                "params.non_gripper_joint_names"
            )

        max_step = self._resolve_joint_vector(
            self.params.get("random_joint_max_step", 0.08),
            count=len(joint_names),
            name="random_joint_max_step",
            minimum=0.0,
        )
        acceleration_std = self._resolve_joint_vector(
            self.params.get("random_joint_acceleration_std", 0.02),
            count=len(joint_names),
            name="random_joint_acceleration_std",
            minimum=0.0,
        )
        randomize_initial_joint_positions = bool(self.params.get("randomize_initial_joint_positions", True))
        seed_raw = self.params.get("random_joint_seed", 0)
        try:
            seed = int(0 if seed_raw is None else seed_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("render_gripper_only requires integer params.random_joint_seed") from exc

        return RandomizedJointConfig(
            enabled=True,
            joint_names=joint_names,
            joint_limits=tuple(joint_limits),
            max_step=tuple(float(value) for value in max_step),
            acceleration_std=tuple(float(value) for value in acceleration_std),
            randomize_initial_joint_positions=randomize_initial_joint_positions,
            seed=seed,
        )

    @staticmethod
    def _normalize_pose_rows(rows: list[Any], *, frame_count: int) -> np.ndarray:
        poses = np.full((frame_count, 7), np.nan, dtype=np.float32)
        for frame_index in range(frame_count):
            row = rows[frame_index] if frame_index < len(rows) else None
            poses[frame_index] = RenderGripperOnlyProcess._normalize_pose_row(row)
        return poses

    @staticmethod
    def _normalize_pose_row(row: Any) -> np.ndarray:
        output = np.full((7,), np.nan, dtype=np.float32)
        if row is None:
            return output
        values = np.asarray(row, dtype=np.float32).reshape(-1)
        if values.size < 6:
            return output
        limit = min(values.size, output.size)
        output[:limit] = values[:limit]
        return output

    def _pose_rows_to_world_transforms(
        self,
        *,
        pose_rows: np.ndarray,
        base_transform: np.ndarray,
        hold_invalid: bool,
    ) -> list[np.ndarray | None]:
        transforms: list[np.ndarray | None] = []
        last_valid: np.ndarray | None = None
        for pose in np.asarray(pose_rows, dtype=np.float64):
            if pose.shape[0] < 6 or not np.isfinite(pose[:6]).all():
                transforms.append(last_valid.copy() if hold_invalid and last_valid is not None else None)
                continue
            local_transform = np.eye(4, dtype=np.float64)
            local_transform[:3, :3] = axis_angle_to_rotation_matrix(pose[3:6]).astype(np.float64)
            local_transform[:3, 3] = pose[:3]
            world_transform = base_transform @ local_transform
            last_valid = world_transform
            transforms.append(world_transform)
        return transforms

    @staticmethod
    def _resolve_signed_range(value: Any, *, name: str) -> tuple[float, float]:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"render_gripper_only requires params.{name} as a length-2 sequence")
        low = float(value[0])
        high = float(value[1])
        if not np.isfinite(low) or not np.isfinite(high) or high < low:
            raise ValueError(f"render_gripper_only requires params.{name} with finite min <= max")
        return low, high

    @staticmethod
    def _resolve_joint_vector(value: Any, *, count: int, name: str, minimum: float) -> tuple[float, ...]:
        if isinstance(value, (int, float)):
            values = [float(value)] * count
        elif isinstance(value, (list, tuple)) and len(value) == count:
            values = [float(item) for item in value]
        else:
            raise ValueError(
                f"render_gripper_only requires params.{name} as a scalar or sequence of length {count}"
            )
        if any((not np.isfinite(item)) or item < minimum for item in values):
            raise ValueError(f"render_gripper_only requires finite params.{name} >= {minimum}")
        return tuple(values)


class _GenesisGripperOnlyRenderer:
    _initialized = False

    def __init__(
        self,
        *,
        urdf_path: str,
        width: int,
        height: int,
        default_fov_deg: float,
        init_backend: str,
        camera_near: float,
        camera_far: float,
        clip_light_config: ClipLightConfig,
        random_joint_config: RandomizedJointConfig,
        use_gripper_signal: bool,
        gripper_open_max: float,
    ) -> None:
        try:
            import genesis as gs
            from genesis.ext import pyrender
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "genesis is required for render_gripper_only process. Install it with `python -m pip install genesis-world`."
            ) from exc

        self._gs = gs
        self._pyrender = pyrender
        self._width = int(width)
        self._height = int(height)
        self._default_fov_deg = float(default_fov_deg)
        self._init_backend = str(init_backend).lower()
        self._camera_near = float(camera_near)
        self._camera_far = float(camera_far)
        self._clip_light_config = clip_light_config
        self._random_joint_config = random_joint_config
        self._use_gripper_signal = bool(use_gripper_signal)
        self._gripper_open_max = float(gripper_open_max)
        self._clip_light = None
        self._clip_light_node = None
        self._left_random_joint_positions = np.zeros((0, len(self._random_joint_config.joint_names)), dtype=np.float32)
        self._right_random_joint_positions = np.zeros((0, len(self._random_joint_config.joint_names)), dtype=np.float32)

        self._ensure_initialized()
        self._scene = self._create_scene()
        self._left_arm_state = self._build_arm_entity_state(urdf_path=urdf_path)
        self._right_arm_state = self._build_arm_entity_state(urdf_path=urdf_path)
        self._camera = self._scene.add_camera(
            res=(self._width, self._height),
            pos=(1.2, 0.0, 0.8),
            lookat=(0.0, 0.0, 0.2),
            up=(0.0, 0.0, 1.0),
            fov=self._default_fov_deg,
            near=self._camera_near,
            far=self._camera_far,
            GUI=False,
        )
        self._scene.build(n_envs=1)

    def prepare_clip(
        self,
        *,
        clip_id: str,
        frame_count: int,
        left_target_tcp_world: np.ndarray | None,
        right_target_tcp_world: np.ndarray | None,
        left_gripper_signal: float | None,
        right_gripper_signal: float | None,
    ) -> None:
        self._left_random_joint_positions = self._build_random_joint_trajectory(
            clip_id=f"{clip_id}:left",
            frame_count=frame_count,
        )
        self._right_random_joint_positions = self._build_random_joint_trajectory(
            clip_id=f"{clip_id}:right",
            frame_count=frame_count,
        )
        self._apply_arm_state(
            arm_state=self._left_arm_state,
            non_gripper_joint_positions=self._joint_row(self._left_random_joint_positions, 0),
            gripper_signal=left_gripper_signal,
            target_world_from_tcp=left_target_tcp_world,
        )
        self._apply_arm_state(
            arm_state=self._right_arm_state,
            non_gripper_joint_positions=self._joint_row(self._right_random_joint_positions, 0),
            gripper_signal=right_gripper_signal,
            target_world_from_tcp=right_target_tcp_world,
        )
        if not self._clip_light_config.enabled:
            return
        self._scene.visualizer.update_visual_states(force_render=True)
        anchor = self._resolve_clip_light_anchor()
        light_position, light_intensity = self._sample_clip_light(clip_id=clip_id, anchor=anchor)
        self._apply_clip_light(position=light_position, intensity=light_intensity)

    def render_rgb_and_mask(
        self,
        *,
        frame_index: int,
        left_target_tcp_world: np.ndarray | None,
        right_target_tcp_world: np.ndarray | None,
        left_gripper_signal: float | None,
        right_gripper_signal: float | None,
        camera_extrinsics: np.ndarray,
        intrinsics: np.ndarray,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        self._apply_arm_state(
            arm_state=self._left_arm_state,
            non_gripper_joint_positions=self._joint_row(self._left_random_joint_positions, frame_index),
            gripper_signal=left_gripper_signal,
            target_world_from_tcp=left_target_tcp_world,
        )
        self._apply_arm_state(
            arm_state=self._right_arm_state,
            non_gripper_joint_positions=self._joint_row(self._right_random_joint_positions, frame_index),
            gripper_signal=right_gripper_signal,
            target_world_from_tcp=right_target_tcp_world,
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
        if self._clip_light_node is not None:
            try:
                self._scene.visualizer.context.remove_node(self._clip_light_node)
            except Exception:
                pass
        self._scene.destroy()
        self._scene = None
        self._camera = None
        self._clip_light = None
        self._clip_light_node = None

    def _build_arm_entity_state(self, *, urdf_path: str) -> _ArmEntityState:
        entity = self._scene.add_entity(
            self._gs.morphs.URDF(
                file=str(Path(urdf_path).resolve()),
                fixed=False,
                merge_fixed_links=False,
            )
        )
        return _ArmEntityState(
            entity=entity,
            root_link=entity.base_link,
            tcp_link=entity.get_link(name="tcp"),
            non_gripper_dofs_idx_local=self._resolve_named_control_indices(
                entity=entity,
                joint_names=list(self._random_joint_config.joint_names),
                index_type="dofs",
            ),
            non_gripper_qs_idx_local=self._resolve_named_control_indices(
                entity=entity,
                joint_names=list(self._random_joint_config.joint_names),
                index_type="qs",
            ),
            gripper_dofs_idx_local=self._resolve_named_control_indices(
                entity=entity,
                joint_names=["joint7", "joint8"],
                index_type="dofs",
            ),
            gripper_qs_idx_local=self._resolve_named_control_indices(
                entity=entity,
                joint_names=["joint7", "joint8"],
                index_type="qs",
            ),
        )

    def _ensure_initialized(self) -> None:
        if _GenesisGripperOnlyRenderer._initialized:
            return
        backend = self._gs.gpu if "gpu" in self._init_backend else self._gs.cpu
        self._gs.init(backend=backend)
        _GenesisGripperOnlyRenderer._initialized = True

    def _create_scene(self) -> Any:
        vis_kwargs: dict[str, Any] = {"segmentation_level": "entity"}
        if self._clip_light_config.enabled:
            vis_kwargs["ambient_light"] = self._clip_light_config.ambient_light
            vis_kwargs["lights"] = []
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
            vis_options=self._gs.options.VisOptions(**vis_kwargs),
            renderer=self._gs.options.renderers.Rasterizer(),
            show_viewer=False,
        )

    def _apply_arm_state(
        self,
        *,
        arm_state: _ArmEntityState,
        non_gripper_joint_positions: np.ndarray,
        gripper_signal: float | None,
        target_world_from_tcp: np.ndarray | None,
    ) -> None:
        if target_world_from_tcp is None:
            return
        if non_gripper_joint_positions.size > 0:
            self._set_joint_subset(
                entity=arm_state.entity,
                values=non_gripper_joint_positions,
                dofs_idx_local=arm_state.non_gripper_dofs_idx_local,
                qs_idx_local=arm_state.non_gripper_qs_idx_local,
            )
        gripper_pair = self._build_gripper_pair(gripper_signal if self._use_gripper_signal else None)
        if gripper_pair is not None:
            self._set_joint_subset(
                entity=arm_state.entity,
                values=gripper_pair,
                dofs_idx_local=arm_state.gripper_dofs_idx_local,
                qs_idx_local=arm_state.gripper_qs_idx_local,
            )

        root_from_tcp = self._compute_root_from_tcp_local_transform(arm_state=arm_state)
        world_from_root = np.asarray(target_world_from_tcp, dtype=np.float64).reshape(4, 4) @ np.linalg.inv(root_from_tcp)
        arm_state.entity.set_quat(
            self._rotation_matrix_to_wxyz(world_from_root[:3, :3]),
            zero_velocity=False,
            relative=False,
        )
        arm_state.entity.set_pos(world_from_root[:3, 3], zero_velocity=False, relative=False)

    def _compute_root_from_tcp_local_transform(self, *, arm_state: _ArmEntityState) -> np.ndarray:
        world_from_root = self._compose_transform(
            pos=self._to_numpy(arm_state.root_link.get_pos()),
            quat_wxyz=self._to_numpy(arm_state.root_link.get_quat()),
        )
        world_from_tcp = self._compose_transform(
            pos=self._to_numpy(arm_state.tcp_link.get_pos()),
            quat_wxyz=self._to_numpy(arm_state.tcp_link.get_quat()),
        )
        return np.linalg.inv(world_from_root) @ world_from_tcp

    def _build_random_joint_trajectory(self, *, clip_id: str, frame_count: int) -> np.ndarray:
        joint_count = len(self._random_joint_config.joint_names)
        if frame_count <= 0 or joint_count <= 0:
            return np.zeros((max(frame_count, 0), joint_count), dtype=np.float32)
        if not self._random_joint_config.enabled:
            return np.zeros((frame_count, joint_count), dtype=np.float32)

        limits = np.asarray(self._random_joint_config.joint_limits, dtype=np.float64)
        low = limits[:, 0]
        high = limits[:, 1]
        max_step = np.asarray(self._random_joint_config.max_step, dtype=np.float64)
        acceleration_std = np.asarray(self._random_joint_config.acceleration_std, dtype=np.float64)
        rng = np.random.default_rng(self._seed_for_clip(clip_id))

        positions = np.empty((frame_count, joint_count), dtype=np.float64)
        velocity = rng.uniform(-max_step, max_step)
        if self._random_joint_config.randomize_initial_joint_positions:
            positions[0] = rng.uniform(low, high)
        else:
            positions[0] = np.clip(np.zeros((joint_count,), dtype=np.float64), low, high)
        for frame_index in range(1, frame_count):
            velocity += rng.normal(0.0, acceleration_std)
            velocity = np.clip(velocity, -max_step, max_step)
            next_position = positions[frame_index - 1] + velocity

            over_mask = next_position > high
            if np.any(over_mask):
                next_position[over_mask] = high[over_mask] - (next_position[over_mask] - high[over_mask])
                velocity[over_mask] *= -1.0
            under_mask = next_position < low
            if np.any(under_mask):
                next_position[under_mask] = low[under_mask] + (low[under_mask] - next_position[under_mask])
                velocity[under_mask] *= -1.0

            positions[frame_index] = np.clip(next_position, low, high)
        return positions.astype(np.float32)

    def _joint_row(self, positions: np.ndarray, frame_index: int) -> np.ndarray:
        if positions.size <= 0:
            return np.zeros((0,), dtype=np.float32)
        safe_index = int(np.clip(frame_index, 0, positions.shape[0] - 1))
        return np.asarray(positions[safe_index], dtype=np.float32).reshape(-1)

    def _resolve_clip_light_anchor(self) -> np.ndarray:
        default_anchor = np.asarray((0.0, 0.0, 0.75), dtype=np.float32)
        try:
            pyrender_scene = self._scene.visualizer.context._scene
            bounds = np.asarray(pyrender_scene.bounds, dtype=np.float32)
        except Exception:
            return default_anchor
        if bounds.shape != (2, 3) or not np.all(np.isfinite(bounds)):
            return default_anchor
        center_xy = 0.5 * (bounds[0, :2] + bounds[1, :2])
        return np.asarray((center_xy[0], center_xy[1], bounds[1, 2]), dtype=np.float32)

    def _sample_clip_light(self, *, clip_id: str, anchor: np.ndarray) -> tuple[np.ndarray, float]:
        rng = np.random.default_rng(self._seed_for_clip(f"clip-light:{clip_id}"))
        x_range, y_range = self._clip_light_config.xy_range
        z_low, z_high = self._clip_light_config.z_range
        intensity_low, intensity_high = self._clip_light_config.intensity_range
        offset = np.asarray(
            (
                rng.uniform(-x_range, x_range),
                rng.uniform(-y_range, y_range),
                rng.uniform(z_low, z_high),
            ),
            dtype=np.float32,
        )
        position = np.asarray(anchor, dtype=np.float32).reshape(3) + offset
        intensity = float(rng.uniform(intensity_low, intensity_high))
        return position, intensity

    def _apply_clip_light(self, *, position: np.ndarray, intensity: float) -> None:
        context = self._scene.visualizer.context
        pose = np.eye(4, dtype=np.float32)
        pose[:3, 3] = np.asarray(position, dtype=np.float32).reshape(3)
        if self._clip_light is None:
            self._clip_light = self._pyrender.PointLight(
                color=np.asarray((1.0, 1.0, 1.0), dtype=np.float32),
                intensity=float(intensity),
            )
            self._clip_light_node = context.add_node(self._clip_light, name="clip_random_light", pose=pose)
            return
        with self._scene.visualizer.viewer_lock:
            self._clip_light.color = np.asarray((1.0, 1.0, 1.0), dtype=np.float32)
            self._clip_light.intensity = float(intensity)
            self._clip_light_node.matrix = pose

    def _seed_for_clip(self, clip_id: str) -> int:
        payload = f"{self._random_joint_config.seed}:{clip_id}".encode("utf-8", "ignore")
        digest = hashlib.blake2b(payload, digest_size=8).digest()
        return int.from_bytes(digest[:4], byteorder="little", signed=False)

    @staticmethod
    def _resolve_named_control_indices(
        *,
        entity: Any,
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
                joint = entity.get_joint(name=joint_name)
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

    @staticmethod
    def _set_joint_subset(
        *,
        entity: Any,
        values: np.ndarray,
        dofs_idx_local: list[int],
        qs_idx_local: list[int],
    ) -> None:
        vector = np.asarray(values, dtype=np.float32).reshape(-1)
        if len(dofs_idx_local) == vector.size:
            entity.set_dofs_position(vector, dofs_idx_local=dofs_idx_local, zero_velocity=False)
            return
        if len(qs_idx_local) == vector.size:
            entity.set_qpos(vector, qs_idx_local=qs_idx_local, zero_velocity=False, skip_forward=False)

    def _build_gripper_pair(self, signal: float | None) -> np.ndarray | None:
        if signal is None:
            return None
        try:
            value = float(signal)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(value):
            return None
        opening = float(np.clip(value, 0.0, 1.0)) * self._gripper_open_max
        return np.asarray([opening, -opening], dtype=np.float32)

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

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=np.float64)

    @staticmethod
    def _compose_transform(*, pos: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
        transform = np.eye(4, dtype=np.float64)
        transform[:3, 3] = np.asarray(pos, dtype=np.float64).reshape(3)
        transform[:3, :3] = _GenesisGripperOnlyRenderer._quat_wxyz_to_rotation_matrix(quat_wxyz)
        return transform

    @staticmethod
    def _quat_wxyz_to_rotation_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
        quat = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
        quat_xyzw = np.asarray([quat[1], quat[2], quat[3], quat[0]], dtype=np.float64)
        return Rscipy.from_quat(quat_xyzw).as_matrix().astype(np.float64)

    @staticmethod
    def _rotation_matrix_to_wxyz(rotation: np.ndarray) -> np.ndarray:
        quat_xyzw = Rscipy.from_matrix(np.asarray(rotation, dtype=np.float64).reshape(3, 3)).as_quat()
        return np.asarray([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float32)

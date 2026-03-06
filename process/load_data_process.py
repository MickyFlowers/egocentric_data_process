from __future__ import annotations

import os
from typing import Any

import imageio.v2 as imageio
import numpy as np

try:
    import h5py
except ModuleNotFoundError:  # pragma: no cover - optional dependency for EgoDex loading
    h5py = None

from utils.image_utils import (
    build_intrinsics,
    draw_hand_keypoints,
    infer_image_size,
    project_points,
    scale_points_2d,
)
from utils.retarget_utils import (
    ensure_bool,
    load_mano_layer,
    load_pose_archive,
    mano_forward,
    resolve_torch_device,
    scale_traj_translation,
    to_numpy,
    to_torch,
    world_to_camera,
)

from .core import BaseProcess, register_process

# EgoDex exposes 25 hand joints. Drop the four non-thumb metacarpals so the
# remaining order matches the 21-joint layout already assumed downstream.
EGODEX_HAND_JOINT_SUFFIXES: tuple[str, ...] = (
    "Hand",
    "ThumbKnuckle",
    "ThumbIntermediateBase",
    "ThumbIntermediateTip",
    "ThumbTip",
    "IndexFingerKnuckle",
    "IndexFingerIntermediateBase",
    "IndexFingerIntermediateTip",
    "IndexFingerTip",
    "MiddleFingerKnuckle",
    "MiddleFingerIntermediateBase",
    "MiddleFingerIntermediateTip",
    "MiddleFingerTip",
    "RingFingerKnuckle",
    "RingFingerIntermediateBase",
    "RingFingerIntermediateTip",
    "RingFingerTip",
    "LittleFingerKnuckle",
    "LittleFingerIntermediateBase",
    "LittleFingerIntermediateTip",
    "LittleFingerTip",
)

EGODEX_REQUIRED_JOINT_SUFFIXES: tuple[str, ...] = (
    "Hand",
    "ThumbKnuckle",
    "ThumbIntermediateBase",
    "ThumbIntermediateTip",
    "ThumbTip",
    "IndexFingerKnuckle",
    "IndexFingerIntermediateBase",
    "IndexFingerIntermediateTip",
    "IndexFingerTip",
    "LittleFingerKnuckle",
)


@register_process("load_data")
class LoadDataProcess(BaseProcess):
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._mano_layers: dict[str, Any] = {}
        self.device = resolve_torch_device(self.params.get("device", "cpu"))

    def run(self, sample: dict[str, Any], context: Any) -> dict[str, Any]:
        archive = load_pose_archive(self.resolve_sample_path(sample["data_path"]))
        slam_data = archive["slam_data"]

        fps_value = archive.get("fps", slam_data.get("fps", self.params.get("default_fps", 30.0)))
        fps = float(np.asarray(to_numpy(fps_value)).reshape(-1)[0])
        intrinsics = build_intrinsics(slam_data["img_focal"], slam_data["img_center"])
        image_size = infer_image_size(slam_data["img_center"])

        frame_count = self._determine_frame_count(archive)
        traj = to_torch(slam_data["traj"], self.device).reshape(-1, 7)
        scale = to_torch(slam_data["scale"], self.device).reshape(1)
        traj = scale_traj_translation(traj[:frame_count], scale)

        left_hand = self._extract_hand(
            archive.get("left_hand"),
            frame_count=frame_count,
            side="left",
            traj=traj,
        )
        right_hand = self._extract_hand(
            archive.get("right_hand"),
            frame_count=frame_count,
            side="right",
            traj=traj,
        )

        sample["left_hand"] = left_hand
        sample["right_hand"] = right_hand
        sample["fps"] = fps
        sample["image_size"] = image_size
        sample["intrinsics"] = intrinsics
        sample["last_process"] = self.name
        return sample

    def _determine_frame_count(self, archive: dict[str, Any]) -> int:
        return int(len(to_numpy(archive["slam_data"]["traj"])))

    def _extract_hand(
        self,
        hand_section: Any,
        *,
        frame_count: int,
        side: str,
        traj: Any,
    ) -> dict[str, Any]:
        keypoints = np.full((frame_count, 21, 3), np.nan, dtype=np.float32)
        vertices = np.full((frame_count, 778, 3), np.nan, dtype=np.float32)
        valid = np.zeros((frame_count,), dtype=np.bool_)

        if hand_section is None:
            return {"keypoints": keypoints, "vertices": vertices, "valid": valid}

        mano_params = hand_section["mano_params"]
        pred_valid = hand_section.get("pred_valid")

        global_orient = to_torch(mano_params["global_orient"], self.device).reshape(-1, 3)
        hand_pose = to_torch(mano_params["hand_pose"], self.device).reshape(-1, 45)
        betas = to_torch(mano_params["betas"], self.device).reshape(-1, 10)
        transl = to_torch(mano_params["transl"], self.device).reshape(-1, 3)

        local_frame_count = min(frame_count, len(global_orient), len(hand_pose), len(betas), len(transl))
        if traj is not None:
            local_frame_count = min(local_frame_count, len(traj))
        if local_frame_count <= 0:
            return {"keypoints": keypoints, "vertices": vertices, "valid": valid}

        valid[:local_frame_count] = True
        if pred_valid is not None:
            valid_mask = ensure_bool(pred_valid).reshape(-1)
            valid[:local_frame_count] &= valid_mask[:local_frame_count]

        mano_layer = self._get_mano_layer(side=side)
        hand_vertices, joints = mano_forward(
            mano_layer,
            global_orient=global_orient[:local_frame_count],
            hand_pose=hand_pose[:local_frame_count],
            betas=betas[:local_frame_count],
            transl=transl[:local_frame_count],
        )
        if traj is not None:
            joints = world_to_camera(joints, traj[:local_frame_count])
            hand_vertices = world_to_camera(hand_vertices, traj[:local_frame_count])

        joints = to_numpy(joints).astype(np.float32, copy=False)
        hand_vertices = to_numpy(hand_vertices).astype(np.float32, copy=False)

        keypoints[:local_frame_count] = joints
        vertices[:local_frame_count] = hand_vertices
        keypoints[~valid] = np.nan
        vertices[~valid] = np.nan
        return {"keypoints": keypoints, "vertices": vertices, "valid": valid}

    def _get_mano_layer(self, *, side: str):
        side_key = side.lower()
        cache_key = f"{side_key}:{self.device.type}"
        if cache_key not in self._mano_layers:
            self._mano_layers[cache_key] = load_mano_layer(
                str(self.resolve_path(self.params["mano_model_dir"])),
                side=side_key,
                use_pca=bool(self.params.get("use_pca", False)),
                flat_hand_mean=bool(self.params.get("flat_hand_mean", True)),
                fix_shapedirs=bool(self.params.get("fix_shapedirs", True)),
                device_name=self.device.type,
            )
        return self._mano_layers[cache_key]

    def _write_visualization(
        self,
        sample: dict[str, Any],
        context: Any,
    ) -> None:
        if context is None:
            return
        if not sample.get("visualize", False):
            return
        output_dir = self.params.get("output_dir")
        if not output_dir:
            return

        left_hand = sample["left_hand"]
        right_hand = sample["right_hand"]
        intrinsics = sample["intrinsics"]
        image_size = sample["image_size"]
        fps = float(sample["fps"])

        left_points_2d, left_depth_valid = project_points(left_hand["keypoints"], intrinsics)
        right_points_2d, right_depth_valid = project_points(right_hand["keypoints"], intrinsics)

        output_root = self.extend_output_dir(output_dir, os.path.join("samples", "hand_pose"))
        output_path, remote_output_path = self.build_output_paths(
            sample,
            output_dir=output_root,
            extension=".mp4",
        )

        reader = imageio.get_reader(self.resolve_sample_path(sample["video_path"]))
        reader_meta = reader.get_meta_data()
        video_fps = float(reader_meta.get("fps") or fps or 30.0)

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
                        if frame_index >= len(left_points_2d) or frame_index >= len(right_points_2d):
                            break

                        rendered = np.asarray(frame).copy()
                        frame_size = rendered.shape[:2]

                        left_frame_points = scale_points_2d(
                            left_points_2d[frame_index],
                            image_size,
                            frame_size,
                        )
                        right_frame_points = scale_points_2d(
                            right_points_2d[frame_index],
                            image_size,
                            frame_size,
                        )

                        if bool(left_hand["valid"][frame_index]):
                            draw_hand_keypoints(
                                rendered,
                                left_frame_points,
                                left_depth_valid[frame_index],
                                color=np.array([0, 255, 0], dtype=np.uint8),
                            )
                        if bool(right_hand["valid"][frame_index]):
                            draw_hand_keypoints(
                                rendered,
                                right_frame_points,
                                right_depth_valid[frame_index],
                                color=np.array([255, 0, 0], dtype=np.uint8),
                            )

                        writer.append_data(rendered)
                finally:
                    writer.close()
            sample["hand_pose_path"] = remote_output_path
        finally:
            reader.close()


@register_process("load_egodex_data")
@register_process("load_ego_dex_data")
class LoadEgoDexDataProcess(LoadDataProcess):
    def run(self, sample: dict[str, Any], context: Any) -> dict[str, Any]:
        if h5py is None:
            raise ModuleNotFoundError(
                "h5py is required for load_egodex_data. Install it with `python -m pip install h5py`."
            )

        archive_path = self.resolve_sample_path(sample["data_path"])
        with h5py.File(archive_path, "r") as archive:
            intrinsics = self._load_egodex_intrinsics(archive)
            frame_count = self._determine_egodex_frame_count(archive)
            camera_transforms = self._read_egodex_transform_sequence(
                archive,
                joint_name="camera",
                frame_count=frame_count,
            )
            camera_inverse = self._invert_egodex_transform_sequence(camera_transforms)

            left_hand = self._extract_egodex_hand(
                archive,
                side="left",
                frame_count=frame_count,
                camera_inverse=camera_inverse,
            )
            right_hand = self._extract_egodex_hand(
                archive,
                side="right",
                frame_count=frame_count,
                camera_inverse=camera_inverse,
            )

        sample["left_hand"] = left_hand
        sample["right_hand"] = right_hand
        sample["fps"] = float(sample.get("fps", self.params.get("default_fps", 30.0)))
        sample["image_size"] = self._resolve_egodex_image_size(intrinsics)
        sample["intrinsics"] = intrinsics
        sample["last_process"] = self.name
        return sample

    def _load_egodex_intrinsics(self, archive: Any) -> np.ndarray:
        dataset = archive.get("camera/intrinsic")
        if dataset is None:
            dataset = archive.get("camera/intrinsics")
        if dataset is None:
            raise KeyError("missing EgoDex camera intrinsics at 'camera/intrinsic'")

        intrinsics = np.asarray(dataset[:], dtype=np.float32)
        if intrinsics.shape != (3, 3):
            raise ValueError(f"unexpected EgoDex intrinsics shape: {intrinsics.shape}")
        return intrinsics

    def _determine_egodex_frame_count(self, archive: Any) -> int:
        camera_dataset = archive.get("transforms/camera")
        if camera_dataset is not None and len(camera_dataset.shape) >= 1:
            return int(camera_dataset.shape[0])

        for group_name in ("transforms", "confidences"):
            group = archive.get(group_name)
            if group is None:
                continue
            for dataset in group.values():
                if len(dataset.shape) >= 1:
                    return int(dataset.shape[0])
        raise ValueError("unable to determine frame_count from EgoDex archive")

    def _read_egodex_transform_sequence(
        self,
        archive: Any,
        *,
        joint_name: str,
        frame_count: int,
    ) -> np.ndarray:
        dataset = archive.get(f"transforms/{joint_name}")
        if dataset is None:
            return np.full((frame_count, 4, 4), np.nan, dtype=np.float32)

        transforms = np.asarray(dataset[:], dtype=np.float32)
        if transforms.ndim != 3 or transforms.shape[1:] != (4, 4):
            raise ValueError(
                f"unexpected EgoDex transform shape for '{joint_name}': {transforms.shape}"
            )
        return self._align_egodex_frame_count(transforms, frame_count=frame_count, fill_value=np.nan)

    def _read_egodex_confidence_sequence(
        self,
        archive: Any,
        *,
        joint_name: str,
        frame_count: int,
    ) -> np.ndarray | None:
        dataset = archive.get(f"confidences/{joint_name}")
        if dataset is None:
            return None

        confidence = np.asarray(dataset[:], dtype=np.float32).reshape(-1)
        return self._align_egodex_frame_count(confidence, frame_count=frame_count, fill_value=np.nan)

    def _align_egodex_frame_count(
        self,
        values: np.ndarray,
        *,
        frame_count: int,
        fill_value: float,
    ) -> np.ndarray:
        output_shape = (frame_count,) + tuple(values.shape[1:])
        aligned = np.full(output_shape, fill_value, dtype=values.dtype)
        local_frame_count = min(frame_count, int(values.shape[0]))
        if local_frame_count > 0:
            aligned[:local_frame_count] = values[:local_frame_count]
        return aligned

    def _invert_egodex_transform_sequence(self, transforms: np.ndarray) -> np.ndarray:
        matrices = np.asarray(transforms, dtype=np.float32)
        inverse = np.full_like(matrices, np.nan, dtype=np.float32)
        valid_mask = np.isfinite(matrices[:, :3, :4]).all(axis=(1, 2))
        for frame_index in np.flatnonzero(valid_mask):
            try:
                inverse[frame_index] = np.linalg.inv(matrices[frame_index]).astype(np.float32, copy=False)
            except np.linalg.LinAlgError:
                continue
        return inverse

    def _extract_egodex_hand(
        self,
        archive: Any,
        *,
        side: str,
        frame_count: int,
        camera_inverse: np.ndarray,
    ) -> dict[str, Any]:
        keypoints = np.full((frame_count, len(EGODEX_HAND_JOINT_SUFFIXES), 3), np.nan, dtype=np.float32)
        vertices = np.full((frame_count, 0, 3), np.nan, dtype=np.float32)
        confidence_by_joint: dict[str, np.ndarray] = {}

        for joint_index, suffix in enumerate(EGODEX_HAND_JOINT_SUFFIXES):
            joint_name = f"{side}{suffix}"
            transforms = self._read_egodex_transform_sequence(
                archive,
                joint_name=joint_name,
                frame_count=frame_count,
            )
            camera_space = self._transform_egodex_sequence_to_camera(
                transforms=transforms,
                camera_inverse=camera_inverse,
            )
            keypoints[:, joint_index] = camera_space[:, :3, 3]

            confidence = self._read_egodex_confidence_sequence(
                archive,
                joint_name=joint_name,
                frame_count=frame_count,
            )
            if confidence is not None:
                confidence_by_joint[suffix] = confidence

        valid = self._compute_egodex_valid_mask(
            keypoints,
            confidence_by_joint=confidence_by_joint,
        )
        keypoints[~valid] = np.nan
        return {"keypoints": keypoints, "vertices": vertices, "valid": valid}

    def _transform_egodex_sequence_to_camera(
        self,
        *,
        transforms: np.ndarray,
        camera_inverse: np.ndarray,
    ) -> np.ndarray:
        joint_transforms = np.asarray(transforms, dtype=np.float32)
        camera_matrices = np.asarray(camera_inverse, dtype=np.float32)
        camera_space = np.full_like(joint_transforms, np.nan, dtype=np.float32)
        valid_mask = np.isfinite(joint_transforms[:, :3, :4]).all(axis=(1, 2))
        valid_mask &= np.isfinite(camera_matrices[:, :3, :4]).all(axis=(1, 2))
        for frame_index in np.flatnonzero(valid_mask):
            camera_space[frame_index] = (
                camera_matrices[frame_index] @ joint_transforms[frame_index]
            ).astype(np.float32, copy=False)
        return camera_space

    def _compute_egodex_valid_mask(
        self,
        keypoints: np.ndarray,
        *,
        confidence_by_joint: dict[str, np.ndarray],
    ) -> np.ndarray:
        required_joint_names = self._resolve_egodex_required_joint_names()
        joint_index_map = {
            name: index
            for index, name in enumerate(EGODEX_HAND_JOINT_SUFFIXES)
        }
        required_indices = [
            joint_index_map[name]
            for name in required_joint_names
            if name in joint_index_map
        ]
        if not required_indices:
            required_indices = list(range(len(EGODEX_HAND_JOINT_SUFFIXES)))

        required_points = np.asarray(keypoints[:, required_indices], dtype=np.float32)
        valid = np.isfinite(required_points).all(axis=(1, 2))

        confidence_threshold = self.params.get("confidence_threshold")
        if confidence_threshold is None:
            return valid.astype(np.bool_, copy=False)

        threshold = float(confidence_threshold)
        min_required_joint_count = int(
            self.params.get("min_required_joint_count", len(required_joint_names))
        )
        available_counts = np.zeros((keypoints.shape[0],), dtype=np.int32)
        confident_counts = np.zeros((keypoints.shape[0],), dtype=np.int32)

        for joint_name in required_joint_names:
            confidence = confidence_by_joint.get(joint_name)
            if confidence is None:
                continue
            confidence_values = np.asarray(confidence, dtype=np.float32).reshape(-1)
            finite_mask = np.isfinite(confidence_values)
            available_counts += finite_mask.astype(np.int32, copy=False)
            confident_counts += (finite_mask & (confidence_values >= threshold)).astype(np.int32, copy=False)

        if np.any(available_counts > 0):
            required_counts = np.minimum(
                available_counts,
                np.full_like(available_counts, max(0, min_required_joint_count)),
            )
            valid &= confident_counts >= required_counts

        return valid.astype(np.bool_, copy=False)

    def _resolve_egodex_required_joint_names(self) -> tuple[str, ...]:
        configured = self.params.get("required_joint_names")
        if not isinstance(configured, (list, tuple)):
            return EGODEX_REQUIRED_JOINT_SUFFIXES

        names = [str(value) for value in configured if str(value)]
        if not names:
            return EGODEX_REQUIRED_JOINT_SUFFIXES
        return tuple(names)

    def _resolve_egodex_image_size(self, intrinsics: np.ndarray) -> tuple[int, int]:
        configured = self.params.get("image_size")
        if isinstance(configured, (list, tuple)) and len(configured) == 2:
            return int(configured[0]), int(configured[1])

        center = np.array([intrinsics[0, 2], intrinsics[1, 2]], dtype=np.float32)
        return infer_image_size(center)

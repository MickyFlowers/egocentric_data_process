from __future__ import annotations

import os
from typing import Any

import imageio.v2 as imageio
import numpy as np

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

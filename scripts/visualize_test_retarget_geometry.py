#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np

if __package__ in {None, ""}:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
except Exception as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "matplotlib is required for this script. Install it with `python -m pip install matplotlib`."
    ) from exc

from process.load_data_process import LoadDataProcess
from utils.image_utils import (
    draw_axes_2d,
    draw_disk,
    draw_line,
    project_points,
    scale_points_2d,
)
from utils.retarget_utils import (
    _compute_cross_accumulated_y_axis,
    _fit_plane_normal_through_axis,
    _fit_weighted_line_to_points,
    _joint_indices,
    _normalize_vector,
    _project_point_onto_line,
    _thumb_index_indices,
)


def _script_finger_weights(point_count: int) -> np.ndarray:
    weights = np.ones((max(0, int(point_count)),), dtype=np.float32)
    if weights.size <= 0:
        return weights
    weights[0] = 4.0
    weights[-1] = max(float(weights[-1]), 2.0)
    return weights


def _collect_input_paths(input_dir: Path, sample_ids: set[str]) -> list[Path]:
    paths = sorted(input_dir.glob("*.pose3d_hand"))
    if sample_ids:
        paths = [path for path in paths if path.stem in sample_ids]
    if not paths:
        raise ValueError(f"no .pose3d_hand files found under: {input_dir}")
    return paths


def _build_loader(project_root: Path) -> LoadDataProcess:
    return LoadDataProcess(
        {
            "params": {
                "mano_model_dir": str(project_root / "assets" / "mano_v1_2"),
                "use_pca": False,
                "flat_hand_mean": True,
                "fix_shapedirs": True,
                "device": "cpu",
            }
        }
    )


def _load_sample(loader: LoadDataProcess, pose_path: Path) -> dict[str, Any]:
    sample = {
        "sample_id": pose_path.stem,
        "data_path": str(pose_path),
        "video_path": str(pose_path.with_suffix(".mp4")),
        "visualize": False,
    }
    return loader(sample, context=None)


def _compute_test_geometry_frame(joints: np.ndarray, side: str) -> dict[str, Any] | None:
    points = np.asarray(joints, dtype=np.float32).reshape(-1, 3)
    if points.shape[0] < 9 or not np.isfinite(points).all():
        return None

    indices = _joint_indices(points.shape[0])
    thumb_idx, index_idx = _thumb_index_indices(points.shape[0])
    wrist = points[indices["wrist"]]

    thumb_points = points[thumb_idx]
    index_points = points[index_idx]
    thumb_weights = _script_finger_weights(len(thumb_idx))
    index_weights = _script_finger_weights(len(index_idx))

    thumb_line_point, thumb_line_direction = _fit_weighted_line_to_points(thumb_points, thumb_weights)
    index_line_point, index_line_direction = _fit_weighted_line_to_points(index_points, index_weights)
    if np.linalg.norm(thumb_line_direction) < 1.0e-8 or np.linalg.norm(index_line_direction) < 1.0e-8:
        return None

    thumb_projected = np.stack(
        [_project_point_onto_line(point, thumb_line_point, thumb_line_direction) for point in thumb_points],
        axis=0,
    ).astype(np.float32, copy=False)
    index_projected = np.stack(
        [_project_point_onto_line(point, index_line_point, index_line_direction) for point in index_points],
        axis=0,
    ).astype(np.float32, copy=False)

    paired_count = min(3, thumb_projected.shape[0], index_projected.shape[0])
    if paired_count <= 0:
        return None
    selected_thumb_projected = thumb_projected[:paired_count]
    selected_index_projected = index_projected[:paired_count]
    pair_midpoints = 0.5 * (selected_thumb_projected + selected_index_projected)
    if side == "left":
        pair_vectors = selected_index_projected - selected_thumb_projected
    else:
        pair_vectors = selected_thumb_projected - selected_index_projected
    pair_norms = np.linalg.norm(pair_vectors, axis=1)
    finite_pair_mask = np.isfinite(pair_vectors).all(axis=1) & np.isfinite(pair_midpoints).all(axis=1) & np.isfinite(pair_norms)
    if int(np.count_nonzero(finite_pair_mask)) <= 0:
        return None
    x_axis = _normalize_vector(pair_vectors[finite_pair_mask].mean(axis=0))
    if np.linalg.norm(x_axis) < 1.0e-8:
        return None

    rough_y_axis = _compute_cross_accumulated_y_axis(
        points,
        wrist,
        thumb_idx,
        index_idx,
        side,
        normalize_inputs=True,
    )
    plane_points = np.concatenate([thumb_projected, index_projected], axis=0)
    plane_weights = np.concatenate([thumb_weights, index_weights], axis=0)
    plane_anchor = pair_midpoints[finite_pair_mask].mean(axis=0)
    y_axis = _fit_plane_normal_through_axis(
        plane_points,
        plane_weights,
        plane_anchor,
        x_axis,
        preferred_normal=rough_y_axis,
    )
    if np.linalg.norm(y_axis) < 1.0e-8:
        return None

    z_axis = _normalize_vector(np.cross(x_axis, y_axis))
    if np.linalg.norm(z_axis) < 1.0e-8:
        return None

    origin = 0.5 * (points[indices["thumb_tip"]] + points[indices["index_tip"]])
    return {
        "origin": origin.astype(np.float32, copy=False),
        "plane_anchor": plane_anchor.astype(np.float32, copy=False),
        "wrist": wrist.astype(np.float32, copy=False),
        "thumb_points": thumb_points.astype(np.float32, copy=False),
        "index_points": index_points.astype(np.float32, copy=False),
        "thumb_projected": thumb_projected,
        "index_projected": index_projected,
        "thumb_line_point": thumb_line_point.astype(np.float32, copy=False),
        "thumb_line_direction": thumb_line_direction.astype(np.float32, copy=False),
        "index_line_point": index_line_point.astype(np.float32, copy=False),
        "index_line_direction": index_line_direction.astype(np.float32, copy=False),
        "rough_y_axis": rough_y_axis.astype(np.float32, copy=False),
        "x_axis": x_axis.astype(np.float32, copy=False),
        "y_axis": y_axis.astype(np.float32, copy=False),
        "z_axis": z_axis.astype(np.float32, copy=False),
    }


def _compute_global_bounds(sample: dict[str, Any], frame_step: int) -> tuple[np.ndarray, float]:
    clouds: list[np.ndarray] = []
    for side_key in ("left_hand", "right_hand"):
        hand = sample.get(side_key)
        if not isinstance(hand, dict):
            continue
        keypoints = np.asarray(hand.get("keypoints"), dtype=np.float32)
        valid = np.asarray(hand.get("valid"), dtype=np.bool_).reshape(-1)
        if keypoints.ndim != 3:
            continue
        for frame_index in range(0, keypoints.shape[0], max(1, int(frame_step))):
            if frame_index >= valid.shape[0] or not bool(valid[frame_index]):
                continue
            frame_points = keypoints[frame_index]
            if np.isfinite(frame_points).all():
                clouds.append(frame_points)
    if not clouds:
        return np.zeros((3,), dtype=np.float32), 0.1

    stacked = np.concatenate(clouds, axis=0)
    mins = stacked.min(axis=0)
    maxs = stacked.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.6 * float(np.max(maxs - mins))
    radius = max(radius, 0.08)
    return center.astype(np.float32, copy=False), radius


def _line_segment(point: np.ndarray, direction: np.ndarray, support_points: np.ndarray) -> np.ndarray:
    anchor = np.asarray(point, dtype=np.float32).reshape(3)
    axis = _normalize_vector(direction)
    support = np.asarray(support_points, dtype=np.float32).reshape(-1, 3)
    if np.linalg.norm(axis) < 1.0e-8 or support.shape[0] <= 0:
        return np.stack([anchor, anchor], axis=0)

    projection = (support - anchor.reshape(1, 3)) @ axis
    min_t = float(np.min(projection))
    max_t = float(np.max(projection))
    pad = 0.1 * max(max_t - min_t, 0.02)
    return np.stack(
        [
            anchor + (min_t - pad) * axis,
            anchor + (max_t + pad) * axis,
        ],
        axis=0,
    ).astype(np.float32, copy=False)


def _plane_polygon(anchor: np.ndarray, x_axis: np.ndarray, z_axis: np.ndarray, support_points: np.ndarray) -> np.ndarray:
    plane_anchor = np.asarray(anchor, dtype=np.float32).reshape(3)
    x_dir = _normalize_vector(x_axis)
    z_dir = _normalize_vector(z_axis)
    support = np.asarray(support_points, dtype=np.float32).reshape(-1, 3)
    if np.linalg.norm(x_dir) < 1.0e-8 or np.linalg.norm(z_dir) < 1.0e-8 or support.shape[0] <= 0:
        return np.zeros((4, 3), dtype=np.float32)

    delta = support - plane_anchor.reshape(1, 3)
    u = delta @ x_dir
    v = delta @ z_dir
    u_pad = 0.1 * max(float(np.max(u) - np.min(u)), 0.02)
    v_pad = 0.15 * max(float(np.max(v) - np.min(v)), 0.02)
    u0, u1 = float(np.min(u) - u_pad), float(np.max(u) + u_pad)
    v0, v1 = float(np.min(v) - v_pad), float(np.max(v) + v_pad)
    return np.stack(
        [
            plane_anchor + u0 * x_dir + v0 * z_dir,
            plane_anchor + u1 * x_dir + v0 * z_dir,
            plane_anchor + u1 * x_dir + v1 * z_dir,
            plane_anchor + u0 * x_dir + v1 * z_dir,
        ],
        axis=0,
    ).astype(np.float32, copy=False)


def _draw_axis(ax: Any, origin: np.ndarray, direction: np.ndarray, *, length: float, color: str, label: str, linestyle: str = "-") -> None:
    start = np.asarray(origin, dtype=np.float32).reshape(3)
    end = start + float(length) * _normalize_vector(direction)
    ax.plot(
        [float(start[0]), float(end[0])],
        [float(start[1]), float(end[1])],
        [float(start[2]), float(end[2])],
        color=color,
        linewidth=2.5,
        linestyle=linestyle,
    )
    ax.text(float(end[0]), float(end[1]), float(end[2]), label, color=color, fontsize=9)


def _render_hand_subplot(
    ax: Any,
    *,
    geometry: dict[str, Any] | None,
    joints: np.ndarray | None,
    side: str,
    frame_index: int,
    center: np.ndarray,
    radius: float,
    axis_length: float,
) -> None:
    ax.set_title(f"{side} frame={frame_index}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=18.0, azim=-58.0)
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.set_xlim(float(center[0] - radius), float(center[0] + radius))
    ax.set_ylim(float(center[1] - radius), float(center[1] + radius))
    ax.set_zlim(float(center[2] - radius), float(center[2] + radius))

    if joints is None or geometry is None:
        ax.text2D(0.05, 0.92, "invalid / no data", transform=ax.transAxes, fontsize=11)
        return

    points = np.asarray(joints, dtype=np.float32).reshape(-1, 3)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=10, color="lightgray", alpha=0.6)

    thumb_points = geometry["thumb_points"]
    index_points = geometry["index_points"]
    thumb_projected = geometry["thumb_projected"]
    index_projected = geometry["index_projected"]
    all_projected = np.concatenate([thumb_projected, index_projected], axis=0)

    ax.plot(thumb_points[:, 0], thumb_points[:, 1], thumb_points[:, 2], color="darkorange", linewidth=2.0)
    ax.plot(index_points[:, 0], index_points[:, 1], index_points[:, 2], color="deepskyblue", linewidth=2.0)
    ax.scatter(thumb_points[:, 0], thumb_points[:, 1], thumb_points[:, 2], s=24, color="orange")
    ax.scatter(index_points[:, 0], index_points[:, 1], index_points[:, 2], s=24, color="cyan")

    thumb_segment = _line_segment(
        geometry["thumb_line_point"],
        geometry["thumb_line_direction"],
        thumb_projected,
    )
    index_segment = _line_segment(
        geometry["index_line_point"],
        geometry["index_line_direction"],
        index_projected,
    )
    ax.plot(thumb_segment[:, 0], thumb_segment[:, 1], thumb_segment[:, 2], color="firebrick", linestyle="--", linewidth=1.8)
    ax.plot(index_segment[:, 0], index_segment[:, 1], index_segment[:, 2], color="navy", linestyle="--", linewidth=1.8)

    ax.scatter(thumb_projected[:, 0], thumb_projected[:, 1], thumb_projected[:, 2], s=26, color="red", marker="x")
    ax.scatter(index_projected[:, 0], index_projected[:, 1], index_projected[:, 2], s=26, color="blue", marker="x")

    plane_vertices = _plane_polygon(
        geometry["plane_anchor"],
        geometry["x_axis"],
        geometry["z_axis"],
        all_projected,
    )
    if np.isfinite(plane_vertices).all():
        plane = Poly3DCollection([plane_vertices], alpha=0.18, facecolor="limegreen", edgecolor="green")
        ax.add_collection3d(plane)

    ax.scatter(
        [float(geometry["origin"][0])],
        [float(geometry["origin"][1])],
        [float(geometry["origin"][2])],
        s=40,
        color="black",
    )
    ax.scatter(
        [float(geometry["plane_anchor"][0])],
        [float(geometry["plane_anchor"][1])],
        [float(geometry["plane_anchor"][2])],
        s=32,
        color="magenta",
        marker="*",
    )
    ax.scatter(
        [float(geometry["wrist"][0])],
        [float(geometry["wrist"][1])],
        [float(geometry["wrist"][2])],
        s=28,
        color="gray",
        marker="s",
    )

    _draw_axis(ax, geometry["origin"], geometry["x_axis"], length=axis_length, color="red", label="x")
    _draw_axis(ax, geometry["origin"], geometry["y_axis"], length=axis_length, color="green", label="y")
    _draw_axis(ax, geometry["origin"], geometry["z_axis"], length=axis_length, color="blue", label="z")
    _draw_axis(
        ax,
        geometry["origin"],
        geometry["rough_y_axis"],
        length=0.8 * axis_length,
        color="goldenrod",
        label="rough y",
        linestyle=":",
    )

    ax.text2D(
        0.03,
        0.97,
        "orange/cyan: thumb/index\nred/blue x: projected\nred/navy --: fitted lines\ngreen patch: fitted plane",
        transform=ax.transAxes,
        va="top",
        fontsize=8,
    )


def _render_frame(
    figure: Any,
    axes: list[Any],
    *,
    sample_id: str,
    frame_index: int,
    left_geometry: dict[str, Any] | None,
    right_geometry: dict[str, Any] | None,
    left_joints: np.ndarray | None,
    right_joints: np.ndarray | None,
    center: np.ndarray,
    radius: float,
    axis_length: float,
) -> np.ndarray:
    figure.clf()
    left_ax = figure.add_subplot(1, 2, 1, projection="3d")
    right_ax = figure.add_subplot(1, 2, 2, projection="3d")
    figure.suptitle(f"{sample_id} | test retarget geometry | frame={frame_index}", fontsize=14)

    _render_hand_subplot(
        left_ax,
        geometry=left_geometry,
        joints=left_joints,
        side="left",
        frame_index=frame_index,
        center=center,
        radius=radius,
        axis_length=axis_length,
    )
    _render_hand_subplot(
        right_ax,
        geometry=right_geometry,
        joints=right_joints,
        side="right",
        frame_index=frame_index,
        center=center,
        radius=radius,
        axis_length=axis_length,
    )
    figure.tight_layout()
    figure.canvas.draw()
    rgba = np.asarray(figure.canvas.buffer_rgba(), dtype=np.uint8)
    return rgba[:, :, :3].copy()


def _project_points_to_frame(points_3d: np.ndarray, intrinsics: np.ndarray, source_size: tuple[int, int], target_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_3d, dtype=np.float32).reshape(-1, 3)
    projected, valid = project_points(points, intrinsics)
    projected = scale_points_2d(projected, source_size, target_size)
    return np.asarray(projected, dtype=np.float32), np.asarray(valid, dtype=np.bool_)


def _draw_polyline_2d(
    frame: np.ndarray,
    points_2d: np.ndarray,
    valid_mask: np.ndarray,
    *,
    color: np.ndarray,
    thickness: int,
    closed: bool = False,
    point_radius: int = 0,
) -> None:
    points = np.asarray(points_2d, dtype=np.float32).reshape(-1, 2)
    valid = np.asarray(valid_mask, dtype=np.bool_).reshape(-1)
    finite = np.isfinite(points).all(axis=1)
    valid = valid & finite
    for index in range(1, points.shape[0]):
        if not bool(valid[index - 1] and valid[index]):
            continue
        draw_line(frame, points[index - 1], points[index], color, thickness=thickness)
    if closed and points.shape[0] >= 3 and bool(valid[0] and valid[-1]):
        draw_line(frame, points[-1], points[0], color, thickness=thickness)
    if point_radius > 0:
        for point, is_valid in zip(points, valid):
            if not bool(is_valid):
                continue
            draw_disk(
                frame,
                int(round(float(point[0]))),
                int(round(float(point[1]))),
                color,
                radius=point_radius,
            )


def _fit_frame_to_source(frame: np.ndarray, source_size: tuple[int, int]) -> np.ndarray:
    target_height, target_width = int(source_size[0]), int(source_size[1])
    if frame.shape[0] == target_height and frame.shape[1] == target_width:
        return frame

    y_indices = np.linspace(0, frame.shape[0] - 1, target_height).round().astype(np.int32)
    x_indices = np.linspace(0, frame.shape[1] - 1, target_width).round().astype(np.int32)
    return frame[y_indices][:, x_indices].copy()


def _render_overlay_hand(
    frame: np.ndarray,
    *,
    geometry: dict[str, Any] | None,
    side: str,
    intrinsics: np.ndarray,
    source_size: tuple[int, int],
    axis_length: float,
) -> None:
    if geometry is None:
        return

    target_size = (frame.shape[0], frame.shape[1])
    colors = {
        "thumb": np.array([255, 165, 0], dtype=np.uint8),
        "index": np.array([0, 255, 255], dtype=np.uint8),
        "thumb_proj": np.array([255, 0, 0], dtype=np.uint8),
        "index_proj": np.array([0, 128, 255], dtype=np.uint8),
        "thumb_line": np.array([180, 0, 0], dtype=np.uint8),
        "index_line": np.array([0, 0, 180], dtype=np.uint8),
        "plane": np.array([0, 220, 0], dtype=np.uint8),
        "rough_y": np.array([255, 215, 0], dtype=np.uint8),
        "origin": np.array([255, 255, 255], dtype=np.uint8),
        "anchor": np.array([255, 0, 255], dtype=np.uint8),
        "wrist": np.array([128, 128, 128], dtype=np.uint8),
    }

    thumb_points_2d, thumb_valid = _project_points_to_frame(geometry["thumb_points"], intrinsics, source_size, target_size)
    index_points_2d, index_valid = _project_points_to_frame(geometry["index_points"], intrinsics, source_size, target_size)
    thumb_projected_2d, thumb_projected_valid = _project_points_to_frame(
        geometry["thumb_projected"], intrinsics, source_size, target_size
    )
    index_projected_2d, index_projected_valid = _project_points_to_frame(
        geometry["index_projected"], intrinsics, source_size, target_size
    )

    thumb_segment = _line_segment(
        geometry["thumb_line_point"],
        geometry["thumb_line_direction"],
        geometry["thumb_projected"],
    )
    index_segment = _line_segment(
        geometry["index_line_point"],
        geometry["index_line_direction"],
        geometry["index_projected"],
    )
    thumb_segment_2d, thumb_segment_valid = _project_points_to_frame(thumb_segment, intrinsics, source_size, target_size)
    index_segment_2d, index_segment_valid = _project_points_to_frame(index_segment, intrinsics, source_size, target_size)

    plane_vertices = _plane_polygon(
        geometry["plane_anchor"],
        geometry["x_axis"],
        geometry["z_axis"],
        np.concatenate([geometry["thumb_projected"], geometry["index_projected"]], axis=0),
    )
    plane_vertices_2d, plane_vertices_valid = _project_points_to_frame(plane_vertices, intrinsics, source_size, target_size)

    axes_points = np.stack(
        [
            geometry["origin"],
            geometry["origin"] + float(axis_length) * geometry["x_axis"],
            geometry["origin"] + float(axis_length) * geometry["y_axis"],
            geometry["origin"] + float(axis_length) * geometry["z_axis"],
        ],
        axis=0,
    ).astype(np.float32, copy=False)
    axes_points_2d, axes_valid = _project_points_to_frame(axes_points, intrinsics, source_size, target_size)

    rough_y_points = np.stack(
        [
            geometry["origin"],
            geometry["origin"] + float(0.8 * axis_length) * geometry["rough_y_axis"],
        ],
        axis=0,
    ).astype(np.float32, copy=False)
    rough_y_points_2d, rough_y_valid = _project_points_to_frame(rough_y_points, intrinsics, source_size, target_size)

    markers = np.stack([geometry["origin"], geometry["plane_anchor"], geometry["wrist"]], axis=0).astype(np.float32, copy=False)
    marker_points_2d, marker_valid = _project_points_to_frame(markers, intrinsics, source_size, target_size)

    _draw_polyline_2d(frame, plane_vertices_2d, plane_vertices_valid, color=colors["plane"], thickness=2, closed=True)
    _draw_polyline_2d(frame, thumb_segment_2d, thumb_segment_valid, color=colors["thumb_line"], thickness=2)
    _draw_polyline_2d(frame, index_segment_2d, index_segment_valid, color=colors["index_line"], thickness=2)
    _draw_polyline_2d(frame, thumb_points_2d, thumb_valid, color=colors["thumb"], thickness=2, point_radius=3)
    _draw_polyline_2d(frame, index_points_2d, index_valid, color=colors["index"], thickness=2, point_radius=3)
    _draw_polyline_2d(frame, thumb_projected_2d, thumb_projected_valid, color=colors["thumb_proj"], thickness=1, point_radius=2)
    _draw_polyline_2d(frame, index_projected_2d, index_projected_valid, color=colors["index_proj"], thickness=1, point_radius=2)
    _draw_polyline_2d(frame, rough_y_points_2d, rough_y_valid, color=colors["rough_y"], thickness=2)

    if bool(np.all(axes_valid)):
        draw_axes_2d(
            frame,
            axes_points_2d,
            origin_color=colors["origin"],
            axis_colors=(
                np.array([255, 0, 0], dtype=np.uint8),
                np.array([0, 255, 0], dtype=np.uint8),
                np.array([0, 0, 255], dtype=np.uint8),
            ),
            origin_radius=3,
            axis_radius=3,
            thickness=2,
        )

    marker_colors = [colors["origin"], colors["anchor"], colors["wrist"]]
    marker_radii = [3, 3, 3]
    for point_2d, is_valid, color, radius in zip(marker_points_2d, marker_valid, marker_colors, marker_radii):
        if not bool(is_valid) or not np.isfinite(point_2d).all():
            continue
        draw_disk(
            frame,
            int(round(float(point_2d[0]))),
            int(round(float(point_2d[1]))),
            color,
            radius=radius,
        )

    label = f"{side}: orange/cyan raw, red/blue proj, green plane"
    text_y = 24 if side == "left" else 48
    for offset, text in enumerate(label.split("\n")):
        y_pos = text_y + offset * 18
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                xx = max(0, 12 + dx)
                yy = max(0, y_pos + dy)
                try:
                    import cv2  # type: ignore

                    cv2.putText(frame, text, (xx, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
                    cv2.putText(frame, text, (12, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, tuple(int(v) for v in colors["origin"]), 1, cv2.LINE_AA)
                except Exception:
                    pass


def _read_video_frame(reader: Any, frame_index: int) -> np.ndarray:
    frame = np.asarray(reader.get_data(frame_index))
    if frame.ndim == 2:
        frame = np.repeat(frame[:, :, None], 3, axis=2)
    if frame.shape[2] > 3:
        frame = frame[:, :, :3]
    return frame.astype(np.uint8, copy=False)


def _write_sample_overlay_video(
    sample: dict[str, Any],
    *,
    output_path: Path,
    frame_step: int,
    max_frames: int | None,
    render_fps: float | None,
    axis_length: float,
) -> None:
    left_hand = sample["left_hand"]
    right_hand = sample["right_hand"]
    left_keypoints = np.asarray(left_hand["keypoints"], dtype=np.float32)
    right_keypoints = np.asarray(right_hand["keypoints"], dtype=np.float32)
    left_valid = np.asarray(left_hand["valid"], dtype=np.bool_).reshape(-1)
    right_valid = np.asarray(right_hand["valid"], dtype=np.bool_).reshape(-1)
    frame_count = int(max(left_keypoints.shape[0], right_keypoints.shape[0]))
    frame_indices = _iter_frame_indices(frame_count, frame_step, max_frames)
    if not frame_indices:
        raise ValueError("no frames selected for overlay rendering")

    source_size = (int(sample["image_size"][0]), int(sample["image_size"][1]))
    intrinsics = np.asarray(sample["intrinsics"], dtype=np.float32).reshape(3, 3)
    effective_fps = float(render_fps) if render_fps is not None else float(sample["fps"]) / max(1, int(frame_step))
    effective_fps = max(effective_fps, 1.0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    video_path = Path(str(sample["video_path"]))
    reader = imageio.get_reader(video_path) if video_path.exists() else None
    try:
        if reader is not None:
            first_frame = _read_video_frame(reader, frame_indices[0])
            target_size = (int(first_frame.shape[0]), int(first_frame.shape[1]))
        else:
            target_size = source_size

        with imageio.get_writer(
            output_path,
            format="FFMPEG",
            fps=effective_fps,
            codec="libx264",
            macro_block_size=1,
            quality=None,
            ffmpeg_log_level="error",
            output_params=["-preset", "ultrafast", "-crf", "20", "-pix_fmt", "yuv420p"],
        ) as writer:
            for frame_index in frame_indices:
                if reader is not None:
                    frame_rgb = _read_video_frame(reader, frame_index)
                else:
                    frame_rgb = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)

                if frame_rgb.shape[0] != target_size[0] or frame_rgb.shape[1] != target_size[1]:
                    frame_rgb = _fit_frame_to_source(frame_rgb, target_size)

                left_geometry = None
                if frame_index < left_keypoints.shape[0] and frame_index < left_valid.shape[0] and bool(left_valid[frame_index]):
                    left_joints = left_keypoints[frame_index]
                    if np.isfinite(left_joints).all():
                        left_geometry = _compute_test_geometry_frame(left_joints, side="left")

                right_geometry = None
                if frame_index < right_keypoints.shape[0] and frame_index < right_valid.shape[0] and bool(right_valid[frame_index]):
                    right_joints = right_keypoints[frame_index]
                    if np.isfinite(right_joints).all():
                        right_geometry = _compute_test_geometry_frame(right_joints, side="right")

                _render_overlay_hand(
                    frame_rgb,
                    geometry=left_geometry,
                    side="left",
                    intrinsics=intrinsics,
                    source_size=source_size,
                    axis_length=axis_length,
                )
                _render_overlay_hand(
                    frame_rgb,
                    geometry=right_geometry,
                    side="right",
                    intrinsics=intrinsics,
                    source_size=source_size,
                    axis_length=axis_length,
                )
                writer.append_data(frame_rgb)
    finally:
        if reader is not None:
            reader.close()


def _iter_frame_indices(frame_count: int, frame_step: int, max_frames: int | None) -> list[int]:
    indices = list(range(0, frame_count, max(1, int(frame_step))))
    if max_frames is not None:
        indices = indices[: max(0, int(max_frames))]
    return indices


def _write_sample_video(
    sample: dict[str, Any],
    *,
    output_path: Path,
    frame_step: int,
    max_frames: int | None,
    render_fps: float | None,
    axis_length: float,
) -> None:
    left_hand = sample["left_hand"]
    right_hand = sample["right_hand"]
    left_keypoints = np.asarray(left_hand["keypoints"], dtype=np.float32)
    right_keypoints = np.asarray(right_hand["keypoints"], dtype=np.float32)
    left_valid = np.asarray(left_hand["valid"], dtype=np.bool_).reshape(-1)
    right_valid = np.asarray(right_hand["valid"], dtype=np.bool_).reshape(-1)
    frame_count = int(max(left_keypoints.shape[0], right_keypoints.shape[0]))
    frame_indices = _iter_frame_indices(frame_count, frame_step, max_frames)
    if not frame_indices:
        raise ValueError("no frames selected for rendering")

    center, radius = _compute_global_bounds(sample, frame_step=max(1, int(frame_step)))
    effective_fps = float(render_fps) if render_fps is not None else float(sample["fps"]) / max(1, int(frame_step))
    effective_fps = max(effective_fps, 1.0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(14, 7), dpi=120)
    try:
        with imageio.get_writer(
            output_path,
            format="FFMPEG",
            fps=effective_fps,
            codec="libx264",
            macro_block_size=1,
            quality=None,
            ffmpeg_log_level="error",
            output_params=["-preset", "ultrafast", "-crf", "24", "-pix_fmt", "yuv420p"],
        ) as writer:
            for frame_index in frame_indices:
                left_joints = None
                left_geometry = None
                if frame_index < left_keypoints.shape[0] and frame_index < left_valid.shape[0] and bool(left_valid[frame_index]):
                    left_joints = left_keypoints[frame_index]
                    if np.isfinite(left_joints).all():
                        left_geometry = _compute_test_geometry_frame(left_joints, side="left")

                right_joints = None
                right_geometry = None
                if frame_index < right_keypoints.shape[0] and frame_index < right_valid.shape[0] and bool(right_valid[frame_index]):
                    right_joints = right_keypoints[frame_index]
                    if np.isfinite(right_joints).all():
                        right_geometry = _compute_test_geometry_frame(right_joints, side="right")

                frame_rgb = _render_frame(
                    figure,
                    [],
                    sample_id=str(sample["sample_id"]),
                    frame_index=frame_index,
                    left_geometry=left_geometry,
                    right_geometry=right_geometry,
                    left_joints=left_joints,
                    right_joints=right_joints,
                    center=center,
                    radius=radius,
                    axis_length=axis_length,
                )
                writer.append_data(frame_rgb)
    finally:
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize the experimental test retarget geometry on .pose3d_hand examples."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="examples",
        help="Directory containing .pose3d_hand files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/test_retarget_geometry",
        help="Output directory for rendered mp4 files.",
    )
    parser.add_argument(
        "--sample-id",
        action="append",
        default=[],
        help="Optional sample_id (file stem) filter. Can be passed multiple times.",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Render every Nth frame.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on rendered frames per sample.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Optional override for output video fps.",
    )
    parser.add_argument(
        "--axis-length",
        type=float,
        default=0.03,
        help="Axis length in meters for the rendered coordinate frame.",
    )
    parser.add_argument(
        "--render-mode",
        type=str,
        default="both",
        choices=("3d", "overlay", "both"),
        help="Render a 3D debug view, an overlay on the original video, or both.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    input_dir = Path(str(args.input_dir)).expanduser()
    if not input_dir.is_absolute():
        input_dir = (project_root / input_dir).resolve()
    output_dir = Path(str(args.output_dir)).expanduser()
    if not output_dir.is_absolute():
        output_dir = (project_root / output_dir).resolve()

    sample_ids = {str(value) for value in args.sample_id if str(value)}
    pose_paths = _collect_input_paths(input_dir, sample_ids)
    loader = _build_loader(project_root)

    print(f"[test-retarget-viz] input_dir={input_dir}")
    print(f"[test-retarget-viz] output_dir={output_dir}")
    print(f"[test-retarget-viz] rendering {len(pose_paths)} sample(s)")

    for pose_path in pose_paths:
        sample = _load_sample(loader, pose_path)
        sample_id = str(sample["sample_id"])
        if args.render_mode in {"3d", "both"}:
            output_path = output_dir / f"{sample_id}.mp4"
            print(f"[test-retarget-viz] rendering 3d {sample_id} -> {output_path}")
            _write_sample_video(
                sample,
                output_path=output_path,
                frame_step=int(args.frame_step),
                max_frames=args.max_frames,
                render_fps=args.fps,
                axis_length=float(args.axis_length),
            )
        if args.render_mode in {"overlay", "both"}:
            overlay_path = output_dir / f"{sample_id}_overlay.mp4"
            print(f"[test-retarget-viz] rendering overlay {sample_id} -> {overlay_path}")
            _write_sample_overlay_video(
                sample,
                output_path=overlay_path,
                frame_step=int(args.frame_step),
                max_frames=args.max_frames,
                render_fps=args.fps,
                axis_length=float(args.axis_length),
            )

    print("[test-retarget-viz] done")


if __name__ == "__main__":
    main()

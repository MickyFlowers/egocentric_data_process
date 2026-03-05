from __future__ import annotations

import numpy as np


def to_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def ensure_float32(value):
    return to_numpy(value).astype(np.float32, copy=False)


def infer_image_size(img_center):
    center = ensure_float32(img_center).reshape(-1)
    return int(round(float(center[1]) * 2.0)), int(round(float(center[0]) * 2.0))


def build_intrinsics(img_focal, img_center):
    focal = ensure_float32(img_focal).reshape(-1)
    center = ensure_float32(img_center).reshape(-1)
    if focal.size == 1:
        fx = fy = float(focal[0])
    else:
        fx, fy = float(focal[0]), float(focal[1])

    intrinsics = np.eye(3, dtype=np.float32)
    intrinsics[0, 0] = fx
    intrinsics[1, 1] = fy
    intrinsics[0, 2] = float(center[0])
    intrinsics[1, 2] = float(center[1])
    return intrinsics


def resize_image_size_and_intrinsics(image_size, intrinsics, short_side):
    height, width = int(image_size[0]), int(image_size[1])
    output_intrinsics = ensure_float32(intrinsics).copy()
    if short_side is None or short_side <= 0:
        return (height, width), output_intrinsics, 1.0

    scale = float(short_side) / float(min(height, width))
    if abs(scale - 1.0) < 1e-6:
        return (height, width), output_intrinsics, 1.0

    output_height = max(1, int(round(height * scale)))
    output_width = max(1, int(round(width * scale)))
    scale_x = float(output_width) / float(width)
    scale_y = float(output_height) / float(height)
    output_intrinsics[0, 0] *= scale_x
    output_intrinsics[0, 2] *= scale_x
    output_intrinsics[1, 1] *= scale_y
    output_intrinsics[1, 2] *= scale_y
    return (output_height, output_width), output_intrinsics, scale


def project_points(points_cam, intrinsics):
    points = ensure_float32(points_cam)
    intr = ensure_float32(intrinsics)
    z = points[..., 2:3]
    safe_z = np.where(np.abs(z) > 1e-6, z, 1e-6)
    xy = points[..., :2] / safe_z
    projected = np.empty(points.shape[:-1] + (2,), dtype=np.float32)
    projected[..., 0] = xy[..., 0] * intr[0, 0] + intr[0, 2]
    projected[..., 1] = xy[..., 1] * intr[1, 1] + intr[1, 2]
    return projected, (z[..., 0] > 1e-6)


def scale_points_2d(points_2d, source_size, target_size):
    points = ensure_float32(points_2d).copy()
    source_height, source_width = int(source_size[0]), int(source_size[1])
    target_height, target_width = int(target_size[0]), int(target_size[1])
    if source_height <= 0 or source_width <= 0:
        return points
    points[..., 0] *= float(target_width) / float(source_width)
    points[..., 1] *= float(target_height) / float(source_height)
    return points


def hand_connections(joint_count):
    if joint_count >= 21:
        return (
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
        )
    return tuple()


def draw_disk(frame, center_x, center_y, color, radius=2):
    height, width = frame.shape[:2]
    x0 = max(0, center_x - radius)
    x1 = min(width, center_x + radius + 1)
    y0 = max(0, center_y - radius)
    y1 = min(height, center_y + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.ogrid[y0:y1, x0:x1]
    mask = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= radius * radius
    frame[y0:y1, x0:x1][mask] = color


def draw_line(frame, start, end, color, thickness=1):
    x0, y0 = int(round(float(start[0]))), int(round(float(start[1])))
    x1, y1 = int(round(float(end[0]))), int(round(float(end[1])))
    steps = max(abs(x1 - x0), abs(y1 - y0), 1)
    xs = np.rint(np.linspace(x0, x1, steps + 1)).astype(np.int32)
    ys = np.rint(np.linspace(y0, y1, steps + 1)).astype(np.int32)
    radius = max(0, int(thickness) - 1)
    for px, py in zip(xs, ys):
        draw_disk(frame, px, py, color, radius=radius)


def draw_hand_keypoints(frame, points_2d, valid_points, color, radius=2, draw_skeleton=True):
    points = ensure_float32(points_2d)
    valid_mask = np.asarray(valid_points, dtype=np.bool_).reshape(-1)
    finite_mask = np.isfinite(points).all(axis=-1)
    valid_mask = valid_mask & finite_mask

    if draw_skeleton:
        for start, end in hand_connections(points.shape[0]):
            if start >= len(valid_mask) or end >= len(valid_mask):
                continue
            if not valid_mask[start] or not valid_mask[end]:
                continue
            draw_line(frame, points[start], points[end], color, thickness=max(1, radius))

    for idx, point in enumerate(points):
        if not valid_mask[idx]:
            continue
        draw_disk(
            frame,
            int(round(float(point[0]))),
            int(round(float(point[1]))),
            color,
            radius=radius,
        )


def draw_axes_2d(
    frame,
    points_2d,
    origin_color,
    axis_colors=None,
    origin_radius=3,
    axis_radius=2,
    thickness=2,
):
    axis_colors = axis_colors or (
        np.array([255, 0, 0], dtype=np.uint8),
        np.array([0, 255, 0], dtype=np.uint8),
        np.array([0, 0, 255], dtype=np.uint8),
    )
    points = ensure_float32(points_2d)
    if points.shape[0] < 4 or not np.isfinite(points).all():
        return

    origin = points[0]
    draw_disk(
        frame,
        int(round(float(origin[0]))),
        int(round(float(origin[1]))),
        origin_color,
        radius=origin_radius,
    )

    for axis_index, color in enumerate(axis_colors, start=1):
        endpoint = points[axis_index]
        draw_line(frame, origin, endpoint, color, thickness=thickness)
        draw_disk(
            frame,
            int(round(float(endpoint[0]))),
            int(round(float(endpoint[1]))),
            color,
            radius=axis_radius,
        )


def build_mano_edges(faces):
    edges: set[tuple[int, int]] = set()
    face_array = to_numpy(faces).astype(np.int32, copy=False)
    for face in face_array:
        a, b, c = int(face[0]), int(face[1]), int(face[2])
        edges.add(tuple(sorted((a, b))))
        edges.add(tuple(sorted((b, c))))
        edges.add(tuple(sorted((a, c))))
    return tuple(sorted(edges))


def render_mano_wireframe(frame, vertices_cam, intrinsics, faces, color, thickness=1):
    points_2d, valid = project_points(vertices_cam, intrinsics)
    vertex_points = points_2d.reshape(-1, 2)
    valid_mask = valid.reshape(-1)
    edges = build_mano_edges(faces)
    height, width = frame.shape[:2]
    for start, end in edges:
        if not valid_mask[start] or not valid_mask[end]:
            continue
        p0 = vertex_points[start]
        p1 = vertex_points[end]
        if (
            (p0[0] < -width and p1[0] < -width)
            or (p0[1] < -height and p1[1] < -height)
            or (p0[0] > width * 2 and p1[0] > width * 2)
            or (p0[1] > height * 2 and p1[1] > height * 2)
        ):
            continue
        draw_line(frame, p0, p1, color, thickness=thickness)
    return frame

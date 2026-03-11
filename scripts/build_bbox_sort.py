#!/usr/bin/env python3
"""Build a bbox ranking for rendered trajectories using render_meta.json."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.retarget_utils import build_transform_matrix, pose_vector_to_matrix
from utils.safe_io import atomic_write_json


def _load_render_sample_ids(render_meta_path: Path) -> list[str]:
    with open(render_meta_path, "r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)

    if not isinstance(payload, list):
        raise ValueError(f"render_meta.json must be a list of sample_id strings: {render_meta_path}")

    sample_ids: list[str] = []
    seen: set[str] = set()
    for item in payload:
        if not isinstance(item, str) or not item:
            continue
        if item in seen:
            continue
        seen.add(item)
        sample_ids.append(item)
    return sample_ids


def _resolve_parquet_path(input_dir: Path, *, data_dir_name: str, sample_id: str) -> Path:
    relative = Path(sample_id)
    return (input_dir / data_dir_name / relative.parent / f"{relative.stem}.parquet").resolve()


def _load_pose_columns(parquet_path: Path) -> dict[str, Any]:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pyarrow is required for reading trajectory parquet files. Install it with `python -m pip install pyarrow`."
        ) from exc

    schema_names = set(pq.read_schema(parquet_path).names)
    columns = [name for name in ("left_pose", "right_pose") if name in schema_names]
    if not columns:
        raise ValueError(f"parquet is missing left_pose/right_pose columns: {parquet_path}")
    return pq.read_table(parquet_path, columns=columns).to_pydict()


def _extract_world_positions(rows: Any, *, base_transform: np.ndarray) -> np.ndarray:
    if not isinstance(rows, list):
        return np.empty((0, 3), dtype=np.float64)

    positions: list[np.ndarray] = []
    for row in rows:
        if row is None:
            continue
        try:
            pose = np.asarray(row, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            continue
        if pose.size < 6 or not np.isfinite(pose[:6]).all():
            continue

        # Stored left_pose/right_pose are expressed in each arm base frame.
        world_from_hand = base_transform @ pose_vector_to_matrix(pose[:6]).astype(np.float64)
        positions.append(np.asarray(world_from_hand[:3, 3], dtype=np.float64))

    if not positions:
        return np.empty((0, 3), dtype=np.float64)
    return np.stack(positions, axis=0)


def _build_bbox_entry(
    *,
    sample_id: str,
    parquet_path: Path,
    left_base_transform: np.ndarray,
    right_base_transform: np.ndarray,
) -> dict[str, Any] | None:
    payload = _load_pose_columns(parquet_path)
    left_positions = _extract_world_positions(payload.get("left_pose"), base_transform=left_base_transform)
    right_positions = _extract_world_positions(payload.get("right_pose"), base_transform=right_base_transform)

    combined = []
    if left_positions.size > 0:
        combined.append(left_positions)
    if right_positions.size > 0:
        combined.append(right_positions)
    if not combined:
        return None

    points = np.concatenate(combined, axis=0)
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    bbox_size = bbox_max - bbox_min
    bbox_volume = float(np.prod(bbox_size))

    return {
        "sample_id": sample_id,
        "trajectory_path": str(parquet_path),
        "bbox_min_world": [float(value) for value in bbox_min],
        "bbox_max_world": [float(value) for value in bbox_max],
        "bbox_size_world": [float(value) for value in bbox_size],
        "bbox_volume_world": bbox_volume,
        "left_valid_pose_count": int(left_positions.shape[0]),
        "right_valid_pose_count": int(right_positions.shape[0]),
        "valid_pose_count": int(points.shape[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read rendered sample IDs from render_meta.json and rank trajectories by combined two-hand world-space bbox volume."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("./outputs"),
        help="Processed output root containing render_meta.json and data/.",
    )
    parser.add_argument(
        "--render-meta-path",
        type=Path,
        default=None,
        help="Optional explicit render_meta.json path. Defaults to <input-dir>/render_meta.json.",
    )
    parser.add_argument(
        "--data-dir-name",
        type=str,
        default="data",
        help="Trajectory parquet directory name under input-dir.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Output json path. Defaults to <input-dir>/bbox_sort.json.",
    )
    parser.add_argument(
        "--left-base-translation",
        type=float,
        nargs=3,
        default=(0.0, 0.3, 0.0),
        metavar=("X", "Y", "Z"),
        help="Left arm base translation used to recover world coordinates from stored left_pose.",
    )
    parser.add_argument(
        "--right-base-translation",
        type=float,
        nargs=3,
        default=(0.0, -0.3, 0.0),
        metavar=("X", "Y", "Z"),
        help="Right arm base translation used to recover world coordinates from stored right_pose.",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    render_meta_path = (
        args.render_meta_path.expanduser().resolve()
        if args.render_meta_path is not None
        else (input_dir / "render_meta.json").resolve()
    )
    output_path = (
        args.output_path.expanduser().resolve()
        if args.output_path is not None
        else (input_dir / "bbox_sort.json").resolve()
    )

    if not render_meta_path.exists():
        raise FileNotFoundError(f"render_meta.json not found: {render_meta_path}")

    sample_ids = _load_render_sample_ids(render_meta_path)
    left_base_transform = build_transform_matrix(translation=args.left_base_translation).astype(np.float64)
    right_base_transform = build_transform_matrix(translation=args.right_base_translation).astype(np.float64)

    entries: list[dict[str, Any]] = []
    missing_count = 0
    empty_count = 0
    for sample_id in sample_ids:
        parquet_path = _resolve_parquet_path(input_dir, data_dir_name=args.data_dir_name, sample_id=sample_id)
        if not parquet_path.exists():
            missing_count += 1
            print(f"skip missing parquet: {parquet_path}")
            continue

        entry = _build_bbox_entry(
            sample_id=sample_id,
            parquet_path=parquet_path,
            left_base_transform=left_base_transform,
            right_base_transform=right_base_transform,
        )
        if entry is None:
            empty_count += 1
            print(f"skip empty bbox: {sample_id}")
            continue
        entries.append(entry)

    entries.sort(key=lambda entry: (-float(entry["bbox_volume_world"]), str(entry["sample_id"])))
    atomic_write_json(output_path, entries, indent=2, sort_keys=False)

    print(f"render_meta_path: {render_meta_path}")
    print(f"output_path: {output_path}")
    print(f"sample_ids_from_render_meta: {len(sample_ids)}")
    print(f"bbox_entries_written: {len(entries)}")
    print(f"missing_parquet: {missing_count}")
    print(f"empty_bbox: {empty_count}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""统计左右手 reachable ratio 都高于阈值的轨迹数量。"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _extract_sample_ids_from_meta(payload: Any) -> list[str]:
    sample_ids: list[str] = []

    def _add(value: Any) -> None:
        if isinstance(value, str) and value:
            sample_ids.append(value)

    if isinstance(payload, dict):
        _add(payload.get("sample_id"))
        values = payload.get("sample_ids")
        if isinstance(values, list):
            for item in values:
                _add(item)
    elif isinstance(payload, list):
        for entry in payload:
            if not isinstance(entry, dict):
                continue
            _add(entry.get("sample_id"))
            values = entry.get("sample_ids")
            if isinstance(values, list):
                for item in values:
                    _add(item)

    return list(dict.fromkeys(sample_ids))


def _extract_ratio_from_ik_section(meta_data: dict[str, Any], side: str) -> float | None:
    ik_section = meta_data.get("ik")
    if not isinstance(ik_section, dict):
        return None
    side_section = ik_section.get(side)
    if not isinstance(side_section, dict):
        return None
    reachable = side_section.get("reachable")
    if not isinstance(reachable, list) or len(reachable) == 0:
        return None
    valid_flags: list[float] = []
    for item in reachable:
        value = _safe_float(item)
        if value is None:
            continue
        valid_flags.append(1.0 if value > 0.5 else 0.0)
    if not valid_flags:
        return None
    return float(sum(valid_flags) / len(valid_flags))


def _extract_left_right_ratio(meta_data: dict[str, Any]) -> tuple[float | None, float | None]:
    left = _safe_float(meta_data.get("left_reachable_ratio"))
    right = _safe_float(meta_data.get("right_reachable_ratio"))
    if left is None:
        left = _safe_float(meta_data.get("left_reachable"))
    if right is None:
        right = _safe_float(meta_data.get("right_reachable"))
    if left is None:
        left = _extract_ratio_from_ik_section(meta_data, "left")
    if right is None:
        right = _extract_ratio_from_ik_section(meta_data, "right")
    return left, right


def _iter_meta_data_paths(input_dir: Path, meta_data_dir_name: str) -> list[Path]:
    meta_data_dir = (input_dir / meta_data_dir_name).resolve()
    if not meta_data_dir.exists():
        return []
    return sorted(meta_data_dir.rglob("*.json"))


def _load_meta_data_path_by_sample_id(input_dir: Path, sample_id: str, meta_data_dir_name: str) -> Path:
    relative = Path(sample_id)
    return (input_dir / meta_data_dir_name / relative.parent / f"{relative.stem}.json").resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Count trajectories with left/right reachable ratio above threshold.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("./outputs"),
        help="Processed output root that contains meta.json and meta_data/",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.95,
        help="Threshold for both left/right reachable ratio.",
    )
    parser.add_argument(
        "--meta-data-dir",
        type=str,
        default="meta_data",
        help="Meta-data directory name under input-dir.",
    )
    parser.add_argument(
        "--print-pass-ids",
        action="store_true",
        help="Print sample_id list that passes threshold.",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    threshold = float(args.threshold)
    meta_json_path = input_dir / "meta.json"

    meta_data_paths: list[Path] = []
    used_meta_json = False
    if meta_json_path.exists():
        with open(meta_json_path, "r", encoding="utf-8") as file_obj:
            payload = json.load(file_obj)
        sample_ids = _extract_sample_ids_from_meta(payload)
        if sample_ids:
            used_meta_json = True
            for sample_id in sample_ids:
                path = _load_meta_data_path_by_sample_id(
                    input_dir=input_dir,
                    sample_id=sample_id,
                    meta_data_dir_name=args.meta_data_dir,
                )
                if path.exists():
                    meta_data_paths.append(path)
    if not meta_data_paths:
        meta_data_paths = _iter_meta_data_paths(input_dir=input_dir, meta_data_dir_name=args.meta_data_dir)

    total = len(meta_data_paths)
    valid = 0
    passed = 0
    missing_ratio = 0
    pass_ids: list[str] = []

    for path in meta_data_paths:
        with open(path, "r", encoding="utf-8") as file_obj:
            meta_data = json.load(file_obj)
        if not isinstance(meta_data, dict):
            continue
        sample_id = meta_data.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            sample_id = path.stem

        left_ratio, right_ratio = _extract_left_right_ratio(meta_data)
        if left_ratio is None or right_ratio is None:
            missing_ratio += 1
            continue
        valid += 1
        if left_ratio > threshold and right_ratio > threshold:
            passed += 1
            pass_ids.append(sample_id)

    fail = max(valid - passed, 0)
    pass_rate = (float(passed) / float(valid)) if valid > 0 else 0.0

    print(f"input_dir: {input_dir}")
    print(f"meta_source: {'meta.json' if used_meta_json else args.meta_data_dir}")
    print(f"threshold: {threshold:.6f}")
    print(f"total_meta_files: {total}")
    print(f"valid_with_ratio: {valid}")
    print(f"missing_ratio: {missing_ratio}")
    print(f"pass_count: {passed}")
    print(f"fail_count: {fail}")
    print(f"pass_rate: {pass_rate:.6f}")

    if args.print_pass_ids:
        for sample_id in pass_ids:
            print(sample_id)


if __name__ == "__main__":
    main()

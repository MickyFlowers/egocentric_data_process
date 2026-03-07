#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any


SQLITE_SUFFIXES = {".db", ".sqlite", ".sqlite3"}


def _resolve_manifest_path(path_value: str) -> tuple[Path, str]:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    else:
        path = path.resolve()

    if path.suffix in SQLITE_SUFFIXES:
        if not path.exists():
            raise FileNotFoundError(f"manifest sqlite does not exist: {path}")
        return path, "sqlite"

    sqlite_path = Path(f"{path}.sqlite3")
    if sqlite_path.exists():
        return sqlite_path.resolve(), "sqlite"
    if path.exists():
        return path, "json"
    raise FileNotFoundError(f"manifest not found: {path} or {sqlite_path}")


def _read_sqlite_manifest(path: Path) -> tuple[dict[str, int], dict[str, list[str]]]:
    counts: dict[str, int] = {}
    samples: dict[str, list[str]] = {}
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute("SELECT status, COUNT(*) AS count FROM tasks GROUP BY status").fetchall()
        for row in rows:
            counts[str(row["status"])] = int(row["count"])

        sample_rows = connection.execute("SELECT sample_id, status FROM tasks ORDER BY sample_id").fetchall()
        for row in sample_rows:
            status = str(row["status"])
            samples.setdefault(status, []).append(str(row["sample_id"]))
    finally:
        connection.close()
    return counts, samples


def _read_json_manifest(path: Path) -> tuple[dict[str, int], dict[str, list[str]]]:
    with open(path, "r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)

    tasks = payload.get("tasks", {}) if isinstance(payload, dict) else {}
    if not isinstance(tasks, dict):
        raise ValueError(f"legacy manifest has invalid tasks payload: {path}")

    counts: dict[str, int] = {}
    samples: dict[str, list[str]] = {}
    for sample_id, entry in tasks.items():
        if not isinstance(sample_id, str) or not sample_id or not isinstance(entry, dict):
            continue
        status = str(entry.get("status", "pending"))
        counts[status] = counts.get(status, 0) + 1
        samples.setdefault(status, []).append(sample_id)
    for sample_ids in samples.values():
        sample_ids.sort()
    return counts, samples


def _load_manifest(path: Path, manifest_type: str) -> tuple[dict[str, int], dict[str, list[str]]]:
    if manifest_type == "sqlite":
        return _read_sqlite_manifest(path)
    if manifest_type == "json":
        return _read_json_manifest(path)
    raise ValueError(f"unsupported manifest type: {manifest_type}")


def _merge_counts(target: dict[str, int], source: dict[str, int]) -> None:
    for status, count in source.items():
        target[status] = target.get(status, 0) + int(count)


def _merge_samples(target: dict[str, list[str]], source: dict[str, list[str]]) -> None:
    for status, sample_ids in source.items():
        bucket = target.setdefault(status, [])
        bucket.extend(sample_ids)


def _print_sample_list(title: str, sample_ids: list[str], limit: int) -> None:
    if not sample_ids:
        return
    print(f"{title} ({len(sample_ids)}):")
    for sample_id in sample_ids[:limit]:
        print(f"  {sample_id}")
    remaining = len(sample_ids) - min(len(sample_ids), limit)
    if remaining > 0:
        print(f"  ... {remaining} more")


def _normalize_statuses(counts: dict[str, int]) -> dict[str, int]:
    normalized = {"pending": 0, "in_progress": 0, "completed": 0}
    normalized.update(counts)
    return normalized


def _deduplicate_paths(path_values: list[str]) -> list[tuple[Path, str]]:
    resolved: list[tuple[Path, str]] = []
    seen: set[tuple[str, str]] = set()
    for value in path_values:
        path, manifest_type = _resolve_manifest_path(value)
        key = (str(path), manifest_type)
        if key in seen:
            continue
        seen.add(key)
        resolved.append((path, manifest_type))
    return resolved


def _discover_default_manifests(cwd: Path) -> list[tuple[Path, str]]:
    discovered = sorted(path.resolve() for path in cwd.glob("*.sqlite3") if path.is_file())
    if not discovered:
        raise FileNotFoundError(f"no *.sqlite3 manifest found in current directory: {cwd}")
    return [(path, "sqlite") for path in discovered]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quickly check whether a dataset is fully processed by reading manifest only.",
    )
    parser.add_argument(
        "manifest_paths",
        nargs="*",
        help="Manifest path(s). If omitted, the script reads all *.sqlite3 files in the current directory.",
    )
    parser.add_argument(
        "--show-samples",
        type=int,
        default=20,
        help="How many sample_ids to print for each non-completed status.",
    )
    parser.add_argument(
        "--show-completed",
        action="store_true",
        help="Also print completed sample_ids.",
    )
    args = parser.parse_args()

    if args.manifest_paths:
        manifest_specs = _deduplicate_paths(list(args.manifest_paths))
    else:
        manifest_specs = _discover_default_manifests(Path.cwd())
    total_counts: dict[str, int] = {}
    total_samples: dict[str, list[str]] = {}

    for manifest_path, manifest_type in manifest_specs:
        counts, samples = _load_manifest(manifest_path, manifest_type)
        counts = _normalize_statuses(counts)
        _merge_counts(total_counts, counts)
        _merge_samples(total_samples, samples)

        total = sum(counts.values())
        completed = counts.get("completed", 0)
        ratio = 0.0 if total == 0 else completed / float(total)
        status_text = "complete" if total > 0 and completed == total else "incomplete"

        print(f"manifest: {manifest_path} ({manifest_type})")
        print(f"  total: {total}")
        print(f"  completed: {completed}")
        print(f"  pending: {counts.get('pending', 0)}")
        print(f"  in_progress: {counts.get('in_progress', 0)}")
        extra_statuses = sorted(status for status in counts if status not in {"pending", "in_progress", "completed"})
        for status in extra_statuses:
            print(f"  {status}: {counts[status]}")
        print(f"  completion_ratio: {ratio:.6f}")
        print(f"  status: {status_text}")

    if len(manifest_specs) <= 1:
        aggregate_counts = _normalize_statuses(total_counts)
    else:
        aggregate_counts = _normalize_statuses(total_counts)
        total = sum(aggregate_counts.values())
        completed = aggregate_counts.get("completed", 0)
        ratio = 0.0 if total == 0 else completed / float(total)
        status_text = "complete" if total > 0 and completed == total else "incomplete"
        print("aggregate:")
        print(f"  manifests: {len(manifest_specs)}")
        print(f"  total: {total}")
        print(f"  completed: {completed}")
        print(f"  pending: {aggregate_counts.get('pending', 0)}")
        print(f"  in_progress: {aggregate_counts.get('in_progress', 0)}")
        extra_statuses = sorted(
            status for status in aggregate_counts if status not in {"pending", "in_progress", "completed"}
        )
        for status in extra_statuses:
            print(f"  {status}: {aggregate_counts[status]}")
        print(f"  completion_ratio: {ratio:.6f}")
        print(f"  status: {status_text}")

    pending_ids = sorted(total_samples.get("pending", []))
    in_progress_ids = sorted(total_samples.get("in_progress", []))
    _print_sample_list("pending_samples", pending_ids, args.show_samples)
    _print_sample_list("in_progress_samples", in_progress_ids, args.show_samples)

    if args.show_completed:
        completed_ids = sorted(total_samples.get("completed", []))
        _print_sample_list("completed_samples", completed_ids, args.show_samples)


if __name__ == "__main__":
    main()

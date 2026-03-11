#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import yaml

if __package__ in {None, ""}:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from process.inverse_kinematics_process import InverseKinematicsProcess
from process.load_data_process import LoadDataProcess
from process.retarget_process import RetargetProcess
from utils.oss_utils import configure_path_mapping

try:
    import ray
except ImportError:  # pragma: no cover - optional dependency
    ray = None

project_root = Path(__file__).resolve().parents[1]

def _load_process_configs(config_path: Path) -> dict[str, dict[str, Any]]:
    with open(config_path, "r", encoding="utf-8") as file_obj:
        payload = yaml.safe_load(file_obj)
    if not isinstance(payload, list):
        raise ValueError(f"process config must be a list: {config_path}")

    configs: dict[str, dict[str, Any]] = {}
    for item in payload:
        if not isinstance(item, dict):
            continue
        process_type = item.get("type")
        if not isinstance(process_type, str) or not process_type:
            continue
        process_config = copy.deepcopy(item)
        process_config["_config_root"] = str(project_root)
        configs[process_type] = process_config
    return configs


def _build_axis_values(lower: float, upper: float, step: float) -> list[float]:
    if step <= 0.0:
        raise ValueError("grid step must be > 0")
    if upper < lower:
        raise ValueError("grid upper bound must be >= lower bound")

    values: list[float] = []
    current = float(lower)
    eps = step * 1.0e-6
    while current <= upper + eps:
        values.append(float(round(current, 10)))
        current += step
    return values


def _infer_sample_id(data_path: str) -> str:
    path = Path(data_path)
    name = path.name
    if name.endswith(".pose3d_hand"):
        return name[: -len(".pose3d_hand")]
    return path.stem


def _infer_video_path(data_path: str) -> str:
    if data_path.endswith(".pose3d_hand"):
        return data_path[: -len(".pose3d_hand")] + ".mp4"
    return str(Path(data_path).with_suffix(".mp4"))


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def _compute_objective(left_ratio: float | None, right_ratio: float | None, objective: str) -> float | None:
    if objective == "left":
        return left_ratio
    if objective == "right":
        return right_ratio
    if left_ratio is None or right_ratio is None:
        return None
    if objective == "mean":
        return float((left_ratio + right_ratio) * 0.5)
    if objective == "min":
        return float(min(left_ratio, right_ratio))
    raise ValueError(f"unsupported objective: {objective}")


def _collect_data_paths_from_input_dir(input_dir: str) -> list[str]:
    root = Path(input_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"input_dir does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"input_dir is not a directory: {root}")

    return sorted(str(path.resolve()) for path in root.rglob("*.pose3d_hand"))


def _build_sample_specs(args: argparse.Namespace) -> list[dict[str, str]]:
    data_paths: list[str] = []
    if args.input_dir is not None:
        data_paths.extend(_collect_data_paths_from_input_dir(args.input_dir))
    if args.data_path is not None:
        data_paths.extend(str(path) for path in args.data_path)
    data_paths = list(dict.fromkeys(data_paths))
    if not data_paths:
        raise ValueError("no input .pose3d_hand files found")

    if args.video_path is not None and len(args.video_path) not in {0, len(data_paths)}:
        raise ValueError("--video-path count must match --data-path count")
    if args.sample_id is not None and len(args.sample_id) not in {0, len(data_paths)}:
        raise ValueError("--sample-id count must match --data-path count")

    video_paths = [str(path) for path in args.video_path] if args.video_path else []
    sample_ids = list(args.sample_id) if args.sample_id else []
    specs: list[dict[str, str]] = []
    for index, data_path in enumerate(data_paths):
        sample_id = sample_ids[index] if index < len(sample_ids) else _infer_sample_id(data_path)
        video_path = video_paths[index] if index < len(video_paths) else _infer_video_path(data_path)
        specs.append(
            {
                "sample_id": sample_id,
                "data_path": data_path,
                "video_path": video_path,
            }
        )
    return specs


def _load_samples(
    *,
    specs: list[dict[str, str]],
    load_process: LoadDataProcess,
) -> list[dict[str, Any]]:
    loaded_samples: list[dict[str, Any]] = []
    for spec in specs:
        sample = {
            "sample_id": spec["sample_id"],
            "data_path": spec["data_path"],
            "video_path": spec["video_path"],
            "visualize": False,
        }
        loaded_samples.append(load_process(sample, context=None))
    return loaded_samples


def _evaluate_workspace_center(
    *,
    loaded_samples: list[dict[str, Any]],
    retarget_template: dict[str, Any],
    ik_process: InverseKinematicsProcess,
    workspace_center: list[float],
    retarget_scheme: str,
    objective: str,
    threshold: float,
) -> dict[str, Any]:
    retarget_config = copy.deepcopy(retarget_template)
    retarget_config.setdefault("params", {})
    retarget_config["params"]["workspace_center"] = workspace_center
    retarget_config["params"]["retarget_scheme"] = str(retarget_scheme).strip().lower() or "test"
    retarget_process = RetargetProcess(retarget_config)

    sample_results: list[dict[str, Any]] = []
    objective_values: list[float] = []
    left_values: list[float] = []
    right_values: list[float] = []
    pass_count = 0

    for loaded_sample in loaded_samples:
        current = retarget_process(dict(loaded_sample), context=None)
        current = ik_process(current, context=None)
        ik_payload = current.get("ik", {})
        ik_section = ik_payload.get("ik", {}) if isinstance(ik_payload, dict) else {}
        left_ratio = _safe_float(
            ((ik_section.get("left") or {}) if isinstance(ik_section, dict) else {}).get("reachable_ratio")
        )
        right_ratio = _safe_float(
            ((ik_section.get("right") or {}) if isinstance(ik_section, dict) else {}).get("reachable_ratio")
        )
        if objective == "pass_ratio":
            passed = bool(
                left_ratio is not None
                and right_ratio is not None
                and left_ratio > threshold
                and right_ratio > threshold
            )
            score = 1.0 if passed else 0.0
        else:
            passed = bool(
                left_ratio is not None
                and right_ratio is not None
                and left_ratio > threshold
                and right_ratio > threshold
            )
            score = _compute_objective(left_ratio, right_ratio, objective)

        if left_ratio is not None:
            left_values.append(left_ratio)
        if right_ratio is not None:
            right_values.append(right_ratio)
        if score is not None:
            objective_values.append(score)
        if passed:
            pass_count += 1

        sample_results.append(
            {
                "sample_id": str(loaded_sample["sample_id"]),
                "left_reachable_ratio": left_ratio,
                "right_reachable_ratio": right_ratio,
                "passed_threshold": passed,
                "objective": score,
            }
        )

    pass_ratio = float(pass_count / len(sample_results)) if sample_results else None
    return {
        "workspace_center": [float(workspace_center[0]), float(workspace_center[1]), float(workspace_center[2])],
        "objective": float(np.mean(objective_values)) if objective_values else None,
        "valid_sample_count": int(len(objective_values)),
        "sample_count": int(len(sample_results)),
        "pass_count": int(pass_count),
        "pass_ratio": pass_ratio,
        "threshold": float(threshold),
        "left_reachable_ratio_mean": float(np.mean(left_values)) if left_values else None,
        "right_reachable_ratio_mean": float(np.mean(right_values)) if right_values else None,
        "samples": sample_results,
    }


def _build_workspace_centers(*, x_values: list[float], workspace_y: float, z_values: list[float]) -> list[list[float]]:
    centers: list[list[float]] = []
    for x_value in x_values:
        for z_value in z_values:
            centers.append([float(x_value), float(workspace_y), float(z_value)])
    return centers


def _batched_workspace_centers(
    workspace_centers: list[list[float]],
    *,
    batch_size: int,
) -> list[list[list[float]]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    return [
        workspace_centers[start_index : start_index + batch_size]
        for start_index in range(0, len(workspace_centers), batch_size)
    ]


def _evaluate_workspace_centers_batch(
    *,
    loaded_samples: list[dict[str, Any]],
    retarget_template: dict[str, Any],
    ik_process: InverseKinematicsProcess,
    workspace_centers: list[list[float]],
    retarget_scheme: str,
    objective: str,
    threshold: float,
) -> list[dict[str, Any]]:
    return [
        _evaluate_workspace_center(
            loaded_samples=loaded_samples,
            retarget_template=retarget_template,
            ik_process=ik_process,
            workspace_center=workspace_center,
            retarget_scheme=retarget_scheme,
            objective=objective,
            threshold=threshold,
        )
        for workspace_center in workspace_centers
    ]


def _print_progress(
    *,
    index: int,
    total: int,
    result: dict[str, Any],
) -> None:
    print(
        f"[{index}/{total}] workspace_center={result['workspace_center']} "
        f"objective={result['objective']} "
        f"pass_ratio={result['pass_ratio']} "
        f"left_mean={result['left_reachable_ratio_mean']} "
        f"right_mean={result['right_reachable_ratio_mean']}"
    )


if ray is not None:

    @ray.remote
    class _WorkspaceSearchWorker:
        def __init__(
            self,
            *,
            retarget_template: dict[str, Any],
            ik_config: dict[str, Any],
            loaded_samples: list[dict[str, Any]],
            local_mount: str | None,
            oss_prefix: str,
        ) -> None:
            if local_mount:
                configure_path_mapping(local_mount, oss_prefix)
            self._retarget_template = copy.deepcopy(retarget_template)
            self._ik_process = InverseKinematicsProcess(copy.deepcopy(ik_config))
            self._loaded_samples = loaded_samples

        def evaluate_batch(
            self,
            *,
            workspace_centers: list[list[float]],
            retarget_scheme: str,
            objective: str,
            threshold: float,
        ) -> list[dict[str, Any]]:
            return _evaluate_workspace_centers_batch(
                loaded_samples=self._loaded_samples,
                retarget_template=self._retarget_template,
                ik_process=self._ik_process,
                workspace_centers=workspace_centers,
                retarget_scheme=retarget_scheme,
                objective=objective,
                threshold=threshold,
            )


def _evaluate_workspace_centers_with_ray(
    *,
    loaded_samples: list[dict[str, Any]],
    retarget_template: dict[str, Any],
    ik_config: dict[str, Any],
    workspace_centers: list[list[float]],
    retarget_scheme: str,
    objective: str,
    threshold: float,
    num_workers: int,
    batch_size: int,
    local_mount: str | None,
    oss_prefix: str,
    ray_address: str | None,
) -> list[dict[str, Any]]:
    if ray is None:
        raise ModuleNotFoundError("ray is required for parallel grid search. Install it with `python -m pip install ray`.")

    worker_count = max(1, int(num_workers))
    batches = _batched_workspace_centers(workspace_centers, batch_size=max(1, int(batch_size)))
    if not batches:
        return []

    ray.init(address=ray_address, ignore_reinit_error=True)
    try:
        loaded_samples_ref = ray.put(loaded_samples)
        workers = [
            _WorkspaceSearchWorker.options(num_cpus=1).remote(
                retarget_template=copy.deepcopy(retarget_template),
                ik_config=copy.deepcopy(ik_config),
                loaded_samples=loaded_samples_ref,
                local_mount=local_mount,
                oss_prefix=oss_prefix,
            )
            for _ in range(worker_count)
        ]

        pending_refs: dict[Any, tuple[int, Any]] = {}
        next_batch_index = 0
        for worker in workers:
            if next_batch_index >= len(batches):
                break
            batch = batches[next_batch_index]
            pending_refs[
                worker.evaluate_batch.remote(
                    workspace_centers=batch,
                    retarget_scheme=retarget_scheme,
                    objective=objective,
                    threshold=threshold,
                )
            ] = (next_batch_index, worker)
            next_batch_index += 1

        ordered_results: list[list[dict[str, Any]] | None] = [None] * len(batches)
        completed = 0
        total = len(workspace_centers)
        while pending_refs:
            ready_refs, _ = ray.wait(list(pending_refs.keys()), num_returns=1)
            ready_ref = ready_refs[0]
            batch_index, worker = pending_refs.pop(ready_ref)
            batch_results = ray.get(ready_ref)
            ordered_results[batch_index] = batch_results
            for result in batch_results:
                completed += 1
                _print_progress(index=completed, total=total, result=result)

            if next_batch_index < len(batches):
                next_batch = batches[next_batch_index]
                pending_refs[
                    worker.evaluate_batch.remote(
                        workspace_centers=next_batch,
                        retarget_scheme=retarget_scheme,
                        objective=objective,
                        threshold=threshold,
                    )
                ] = (next_batch_index, worker)
                next_batch_index += 1

        flattened: list[dict[str, Any]] = []
        for batch_results in ordered_results:
            if batch_results:
                flattened.extend(batch_results)
        return flattened
    finally:
        ray.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid search workspace_center[x, z] by IK reachable_ratio.")
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Directory to recursively search for .pose3d_hand files.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        nargs="+",
        default=None,
        help="One or more input .pose3d_hand paths.",
    )
    parser.add_argument(
        "--video-path",
        type=str,
        nargs="*",
        default=None,
        help="Optional mp4 paths. If omitted, inferred from data-path.",
    )
    parser.add_argument(
        "--sample-id",
        type=str,
        nargs="*",
        default=None,
        help="Optional sample ids. If omitted, inferred from data-path.",
    )
    parser.add_argument(
        "--process-config",
        type=Path,
        default=Path("./config/processes/process.yaml"),
        help="Process config YAML that contains load_data / retarget / inverse_kinematics.",
    )
    parser.add_argument("--x-min", type=float, required=True, help="Grid lower bound for workspace_center x.")
    parser.add_argument("--x-max", type=float, required=True, help="Grid upper bound for workspace_center x.")
    parser.add_argument("--x-step", type=float, required=True, help="Grid step for workspace_center x.")
    parser.add_argument("--z-min", type=float, required=True, help="Grid lower bound for workspace_center z.")
    parser.add_argument("--z-max", type=float, required=True, help="Grid upper bound for workspace_center z.")
    parser.add_argument("--z-step", type=float, required=True, help="Grid step for workspace_center z.")
    parser.add_argument(
        "--workspace-y",
        type=float,
        default=None,
        help="Workspace center y. Defaults to retarget config value.",
    )
    parser.add_argument(
        "--retarget-scheme",
        type=str,
        default="test",
        help="Retarget scheme passed into RetargetProcess. Defaults to 'test'.",
    )
    parser.add_argument(
        "--objective",
        choices=("pass_ratio", "min", "mean", "left", "right"),
        default="pass_ratio",
        help="Optimization target. 'pass_ratio' means proportion of samples with left/right ratio > threshold.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.95,
        help="Pass threshold used by pass_ratio and reported sample pass/fail.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="How many best grid points to print.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional output json path for full search results.",
    )
    parser.add_argument(
        "--local-mount",
        type=str,
        default=None,
        help="Optional path mapping local_mount for oss:// inputs.",
    )
    parser.add_argument(
        "--oss-prefix",
        type=str,
        default="oss://",
        help="Optional path mapping oss_prefix for oss:// inputs.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=max(1, os.cpu_count() or 1),
        help="Parallel Ray workers for workspace search. Use 1 to run serially.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="How many workspace centers each Ray task evaluates before returning.",
    )
    parser.add_argument(
        "--ray-address",
        type=str,
        default=None,
        help="Optional Ray cluster address. Defaults to starting a local Ray instance.",
    )
    args = parser.parse_args()
    if args.input_dir is None and not args.data_path:
        raise ValueError("either --input-dir or --data-path is required")

    if args.local_mount:
        configure_path_mapping(args.local_mount, args.oss_prefix)

    process_config_path = args.process_config.expanduser().resolve()
    process_configs = _load_process_configs(process_config_path)
    if "load_data" not in process_configs or "retarget" not in process_configs or "inverse_kinematics" not in process_configs:
        raise KeyError("process config must contain load_data, retarget, and inverse_kinematics")

    load_process = LoadDataProcess(copy.deepcopy(process_configs["load_data"]))
    retarget_config = copy.deepcopy(process_configs["retarget"])
    retarget_config.setdefault("params", {})
    retarget_config["params"]["retarget_scheme"] = str(args.retarget_scheme).strip().lower() or "test"
    ik_process = InverseKinematicsProcess(copy.deepcopy(process_configs["inverse_kinematics"]))

    base_workspace_center = list(retarget_config.get("params", {}).get("workspace_center", [0.25, 0.0, 0.4]))
    if len(base_workspace_center) != 3:
        raise ValueError("retarget workspace_center must have 3 values")
    workspace_y = float(args.workspace_y) if args.workspace_y is not None else float(base_workspace_center[1])

    sample_specs = _build_sample_specs(args)
    loaded_samples = _load_samples(specs=sample_specs, load_process=load_process)
    if not loaded_samples:
        raise RuntimeError("no samples loaded")

    x_values = _build_axis_values(args.x_min, args.x_max, args.x_step)
    z_values = _build_axis_values(args.z_min, args.z_max, args.z_step)
    workspace_centers = _build_workspace_centers(x_values=x_values, workspace_y=workspace_y, z_values=z_values)
    total = len(workspace_centers)

    if int(args.num_workers) > 1:
        results = _evaluate_workspace_centers_with_ray(
            loaded_samples=loaded_samples,
            retarget_template=retarget_config,
            ik_config=copy.deepcopy(process_configs["inverse_kinematics"]),
            workspace_centers=workspace_centers,
            retarget_scheme=str(args.retarget_scheme),
            objective=str(args.objective),
            threshold=float(args.threshold),
            num_workers=int(args.num_workers),
            batch_size=int(args.batch_size),
            local_mount=args.local_mount,
            oss_prefix=str(args.oss_prefix),
            ray_address=args.ray_address,
        )
    else:
        results = []
        for index, workspace_center in enumerate(workspace_centers, start=1):
            result = _evaluate_workspace_center(
                loaded_samples=loaded_samples,
                retarget_template=retarget_config,
                ik_process=ik_process,
                workspace_center=workspace_center,
                retarget_scheme=str(args.retarget_scheme),
                objective=str(args.objective),
                threshold=float(args.threshold),
            )
            results.append(result)
            _print_progress(index=index, total=total, result=result)

    valid_results = [item for item in results if item.get("objective") is not None]
    if not valid_results:
        raise RuntimeError("all grid points produced invalid reachable_ratio")

    valid_results.sort(key=lambda item: float(item["objective"]), reverse=True)
    best = valid_results[0]

    print("")
    print(f"objective: {args.objective}")
    print(f"threshold: {args.threshold}")
    print(f"sample_count: {len(loaded_samples)}")
    print(f"best_workspace_center: {best['workspace_center']}")
    print(f"best_objective: {best['objective']}")
    print(f"best_pass_count: {best['pass_count']}")
    print(f"best_pass_ratio: {best['pass_ratio']}")
    print(f"best_left_reachable_ratio_mean: {best['left_reachable_ratio_mean']}")
    print(f"best_right_reachable_ratio_mean: {best['right_reachable_ratio_mean']}")

    top_k = max(1, int(args.top_k))
    print("")
    print(f"top_{top_k}:")
    for rank, item in enumerate(valid_results[:top_k], start=1):
        print(
            f"{rank}. center={item['workspace_center']} "
            f"objective={item['objective']} "
            f"pass_ratio={item['pass_ratio']} "
            f"left_mean={item['left_reachable_ratio_mean']} "
            f"right_mean={item['right_reachable_ratio_mean']}"
        )

    if args.output_json is not None:
        output_path = args.output_json.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "process_config": str(process_config_path),
            "retarget_scheme": str(args.retarget_scheme),
            "objective": str(args.objective),
            "threshold": float(args.threshold),
            "workspace_y": float(workspace_y),
            "x_values": x_values,
            "z_values": z_values,
            "sample_specs": sample_specs,
            "best": best,
            "results": valid_results,
        }
        with open(output_path, "w", encoding="utf-8") as file_obj:
            json.dump(payload, file_obj, indent=2, ensure_ascii=False)
        print("")
        print(f"saved_json: {output_path}")


if __name__ == "__main__":
    main()

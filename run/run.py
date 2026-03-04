from __future__ import annotations

import os
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from data_loader import build_data_loader
from process.process import build_pipeline
from utils.manifest import ManifestStore
from utils.oss_utils import oss_to_local
from utils.pipeline_context import PipelineContext
from utils.progress import build_progress_bar

PROJECT_ROOT = Path(__file__).resolve().parents[1]

try:
    import hydra
    from omegaconf import DictConfig, OmegaConf
except ImportError:
    hydra = None
    DictConfig = Any  # type: ignore[assignment]
    OmegaConf = None

try:
    import ray
except ImportError:
    ray = None


def resolve_config_path(base_root: Path, path_value: str) -> Path:
    if path_value.startswith("oss://"):
        return Path(oss_to_local(path_value)).resolve()

    path = Path(path_value)
    if path.is_absolute():
        return path.resolve()
    return (base_root / path).resolve()


@dataclass
class RuntimeSettings:
    manifest_path: Path
    num_workers: int
    limit: int | None
    recover_in_progress: bool
    show_progress: bool
    ray_address: str | None
    ray_init_kwargs: dict[str, Any]
    ray_worker_options: dict[str, Any]


def _require_hydra() -> None:
    if hydra is None or OmegaConf is None:
        raise ImportError("hydra-core is required for config management. Install it with `pip install hydra-core`.")


def _cfg_to_dict(cfg: DictConfig) -> dict[str, Any]:
    _require_hydra()
    raw_config = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(raw_config, dict):
        raise TypeError("pipeline config must resolve to a mapping")
    return raw_config


class PipelineManager:
    def __init__(self, raw_config: dict[str, Any], project_root: Path | None = None) -> None:
        self.project_root = (project_root or PROJECT_ROOT).resolve()
        self.raw_config = self._normalise_config(raw_config)
        self.runtime = self._build_runtime_settings()
        self.data_loader = self._build_data_loader()
        self.manifest = ManifestStore(self.runtime.manifest_path)

    def _normalise_config(self, raw_config: dict[str, Any]) -> dict[str, Any]:
        config = dict(raw_config)
        config.setdefault("runtime", {})
        config["runtime"].setdefault("manifest_path", "./workspace/manifest.json")
        config["runtime"].setdefault("num_workers", os.cpu_count() or 1)
        config["runtime"].setdefault("limit", None)
        config["runtime"].setdefault("recover_in_progress", True)
        config["runtime"].setdefault("show_progress", True)
        config["runtime"].setdefault("ray", {})
        config["runtime"]["ray"].setdefault("address", None)
        config["runtime"]["ray"].setdefault("init", {})
        config["runtime"]["ray"].setdefault("worker", {})
        config.setdefault("data", {})
        if "type" not in config["data"]:
            legacy_data = dict(config["data"])
            legacy_data.setdefault("input_glob", "**/*")
            legacy_data.setdefault("extensions", [])
            config["data"] = {
                "type": "glob",
                "params": legacy_data,
            }
        config["data"].setdefault("type", "glob")
        config["data"].setdefault("params", {})
        config["data"]["_config_root"] = str(self.project_root)
        config.setdefault("processes", [])
        config["_config_root"] = str(self.project_root)
        for process_config in config["processes"]:
            process_config["_config_root"] = str(self.project_root)
        return config

    def _build_runtime_settings(self) -> RuntimeSettings:
        runtime_config = self.raw_config.get("runtime", {})
        ray_config = runtime_config.get("ray", {})
        manifest_path = resolve_config_path(
            self.project_root, runtime_config.get("manifest_path", "./workspace/manifest.json")
        )
        num_workers = int(runtime_config.get("num_workers", os.cpu_count() or 1))
        limit = runtime_config.get("limit")
        recover_in_progress = bool(runtime_config.get("recover_in_progress", True))
        show_progress = bool(runtime_config.get("show_progress", True))
        ray_worker_options = dict(ray_config.get("worker", {}))
        if ray_worker_options.get("resources") is None:
            ray_worker_options.pop("resources", None)
        elif not isinstance(ray_worker_options["resources"], dict):
            raise TypeError("runtime.ray.worker.resources must be a mapping")
        return RuntimeSettings(
            manifest_path=manifest_path,
            num_workers=max(1, num_workers),
            limit=None if limit is None else int(limit),
            recover_in_progress=recover_in_progress,
            show_progress=show_progress,
            ray_address=ray_config.get("address"),
            ray_init_kwargs=dict(ray_config.get("init", {})),
            ray_worker_options=ray_worker_options,
        )

    def _build_data_loader(self):
        return build_data_loader(self.raw_config["data"])

    def discover_samples(self) -> list[dict[str, Any]]:
        return self.data_loader()

    def prepare(self) -> list[dict[str, Any]]:
        if self.runtime.recover_in_progress:
            recovered = self.manifest.recover_in_progress()
            if recovered:
                print(f"Recovered {len(recovered)} unfinished task(s).")

        samples = self.discover_samples()
        sync_result = self.manifest.sync_tasks(samples)
        print(f"Discovered {len(samples)} sample(s), new={sync_result['new']}, refreshed={sync_result['refreshed']}.")
        discovered_ids = {sample["sample_id"] for sample in samples}
        return self.manifest.pending_samples(sample_ids=discovered_ids)

    def run(self) -> dict[str, int]:
        pending_samples = self.prepare()
        if self.runtime.limit is not None:
            pending_samples = pending_samples[: self.runtime.limit]

        if not pending_samples:
            print("No pending samples.")
            return self.manifest.summary()

        worker_count = max(1, self.runtime.num_workers)
        print(
            f"Processing {len(pending_samples)} sample(s) with {worker_count} Ray worker actor(s), "
            f"worker_options={self.runtime.ray_worker_options}."
        )

        self._ensure_ray()
        results = self._run_with_ray(pending_samples, worker_count)
        recovered_after_run = self.manifest.recover_in_progress()
        if recovered_after_run:
            print(f"Recovered {len(recovered_after_run)} stale in-progress task(s) after Ray execution.")

        success_count = sum(1 for result in results if result["status"] == "completed")
        failed_count = sum(1 for result in results if result["status"] == "failed")
        skipped_count = sum(1 for result in results if result["status"] == "skipped")
        print(
            f"Run finished: completed={success_count}, failed={failed_count}, skipped={skipped_count}, "
            f"manifest={self.manifest.summary()}."
        )
        return self.manifest.summary()

    def _ensure_ray(self) -> None:
        if ray is None:
            raise ImportError("ray is required for pipeline execution. Install it with `pip install ray`.")

        if ray.is_initialized():
            return

        init_kwargs = dict(self.runtime.ray_init_kwargs)
        if self.runtime.ray_address is not None:
            init_kwargs["address"] = self.runtime.ray_address
        ray.init(**init_kwargs)

    def _run_with_ray(self, pending_samples: list[dict[str, Any]], worker_count: int) -> list[dict[str, Any]]:
        worker_remote = ray.remote(PipelineWorker)
        actors = [
            worker_remote.options(**self.runtime.ray_worker_options).remote(self.raw_config, worker_index=index)
            for index in range(worker_count)
        ]

        results: list[dict[str, Any]] = []
        inflight: dict[Any, dict[str, Any]] = {}
        sample_iter = iter(pending_samples)
        next_actor_index = 0
        status_counts = {"completed": 0, "failed": 0, "skipped": 0}
        progress_bar = build_progress_bar(
            enabled=self.runtime.show_progress,
            total=len(pending_samples),
            desc="Processing",
            unit="sample",
        )

        def schedule_next() -> None:
            nonlocal next_actor_index
            while len(inflight) < len(actors):
                try:
                    sample = next(sample_iter)
                except StopIteration:
                    break
                actor = actors[next_actor_index % len(actors)]
                next_actor_index += 1
                ref = actor.process_sample.remote(sample)
                inflight[ref] = sample

        schedule_next()
        try:
            while inflight:
                done_refs, _ = ray.wait(list(inflight), num_returns=1)
                done_ref = done_refs[0]
                sample = inflight.pop(done_ref)
                try:
                    result = ray.get(done_ref)
                except Exception as exc:
                    result = {
                        "sample_id": sample["sample_id"],
                        "status": "failed",
                        "error": f"Ray execution failed: {exc}",
                    }
                results.append(result)
                status = result.get("status", "failed")
                status_counts.setdefault(status, 0)
                status_counts[status] += 1
                progress_bar.update(1)
                progress_bar.set_postfix(
                    completed=status_counts.get("completed", 0),
                    failed=status_counts.get("failed", 0),
                    skipped=status_counts.get("skipped", 0),
                )
                progress_bar.refresh()
                schedule_next()
        finally:
            progress_bar.close()

        return results


class PipelineWorker:
    def __init__(self, config: dict[str, Any], worker_index: int = 0) -> None:
        self.config = config
        self.worker_index = worker_index
        self.pipeline = build_pipeline(config.get("processes", []))
        manifest_path = resolve_config_path(Path(config["_config_root"]), config["runtime"]["manifest_path"])
        self.manifest = ManifestStore(manifest_path)

    def process_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        sample_id = sample["sample_id"]
        worker_info = {
            "name": f"ray-worker-{self.worker_index}",
            "pid": os.getpid(),
        }
        claimed = self.manifest.claim_task(sample_id, worker=worker_info)
        if not claimed:
            return {"sample_id": sample_id, "status": "skipped"}

        context = PipelineContext(sample_id, self.manifest)
        try:
            result = self.pipeline(sample, context)
            summary = {"last_process": result.get("last_process")}
            self.manifest.mark_completed(sample_id, summary=summary)
            context.finish()
            return {"sample_id": sample_id, "status": "completed"}
        except Exception as exc:
            context.cleanup()
            self.manifest.reset_to_pending(sample_id, last_error=traceback.format_exc())
            return {"sample_id": sample_id, "status": "failed", "error": str(exc)}


if hydra is not None:
    @hydra.main(version_base=None, config_path="../config", config_name="pipeline")
    def main(cfg: DictConfig) -> None:
        _require_hydra()
        manager = PipelineManager(_cfg_to_dict(cfg), project_root=PROJECT_ROOT)
        manager.run()
else:
    def main() -> None:
        _require_hydra()


if __name__ == "__main__":
    main()

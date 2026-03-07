from __future__ import annotations
import os
import signal
import traceback
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from data_loader import build_data_loader
from process.process import build_pipeline
from utils.manifest import ManifestStore
from utils.oss_utils import configure_path_mapping, local_to_oss, oss_to_local
from utils.pipeline_context import PipelineContext
from utils.progress import build_progress_bar
from utils.safe_io import atomic_write_json

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


def _normalize_remote_root(remote_root: str) -> str:
    if remote_root.endswith("://"):
        return remote_root
    return remote_root.rstrip("/")


def _resolve_remote_root(project_root: Path, remote_value: str) -> str:
    if "://" in remote_value:
        return _normalize_remote_root(remote_value)
    remote_path = Path(remote_value)
    if remote_path.is_absolute():
        return _normalize_remote_root(local_to_oss(str(remote_path)))
    return _normalize_remote_root(local_to_oss(str((project_root / remote_path).resolve())))


def _join_remote_path(remote_root: str, relative_path: Path) -> str:
    normalized_remote_root = _normalize_remote_root(remote_root)
    relative_value = relative_path.as_posix().lstrip("/")
    if not relative_value:
        return normalized_remote_root
    if normalized_remote_root.endswith("://"):
        return f"{normalized_remote_root}{relative_value}"
    return f"{normalized_remote_root}/{relative_value}"


def _configure_path_mapping_from_runtime(project_root: Path, raw_config: dict[str, Any]) -> None:
    path_mapping_config = dict(raw_config.get("runtime", {}).get("path_mapping", {}))
    local_mount_value = path_mapping_config.get("local_mount", str(project_root))
    if not isinstance(local_mount_value, str) or not local_mount_value:
        local_mount_value = str(project_root)
    local_mount = resolve_config_path(project_root, local_mount_value)

    oss_prefix_value = path_mapping_config.get("oss_prefix", "oss://")
    if not isinstance(oss_prefix_value, str) or not oss_prefix_value:
        oss_prefix_value = "oss://"

    configure_path_mapping(str(local_mount), oss_prefix_value)


@dataclass
class RuntimeSettings:
    manifest_path: Path
    num_workers: int
    limit: int | None
    resume: bool
    cleanup_artifacts_on_reset: bool
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
        self._configure_path_mapping()
        self.runtime = self._build_runtime_settings()
        self.data_loader = self._build_data_loader()
        self.manifest = ManifestStore(self.runtime.manifest_path)
        self._interruption_signal: str | None = None

    def _normalise_config(self, raw_config: dict[str, Any]) -> dict[str, Any]:
        config = dict(raw_config)
        config.setdefault("runtime", {})
        config["runtime"].setdefault("manifest_path", "./workspace/manifest.json")
        config["runtime"].setdefault("num_workers", os.cpu_count() or 1)
        config["runtime"].setdefault("limit", None)
        config["runtime"].setdefault("resume", False)
        config["runtime"].setdefault("cleanup_artifacts_on_reset", True)
        config["runtime"].setdefault("recover_in_progress", True)
        config["runtime"].setdefault("show_progress", True)
        config["runtime"].setdefault("path_mapping", {})
        config["runtime"]["path_mapping"].setdefault("local_mount", str(self.project_root))
        config["runtime"]["path_mapping"].setdefault("oss_prefix", "oss://")
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
        config["data"]["_path_mapping"] = dict(config["runtime"].get("path_mapping", {}))
        config.setdefault("processes", [])
        config["_config_root"] = str(self.project_root)
        for process_config in config["processes"]:
            process_config["_config_root"] = str(self.project_root)
        return config

    def _configure_path_mapping(self) -> None:
        _configure_path_mapping_from_runtime(self.project_root, self.raw_config)

    def _build_runtime_settings(self) -> RuntimeSettings:
        runtime_config = self.raw_config.get("runtime", {})
        ray_config = runtime_config.get("ray", {})
        manifest_path = resolve_config_path(
            self.project_root, runtime_config.get("manifest_path", "./workspace/manifest.json")
        )
        num_workers = int(runtime_config.get("num_workers", os.cpu_count() or 1))
        limit = runtime_config.get("limit")
        resume = bool(runtime_config.get("resume", False))
        cleanup_artifacts_on_reset = bool(runtime_config.get("cleanup_artifacts_on_reset", True))
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
            resume=resume,
            cleanup_artifacts_on_reset=cleanup_artifacts_on_reset,
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
        if self.runtime.resume and self.runtime.recover_in_progress:
            recovered = self.manifest.recover_in_progress()
            if recovered:
                print(f"Recovered {len(recovered)} unfinished task(s).")

        samples = self.discover_samples()
        if self.runtime.resume:
            existing_payloads = self.manifest.get_payloads(sample["sample_id"] for sample in samples)
            for sample in samples:
                existing_payload = existing_payloads.get(sample["sample_id"])
                if existing_payload is not None and "visualize" in existing_payload:
                    sample["visualize"] = existing_payload["visualize"]
        sync_result = self.manifest.sync_tasks(samples)
        print(f"Discovered {len(samples)} sample(s), new={sync_result['new']}, refreshed={sync_result['refreshed']}.")
        discovered_ids = {sample["sample_id"] for sample in samples}
        if not self.runtime.resume:
            reset_count = self.manifest.reset_samples(
                discovered_ids,
                cleanup_artifacts=self.runtime.cleanup_artifacts_on_reset,
                last_error="Reset because runtime.resume=false.",
            )
            print(f"Reset {reset_count} sample(s) to pending because runtime.resume=false.")
        return self.manifest.pending_samples(sample_ids=discovered_ids)

    def run(self) -> dict[str, int]:
        final_summary: dict[str, int] | None = None
        try:
            pending_samples = self.prepare()
            if self.runtime.limit is not None:
                pending_samples = pending_samples[: self.runtime.limit]

            if not pending_samples:
                print("No pending samples.")
                final_summary = self.manifest.summary()
                return final_summary

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
            final_summary = self.manifest.summary()
            print(
                f"Run finished: completed={success_count}, failed={failed_count}, skipped={skipped_count}, "
                f"manifest={final_summary}."
            )
            return final_summary
        except KeyboardInterrupt:
            print("Pipeline interrupted, preparing meta.json for processed outputs.")
            raise
        finally:
            self._write_meta_file()
            self._write_render_meta_file()

    def mark_interrupted(self, signal_name: str) -> None:
        self._interruption_signal = signal_name

    def _resolve_write_data_layout(self) -> dict[str, Path] | None:
        output_dir_value: str | dict[str, str] | None = None
        for process_config in self.raw_config.get("processes", []):
            if not process_config.get("enabled", True):
                continue
            if process_config.get("type") != "write_data":
                continue
            output_dir_value = dict(process_config.get("params", {})).get("output_dir")
            if output_dir_value is not None:
                break

        if output_dir_value is None:
            return None

        if isinstance(output_dir_value, dict):
            local_value = output_dir_value.get("local") or output_dir_value.get("remote")
            remote_value = output_dir_value.get("remote") or local_value
            if not isinstance(local_value, str) or not local_value:
                return None
            if not isinstance(remote_value, str) or not remote_value:
                return None
            output_root = resolve_config_path(self.project_root, local_value)
            remote_root = _resolve_remote_root(self.project_root, remote_value)
        elif isinstance(output_dir_value, str):
            output_root = resolve_config_path(self.project_root, output_dir_value)
            remote_root = _resolve_remote_root(self.project_root, output_dir_value)
        else:
            return None

        data_relative_dir = Path("data")
        samples_relative_dir = Path("samples")
        sample_metadata_relative_dir = Path("meta_data")
        meta_relative_path = Path("meta.json")
        return {
            "output_root": output_root.resolve(),
            "remote_root": remote_root,
            "data_root": (output_root / data_relative_dir).resolve(),
            "samples_root": (output_root / samples_relative_dir).resolve(),
            "sample_metadata_root": (output_root / sample_metadata_relative_dir).resolve(),
            "meta_path": (output_root / meta_relative_path).resolve(),
        }

    def _write_meta_file(self) -> None:
        layout = self._resolve_write_data_layout()
        if layout is None:
            return

        meta_path = layout["meta_path"]
        data_root = Path(layout["data_root"])
        sample_metadata_root = Path(layout["sample_metadata_root"])

        completed_ids = self.manifest.completed_sample_ids()
        processed_entries: list[dict[str, Any]] = []
        sample_ids: list[str] = []
        for sample_id in completed_ids:
            relative = Path(sample_id)
            parquet_path = (data_root / relative.parent / f"{relative.stem}.parquet").resolve()
            sample_metadata_path = (sample_metadata_root / relative.parent / f"{relative.stem}.json").resolve()
            if not parquet_path.exists() or not sample_metadata_path.exists():
                continue
            sample_ids.append(sample_id)
        processed_entries.append({
            "sample_ids": sample_ids,
            "output_dir": layout["remote_root"],
        })

        try:
            atomic_write_json(meta_path, processed_entries, indent=2, sort_keys=True)
            print(f"Wrote processed data meta: {meta_path} (count={len(processed_entries)}).")
        except Exception as exc:
            print(f"Failed to write meta.json: {exc}")

    def _resolve_render_layout(self) -> dict[str, Path] | None:
        output_dir_value: str | dict[str, str] | None = None
        render_relative_dir = Path("render")
        render_meta_relative_path = Path("render_meta.json")

        for process_config in self.raw_config.get("processes", []):
            if not process_config.get("enabled", True):
                continue
            if process_config.get("type") != "render":
                continue
            params = dict(process_config.get("params", {}))
            output_dir_value = params.get("output_dir")
            render_relative_dir = Path(str(params.get("render_relative_dir", "render")))
            render_meta_relative_path = Path(str(params.get("render_meta_relative_path", "render_meta.json")))
            break

        if output_dir_value is None:
            return None

        if isinstance(output_dir_value, dict):
            local_value = output_dir_value.get("local") or output_dir_value.get("remote")
            if not isinstance(local_value, str) or not local_value:
                return None
            output_root = resolve_config_path(self.project_root, local_value)
        elif isinstance(output_dir_value, str):
            output_root = resolve_config_path(self.project_root, output_dir_value)
        else:
            return None

        return {
            "output_root": output_root.resolve(),
            "render_root": (output_root / render_relative_dir).resolve(),
            "render_meta_path": (output_root / render_meta_relative_path).resolve(),
        }

    def _write_render_meta_file(self) -> None:
        layout = self._resolve_render_layout()
        if layout is None:
            return

        render_root = Path(layout["render_root"])
        render_meta_path = Path(layout["render_meta_path"])
        completed_ids = self.manifest.completed_sample_ids()
        rendered_entries: list = []
        for sample_id in completed_ids:
            relative = Path(sample_id)
            render_video_path = (render_root / relative.parent / f"{relative.stem}.mp4").resolve()
            if not render_video_path.exists():
                continue
            rendered_entries.append(sample_id)

        try:
            atomic_write_json(render_meta_path, rendered_entries, indent=2, sort_keys=True)
            print(f"Wrote render meta: {render_meta_path} (count={len(rendered_entries)}).")
        except Exception as exc:
            print(f"Failed to write render_meta.json: {exc}")

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
                    print(f"Ray execution failed for sample {sample['sample_id']}: {exc}")
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
        config_root = Path(str(config.get("_config_root", PROJECT_ROOT))).resolve()
        _configure_path_mapping_from_runtime(config_root, config)
        self.pipeline = build_pipeline(config.get("processes", []))
        manifest_path = resolve_config_path(config_root, config["runtime"]["manifest_path"])
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
        except BaseException as exc:
            error_text = traceback.format_exc()
            print(f"Processing failed for sample {sample_id}:\n{error_text}")
            context.cleanup()
            self.manifest.reset_to_pending(sample_id, last_error=error_text)
            if isinstance(exc, Exception):
                return {"sample_id": sample_id, "status": "failed", "error": str(exc)}
            raise


if hydra is not None:
    @hydra.main(version_base=None, config_path="../config", config_name="pipeline")
    def main(cfg: DictConfig) -> None:
        _require_hydra()
        manager = PipelineManager(_cfg_to_dict(cfg), project_root=PROJECT_ROOT)
        managed_signals: list[int] = []
        for signal_name in ("SIGINT", "SIGTERM"):
            if hasattr(signal, signal_name):
                managed_signals.append(int(getattr(signal, signal_name)))
        previous_handlers: dict[int, Any] = {}

        def _interrupt_handler(signum: int, _frame: Any) -> None:
            signal_name = signal.Signals(signum).name
            manager.mark_interrupted(signal_name)
            print(f"Received {signal_name}, stopping pipeline...")
            raise KeyboardInterrupt

        try:
            for sig in managed_signals:
                previous_handlers[sig] = signal.getsignal(sig)
                signal.signal(sig, _interrupt_handler)
            manager.run()
        except KeyboardInterrupt:
            pass
        finally:
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)
else:
    def main() -> None:
        _require_hydra()


if __name__ == "__main__":
    main()

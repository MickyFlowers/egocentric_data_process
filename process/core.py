from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from utils.oss_utils import local_to_oss, oss_to_local

PROCESS_REGISTRY: dict[str, type["BaseProcess"]] = {}


def register_process(name: str) -> Callable[[type["BaseProcess"]], type["BaseProcess"]]:
    def decorator(process_cls: type["BaseProcess"]) -> type["BaseProcess"]:
        if name in PROCESS_REGISTRY:
            raise ValueError(f"process '{name}' is already registered")
        PROCESS_REGISTRY[name] = process_cls
        process_cls.process_type = name
        return process_cls

    return decorator


class BaseProcess:
    process_type = "base"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.params = dict(self.config.get("params", {}))
        self.name = self.config.get("name", self.process_type)
        self._config_root = Path(self.config.get("_config_root", ".")).resolve()

    def __call__(self, sample: dict[str, Any], context: Any) -> dict[str, Any]:
        return self.run(dict(sample), context)

    def run(self, sample: dict[str, Any], context: Any) -> dict[str, Any]:
        raise NotImplementedError

    def resolve_path(self, path_value: str) -> Path:
        if path_value.startswith("oss://"):
            return Path(oss_to_local(path_value)).resolve()

        path = Path(path_value)
        if path.is_absolute():
            return path
        return (self._config_root / path).resolve()

    def resolve_sample_path(self, path_value: str) -> str:
        return str(self.resolve_path(path_value))

    def build_output_path(
        self,
        sample: dict[str, Any],
        output_dir: str,
        extension: str | None = None,
        suffix: str = "",
    ) -> Path:
        base_dir = self.resolve_path(output_dir)
        relative_path = Path(sample["sample_id"])
        source_suffix = Path(self.resolve_sample_path(sample["video_path"])).suffix

        if extension is None:
            target_name = f"{relative_path.stem}{suffix}{source_suffix}"
        else:
            normalized_extension = extension if extension.startswith(".") else f".{extension}"
            target_name = f"{relative_path.stem}{suffix}{normalized_extension}"

        return base_dir / relative_path.parent / target_name

    def build_output_paths(
        self,
        sample: dict[str, Any],
        output_dir: str | dict[str, str],
        extension: str | None = None,
        suffix: str = "",
    ) -> tuple[Path, str]:
        local_root, remote_root = self._resolve_output_roots(output_dir)
        relative_path = Path(sample["sample_id"])
        source_suffix = Path(self.resolve_sample_path(sample["video_path"])).suffix

        if extension is None:
            target_name = f"{relative_path.stem}{suffix}{source_suffix}"
        else:
            normalized_extension = extension if extension.startswith(".") else f".{extension}"
            target_name = f"{relative_path.stem}{suffix}{normalized_extension}"

        relative_output = relative_path.parent / target_name
        local_path = local_root / relative_output
        remote_path = self._join_remote_path(remote_root, relative_output)
        return local_path, remote_path

    def extend_output_dir(self, output_dir: str | dict[str, str], subdir: str) -> str | dict[str, str]:
        if isinstance(output_dir, dict):
            return {
                key: self._join_root_value(value, subdir)
                for key, value in output_dir.items()
                if value is not None
            }
        return self._join_root_value(output_dir, subdir)

    def _resolve_output_roots(self, output_dir: str | dict[str, str]) -> tuple[Path, str]:
        if isinstance(output_dir, dict):
            local_value = output_dir.get("local")
            if not local_value:
                raise ValueError("output_dir.local is required")
            remote_value = output_dir.get("remote", local_value)
        else:
            local_value = output_dir
            remote_value = output_dir

        return self.resolve_path(local_value), self._resolve_remote_root(remote_value)

    def _resolve_remote_root(self, remote_value: str) -> str:
        if "://" in remote_value:
            return self._normalize_remote_root(remote_value)
        remote_path = Path(remote_value)
        if remote_path.is_absolute():
            return local_to_oss(str(remote_path))
        return local_to_oss(str((self._config_root / remote_path).resolve()))

    def _join_remote_path(self, remote_root: str, relative_output: Path) -> str:
        normalized_remote_root = self._normalize_remote_root(remote_root)
        relative_value = relative_output.as_posix().lstrip("/")
        if not relative_value:
            return normalized_remote_root
        if normalized_remote_root.endswith("://"):
            return f"{normalized_remote_root}{relative_value}"
        return f"{normalized_remote_root}/{relative_value}"

    def _join_root_value(self, root_value: str, subdir: str) -> str:
        if "://" in root_value:
            normalized_root = self._normalize_remote_root(root_value)
            if normalized_root.endswith("://"):
                return f"{normalized_root}{subdir.strip('/')}"
            return f"{normalized_root}/{subdir.strip('/')}"
        return str(Path(root_value) / subdir)

    def _normalize_remote_root(self, remote_root: str) -> str:
        if remote_root.endswith("://"):
            return remote_root
        return remote_root.rstrip("/")


class Pipeline:
    def __init__(self, processes: list[BaseProcess]) -> None:
        self.processes = processes

    def __call__(self, sample: dict[str, Any], context: Any) -> dict[str, Any]:
        current = dict(sample)
        for process in self.processes:
            current = process(current, context)
        return current


def build_pipeline(process_configs: list[dict[str, Any]]) -> Pipeline:
    processes: list[BaseProcess] = []
    for process_config in process_configs:
        if not process_config.get("enabled", True):
            continue

        process_type = process_config["type"]
        process_cls = PROCESS_REGISTRY.get(process_type)
        if process_cls is None:
            available = ", ".join(sorted(PROCESS_REGISTRY))
            raise KeyError(f"unknown process '{process_type}', available: {available}")

        processes.append(process_cls(process_config))

    return Pipeline(processes)

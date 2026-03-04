from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Callable

from utils.oss_utils import oss_to_local


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

    def build_output_path(
        self,
        sample: dict[str, Any],
        output_dir: str,
        extension: str | None = None,
        suffix: str = "",
    ) -> Path:
        base_dir = self.resolve_path(output_dir)
        relative_path = Path(sample["relative_path"])

        if extension is None:
            target_name = f"{relative_path.stem}{suffix}{relative_path.suffix}"
        else:
            normalized_extension = extension if extension.startswith(".") else f".{extension}"
            target_name = f"{relative_path.stem}{suffix}{normalized_extension}"

        return base_dir / relative_path.parent / target_name


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


@register_process("add_fields")
class AddFieldsProcess(BaseProcess):
    def run(self, sample: dict[str, Any], context: Any) -> dict[str, Any]:
        fields = self.params.get("fields", {})
        if not isinstance(fields, dict):
            raise TypeError("'fields' must be a mapping")

        sample.update(fields)
        sample["last_process"] = self.name
        return sample


@register_process("write_metadata_json")
class WriteMetadataJsonProcess(BaseProcess):
    def run(self, sample: dict[str, Any], context: Any) -> dict[str, Any]:
        output_dir = self.params["output_dir"]
        include_keys = self.params.get("include_keys")

        payload = dict(sample)
        if include_keys is not None:
            payload = {key: payload.get(key) for key in include_keys}

        output_path = self.build_output_path(
            sample,
            output_dir=output_dir,
            extension=self.params.get("extension", ".json"),
            suffix=self.params.get("suffix", ""),
        )
        context.write_json(str(output_path), payload, indent=self.params.get("indent", 2))
        sample["last_process"] = self.name
        return sample


@register_process("copy_input_file")
class CopyInputFileProcess(BaseProcess):
    def run(self, sample: dict[str, Any], context: Any) -> dict[str, Any]:
        output_path = self.build_output_path(
            sample,
            output_dir=self.params["output_dir"],
            extension=self.params.get("extension"),
            suffix=self.params.get("suffix", ""),
        )
        context.copy_file(sample["input_path"], str(output_path))
        sample["last_process"] = self.name
        return sample


@register_process("run_command")
class RunCommandProcess(BaseProcess):
    def run(self, sample: dict[str, Any], context: Any) -> dict[str, Any]:
        output_path = self.build_output_path(
            sample,
            output_dir=self.params["output_dir"],
            extension=self.params.get("extension"),
            suffix=self.params.get("suffix", ""),
        )
        command_template = self.params["command"]
        if not isinstance(command_template, list) or not command_template:
            raise TypeError("'command' must be a non-empty list")

        format_values = dict(sample)
        format_values["output_path"] = str(output_path)

        with context.staged_output(str(output_path)) as temp_output_path:
            format_values["temp_output_path"] = temp_output_path
            command = [part.format(**format_values) for part in command_template]
            subprocess.run(command, check=True)

        sample["last_process"] = self.name
        return sample

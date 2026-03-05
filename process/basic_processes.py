from __future__ import annotations

import subprocess
from typing import Any

from .core import BaseProcess, register_process


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

        output_path, remote_output_path = self.build_output_paths(
            sample,
            output_dir=output_dir,
            extension=self.params.get("extension", ".json"),
            suffix=self.params.get("suffix", ""),
        )
        context.write_json(str(output_path), payload, indent=self.params.get("indent", 2))
        sample["metadata_path"] = remote_output_path
        sample["last_process"] = self.name
        return sample


@register_process("copy_input_file")
class CopyInputFileProcess(BaseProcess):
    def run(self, sample: dict[str, Any], context: Any) -> dict[str, Any]:
        output_path, remote_output_path = self.build_output_paths(
            sample,
            output_dir=self.params["output_dir"],
            extension=self.params.get("extension"),
            suffix=self.params.get("suffix", ""),
        )
        context.copy_file(self.resolve_sample_path(sample["video_path"]), str(output_path))
        sample["media_path"] = remote_output_path
        sample["last_process"] = self.name
        return sample


@register_process("run_command")
class RunCommandProcess(BaseProcess):
    def run(self, sample: dict[str, Any], context: Any) -> dict[str, Any]:
        output_path, remote_output_path = self.build_output_paths(
            sample,
            output_dir=self.params["output_dir"],
            extension=self.params.get("extension"),
            suffix=self.params.get("suffix", ""),
        )
        command_template = self.params["command"]
        if not isinstance(command_template, list) or not command_template:
            raise TypeError("'command' must be a non-empty list")

        format_values = dict(sample)
        for key, value in list(sample.items()):
            if key.endswith("_path") and isinstance(value, str):
                format_values[f"{key}_remote"] = value
                format_values[key] = self.resolve_sample_path(value)
        format_values["output_path"] = str(output_path)
        format_values["remote_output_path"] = remote_output_path

        with context.staged_output(str(output_path)) as temp_output_path:
            format_values["temp_output_path"] = temp_output_path
            command = [part.format(**format_values) for part in command_template]
            subprocess.run(command, check=True)

        sample[f"{self.name}_path"] = remote_output_path
        sample["last_process"] = self.name
        return sample

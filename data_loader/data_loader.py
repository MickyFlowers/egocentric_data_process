from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from utils.oss_utils import local_to_oss, oss_to_local
from supabase import create_client, Client
from utils.oss_utils import oss_to_local

DATA_LOADER_REGISTRY: dict[str, type["BaseDataLoader"]] = {}


def register_data_loader(
    name: str,
) -> Callable[[type["BaseDataLoader"]], type["BaseDataLoader"]]:
    def decorator(loader_cls: type["BaseDataLoader"]) -> type["BaseDataLoader"]:
        if name in DATA_LOADER_REGISTRY:
            raise ValueError(f"data loader '{name}' is already registered")
        DATA_LOADER_REGISTRY[name] = loader_cls
        loader_cls.loader_type = name
        return loader_cls

    return decorator


class BaseDataLoader:
    loader_type = "base"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.params = dict(self.config.get("params", {}))
        self.name = self.config.get("name", self.loader_type)
        self._config_root = Path(self.config.get("_config_root", ".")).resolve()

    def __call__(self) -> list[dict[str, Any]]:
        return self.load()

    def load(self) -> list[dict[str, Any]]:
        raise NotImplementedError

    def resolve_path(self, path_value: str) -> Path:
        if path_value.startswith("oss://"):
            return Path(oss_to_local(path_value)).resolve()

        path = Path(path_value)
        if path.is_absolute():
            return path.resolve()
        return (self._config_root / path_value).resolve()

    def build_sample(
        self,
        path: Path,
        input_root: Path | None = None,
        sample_id: str | None = None,
        extra_fields: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        resolved_path = path.resolve()
        if input_root is not None:
            relative_path = resolved_path.relative_to(input_root.resolve()).as_posix()
        else:
            relative_path = path.name

        sample = {
            "sample_id": sample_id or relative_path,
            "input_path": str(resolved_path),
            "input_oss_path": (
                local_to_oss(str(resolved_path))
                if str(resolved_path).startswith("/home/")
                else None
            ),
            "relative_path": relative_path,
            "stem": resolved_path.stem,
            "extension": resolved_path.suffix,
        }
        if extra_fields:
            sample.update(extra_fields)
        return sample


def build_data_loader(loader_config: dict[str, Any]) -> BaseDataLoader:
    loader_type = loader_config["type"]
    loader_cls = DATA_LOADER_REGISTRY.get(loader_type)
    if loader_cls is None:
        available = ", ".join(sorted(DATA_LOADER_REGISTRY))
        raise KeyError(f"unknown data loader '{loader_type}', available: {available}")
    return loader_cls(loader_config)


@register_data_loader("glob")
class GlobDataLoader(BaseDataLoader):
    def load(self) -> list[dict[str, Any]]:
        input_root_value = self.params.get("input_root")
        if not input_root_value:
            raise ValueError("data.params.input_root is required for glob data loader")

        input_root = self.resolve_path(input_root_value)
        if not input_root.exists():
            raise FileNotFoundError(f"input_root does not exist: {input_root}")

        input_glob = self.params.get("input_glob", "**/*")
        extensions = [
            extension.lower() for extension in self.params.get("extensions", [])
        ]

        samples: list[dict[str, Any]] = []
        for path in sorted(input_root.glob(input_glob)):
            if not path.is_file():
                continue
            if extensions and path.suffix.lower() not in extensions:
                continue
            samples.append(self.build_sample(path, input_root=input_root))

        return samples


@register_data_loader("json_list")
class JsonListDataLoader(BaseDataLoader):
    def load(self) -> list[dict[str, Any]]:
        samples_path_value = self.params.get("samples_path")
        if not samples_path_value:
            raise ValueError(
                "data.params.samples_path is required for json_list data loader"
            )

        samples_path = self.resolve_path(samples_path_value)
        if not samples_path.exists():
            raise FileNotFoundError(f"samples_path does not exist: {samples_path}")

        relative_root_value = self.params.get("relative_root")
        relative_root = (
            self.resolve_path(relative_root_value) if relative_root_value else None
        )
        file_format = self.params.get("format")

        if file_format == "jsonl" or samples_path.suffix.lower() == ".jsonl":
            rows = []
            with open(samples_path, "r", encoding="utf-8") as file_obj:
                for line in file_obj:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
        else:
            with open(samples_path, "r", encoding="utf-8") as file_obj:
                rows = json.load(file_obj)

        if not isinstance(rows, list):
            raise TypeError("json_list loader expects a JSON array or JSONL rows")

        samples: list[dict[str, Any]] = []
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise TypeError(f"sample row at index {index} must be a mapping")

            input_path_value = row.get("input_path")
            if input_path_value is None:
                raise ValueError(f"sample row at index {index} is missing 'input_path'")

            input_path = self.resolve_path(str(input_path_value))
            if not input_path.exists():
                raise FileNotFoundError(f"input_path does not exist: {input_path}")

            sample = self.build_sample(
                input_path,
                input_root=relative_root,
                sample_id=row.get("sample_id"),
                extra_fields={
                    key: value
                    for key, value in row.items()
                    if key not in {"sample_id", "input_path"}
                },
            )
            if "relative_path" in row:
                sample["relative_path"] = str(row["relative_path"])
                if "sample_id" not in row:
                    sample["sample_id"] = sample["relative_path"]
            samples.append(sample)

        return samples


@register_data_loader("database_loader")
class DatabaseDataLoader(BaseDataLoader):
    def load(self) -> list[dict[str, Any]]:
        database_url = self.params.get("database_url")
        database_key = self.params.get("database_key")
        dataset_name = self.params.get("dataset_name")
        table_name = self.params.get("database_table")
        database: Client = create_client(database_url, database_key)
        start = 0
        samples: list[dict[str, Any]] = []
        while True:
            response = (
                database.table(table_name)
                .select("path")
                .eq("dataset_name", dataset_name)
                .not_.is_("pose3d_hand_path", "null")
                .is_("multi_hand_flag", "False")
                .order("path")
                .range(start, start + 1000 - 1)
                .execute()
            )
            if len(response.data) == 0:
                break
            start += len(response.data)
            print("Fetched {} samples from database".format(start))
            sample = [
                {
                    "video_path": oss_to_local(
                        response.data[i]["path"],
                    ),
                    "data_path": oss_to_local(
                        response.data[i]['path'].replace(".mp4", ".pose3d_hand")
                    )
                }
                for i in range(len(response.data))
            ]
            samples.extend(sample)
        return samples
            

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable

from supabase import create_client, Client
from utils.oss_utils import ensure_oss_path, oss_to_local

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
        self._config_root = Path(self.config.get("_config_root", ".")).resolve()

    def __call__(self) -> list[dict[str, Any]]:
        return self.load()

    def load(self) -> list[dict[str, Any]]:
        raise NotImplementedError


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
        input_dir_value = self.params.get("input_dir") or self.params.get("input_root")
        if not input_dir_value:
            raise ValueError("data.params.input_dir is required for glob data loader")

        config_root = Path(self.config.get("_config_root", ".")).resolve()
        if isinstance(input_dir_value, str) and input_dir_value.startswith("oss://"):
            input_root = Path(oss_to_local(input_dir_value)).resolve()
        else:
            input_root = Path(input_dir_value)
            if not input_root.is_absolute():
                input_root = (config_root / input_root).resolve()

        if not input_root.exists():
            raise FileNotFoundError(f"input_dir does not exist: {input_root}")
        if not input_root.is_dir():
            raise NotADirectoryError(f"input_dir is not a directory: {input_root}")

        visualize_ratio = float(self.params.get("visualize_ratio", 0.0))
        if not 0.0 <= visualize_ratio <= 1.0:
            raise ValueError("data.params.visualize_ratio must be between 0.0 and 1.0")

        samples: list[dict[str, Any]] = []
        for path in input_root.rglob("*.mp4"):
            resolved_path = path.resolve()
            video_path = ensure_oss_path(str(resolved_path), config_root=self._config_root)
            samples.append(
                {
                    "video_path": video_path,
                    "data_path": video_path.replace(".mp4", ".pose3d_hand"),
                    "sample_id": Path(str(resolved_path)).name.replace(".mp4", ""),
                    "visualize": self._should_visualize(Path(str(resolved_path)).name.replace(".mp4", ""), visualize_ratio),
                }
            )

        return samples

    def _should_visualize(self, sample_id: str, visualize_ratio: float) -> bool:
        if visualize_ratio <= 0.0:
            return False
        if visualize_ratio >= 1.0:
            return True

        digest = hashlib.blake2b(sample_id.encode("utf-8"), digest_size=8).digest()
        random_value = int.from_bytes(digest, "big") / float(1 << 64)
        return random_value < visualize_ratio

@register_data_loader("database_loader")
class DatabaseDataLoader(BaseDataLoader):
    def load(self) -> list[dict[str, Any]]:
        database_url = self.params.get("database_url")
        database_key = self.params.get("database_key")
        dataset_name = self.params.get("dataset_name")
        table_name = self.params.get("database_table")
        visualize_ratio = float(self.params.get("visualize_ratio", 0.0))
        if not 0.0 <= visualize_ratio <= 1.0:
            raise ValueError("data.params.visualize_ratio must be between 0.0 and 1.0")
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
                    "video_path": ensure_oss_path(response.data[i]["path"], config_root=self._config_root),
                    "data_path": ensure_oss_path(
                        response.data[i]["path"].replace(".mp4", ".pose3d_hand"),
                        config_root=self._config_root,
                    ),
                    "sample_id": Path(response.data[i]["path"]).name.replace(".mp4", ""),
                    "visualize": self._should_visualize(
                        Path(response.data[i]["path"]).name.replace(".mp4", ""),
                        visualize_ratio,
                    ),
                }
                for i in range(len(response.data))
            ]
            samples.extend(sample)
        return samples

    def _should_visualize(self, sample_id: str, visualize_ratio: float) -> bool:
        if visualize_ratio <= 0.0:
            return False
        if visualize_ratio >= 1.0:
            return True

        digest = hashlib.blake2b(sample_id.encode("utf-8"), digest_size=8).digest()
        random_value = int.from_bytes(digest, "big") / float(1 << 64)
        return random_value < visualize_ratio

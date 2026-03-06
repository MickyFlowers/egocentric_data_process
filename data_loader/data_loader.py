from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from utils.oss_utils import ensure_oss_path, oss_to_local
from utils.oss_utils import configure_path_mapping

try:
    from supabase import create_client, Client
except ModuleNotFoundError:  # pragma: no cover - optional dependency for database loader
    create_client = None
    Client = Any  # type: ignore[assignment]

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
        self._configure_path_mapping()

    def __call__(self) -> list[dict[str, Any]]:
        return self._finalize_samples(self.load())

    def load(self) -> list[dict[str, Any]]:
        raise NotImplementedError

    def _finalize_samples(self, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return self._partition_samples(samples)

    def _partition_samples(self, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        raw_num_parts = self.params.get("num_parts", 1)
        raw_part = self.params.get("part", 0)

        num_parts = int(raw_num_parts)
        part = int(raw_part)
        if num_parts <= 0:
            raise ValueError("data.params.num_parts must be a positive integer")
        if part < 0 or part >= num_parts:
            raise ValueError(
                f"data.params.part must satisfy 0 <= part < num_parts, got part={part}, num_parts={num_parts}"
            )
        if num_parts == 1:
            return samples

        sorted_samples = sorted(
            samples,
            key=lambda sample: str(sample.get("sample_id", "")),
        )
        partitioned: list[dict[str, Any]] = []
        for index, sample in enumerate(sorted_samples):
            sample_id = sample.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id:
                raise KeyError("partitioned data loader requires non-empty string sample_id for every sample")
            if (index % num_parts) != part:
                continue
            partitioned.append(sample)

        print(
            f"[data_loader] partition enabled: part={part}, num_parts={num_parts}, "
            f"kept={len(partitioned)}/{len(samples)} sample(s)"
        )
        return partitioned

    def _configure_path_mapping(self) -> None:
        mapping = self.config.get("_path_mapping")
        if not isinstance(mapping, dict):
            return

        local_mount_value = mapping.get("local_mount")
        if not isinstance(local_mount_value, str) or not local_mount_value:
            return
        local_mount = Path(local_mount_value).expanduser()
        if not local_mount.is_absolute():
            local_mount = (self._config_root / local_mount).resolve()
        else:
            local_mount = local_mount.resolve()

        oss_prefix = mapping.get("oss_prefix", "oss://")
        if not isinstance(oss_prefix, str) or not oss_prefix:
            oss_prefix = "oss://"
        configure_path_mapping(str(local_mount), oss_prefix)


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


@register_data_loader("egodex")
@register_data_loader("ego_dex")
class EgoDexDataLoader(BaseDataLoader):
    def load(self) -> list[dict[str, Any]]:
        input_dir_value = self.params.get("input_dir") or self.params.get("input_root")
        if not input_dir_value:
            raise ValueError("data.params.input_dir is required for egodex data loader")

        config_root = Path(self.config.get("_config_root", ".")).resolve()
        if isinstance(input_dir_value, str) and input_dir_value.startswith("oss://"):
            input_root = Path(oss_to_local(input_dir_value)).resolve()
        else:
            input_root = Path(str(input_dir_value))
            if not input_root.is_absolute():
                input_root = (config_root / input_root).resolve()
            else:
                input_root = input_root.resolve()

        if not input_root.exists():
            raise FileNotFoundError(f"input_dir does not exist: {input_root}")
        if not input_root.is_dir():
            raise NotADirectoryError(f"input_dir is not a directory: {input_root}")

        visualize_ratio = float(self.params.get("visualize_ratio", 0.0))
        if not 0.0 <= visualize_ratio <= 1.0:
            raise ValueError("data.params.visualize_ratio must be between 0.0 and 1.0")

        annotation_glob = str(self.params.get("annotation_glob") or self.params.get("input_glob") or "*.hdf5")
        recursive = bool(self.params.get("recursive", True))
        video_extension = str(self.params.get("video_extension", ".mp4"))
        strict_pairing = bool(self.params.get("strict_pairing", True))

        iterator = input_root.rglob(annotation_glob) if recursive else input_root.glob(annotation_glob)
        samples: list[dict[str, Any]] = []
        for annotation_path in sorted(iterator):
            if not annotation_path.is_file():
                continue

            video_path = annotation_path.with_suffix(video_extension)
            if not video_path.exists():
                if strict_pairing:
                    raise FileNotFoundError(f"paired video not found for {annotation_path}: {video_path}")
                continue

            sample_id = annotation_path.relative_to(input_root).with_suffix("").as_posix()
            resolved_annotation = annotation_path.resolve()
            resolved_video = video_path.resolve()

            samples.append(
                {
                    "video_path": ensure_oss_path(str(resolved_video), config_root=self._config_root),
                    "data_path": ensure_oss_path(str(resolved_annotation), config_root=self._config_root),
                    "sample_id": sample_id,
                    "visualize": self._should_visualize(sample_id, visualize_ratio),
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
        if create_client is None:
            raise ModuleNotFoundError(
                "supabase is required for database_loader. Install it with `python -m pip install supabase`."
            )
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


@register_data_loader("processed")
@register_data_loader("processd")
class ProcessedDataLoader(BaseDataLoader):
    def load(self) -> list[dict[str, Any]]:
        input_dir_value = self.params.get("input_dir")
        if not input_dir_value:
            raise ValueError("data.params.input_dir is required for processed data loader")

        if isinstance(input_dir_value, str) and input_dir_value.startswith("oss://"):
            input_root = Path(oss_to_local(input_dir_value)).resolve()
        else:
            input_root = Path(str(input_dir_value))
            if not input_root.is_absolute():
                input_root = (self._config_root / input_root).resolve()
            else:
                input_root = input_root.resolve()

        meta_path = input_root / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"processed meta.json not found: {meta_path}")

        with open(meta_path, "r", encoding="utf-8") as file_obj:
            payload = json.load(file_obj)
        sample_ids = self._extract_sample_ids(payload)
        if not sample_ids:
            raise ValueError(f"no sample_id found in meta.json: {meta_path}")

        samples: list[dict[str, Any]] = []
        for sample_id in sample_ids:
            relative = Path(sample_id)
            parquet_path = (input_root / "data" / relative.parent / f"{relative.stem}.parquet").resolve()
            meta_data_path = (input_root / "meta_data" / relative.parent / f"{relative.stem}.json").resolve()
            if not parquet_path.exists() or not meta_data_path.exists():
                continue

            with open(meta_data_path, "r", encoding="utf-8") as file_obj:
                sample_meta = json.load(file_obj)
            if not isinstance(sample_meta, dict):
                continue

            sample = {
                "sample_id": sample_id,
                "trajectory_path": ensure_oss_path(str(parquet_path), config_root=self._config_root),
                "meta_data_path": ensure_oss_path(str(meta_data_path), config_root=self._config_root),
                "meta_data": sample_meta,
                "video_path": sample_meta.get("video_path"),
            }
            samples.append(sample)

        return samples

    @staticmethod
    def _extract_sample_ids(payload: Any) -> list[str]:
        collected: list[str] = []

        def _add_sample_id(value: Any) -> None:
            if isinstance(value, str) and value:
                collected.append(value)

        if isinstance(payload, dict):
            _add_sample_id(payload.get("sample_id"))
            sample_ids = payload.get("sample_ids")
            if isinstance(sample_ids, list):
                for item in sample_ids:
                    _add_sample_id(item)
        elif isinstance(payload, list):
            for entry in payload:
                if not isinstance(entry, dict):
                    continue
                _add_sample_id(entry.get("sample_id"))
                sample_ids = entry.get("sample_ids")
                if isinstance(sample_ids, list):
                    for item in sample_ids:
                        _add_sample_id(item)

        deduplicated = list(dict.fromkeys(collected))
        return deduplicated

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
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

    def _resolve_sample_id_tail_parts(
        self,
        *,
        default: int | None = None,
        dataset_name: str | None = None,
    ) -> int | None:
        raw_value = self.params.get("sample_id_tail_parts")
        if raw_value is not None:
            tail_parts = int(raw_value)
            if tail_parts <= 0:
                raise ValueError("data.params.sample_id_tail_parts must be a positive integer")
            return tail_parts
        if isinstance(dataset_name, str) and "egodex" in dataset_name.lower():
            return 3
        return default

    def _build_sample_id(
        self,
        path_value: str | Path,
        *,
        relative_to: Path | None = None,
        tail_parts: int | None = None,
    ) -> str:
        parts: list[str]
        if relative_to is not None:
            resolved_path = Path(path_value).resolve()
            relative_root = Path(relative_to).resolve()
            try:
                relative_path = resolved_path.relative_to(relative_root)
            except ValueError:
                relative_path = resolved_path
            parts = [part for part in relative_path.with_suffix("").parts if part not in ("", ".", "..")]
        else:
            raw_path = str(path_value).strip().replace("\\", "/")
            if raw_path.startswith("oss://"):
                raw_path = raw_path[len("oss://") :]
            parts = [part for part in raw_path.split("/") if part not in ("", ".", "..")]
            if parts:
                parts[-1] = PurePosixPath(parts[-1]).stem

        if tail_parts is not None and len(parts) > tail_parts:
            parts = parts[-tail_parts:]
        sample_id = "_".join(parts)
        if not sample_id:
            raise ValueError(f"failed to derive sample_id from path: {path_value}")
        return sample_id


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
# @register_data_loader("ego_dex")
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
        sample_id_tail_parts = self._resolve_sample_id_tail_parts(
            dataset_name=str(self.params.get("dataset_name", "")),
        )

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

            sample_id = self._build_sample_id(
                annotation_path,
                relative_to=input_root,
                tail_parts=sample_id_tail_parts,
            )
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


@register_data_loader("csv")
class CsvDataLoader(BaseDataLoader):
    def load(self) -> list[dict[str, Any]]:
        csv_path = self._resolve_csv_path()
        path_field = str(self.params.get("path_field", "path"))
        data_path_field = self.params.get("data_path_field")
        visualize_ratio = self._require_visualize_ratio()
        is_ml_egodex = self._require_bool_param("is_ml_egodex", default=False)
        match_token = str(self.params.get("ml_egodex_match_token", "/ml-egodex/")).strip().lower()
        if not match_token:
            raise ValueError("data.params.ml_egodex_match_token must be a non-empty string")

        default_data_extension = ".hdf5" if is_ml_egodex else ".pose3d_hand"
        data_extension = str(self.params.get("data_extension", default_data_extension))
        default_tail_parts = 3 if is_ml_egodex else 2
        sample_id_tail_parts = self._resolve_sample_id_tail_parts(
            default=default_tail_parts,
            dataset_name="ml-egodex" if is_ml_egodex else None,
        )
        encoding = str(self.params.get("encoding", "utf-8"))
        delimiter = str(self.params.get("delimiter", ","))

        samples: list[dict[str, Any]] = []
        with csv_path.open("r", encoding=encoding, newline="") as file_obj:
            reader = csv.DictReader(file_obj, delimiter=delimiter)
            fieldnames = list(reader.fieldnames or [])
            if path_field not in fieldnames:
                raise KeyError(f"CSV is missing required column '{path_field}': {csv_path}")
            if isinstance(data_path_field, str) and data_path_field and data_path_field not in fieldnames:
                raise KeyError(f"CSV is missing configured data_path_field '{data_path_field}': {csv_path}")

            for row in reader:
                raw_video_path = row.get(path_field)
                if not isinstance(raw_video_path, str):
                    continue

                video_path_value = raw_video_path.strip()
                if not video_path_value:
                    continue

                row_is_ml_egodex = match_token in video_path_value.lower()
                if row_is_ml_egodex != is_ml_egodex:
                    continue

                if isinstance(data_path_field, str) and data_path_field:
                    raw_data_path = row.get(data_path_field)
                    if not isinstance(raw_data_path, str) or not raw_data_path.strip():
                        continue
                    data_path_value = raw_data_path.strip()
                else:
                    data_path_value = self._replace_path_suffix(video_path_value, data_extension)

                sample_id = self._build_sample_id(video_path_value, tail_parts=sample_id_tail_parts)
                samples.append(
                    {
                        "video_path": ensure_oss_path(video_path_value, config_root=self._config_root),
                        "data_path": ensure_oss_path(data_path_value, config_root=self._config_root),
                        "sample_id": sample_id,
                        "visualize": self._should_visualize(sample_id, visualize_ratio),
                    }
                )

        samples = self._ensure_unique_sample_ids(samples, visualize_ratio=visualize_ratio)
        print(f"[data_loader] loaded {len(samples)} sample(s) from csv_path={csv_path}, is_ml_egodex={is_ml_egodex}")
        return samples

    def _resolve_csv_path(self) -> Path:
        csv_path_value = self.params.get("csv_path")
        if not isinstance(csv_path_value, str) or not csv_path_value:
            raise ValueError("data.params.csv_path is required for csv data loader")

        csv_path = Path(csv_path_value).expanduser()
        if not csv_path.is_absolute():
            csv_path = (self._config_root / csv_path).resolve()
        else:
            csv_path = csv_path.resolve()

        if not csv_path.exists():
            raise FileNotFoundError(f"csv_path does not exist: {csv_path}")
        if not csv_path.is_file():
            raise FileNotFoundError(f"csv_path is not a file: {csv_path}")
        return csv_path

    def _require_bool_param(self, name: str, *, default: bool) -> bool:
        raw_value = self.params.get(name, default)
        if isinstance(raw_value, bool):
            return raw_value
        if isinstance(raw_value, (int, float)) and raw_value in {0, 1}:
            return bool(raw_value)
        if isinstance(raw_value, str):
            normalized = raw_value.strip().lower()
            if normalized in {"1", "true", "yes", "y", "on"}:
                return True
            if normalized in {"0", "false", "no", "n", "off"}:
                return False
        raise ValueError(f"data.params.{name} must be a boolean, got {raw_value!r}")

    def _require_visualize_ratio(self) -> float:
        visualize_ratio = float(self.params.get("visualize_ratio", 0.0))
        if not 0.0 <= visualize_ratio <= 1.0:
            raise ValueError("data.params.visualize_ratio must be between 0.0 and 1.0")
        return visualize_ratio

    def _replace_path_suffix(self, path_value: str, suffix: str) -> str:
        normalized_path = path_value.strip().replace("\\", "/")
        prefix = ""
        path_without_prefix = normalized_path
        if normalized_path.startswith("oss://"):
            prefix = "oss://"
            path_without_prefix = normalized_path[len(prefix) :]
        return f"{prefix}{PurePosixPath(path_without_prefix).with_suffix(suffix)}"

    def _ensure_unique_sample_ids(
        self,
        samples: list[dict[str, Any]],
        *,
        visualize_ratio: float,
    ) -> list[dict[str, Any]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for sample in samples:
            sample_id = sample.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id:
                raise KeyError("csv loader requires non-empty string sample_id for every sample")
            grouped.setdefault(sample_id, []).append(sample)

        collision_count = 0
        for sample_id, group in grouped.items():
            if len(group) <= 1:
                continue

            collision_count += len(group)
            for sample in group:
                video_path = str(sample.get("video_path", ""))
                digest = hashlib.blake2b(video_path.encode("utf-8", "ignore"), digest_size=4).hexdigest()
                unique_id = f"{sample_id}_{digest}"
                sample["sample_id"] = unique_id
                sample["visualize"] = self._should_visualize(unique_id, visualize_ratio)

        if collision_count > 0:
            print(f"[data_loader] resolved {collision_count} colliding sample_id(s) in csv loader")
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
        database = self._create_database_client()
        dataset_name = self.params.get("dataset_name")
        table_name = self._require_non_empty_param("database_table")
        path_field = str(self.params.get("path_field", "path"))
        data_path_field = self.params.get("data_path_field")
        required_data_field = self.params.get("required_data_field", "pose3d_hand_path")
        apply_multi_hand_filter = bool(self.params.get("apply_multi_hand_filter", True))
        multi_hand_field = self.params.get("multi_hand_field", "multi_hand_flag")
        multi_hand_value = self.params.get("multi_hand_value", "False")
        order_field = str(self.params.get("order_field", path_field))
        page_size = self._require_positive_int_param("page_size", default=1000)
        data_extension = str(self.params.get("data_extension", ".pose3d_hand"))
        visualize_ratio = self._require_visualize_ratio()
        sample_id_tail_parts = self._resolve_sample_id_tail_parts(
            default=1,
            dataset_name=str(dataset_name) if dataset_name is not None else None,
        )
        select_fields = self._build_database_select_fields(path_field=path_field, data_path_field=data_path_field)

        start = 0
        samples: list[dict[str, Any]] = []
        while True:
            query = database.table(table_name).select(",".join(select_fields))
            query = self._apply_database_common_filters(
                query,
                dataset_name=dataset_name,
                required_data_field=required_data_field,
                apply_multi_hand_filter=apply_multi_hand_filter,
                multi_hand_field=multi_hand_field,
                multi_hand_value=multi_hand_value,
            )
            response = query.order(order_field).range(start, start + page_size - 1).execute()
            rows = list(response.data)
            if len(rows) == 0:
                break
            start += len(rows)
            print(f"Fetched {start} samples from database")
            samples.extend(
                self._build_database_samples(
                    rows,
                    path_field=path_field,
                    data_path_field=data_path_field,
                    data_extension=data_extension,
                    sample_id_tail_parts=sample_id_tail_parts,
                    visualize_ratio=visualize_ratio,
                )
            )
        return samples

    def _create_database_client(self) -> Client:
        if create_client is None:
            raise ModuleNotFoundError(
                "supabase is required for database_loader. Install it with `python -m pip install supabase`."
            )
        database_url = self._require_non_empty_param("database_url")
        database_key = self._require_non_empty_param("database_key")
        return create_client(database_url, database_key)

    def _require_non_empty_param(self, name: str) -> str:
        value = self.params.get(name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"data.params.{name} is required and must be a non-empty string")
        return value

    def _require_positive_int_param(self, name: str, *, default: int) -> int:
        raw_value = self.params.get(name, default)
        value = int(raw_value)
        if value <= 0:
            raise ValueError(f"data.params.{name} must be a positive integer")
        return value

    def _require_visualize_ratio(self) -> float:
        visualize_ratio = float(self.params.get("visualize_ratio", 0.0))
        if not 0.0 <= visualize_ratio <= 1.0:
            raise ValueError("data.params.visualize_ratio must be between 0.0 and 1.0")
        return visualize_ratio

    @staticmethod
    def _build_database_select_fields(*, path_field: str, data_path_field: Any) -> list[str]:
        select_fields = [path_field]
        if isinstance(data_path_field, str) and data_path_field and data_path_field not in select_fields:
            select_fields.append(data_path_field)
        return select_fields

    @staticmethod
    def _apply_database_common_filters(
        query: Any,
        *,
        dataset_name: Any,
        required_data_field: Any,
        apply_multi_hand_filter: bool,
        multi_hand_field: Any,
        multi_hand_value: Any,
    ) -> Any:
        if dataset_name is not None:
            query = query.eq("dataset_name", dataset_name)
        if isinstance(required_data_field, str) and required_data_field:
            query = query.not_.is_(required_data_field, "null")
        if apply_multi_hand_filter and isinstance(multi_hand_field, str) and multi_hand_field:
            query = query.is_(multi_hand_field, str(multi_hand_value))
        return query

    def _build_database_samples(
        self,
        rows: list[dict[str, Any]],
        *,
        path_field: str,
        data_path_field: Any,
        data_extension: str,
        sample_id_tail_parts: int | None,
        visualize_ratio: float,
    ) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        for row in rows:
            video_path_value = row.get(path_field)
            if not isinstance(video_path_value, str) or not video_path_value:
                continue
            if isinstance(data_path_field, str) and data_path_field:
                data_path_value = row.get(data_path_field)
            else:
                data_path_value = None
            if not isinstance(data_path_value, str) or not data_path_value:
                data_path_value = video_path_value.replace(".mp4", data_extension)
            sample_id = self._build_sample_id(video_path_value, tail_parts=sample_id_tail_parts)
            samples.append(
                {
                    "video_path": ensure_oss_path(video_path_value, config_root=self._config_root),
                    "data_path": ensure_oss_path(data_path_value, config_root=self._config_root),
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


@register_data_loader("random_database_loader")
class RandomDatabaseDataLoader(DatabaseDataLoader):
    def load(self) -> list[dict[str, Any]]:
        database = self._create_database_client()
        dataset_name = self.params.get("dataset_name")
        dataset_name_filter = self._resolve_random_dataset_filter(dataset_name)
        table_name = self._require_non_empty_param("database_table")
        path_field = str(self.params.get("path_field", "path"))
        data_path_field = self.params.get("data_path_field")
        required_data_field = self.params.get("required_data_field", "pose3d_hand_path")
        apply_multi_hand_filter = bool(self.params.get("apply_multi_hand_filter", True))
        multi_hand_field = self.params.get("multi_hand_field", "multi_hand_flag")
        multi_hand_value = self.params.get("multi_hand_value", "False")
        data_extension = str(self.params.get("data_extension", ".pose3d_hand"))
        visualize_ratio = self._require_visualize_ratio()
        sample_id_tail_parts = self._resolve_sample_id_tail_parts(default=None)
        random_number_field = str(self.params.get("random_number_field", "random_number"))
        random_threshold = float(self.params.get("random_threshold", 0.01))
        query_limit_raw = self.params.get("query_limit", 100)
        select_fields = self._build_database_select_fields(path_field=path_field, data_path_field=data_path_field)

        if not 0.0 <= random_threshold <= 1.0:
            raise ValueError("data.params.random_threshold must be between 0.0 and 1.0")

        query = database.table(table_name).select(",".join(select_fields))
        query = self._apply_database_common_filters(
            query,
            dataset_name=dataset_name_filter,
            required_data_field=required_data_field,
            apply_multi_hand_filter=apply_multi_hand_filter,
            multi_hand_field=multi_hand_field,
            multi_hand_value=multi_hand_value,
        )
        query = query.lt(random_number_field, random_threshold)
        if query_limit_raw is not None:
            query_limit = int(query_limit_raw)
            if query_limit <= 0:
                raise ValueError("data.params.query_limit must be a positive integer when provided")
            query = query.limit(query_limit)

        response = query.execute()
        rows = list(response.data)
        if dataset_name_filter is None:
            print(f"Fetched {len(rows)} random sample(s) from database")
        else:
            print(f"Fetched {len(rows)} random sample(s) from database for dataset={dataset_name_filter}")
        samples = self._build_database_samples(
            rows,
            path_field=path_field,
            data_path_field=data_path_field,
            data_extension=data_extension,
            sample_id_tail_parts=sample_id_tail_parts,
            visualize_ratio=visualize_ratio,
        )
        return self._ensure_unique_sample_ids(samples, visualize_ratio=visualize_ratio)

    @staticmethod
    def _resolve_random_dataset_filter(dataset_name: Any) -> str | None:
        if not isinstance(dataset_name, str):
            return None
        normalized = dataset_name.strip()
        if not normalized:
            return None
        if normalized.lower() in {"all_datasets", "all", "*", "none", "null"}:
            return None
        return normalized

    def _ensure_unique_sample_ids(
        self,
        samples: list[dict[str, Any]],
        *,
        visualize_ratio: float,
    ) -> list[dict[str, Any]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for sample in samples:
            sample_id = sample.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id:
                raise KeyError("random_database_loader requires non-empty string sample_id for every sample")
            grouped.setdefault(sample_id, []).append(sample)

        collision_count = 0
        for sample_id, group in grouped.items():
            if len(group) <= 1:
                continue
            collision_count += len(group)
            for sample in group:
                video_path = str(sample.get("video_path", ""))
                digest = hashlib.blake2b(video_path.encode("utf-8", "ignore"), digest_size=4).hexdigest()
                unique_id = f"{sample_id}_{digest}"
                sample["sample_id"] = unique_id
                sample["visualize"] = self._should_visualize(unique_id, visualize_ratio)

        if collision_count > 0:
            print(f"[data_loader] resolved {collision_count} colliding sample_id(s) in random_database_loader")
        return samples


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

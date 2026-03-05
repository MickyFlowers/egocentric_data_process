from __future__ import annotations

from pathlib import Path


_PATH_MAPPING = {
    "local_mount": str(Path(".").resolve()),
    "oss_prefix": "oss://",
}


def _normalize_oss_prefix(oss_prefix: str) -> str:
    if oss_prefix.endswith("://"):
        return oss_prefix
    return oss_prefix.rstrip("/")


def configure_path_mapping(local_mount: str, oss_prefix: str = "oss://") -> None:
    if not local_mount:
        raise ValueError("local_mount is required")
    if not oss_prefix:
        raise ValueError("oss_prefix is required")

    resolved_local_mount = str(Path(local_mount).expanduser().resolve())
    normalized_oss_prefix = _normalize_oss_prefix(oss_prefix)

    _PATH_MAPPING["local_mount"] = resolved_local_mount
    _PATH_MAPPING["oss_prefix"] = normalized_oss_prefix


def get_path_mapping() -> dict[str, str]:
    return dict(_PATH_MAPPING)


def oss_to_local(oss_path: str, mnt_path: str | None = None, oss_prefix: str | None = None) -> str:
    if not oss_path.startswith("oss://"):
        # Keep non-OSS paths stable and avoid binding to current runtime cwd.
        return str(Path(oss_path).expanduser())

    local_mount_path = Path(mnt_path or _PATH_MAPPING["local_mount"]).expanduser()
    if not local_mount_path.is_absolute():
        raise ValueError(
            f"local_mount must be absolute to avoid runtime-dir dependent paths: {local_mount_path}"
        )
    local_mount = str(local_mount_path).rstrip("/")
    normalized_prefix = _normalize_oss_prefix(oss_prefix or _PATH_MAPPING["oss_prefix"])
    if not oss_path.startswith(normalized_prefix):
        raise ValueError(f"oss_path does not start with configured oss_prefix: {oss_path}")

    relative_path = oss_path[len(normalized_prefix) :].lstrip("/")
    return str(Path(local_mount) / relative_path)


def local_to_oss(local_path: str, mnt_path: str | None = None, oss_prefix: str | None = None) -> str:
    if local_path.startswith("oss://"):
        return local_path

    local_mount = Path(mnt_path or _PATH_MAPPING["local_mount"]).expanduser().resolve()
    resolved_local_path = Path(local_path).expanduser().resolve()
    normalized_prefix = _normalize_oss_prefix(oss_prefix or _PATH_MAPPING["oss_prefix"])

    try:
        relative_path = resolved_local_path.relative_to(local_mount)
    except ValueError as exc:
        raise ValueError(
            f"local_path must be under local_mount. local_path={resolved_local_path}, local_mount={local_mount}"
        ) from exc

    if normalized_prefix.endswith("://"):
        return f"{normalized_prefix}{relative_path.as_posix()}"
    return f"{normalized_prefix}/{relative_path.as_posix()}"


def ensure_oss_path(path_value: str, config_root: str | Path | None = None) -> str:
    if path_value.startswith("oss://"):
        return path_value

    path = Path(path_value).expanduser()
    if not path.is_absolute():
        base_root = Path(config_root or ".").expanduser().resolve()
        path = (base_root / path).resolve()
    else:
        path = path.resolve()
    return local_to_oss(str(path))

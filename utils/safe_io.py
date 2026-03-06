from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any


def ensure_parent_dir(path: str | Path) -> None:
    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def build_temp_path(final_path: str | Path, temp_dir: str | Path | None = None) -> str:
    final_path = Path(final_path).expanduser()
    base_dir = Path(temp_dir).expanduser() if temp_dir else final_path.parent
    base_dir.mkdir(parents=True, exist_ok=True)

    if final_path.suffix:
        temp_name = f".{final_path.stem}.{os.getpid()}.{uuid.uuid4().hex}.tmp{final_path.suffix}"
    else:
        temp_name = f".{final_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    return str((base_dir / temp_name).resolve())


def commit_temp_path(temp_path: str | Path, final_path: str | Path) -> None:
    ensure_parent_dir(final_path)
    os.replace(str(temp_path), str(final_path))


def remove_path(path: str | Path) -> None:
    target = Path(path).expanduser()
    if not target.exists() and not target.is_symlink():
        return

    if target.is_dir() and not target.is_symlink():
        shutil.rmtree(target, ignore_errors=True)
        return

    try:
        target.unlink()
    except FileNotFoundError:
        return


def atomic_write_text(final_path: str | Path, content: str, encoding: str = "utf-8") -> None:
    temp_path = build_temp_path(final_path)
    try:
        with open(temp_path, "w", encoding=encoding) as file_obj:
            file_obj.write(content)
        commit_temp_path(temp_path, final_path)
    except BaseException:
        remove_path(temp_path)
        raise


def atomic_write_json(final_path: str | Path, payload: Any, indent: int = 2, *, sort_keys: bool = True) -> None:
    temp_path = build_temp_path(final_path)
    try:
        with open(temp_path, "w", encoding="utf-8") as file_obj:
            json.dump(payload, file_obj, ensure_ascii=False, indent=indent, sort_keys=sort_keys)
        commit_temp_path(temp_path, final_path)
    except BaseException:
        remove_path(temp_path)
        raise

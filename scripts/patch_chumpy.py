#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path


LEGACY_IMPORT = "from numpy import bool, int, float, complex, object, unicode, str, nan, inf"
PATCH_BLOCK = """import numpy as _np
_np_dict = _np.__dict__
bool = _np_dict.get("bool", bool)
int = _np_dict.get("int", int)
float = _np_dict.get("float", float)
complex = _np_dict.get("complex", complex)
object = _np_dict.get("object", object)
unicode = _np_dict.get("unicode", str)
str = _np_dict.get("str", str)
nan = _np.nan
inf = _np.inf"""
PATCH_MARKER = "_np_dict = _np.__dict__"


def resolve_chumpy_init() -> Path:
    spec = importlib.util.find_spec("chumpy")
    if spec is None or not spec.origin:
        raise ModuleNotFoundError("chumpy is not installed, skip patch")
    return Path(spec.origin).resolve()


def apply_patch(chumpy_init: Path) -> str:
    text = chumpy_init.read_text(encoding="utf-8")
    if PATCH_MARKER in text:
        return "already_patched"
    if LEGACY_IMPORT not in text:
        raise RuntimeError(f"unsupported chumpy layout in {chumpy_init}")
    updated = text.replace(LEGACY_IMPORT, PATCH_BLOCK, 1)
    chumpy_init.write_text(updated, encoding="utf-8")
    return "patched"


def main() -> None:
    try:
        chumpy_init = resolve_chumpy_init()
    except ModuleNotFoundError as exc:
        print(f"[patch_chumpy] {exc}")
        return

    status = apply_patch(chumpy_init)
    print(f"[patch_chumpy] {status}: {chumpy_init}")


if __name__ == "__main__":
    main()

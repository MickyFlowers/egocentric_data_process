from __future__ import annotations

import sys
from typing import Any

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


class NullProgressBar:
    def update(self, n: int = 1) -> None:
        return

    def set_postfix(self, **kwargs: Any) -> None:
        return

    def refresh(self) -> None:
        return

    def close(self) -> None:
        return


class SimpleProgressBar:
    def __init__(self, total: int | None = None, desc: str = "", unit: str = "item") -> None:
        self.total = total
        self.desc = desc
        self.unit = unit
        self.n = 0
        self.postfix = ""
        self._render()

    def update(self, n: int = 1) -> None:
        self.n += n
        self._render()

    def set_postfix(self, **kwargs: Any) -> None:
        parts = [f"{key}={value}" for key, value in kwargs.items()]
        self.postfix = f" [{' '.join(parts)}]" if parts else ""
        self._render()

    def refresh(self) -> None:
        self._render()

    def close(self) -> None:
        self._render()
        sys.stderr.write("\n")
        sys.stderr.flush()

    def _render(self) -> None:
        total_text = str(self.total) if self.total is not None else "?"
        prefix = f"{self.desc}: " if self.desc else ""
        sys.stderr.write(f"\r{prefix}{self.n}/{total_text} {self.unit}{self.postfix}")
        sys.stderr.flush()


def build_progress_bar(*, enabled: bool, total: int | None, desc: str, unit: str):
    if not enabled:
        return NullProgressBar()
    if tqdm is not None:
        return tqdm(total=total, desc=desc, unit=unit, dynamic_ncols=True, leave=True)
    return SimpleProgressBar(total=total, desc=desc, unit=unit)

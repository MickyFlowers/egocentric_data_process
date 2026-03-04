from __future__ import annotations

import json
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from utils.manifest import ManifestStore
from utils.safe_io import build_temp_path, commit_temp_path, remove_path


class PipelineContext:
    def __init__(self, sample_id: str, manifest: ManifestStore) -> None:
        self.sample_id = sample_id
        self.manifest = manifest
        self._cleanup_paths: set[str] = set()
        self._temp_paths: set[str] = set()

    @contextmanager
    def staged_output(self, final_path: str) -> Iterator[str]:
        final_path = str(Path(final_path).expanduser().resolve())
        temp_path = build_temp_path(final_path)

        self.manifest.add_temp_artifact(self.sample_id, temp_path)
        self._temp_paths.add(temp_path)
        self._cleanup_paths.add(temp_path)

        try:
            yield temp_path
            self.manifest.add_artifact(self.sample_id, final_path)
            self._cleanup_paths.add(final_path)
            commit_temp_path(temp_path, final_path)
            self.manifest.remove_temp_artifact(self.sample_id, temp_path)
            self._temp_paths.discard(temp_path)
            self._cleanup_paths.discard(temp_path)
        except Exception:
            remove_path(temp_path)
            self.manifest.discard_paths(self.sample_id, [temp_path, final_path])
            self._temp_paths.discard(temp_path)
            self._cleanup_paths.discard(temp_path)
            raise

    def copy_file(self, source_path: str, final_path: str) -> None:
        with self.staged_output(final_path) as temp_path:
            shutil.copy2(source_path, temp_path)

    def write_json(self, final_path: str, payload: Any, indent: int = 2) -> None:
        with self.staged_output(final_path) as temp_path:
            with open(temp_path, "w", encoding="utf-8") as file_obj:
                json.dump(payload, file_obj, ensure_ascii=False, indent=indent, sort_keys=True)

    def cleanup(self) -> None:
        cleanup_paths = sorted(self._cleanup_paths | self._temp_paths, reverse=True)
        for path in cleanup_paths:
            remove_path(path)
        self.manifest.discard_paths(self.sample_id, cleanup_paths)
        self._cleanup_paths.clear()
        self._temp_paths.clear()

    def finish(self) -> None:
        self._cleanup_paths.clear()
        self._temp_paths.clear()

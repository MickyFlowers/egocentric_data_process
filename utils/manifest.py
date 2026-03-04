from __future__ import annotations

import fcntl
import json
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from utils.safe_io import atomic_write_json, remove_path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_manifest() -> dict[str, Any]:
    return {"version": 1, "tasks": {}}


class ManifestStore:
    def __init__(self, manifest_path: str | Path) -> None:
        self.path = Path(manifest_path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.path.with_suffix(f"{self.path.suffix}.lock" if self.path.suffix else ".lock")
        self._ensure_manifest_file()

    def _ensure_manifest_file(self) -> None:
        if self.path.exists():
            return
        atomic_write_json(self.path, default_manifest())

    def _read_state(self) -> dict[str, Any]:
        if not self.path.exists():
            return default_manifest()

        raw_text = self.path.read_text(encoding="utf-8").strip()
        if not raw_text:
            return default_manifest()
        return json.loads(raw_text)

    @contextmanager
    def _locked_state(self) -> Iterator[dict[str, Any]]:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.lock_path, "a+", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            state = self._read_state()
            yield state
            atomic_write_json(self.path, state)
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _task_entry(self, state: dict[str, Any], sample: dict[str, Any]) -> dict[str, Any]:
        sample_id = sample["sample_id"]
        tasks = state.setdefault("tasks", {})
        if sample_id not in tasks:
            tasks[sample_id] = {
                "status": "pending",
                "payload": sample,
                "artifacts": [],
                "temp_artifacts": [],
                "attempts": 0,
                "created_at": utc_now(),
                "updated_at": utc_now(),
                "last_error": None,
                "worker": None,
            }
        return tasks[sample_id]

    def sync_tasks(self, samples: list[dict[str, Any]]) -> dict[str, int]:
        new_count = 0
        refreshed_count = 0
        with self._locked_state() as state:
            tasks = state.setdefault("tasks", {})
            for sample in samples:
                sample_id = sample["data_path"]
                if sample_id not in tasks:
                    tasks[sample_id] = {
                        "status": "pending",
                        "payload": sample,
                        "artifacts": [],
                        "temp_artifacts": [],
                        "attempts": 0,
                        "created_at": utc_now(),
                        "updated_at": utc_now(),
                        "last_error": None,
                        "worker": None,
                    }
                    new_count += 1
                    continue

                entry = tasks[sample_id]
                entry["payload"] = sample
                entry["updated_at"] = utc_now()
                refreshed_count += 1
        return {"new": new_count, "refreshed": refreshed_count}

    def claim_task(self, sample_id: str, worker: dict[str, Any]) -> bool:
        with self._locked_state() as state:
            entry = state["tasks"].get(sample_id)
            if entry is None or entry["status"] != "pending":
                return False

            entry["status"] = "in_progress"
            entry["worker"] = worker
            entry["started_at"] = utc_now()
            entry["updated_at"] = utc_now()
            entry["last_error"] = None
            entry["attempts"] = int(entry.get("attempts", 0)) + 1
            return True

    def add_temp_artifact(self, sample_id: str, path: str) -> None:
        with self._locked_state() as state:
            entry = state["tasks"][sample_id]
            if path not in entry["temp_artifacts"]:
                entry["temp_artifacts"].append(path)
                entry["updated_at"] = utc_now()

    def add_artifact(self, sample_id: str, path: str) -> None:
        with self._locked_state() as state:
            entry = state["tasks"][sample_id]
            if path not in entry["artifacts"]:
                entry["artifacts"].append(path)
                entry["updated_at"] = utc_now()

    def remove_temp_artifact(self, sample_id: str, path: str) -> None:
        with self._locked_state() as state:
            entry = state["tasks"][sample_id]
            entry["temp_artifacts"] = [item for item in entry["temp_artifacts"] if item != path]
            entry["updated_at"] = utc_now()

    def discard_paths(self, sample_id: str, paths: list[str]) -> None:
        drop_set = set(paths)
        if not drop_set:
            return

        with self._locked_state() as state:
            entry = state["tasks"][sample_id]
            entry["artifacts"] = [item for item in entry["artifacts"] if item not in drop_set]
            entry["temp_artifacts"] = [item for item in entry["temp_artifacts"] if item not in drop_set]
            entry["updated_at"] = utc_now()

    def mark_completed(self, sample_id: str, summary: dict[str, Any] | None = None) -> None:
        with self._locked_state() as state:
            entry = state["tasks"][sample_id]
            entry["status"] = "completed"
            entry["worker"] = None
            entry["completed_at"] = utc_now()
            entry["updated_at"] = utc_now()
            entry["last_error"] = None
            if summary is not None:
                entry["summary"] = summary

    def reset_to_pending(self, sample_id: str, last_error: str | None = None) -> None:
        with self._locked_state() as state:
            entry = state["tasks"][sample_id]
            entry["status"] = "pending"
            entry["worker"] = None
            entry["artifacts"] = []
            entry["temp_artifacts"] = []
            entry["updated_at"] = utc_now()
            entry["last_error"] = last_error

    def pending_samples(self, limit: int | None = None) -> list[dict[str, Any]]:
        with self._locked_state() as state:
            pending = [
                entry["payload"]
                for _, entry in sorted(state.get("tasks", {}).items())
                if entry.get("status") == "pending"
            ]
        return pending[:limit] if limit is not None else pending

    def summary(self) -> dict[str, int]:
        with self._locked_state() as state:
            counts = {"pending": 0, "in_progress": 0, "completed": 0}
            for entry in state.get("tasks", {}).values():
                status = entry.get("status", "pending")
                counts.setdefault(status, 0)
                counts[status] += 1
        return counts

    def recover_in_progress(self) -> list[str]:
        cleanup_paths: list[str] = []
        recovered_samples: list[str] = []

        with self._locked_state() as state:
            for sample_id, entry in state.get("tasks", {}).items():
                if entry.get("status") != "in_progress":
                    continue

                cleanup_paths.extend(entry.get("artifacts", []))
                cleanup_paths.extend(entry.get("temp_artifacts", []))
                entry["status"] = "pending"
                entry["worker"] = None
                entry["artifacts"] = []
                entry["temp_artifacts"] = []
                entry["updated_at"] = utc_now()
                entry["last_error"] = "Recovered from unfinished in_progress task."
                recovered_samples.append(sample_id)

        for path in sorted(set(cleanup_paths), reverse=True):
            remove_path(path)

        return recovered_samples

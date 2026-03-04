from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from utils.safe_io import remove_path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_manifest() -> dict[str, Any]:
    return {"version": 1, "tasks": {}}


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _json_load(value: str | None, default: Any) -> Any:
    if value in {None, ""}:
        return default
    return json.loads(value)


def _batched(items: Iterable[Any], batch_size: int) -> Iterable[list[Any]]:
    batch: list[Any] = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


class ManifestStore:
    def __init__(self, manifest_path: str | Path) -> None:
        self.path = Path(manifest_path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = self._build_db_path(self.path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, timeout=30.0, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._initialise_database()
        self._migrate_legacy_json_if_needed()

    def _build_db_path(self, path: Path) -> Path:
        if path.suffix in {".db", ".sqlite", ".sqlite3"}:
            return path
        return Path(f"{path}.sqlite3")

    def _initialise_database(self) -> None:
        with self._conn:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA temp_store=MEMORY")
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    sample_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    artifacts TEXT NOT NULL,
                    temp_artifacts TEXT NOT NULL,
                    attempts INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_error TEXT,
                    worker TEXT,
                    started_at TEXT,
                    completed_at TEXT,
                    summary TEXT
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_tasks_status_sample_id ON tasks(status, sample_id)"
            )

    def _migrate_legacy_json_if_needed(self) -> None:
        if self.db_path == self.path or not self.path.exists():
            return

        cursor = self._conn.execute("SELECT COUNT(*) AS count FROM tasks")
        if int(cursor.fetchone()["count"]) > 0:
            return

        try:
            raw_text = self.path.read_text(encoding="utf-8").strip()
        except OSError:
            return
        if not raw_text:
            return

        try:
            state = json.loads(raw_text)
        except json.JSONDecodeError:
            return
        if not isinstance(state, dict):
            return

        tasks = state.get("tasks", {})
        if not isinstance(tasks, dict) or not tasks:
            return

        rows = []
        for sample_id, entry in tasks.items():
            if not isinstance(entry, dict):
                continue
            rows.append(self._row_from_entry(sample_id, entry))

        if not rows:
            return

        with self._conn:
            self._conn.executemany(
                """
                INSERT OR REPLACE INTO tasks (
                    sample_id, status, payload, artifacts, temp_artifacts, attempts,
                    created_at, updated_at, last_error, worker, started_at, completed_at, summary
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    def _empty_task_entry(self, sample: dict[str, Any]) -> dict[str, Any]:
        timestamp = utc_now()
        return {
            "status": "pending",
            "payload": sample,
            "artifacts": [],
            "temp_artifacts": [],
            "attempts": 0,
            "created_at": timestamp,
            "updated_at": timestamp,
            "last_error": None,
            "worker": None,
            "started_at": None,
            "completed_at": None,
            "summary": None,
        }

    def _row_from_entry(self, sample_id: str, entry: dict[str, Any]) -> tuple[Any, ...]:
        return (
            sample_id,
            entry.get("status", "pending"),
            _json_dump(entry.get("payload", {})),
            _json_dump(entry.get("artifacts", [])),
            _json_dump(entry.get("temp_artifacts", [])),
            int(entry.get("attempts", 0)),
            entry.get("created_at", utc_now()),
            entry.get("updated_at", utc_now()),
            entry.get("last_error"),
            _json_dump(entry.get("worker")) if entry.get("worker") is not None else None,
            entry.get("started_at"),
            entry.get("completed_at"),
            _json_dump(entry.get("summary")) if entry.get("summary") is not None else None,
        )

    def _fetch_json_list_column(self, sample_id: str, column: str) -> list[str]:
        row = self._conn.execute(f"SELECT {column} FROM tasks WHERE sample_id = ?", (sample_id,)).fetchone()
        if row is None:
            raise KeyError(f"sample_id not found in manifest: {sample_id}")
        return list(_json_load(row[column], []))

    def sync_tasks(self, samples: list[dict[str, Any]]) -> dict[str, int]:
        if not samples:
            return {"new": 0, "refreshed": 0}

        sample_ids = [sample["sample_id"] for sample in samples]
        existing_ids: set[str] = set()
        for batch in _batched(sample_ids, 500):
            placeholders = ",".join("?" for _ in batch)
            rows = self._conn.execute(
                f"SELECT sample_id FROM tasks WHERE sample_id IN ({placeholders})",
                tuple(batch),
            ).fetchall()
            existing_ids.update(row["sample_id"] for row in rows)

        new_rows = []
        refresh_rows = []
        for sample in samples:
            sample_id = sample["sample_id"]
            timestamp = utc_now()
            if sample_id in existing_ids:
                refresh_rows.append((_json_dump(sample), timestamp, sample_id))
            else:
                entry = self._empty_task_entry(sample)
                entry["created_at"] = timestamp
                entry["updated_at"] = timestamp
                new_rows.append(self._row_from_entry(sample_id, entry))

        with self._conn:
            if new_rows:
                self._conn.executemany(
                    """
                    INSERT INTO tasks (
                        sample_id, status, payload, artifacts, temp_artifacts, attempts,
                        created_at, updated_at, last_error, worker, started_at, completed_at, summary
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    new_rows,
                )
            if refresh_rows:
                self._conn.executemany(
                    "UPDATE tasks SET payload = ?, updated_at = ? WHERE sample_id = ?",
                    refresh_rows,
                )

        return {"new": len(new_rows), "refreshed": len(refresh_rows)}

    def claim_task(self, sample_id: str, worker: dict[str, Any]) -> bool:
        timestamp = utc_now()
        worker_json = _json_dump(worker)
        with self._conn:
            cursor = self._conn.execute(
                """
                UPDATE tasks
                SET status = ?,
                    worker = ?,
                    started_at = ?,
                    updated_at = ?,
                    last_error = NULL,
                    attempts = attempts + 1
                WHERE sample_id = ? AND status = ?
                """,
                ("in_progress", worker_json, timestamp, timestamp, sample_id, "pending"),
            )
        return cursor.rowcount > 0

    def add_temp_artifact(self, sample_id: str, path: str) -> None:
        temp_artifacts = self._fetch_json_list_column(sample_id, "temp_artifacts")
        if path in temp_artifacts:
            return
        temp_artifacts.append(path)
        with self._conn:
            self._conn.execute(
                "UPDATE tasks SET temp_artifacts = ?, updated_at = ? WHERE sample_id = ?",
                (_json_dump(temp_artifacts), utc_now(), sample_id),
            )

    def add_artifact(self, sample_id: str, path: str) -> None:
        artifacts = self._fetch_json_list_column(sample_id, "artifacts")
        if path in artifacts:
            return
        artifacts.append(path)
        with self._conn:
            self._conn.execute(
                "UPDATE tasks SET artifacts = ?, updated_at = ? WHERE sample_id = ?",
                (_json_dump(artifacts), utc_now(), sample_id),
            )

    def remove_temp_artifact(self, sample_id: str, path: str) -> None:
        temp_artifacts = self._fetch_json_list_column(sample_id, "temp_artifacts")
        filtered = [item for item in temp_artifacts if item != path]
        with self._conn:
            self._conn.execute(
                "UPDATE tasks SET temp_artifacts = ?, updated_at = ? WHERE sample_id = ?",
                (_json_dump(filtered), utc_now(), sample_id),
            )

    def discard_paths(self, sample_id: str, paths: list[str]) -> None:
        drop_set = set(paths)
        if not drop_set:
            return

        artifacts = [item for item in self._fetch_json_list_column(sample_id, "artifacts") if item not in drop_set]
        temp_artifacts = [
            item for item in self._fetch_json_list_column(sample_id, "temp_artifacts") if item not in drop_set
        ]
        with self._conn:
            self._conn.execute(
                """
                UPDATE tasks
                SET artifacts = ?, temp_artifacts = ?, updated_at = ?
                WHERE sample_id = ?
                """,
                (_json_dump(artifacts), _json_dump(temp_artifacts), utc_now(), sample_id),
            )

    def mark_completed(self, sample_id: str, summary: dict[str, Any] | None = None) -> None:
        timestamp = utc_now()
        with self._conn:
            self._conn.execute(
                """
                UPDATE tasks
                SET status = ?,
                    worker = NULL,
                    completed_at = ?,
                    updated_at = ?,
                    last_error = NULL,
                    summary = ?
                WHERE sample_id = ?
                """,
                ("completed", timestamp, timestamp, _json_dump(summary) if summary is not None else None, sample_id),
            )

    def reset_to_pending(self, sample_id: str, last_error: str | None = None) -> None:
        with self._conn:
            self._conn.execute(
                """
                UPDATE tasks
                SET status = ?,
                    worker = NULL,
                    artifacts = ?,
                    temp_artifacts = ?,
                    updated_at = ?,
                    last_error = ?
                WHERE sample_id = ?
                """,
                ("pending", _json_dump([]), _json_dump([]), utc_now(), last_error, sample_id),
            )

    def pending_samples(self, sample_ids: set[str] | None = None, limit: int | None = None) -> list[dict[str, Any]]:
        query = "SELECT payload FROM tasks WHERE status = ?"
        params: list[Any] = ["pending"]

        if sample_ids:
            ids = sorted(sample_ids)
            pending: list[dict[str, Any]] = []
            collected = 0
            for batch in _batched(ids, 500):
                placeholders = ",".join("?" for _ in batch)
                batch_query = f"{query} AND sample_id IN ({placeholders}) ORDER BY sample_id"
                if limit is not None:
                    remaining = limit - collected
                    if remaining <= 0:
                        break
                    batch_query += " LIMIT ?"
                    batch_params = params + batch + [remaining]
                else:
                    batch_params = params + batch
                rows = self._conn.execute(batch_query, tuple(batch_params)).fetchall()
                decoded = [_json_load(row["payload"], {}) for row in rows]
                pending.extend(decoded)
                collected += len(decoded)
            return pending

        query += " ORDER BY sample_id"
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        rows = self._conn.execute(query, tuple(params)).fetchall()
        return [_json_load(row["payload"], {}) for row in rows]

    def summary(self) -> dict[str, int]:
        counts = {"pending": 0, "in_progress": 0, "completed": 0}
        rows = self._conn.execute("SELECT status, COUNT(*) AS count FROM tasks GROUP BY status").fetchall()
        for row in rows:
            counts[row["status"]] = int(row["count"])
        return counts

    def recover_in_progress(self) -> list[str]:
        rows = self._conn.execute(
            "SELECT sample_id, artifacts, temp_artifacts FROM tasks WHERE status = ?",
            ("in_progress",),
        ).fetchall()
        if not rows:
            return []

        cleanup_paths: list[str] = []
        recovered_samples: list[str] = []
        for row in rows:
            recovered_samples.append(row["sample_id"])
            cleanup_paths.extend(_json_load(row["artifacts"], []))
            cleanup_paths.extend(_json_load(row["temp_artifacts"], []))

        with self._conn:
            self._conn.execute(
                """
                UPDATE tasks
                SET status = ?,
                    worker = NULL,
                    artifacts = ?,
                    temp_artifacts = ?,
                    updated_at = ?,
                    last_error = ?
                WHERE status = ?
                """,
                (
                    "pending",
                    _json_dump([]),
                    _json_dump([]),
                    utc_now(),
                    "Recovered from unfinished in_progress task.",
                    "in_progress",
                ),
            )

        for path in sorted(set(cleanup_paths), reverse=True):
            remove_path(path)

        return recovered_samples

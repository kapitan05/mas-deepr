"""SQLite HTTP cache shared by search and fetch tools.

Caching every outbound call makes eval runs reproducible (BrowseComp answers
don't drift as the live web changes between milestones) and keeps dev-loop
cost down. Keyed by a normalized request signature, not the raw URL, so
search queries and page fetches share one table with no collisions.
"""

import hashlib
import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS http_cache (
    key TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at REAL NOT NULL DEFAULT (unixepoch('subsec'))
);
"""


class WebCache:
    """Thread-safe key-value cache over SQLite. One process, many callers."""

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.execute(_SCHEMA)

    @contextmanager
    def _connect(self) -> Any:
        conn = sqlite3.connect(self._db_path, timeout=30)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def make_key(kind: str, **params: object) -> str:
        blob = json.dumps({"kind": kind, **params}, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def get(self, key: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM http_cache WHERE key = ?", (key,)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def set(self, key: str, kind: str, payload: dict[str, Any]) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO http_cache (key, kind, payload) "
                "VALUES (?, ?, ?)",
                (key, kind, json.dumps(payload, ensure_ascii=False)),
            )

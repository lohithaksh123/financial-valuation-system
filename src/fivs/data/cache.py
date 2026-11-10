"""A small SQLite key-value cache with expiry, so repeat runs stay off the network."""
from __future__ import annotations

import json
import sqlite3
import time
import zlib
from contextlib import closing
from pathlib import Path
from typing import Any, Optional

from fivs.config import CACHE_PATH, CACHE_TTL_SECONDS


class Cache:
    def __init__(self, path: Path = CACHE_PATH, ttl_seconds: int = CACHE_TTL_SECONDS):
        self.path = Path(path)
        self.ttl = ttl_seconds
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as db, db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS cache "
                "(key TEXT PRIMARY KEY, value BLOB NOT NULL, stored_at REAL NOT NULL)"
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=30)

    def get(self, key: str) -> Optional[Any]:
        with closing(self._connect()) as db:
            row = db.execute("SELECT value, stored_at FROM cache WHERE key = ?", (key,)).fetchone()
        if row is None or time.time() - row[1] > self.ttl:
            return None
        return json.loads(zlib.decompress(row[0]))

    def set(self, key: str, value: Any) -> None:
        blob = zlib.compress(json.dumps(value).encode())
        with closing(self._connect()) as db, db:
            db.execute(
                "INSERT OR REPLACE INTO cache (key, value, stored_at) VALUES (?, ?, ?)",
                (key, blob, time.time()),
            )

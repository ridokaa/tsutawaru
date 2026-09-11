"""L1 LRU + L2 SQLite persistent translation cache."""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import sqlite3
import threading
import time
from typing import Optional
from platformdirs import user_cache_dir


class LRU:
    def __init__(self, cap: int):
        self.cap = cap
        self.d: OrderedDict[str, str] = OrderedDict()
        self.lock = threading.Lock()

    def get(self, k: str) -> Optional[str]:
        with self.lock:
            if k in self.d:
                self.d.move_to_end(k)
                return self.d[k]
        return None

    def put(self, k: str, v: str) -> None:
        with self.lock:
            self.d[k] = v
            self.d.move_to_end(k)
            if len(self.d) > self.cap:
                self.d.popitem(last=False)


class GlossCache:
    """L1 in-memory LRU -> L2 SQLite (survives restarts) -> miss."""

    def __init__(self, cap: int = 20000, persist: bool = True, db_path: Optional[str] = None):
        self.l1 = LRU(cap)
        self.db: Optional[sqlite3.Connection] = None
        self.lock = threading.Lock()
        self._pending: list[tuple[str, str]] = []
        self._last_commit = time.monotonic()

        if persist:
            if db_path is not None:
                p = Path(db_path)
                p.parent.mkdir(parents=True, exist_ok=True)
                db_file = p
            else:
                p = Path(user_cache_dir("tsutawaru"))
                p.mkdir(parents=True, exist_ok=True)
                db_file = p / "gloss.db"
            self.db = sqlite3.connect(str(db_file), check_same_thread=False)
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute("CREATE TABLE IF NOT EXISTS g(k TEXT PRIMARY KEY, v TEXT)")
            self.db.commit()

    def get(self, key: str) -> Optional[str]:
        v = self.l1.get(key)
        if v is not None:
            return v
        if self.db:
            with self.lock:
                row = self.db.execute("SELECT v FROM g WHERE k=?", (key,)).fetchone()
            if row:
                self.l1.put(key, row[0])
                return row[0]
        return None

    def put(self, key: str, val: str) -> None:
        self.l1.put(key, val)
        if not self.db:
            return
        with self.lock:
            self._pending.append((key, val))
            if len(self._pending) >= 32 or time.monotonic() - self._last_commit > 5.0:
                self.db.executemany("INSERT OR REPLACE INTO g VALUES(?,?)", self._pending)
                self.db.commit()
                self._pending.clear()
                self._last_commit = time.monotonic()

    def flush(self) -> None:
        with self.lock:
            if self.db and self._pending:
                self.db.executemany("INSERT OR REPLACE INTO g VALUES(?,?)", self._pending)
                self.db.commit()
                self._pending.clear()

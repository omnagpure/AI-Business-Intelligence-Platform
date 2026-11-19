"""SQLite storage for accounts, sessions and per-user records.

One database file under the data directory (BI_DATA_DIR, default ./data_store, which is git-ignored).
Every connection enables foreign keys, so deleting a user also deletes their sessions, datasets and reports.
All queries elsewhere use parameter binding; nothing builds SQL from user input.
"""
from __future__ import annotations

import os
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from config.settings import PROJECT_ROOT

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    email           TEXT    NOT NULL UNIQUE COLLATE NOCASE,
    full_name       TEXT    NOT NULL,
    password_hash   TEXT    NOT NULL,
    created_at      REAL    NOT NULL,
    last_login_at   REAL
);

CREATE TABLE IF NOT EXISTS login_failures (    -- per email, whether or not an account exists, so throttling leaks nothing
    email           TEXT    NOT NULL COLLATE NOCASE,
    at              REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_login_failures ON login_failures(email, at);

CREATE TABLE IF NOT EXISTS auth_sessions (
    id              TEXT    PRIMARY KEY,          -- SHA-256 of the session token; the token itself is never stored
    user_id         INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at      REAL    NOT NULL,
    expires_at      REAL    NOT NULL,
    last_seen_at    REAL    NOT NULL,
    remember        INTEGER NOT NULL DEFAULT 0,
    revoked_at      REAL
);
CREATE INDEX IF NOT EXISTS ix_sessions_user ON auth_sessions(user_id);

CREATE TABLE IF NOT EXISTS auth_handoffs (
    code_hash       TEXT    PRIMARY KEY,          -- SHA-256 of a single-use code, valid for seconds
    user_id         INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    remember        INTEGER NOT NULL,
    browser_binding TEXT,                         -- hash of the browser's XSRF token, when available
    expires_at      REAL    NOT NULL
);

CREATE TABLE IF NOT EXISTS datasets (
    id              TEXT    PRIMARY KEY,          -- random id, also the stored file name
    user_id         INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    filename        TEXT    NOT NULL,
    sheet           TEXT,
    fingerprint     TEXT    NOT NULL,
    size_bytes      INTEGER NOT NULL,
    rows            INTEGER NOT NULL,
    columns         INTEGER NOT NULL,
    stored_name     TEXT    NOT NULL,
    created_at      REAL    NOT NULL,
    last_opened_at  REAL    NOT NULL,
    UNIQUE (user_id, fingerprint)
);
CREATE INDEX IF NOT EXISTS ix_datasets_user ON datasets(user_id, last_opened_at DESC);

CREATE TABLE IF NOT EXISTS reports (
    id              TEXT    PRIMARY KEY,
    user_id         INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    dataset_id      TEXT    REFERENCES datasets(id) ON DELETE SET NULL,
    dataset_name    TEXT    NOT NULL,
    title           TEXT    NOT NULL,
    fmt             TEXT    NOT NULL CHECK (fmt IN ('pdf', 'html')),
    size_bytes      INTEGER NOT NULL,
    stored_name     TEXT    NOT NULL,
    created_at      REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_reports_user ON reports(user_id, created_at DESC);
"""

_init_lock = threading.Lock()
_initialised: set[str] = set()


def data_dir() -> Path:
    """Root folder for the database and stored user files. Read on every call so tests can point it elsewhere."""
    raw = os.getenv("BI_DATA_DIR", "data_store")
    path = Path(raw)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def db_path() -> Path:
    return data_dir() / "app.db"


def _ensure_schema(path: Path) -> None:
    key = str(path)
    if key in _initialised and path.exists():
        return
    with _init_lock:
        if key in _initialised and path.exists():
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(_SCHEMA)
            conn.commit()
        finally:
            conn.close()
        _initialised.add(key)


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    """A short-lived connection in one transaction: committed on success, rolled back on error."""
    path = db_path()
    _ensure_schema(path)
    conn = sqlite3.connect(path, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

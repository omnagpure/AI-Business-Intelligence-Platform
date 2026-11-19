"""Per-user datasets and reports: metadata in SQLite, file contents under <data dir>/users/<user id>/.

Ownership is enforced here, not in the UI: every function takes the signed-in user's id and every query filters on
it, so a dataset or report id belonging to someone else behaves exactly like an id that does not exist.
Stored file names are random ids generated here; nothing from the user (file names, ids) is ever used as a path.
"""
from __future__ import annotations

import os
import secrets
import time
from dataclasses import dataclass
from pathlib import Path

from auth.db import connect, data_dir
from core.logging_config import get_logger

log = get_logger("storage")
_KINDS = ("datasets", "reports")


class NotFound(Exception):
    """The record does not exist or is not owned by this user (deliberately indistinguishable)."""


@dataclass(frozen=True)
class DatasetRecord:
    id: str
    filename: str
    sheet: str | None
    fingerprint: str
    size_bytes: int
    rows: int
    columns: int
    created_at: float
    last_opened_at: float


@dataclass(frozen=True)
class ReportRecord:
    id: str
    dataset_id: str | None
    dataset_name: str
    title: str
    fmt: str
    size_bytes: int
    created_at: float


def _require_user(user_id: int) -> int:
    if not isinstance(user_id, int) or isinstance(user_id, bool) or user_id <= 0:
        raise PermissionError("A signed-in user is required.")
    return user_id


def _user_dir(user_id: int, kind: str) -> Path:
    assert kind in _KINDS
    return data_dir() / "users" / str(_require_user(user_id)) / kind


def _file_path(user_id: int, kind: str, stored_name: str) -> Path:
    base = _user_dir(user_id, kind).resolve()
    path = (base / stored_name).resolve()
    if path.parent != base:                       # defence in depth: stored names are generated, never user input
        raise PermissionError("Invalid storage path.")
    return path


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _remove(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        log.warning("Could not delete a stored file; it is no longer referenced")


# ---- datasets ----------------------------------------------------------------------------------------------------
def _ds(row) -> DatasetRecord:
    return DatasetRecord(row["id"], row["filename"], row["sheet"], row["fingerprint"], row["size_bytes"], row["rows"],
                         row["columns"], row["created_at"], row["last_opened_at"])


def save_dataset(user_id: int, data: bytes, filename: str, sheet: str | None, fingerprint: str,
                 rows: int, columns: int) -> DatasetRecord:
    """Store an uploaded file for this user. The same file uploaded again is not stored twice."""
    _require_user(user_id)
    now = time.time()
    with connect() as conn:
        row = conn.execute("SELECT * FROM datasets WHERE user_id = ? AND fingerprint = ?", (user_id, fingerprint)).fetchone()
        if row is not None:
            conn.execute("UPDATE datasets SET last_opened_at = ?, filename = ? WHERE id = ? AND user_id = ?",
                         (now, filename[:255], row["id"], user_id))
            return _ds(conn.execute("SELECT * FROM datasets WHERE id = ? AND user_id = ?", (row["id"], user_id)).fetchone())
    ds_id = secrets.token_hex(16)
    stored = f"{ds_id}.bin"
    path = _file_path(user_id, "datasets", stored)
    _write_atomic(path, data)
    try:
        with connect() as conn:
            conn.execute("INSERT INTO datasets (id, user_id, filename, sheet, fingerprint, size_bytes, rows, columns, stored_name, "
                         "created_at, last_opened_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                         (ds_id, user_id, filename[:255], sheet, fingerprint, len(data), rows, columns, stored, now, now))
            row = conn.execute("SELECT * FROM datasets WHERE id = ? AND user_id = ?", (ds_id, user_id)).fetchone()
    except Exception:
        _remove(path)
        raise
    log.info("Dataset stored for user %s", user_id)
    return _ds(row)


def list_datasets(user_id: int) -> list[DatasetRecord]:
    with connect() as conn:
        rows = conn.execute("SELECT * FROM datasets WHERE user_id = ? ORDER BY last_opened_at DESC", (_require_user(user_id),)).fetchall()
    return [_ds(r) for r in rows]


def get_dataset(user_id: int, dataset_id: str) -> DatasetRecord:
    with connect() as conn:
        row = conn.execute("SELECT * FROM datasets WHERE id = ? AND user_id = ?", (str(dataset_id), _require_user(user_id))).fetchone()
    if row is None:
        raise NotFound(dataset_id)
    return _ds(row)


def read_dataset(user_id: int, dataset_id: str) -> tuple[DatasetRecord, bytes]:
    """The record and file contents, only if this user owns it. Marks it as opened."""
    with connect() as conn:
        row = conn.execute("SELECT * FROM datasets WHERE id = ? AND user_id = ?", (str(dataset_id), _require_user(user_id))).fetchone()
        if row is None:
            raise NotFound(dataset_id)
        conn.execute("UPDATE datasets SET last_opened_at = ? WHERE id = ? AND user_id = ?", (time.time(), row["id"], user_id))
    try:
        data = _file_path(user_id, "datasets", row["stored_name"]).read_bytes()
    except FileNotFoundError:
        raise NotFound(dataset_id) from None
    return _ds(row), data


def find_dataset(user_id: int, fingerprint: str) -> DatasetRecord | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM datasets WHERE user_id = ? AND fingerprint = ?", (_require_user(user_id), fingerprint)).fetchone()
    return _ds(row) if row else None


def delete_dataset(user_id: int, dataset_id: str) -> None:
    with connect() as conn:
        row = conn.execute("DELETE FROM datasets WHERE id = ? AND user_id = ? RETURNING stored_name",
                           (str(dataset_id), _require_user(user_id))).fetchone()
    if row is None:
        raise NotFound(dataset_id)
    _remove(_file_path(user_id, "datasets", row["stored_name"]))


# ---- reports -----------------------------------------------------------------------------------------------------
def _rp(row) -> ReportRecord:
    return ReportRecord(row["id"], row["dataset_id"], row["dataset_name"], row["title"], row["fmt"], row["size_bytes"], row["created_at"])


def save_report(user_id: int, dataset_id: str | None, dataset_name: str, title: str, fmt: str, data: bytes) -> ReportRecord:
    _require_user(user_id)
    if fmt not in ("pdf", "html"):
        raise ValueError(fmt)
    if dataset_id is not None:
        get_dataset(user_id, dataset_id)            # a report can only reference the user's own dataset
    rep_id = secrets.token_hex(16)
    stored = f"{rep_id}.{fmt}"
    path = _file_path(user_id, "reports", stored)
    _write_atomic(path, data)
    try:
        with connect() as conn:
            conn.execute("INSERT INTO reports (id, user_id, dataset_id, dataset_name, title, fmt, size_bytes, stored_name, created_at) "
                         "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                         (rep_id, user_id, dataset_id, dataset_name[:255], title[:200], fmt, len(data), stored, time.time()))
            row = conn.execute("SELECT * FROM reports WHERE id = ? AND user_id = ?", (rep_id, user_id)).fetchone()
    except Exception:
        _remove(path)
        raise
    return _rp(row)


def list_reports(user_id: int, limit: int = 50) -> list[ReportRecord]:
    with connect() as conn:
        rows = conn.execute("SELECT * FROM reports WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
                            (_require_user(user_id), int(limit))).fetchall()
    return [_rp(r) for r in rows]


def read_report(user_id: int, report_id: str) -> tuple[ReportRecord, bytes]:
    with connect() as conn:
        row = conn.execute("SELECT * FROM reports WHERE id = ? AND user_id = ?", (str(report_id), _require_user(user_id))).fetchone()
    if row is None:
        raise NotFound(report_id)
    try:
        data = _file_path(user_id, "reports", row["stored_name"]).read_bytes()
    except FileNotFoundError:
        raise NotFound(report_id) from None
    return _rp(row), data


def delete_report(user_id: int, report_id: str) -> None:
    with connect() as conn:
        row = conn.execute("DELETE FROM reports WHERE id = ? AND user_id = ? RETURNING stored_name",
                           (str(report_id), _require_user(user_id))).fetchone()
    if row is None:
        raise NotFound(report_id)
    _remove(_file_path(user_id, "reports", row["stored_name"]))

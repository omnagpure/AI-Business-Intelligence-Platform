"""Application settings, read from environment variables (optionally a .env file)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEXT_EXTENSIONS: tuple[str, ...] = (".csv", ".tsv", ".txt")
EXCEL_EXTENSIONS: tuple[str, ...] = (".xlsx", ".xlsm", ".xls")
SUPPORTED_EXTENSIONS: tuple[str, ...] = TEXT_EXTENSIONS + EXCEL_EXTENSIONS
APP_NAME = "AI Business Intelligence Platform"
APP_VERSION = "0.1.0"


@dataclass(frozen=True)
class Settings:
    max_upload_mb: int = 200
    preview_rows: int = 100
    log_level: str = "INFO"
    log_dir: Path = PROJECT_ROOT / "logs"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


def _int_env(name: str, default: int, minimum: int = 1) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value >= minimum else default


def load_settings() -> Settings:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    log_dir = Path(os.getenv("BI_LOG_DIR", "logs"))
    if not log_dir.is_absolute():
        log_dir = PROJECT_ROOT / log_dir
    return Settings(
        max_upload_mb=_int_env("BI_MAX_UPLOAD_MB", 200),
        preview_rows=_int_env("BI_PREVIEW_ROWS", 100),
        log_level=os.getenv("BI_LOG_LEVEL", "INFO").upper(),
        log_dir=log_dir,
    )

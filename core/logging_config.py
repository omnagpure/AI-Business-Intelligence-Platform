"""Logging setup. Never log raw data records; log metadata (names, counts) only."""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from config.settings import Settings

_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def setup_logging(settings: Settings) -> None:
    root = logging.getLogger("bi")
    if getattr(root, "_bi_configured", False):
        return
    root.setLevel(getattr(logging, settings.log_level, logging.INFO))
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(_FORMAT))
    root.addHandler(console)
    try:
        settings.log_dir.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(
            settings.log_dir / "app.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8"
        )
        fh.setFormatter(logging.Formatter(_FORMAT))
        root.addHandler(fh)
    except OSError:
        root.warning("File logging unavailable; using console only.")
    root.propagate = False
    root._bi_configured = True  # type: ignore[attr-defined]


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"bi.{name}")

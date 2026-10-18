"""Upload validation that does not depend on Streamlit or file parsing."""
from __future__ import annotations

import re
from pathlib import PurePath

from config.settings import SUPPORTED_EXTENSIONS
from core.exceptions import DataValidationError, FileTooLargeError, UnsupportedFileFormatError

_SAFE = re.compile(r"[^A-Za-z0-9._ -]")


def sanitize_filename(name: str) -> str:
    """Strip any path component and unsafe characters (prevents path traversal)."""
    base = PurePath(name.replace("\\", "/")).name
    base = _SAFE.sub("_", base).strip(". ")
    if not base:
        raise DataValidationError("The file name is empty or invalid.")
    return base[:150]


def validate_extension(filename: str) -> str:
    ext = PurePath(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileFormatError(
            f"Unsupported file type '{ext or 'none'}'. Please upload one of: "
            f"{', '.join(SUPPORTED_EXTENSIONS)}."
        )
    return ext


def validate_size(size_bytes: int, max_bytes: int) -> None:
    if size_bytes <= 0:
        raise DataValidationError("The uploaded file is empty.")
    if size_bytes > max_bytes:
        raise FileTooLargeError(
            f"File is {size_bytes / 1024 / 1024:.1f} MB; the limit is "
            f"{max_bytes / 1024 / 1024:.0f} MB."
        )

"""Domain exceptions. `user_message` is safe to show in the UI; details go to logs."""
from __future__ import annotations


class BIError(Exception):
    """Base class for all expected application errors."""

    def __init__(self, user_message: str, *, detail: str | None = None) -> None:
        super().__init__(user_message)
        self.user_message = user_message
        self.detail = detail


class UnsupportedFileFormatError(BIError):
    """File type is not supported or its content does not match its extension."""


class FileTooLargeError(BIError):
    """File exceeds the configured size limit."""


class DataValidationError(BIError):
    """Dataset is empty, malformed, or otherwise unusable."""


class InsufficientDataError(BIError):
    """Not enough data for the requested analysis."""


class ProfilingError(BIError):
    """Profiling failed unexpectedly."""

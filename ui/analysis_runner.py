"""Shared wrapper for analysis pages: cached, and a failure becomes a message instead of breaking the page."""
from __future__ import annotations

from core.exceptions import BIError, InsufficientDataError
from core.logging_config import get_logger
from ui.analytics_state import cached

log = get_logger("ui.analysis")


def safe_cached(name: str, params, fn, failure_text: str = "The analysis could not be completed for this selection."):
    """Returns (status, value): ('ok', result) | ('insufficient', message) | ('error', message)."""
    def compute():
        try:
            return "ok", fn()
        except InsufficientDataError as exc:
            return "insufficient", exc.user_message
        except BIError as exc:
            return "error", exc.user_message
        except Exception:
            log.exception("%s failed", name)
            return "error", failure_text
    return cached(name, params, compute)

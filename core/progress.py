"""Optional progress reporting for long-running work. No Streamlit: only the UI turns these calls into a bar.

A progress callback is `progress(step, total, message)`:
  step     steps FINISHED so far (0..total); it never goes down, and a step that fails still counts as finished
  total    steps planned for this run
  message  what is happening now, e.g. "Comparing models, 3 of 7"; "<name> - skipped: <reason>" right after a step
           failed (see `split_skip`); "Done" once at the end of a run that uses `StepRunner`

Work that cannot report partial progress (one model fit, one clustering) just shows its step name; nothing here invents
a percentage. A callback that raises is ignored, so a broken progress display can never break an analysis.
"""
from __future__ import annotations

from typing import Any, Callable, Sequence

from core.exceptions import BIError
from core.logging_config import get_logger

log = get_logger("core.progress")

ProgressFn = Callable[[int, int, str], None]
SKIP_MARK = " - skipped: "


def notify(progress: ProgressFn | None, step: int, total: int, message: str) -> None:
    if progress is None:
        return
    try:
        progress(max(0, min(step, total)), max(total, 1), message)
    except Exception:                                       # a failing display must not stop the analysis
        log.warning("Progress callback failed", exc_info=True)


def skip_message(name: str, reason: str) -> str:
    return f"{name}{SKIP_MARK}{reason}"


def split_skip(message: str) -> tuple[str, str] | None:
    """(name, reason) when `message` reports a skipped step, else None."""
    if SKIP_MARK in message:
        name, reason = message.split(SKIP_MARK, 1)
        return name, reason
    return None


def failure_reason(exc: BaseException, default: str = "it could not be completed for this dataset") -> str:
    """User-safe reason: the app's own messages are kept, anything else becomes the generic text (never a stack trace)."""
    if isinstance(exc, BIError):
        return exc.user_message
    return default


class StepRunner:
    """Runs named steps in order, reports each one, and carries on when a step fails.

    Calls, for steps A, B, C where B fails:  (0,3,"A") (1,3,"B") (2,3,"B - skipped: why") (2,3,"C") (3,3,"Done")
    """

    def __init__(self, total: int, progress: ProgressFn | None = None):
        self.total, self.progress, self.finished = total, progress, 0
        self.skipped: list[tuple[str, str]] = []

    def run(self, name: str, fn: Callable[[], Any], default_reason: str | None = None) -> tuple[bool, Any, str]:
        """(ok, value, reason). `fn` may itself return a ("ok" | other, payload) pair; see run_status for that form."""
        notify(self.progress, self.finished, self.total, name)
        try:
            value = fn()
        except Exception as exc:
            reason = failure_reason(exc, default_reason or "it could not be completed for this dataset")
            if not isinstance(exc, BIError):
                log.exception("Step %s failed", name)
            return self._fail(name, reason)
        self.finished += 1
        return True, value, ""

    def run_status(self, name: str, fn: Callable[[], tuple[str, Any]]) -> tuple[bool, Any, str]:
        """For callables that already catch their own errors and return (status, payload), like the UI's cached wrappers."""
        notify(self.progress, self.finished, self.total, name)
        try:
            status, payload = fn()
        except Exception as exc:
            return self._fail(name, failure_reason(exc))
        if status == "ok":
            self.finished += 1
            return True, payload, ""
        return self._fail(name, str(payload))

    def _fail(self, name: str, reason: str) -> tuple[bool, Any, str]:
        self.finished += 1                                   # a failed step still advances the count
        self.skipped.append((name, reason))
        notify(self.progress, self.finished, self.total, skip_message(name, reason))
        return False, None, reason

    def reuse(self, name: str, detail: str) -> None:
        """Step satisfied by a result that already exists (nothing computed). Counts as finished."""
        notify(self.progress, self.finished, self.total, f"{name} ({detail})")
        self.finished += 1

    def skip(self, name: str, reason: str) -> None:
        """Step not attempted (switched off or impossible). Counts as finished and is reported as skipped."""
        self._fail(name, reason)

    def done(self) -> None:
        self.finished = self.total
        notify(self.progress, self.total, self.total, "Done")


def sub_progress(progress: ProgressFn | None, offset: int, total: int, label: str = "") -> ProgressFn | None:
    """Wrap a callback so an inner loop's (i, n, msg) reports into steps offset..offset+1 of a bigger run."""
    if progress is None:
        return None
    def inner(step: int, n: int, message: str) -> None:
        notify(progress, offset, total, f"{label}: {message}" if label else message)
    return inner


def loop_items(progress: ProgressFn | None, items: Sequence[Any], message: str):
    """Iterate `items`, reporting "<message>, k of n" before each one. Yields the items unchanged."""
    n = len(items)
    for i, item in enumerate(items):
        notify(progress, i, n, f"{message}, {i + 1} of {n}")
        yield item

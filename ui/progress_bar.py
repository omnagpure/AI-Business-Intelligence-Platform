"""Turns core progress callbacks (core/progress.py) into a Streamlit bar with elapsed time. The only place progress meets Streamlit."""
from __future__ import annotations

import time

import streamlit as st

from core.progress import split_skip


class ProgressBar:
    """Use as a context manager and pass the instance as `progress=`. The bar is removed when the block ends.

    numbered=True shows "Step k of n: message" (named steps); False shows just the message (one analysis)."""

    def __init__(self, title: str, numbered: bool = False):
        self.title, self.numbered, self.skipped = title, numbered, []
        self._t0 = time.perf_counter()
        self._bar = None

    def __enter__(self) -> "ProgressBar":
        self._t0 = time.perf_counter()
        self._bar = st.progress(0.0, text=f"{self.title}...")
        return self

    def __exit__(self, *exc) -> None:
        if self._bar is not None:
            self._bar.empty()

    def elapsed(self) -> float:
        return time.perf_counter() - self._t0

    def __call__(self, step: int, total: int, message: str) -> None:
        if self._bar is None:
            return
        skip = split_skip(message)
        if skip is not None:
            self.skipped.append(f"{skip[0]}: skipped ({skip[1]})")
            return
        if message == "Done":
            text = f"Finished in {self.elapsed():.0f} s"
        elif self.numbered:
            text = f"Step {min(step + 1, total)} of {total}: {message} · {self.elapsed():.0f} s"
        else:
            text = f"{message} · {self.elapsed():.0f} s"
        self._bar.progress(min(max(step / max(total, 1), 0.0), 1.0), text=text)

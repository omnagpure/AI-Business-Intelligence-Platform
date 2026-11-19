"""How many worker threads heavy analyses may use. No Streamlit. Results never depend on the number chosen."""
from __future__ import annotations

import os

DEFAULT_CAP = 4          # leave the rest of the machine usable while an analysis runs


def worker_count(cap: int = DEFAULT_CAP) -> int:
    """min(cap, CPU count), at least 1. BI_WORKERS=<n> overrides it (1 switches threading off)."""
    raw = os.getenv("BI_WORKERS", "").strip()
    if raw.isdigit() and int(raw) >= 1:
        return int(raw)
    return max(1, min(cap, os.cpu_count() or 1))

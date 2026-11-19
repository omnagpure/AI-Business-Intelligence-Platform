"""Rough run-time guide shown before heavy work. No Streamlit.

Rates are seconds per 100,000 rows, taken from docs/PERFORMANCE.md (single-CPU sandbox, DataCo and supply_chain_dataset1).
They are a guide only: another computer can be faster or slower, so the text always says so and gives a range.
"""
from __future__ import annotations

RATE_PER_100K = {"report": 14.0, "insights": 11.0, "forecast": 0.5, "anomalies": 0.25, "segmentation": 2.0, "drivers": 7.5}
SHOW_FROM_SECONDS = 8.0          # below this the run is quick enough that an estimate is only noise


def estimate_range(kind: str, rows: int) -> tuple[float, float] | None:
    rate = RATE_PER_100K.get(kind)
    if rate is None or rows <= 0:
        return None
    mid = rate * rows / 100_000
    return 0.6 * mid, 1.6 * mid


def describe_estimate(kind: str, rows: int) -> str | None:
    """"Estimated time: about 15 to 40 seconds ..." or None when the run is expected to be quick."""
    rng = estimate_range(kind, rows)
    if rng is None or rng[1] < SHOW_FROM_SECONDS:
        return None
    lo, hi = rng
    fmt = lambda s: f"{s:.0f} seconds" if s < 90 else f"{s / 60:.1f} minutes"          # noqa: E731
    return (f"Estimated time: about {fmt(max(lo, 1))} to {fmt(hi)} for {rows:,} rows. This is a rough guide from timing "
            "on one computer; yours may be faster or slower.")

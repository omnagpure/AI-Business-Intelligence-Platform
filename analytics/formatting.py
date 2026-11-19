"""Display formatting for KPI values. No currency symbol is assumed because the data does not state one."""
from __future__ import annotations

import math


def format_value(value: float | None, unit: str) -> str:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "Not calculable"
    if unit == "percent":
        return f"{value:,.1f}%"
    if unit == "count":
        return f"{value:,.0f}"
    a = abs(value)
    for limit, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if a >= limit:
            return f"{value / limit:,.2f}{suffix}"
    return f"{value:,.0f}" if a >= 1000 else f"{value:,.2f}"

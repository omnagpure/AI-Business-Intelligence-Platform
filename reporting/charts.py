"""Static report charts (Matplotlib, PNG bytes). Every builder returns None when there is nothing honest to draw.

Style: white background, accent-coloured bars (the highlighted one full strength, the rest lighter), light gridlines
only where a value axis is needed, direct value labels, IBM Plex Sans. `half=True` draws a chart sized for a two-up
row. The accent comes from `theme.current()`, which the report builder sets for one report.
"""
from __future__ import annotations

import io
import math
import re

import matplotlib
import matplotlib.dates
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from reporting import theme  # noqa: E402
from reporting.fonts import chart_family  # noqa: E402
from reporting.theme import INK, INK_20, INK_40, INK_60, PAPER  # noqa: E402

DPI = 200
_W = 6.6          # inches; fits the PDF text column and the HTML page
_HALF = 3.2


def _fig(height: float, half: bool = False):
    plt.rcParams["font.family"] = chart_family()
    fig, ax = plt.subplots(figsize=(_HALF if half else _W, height), dpi=DPI)
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(INK_20)
    ax.tick_params(colors=INK_60, labelsize=7.5, length=0)
    return fig, ax


def _png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.12,
                transparent=fig.get_facecolor()[3] == 0)
    plt.close(fig)
    return buf.getvalue()


def _short(text: object, n: int = 28) -> str:
    s = str(text)
    return s if len(s) <= n else s[: n - 1] + "…"


def _num(v: float) -> str:
    a = abs(v)
    for limit, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if a >= limit:
            return f"{v / limit:.1f}{suffix}"
    return f"{v:,.0f}" if a >= 100 else f"{v:,.2f}"


def hbar(labels: list[str], values: list[float], fmt=_num, highlight: int | None = 0, unit: str = "", ref: float | None = None,
         ref_label: str = "", half: bool = False) -> bytes | None:
    """Horizontal bars in the given order, first on top. One bar (the highlight) takes the accent colour; `ref` draws
    a dashed reference line (for example the overall rate)."""
    pairs = [(l, float(v)) for l, v in zip(labels, values) if v is not None and math.isfinite(float(v))]
    if not pairs:
        return None
    p = theme.current()
    labels, values = [x[0] for x in pairs], [x[1] for x in pairs]
    fig, ax = _fig(0.3 * len(values) + 0.6, half)
    y = list(range(len(values)))[::-1]
    colors = [p.accent if i == highlight else p.light for i in range(len(values))]
    ax.barh(y, values, color=colors, height=0.58 if len(values) > 2 else 0.4)
    ax.set_yticks(y)
    ax.set_yticklabels([_short(l, 20 if half else 28) for l in labels], color=INK, fontsize=7.5 if half else 8)
    ax.set_xticks([])
    span = max(abs(v) for v in values + ([ref] if ref is not None else [])) or 1.0
    for yi, v in zip(y, values):
        ax.text(v + span * 0.015, yi, fmt(v) + unit, va="center", ha="left", fontsize=7 if half else 7.5, color=INK)
    if ref is not None and math.isfinite(ref):
        ax.axvline(ref, color=INK_40, linewidth=0.9, linestyle=(0, (3, 2)))
        ax.text(ref, len(values) - 0.35, f" {ref_label or 'Overall'} {fmt(ref)}{unit}", fontsize=6.8, color=INK_60, va="bottom", ha="left")
    ax.set_xlim(min(0, min(values)), max(0, max(values + ([ref] if ref is not None else []))) + span * (0.24 if half else 0.16))
    ax.set_ylim(-0.6, len(values) - 0.1 + (0.5 if ref is not None else 0))
    ax.spines["bottom"].set_visible(False)
    return _png(fig)


def forecast_chart(history: pd.DataFrame, forecast: pd.DataFrame, value_label: str = "") -> bytes | None:
    if history is None or forecast is None or len(history) < 2 or len(forecast) == 0:
        return None
    p = theme.current()
    fig, ax = _fig(2.9)
    ax.plot(history["period"], history["value"], color=INK, linewidth=1.4, label="Actual")
    if {"lower", "upper"} <= set(forecast.columns) and forecast[["lower", "upper"]].notna().all().all():
        ax.fill_between(forecast["period"], forecast["lower"], forecast["upper"], color=p.accent, alpha=0.14, linewidth=0,
                        label="Likely range")
    ax.plot(forecast["period"], forecast["forecast"], color=p.accent, linewidth=1.9, label="Forecast")
    ax.axvline(history["period"].iloc[-1], color=INK_40, linewidth=0.8, linestyle=(0, (3, 3)))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: _num(v)))
    ax.grid(axis="y", color=INK_20, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK_60, loc="upper left", ncol=3)
    if value_label:
        ax.set_ylabel(_short(value_label, 40), color=INK_60, fontsize=7.5)
    return _png(fig)


def line_chart(x, y, label: str = "") -> bytes | None:
    s = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(s) < 2:
        return None
    p = theme.current()
    fig, ax = _fig(2.6)
    ax.plot(s["x"], s["y"], color=INK, linewidth=1.5)
    ax.scatter([s["x"].iloc[-1]], [s["y"].iloc[-1]], color=p.accent, s=22, zorder=3)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: _num(v)))
    ax.grid(axis="y", color=INK_20, linewidth=0.5)
    ax.set_axisbelow(True)
    if label:
        ax.set_ylabel(_short(label, 40), color=INK_60, fontsize=7.5)
    return _png(fig)


def trend(periods: list, values: list[float], fmt=_num, ref: float | None = None, ref_label: str = "", half: bool = True) -> bytes | None:
    """A KPI over time: line with a light area, the latest value labelled, optional dashed reference level."""
    s = pd.DataFrame({"x": pd.to_datetime([str(x) for x in periods], errors="coerce"), "y": values}).dropna()
    if len(s) < 3:
        return None
    p = theme.current()
    fig, ax = _fig(2.0, half)
    lo = min(s["y"].min(), ref if ref is not None else s["y"].min())
    base = lo - (s["y"].max() - lo) * 0.15 if lo > 0 else lo
    ax.fill_between(s["x"], s["y"], base, color=p.accent, alpha=0.08, linewidth=0)
    ax.plot(s["x"], s["y"], color=p.accent, linewidth=1.6)
    ax.scatter([s["x"].iloc[-1]], [s["y"].iloc[-1]], color=p.accent, s=16, zorder=3)
    ax.annotate(fmt(float(s["y"].iloc[-1])), (s["x"].iloc[-1], s["y"].iloc[-1]), textcoords="offset points", xytext=(0, 6),
                ha="center", fontsize=7, color=INK, fontweight="bold")
    if ref is not None and math.isfinite(ref):
        ax.axhline(ref, color=INK_40, linewidth=0.8, linestyle=(0, (3, 2)))
        ax.text(s["x"].iloc[0], ref, f"{ref_label} ", fontsize=6.5, color=INK_60, va="bottom", ha="left")
    ax.set_ylim(bottom=base)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(_tick_formatter(fmt, ax)))
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %y"))
    ax.grid(axis="y", color=INK_20, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=6.5)
    return _png(fig)


def _tick_formatter(fmt, ax):
    """Axis labels with just enough decimals that neighbouring ticks never read the same ('1.55 days', '1.60 days')."""
    sample = fmt(1.0)
    m = re.match(r"^-?[\d.,]+(.*)$", sample)
    suffix = m.group(1) if m else ""
    if suffix.strip() not in ("%", "days", "day", "pts", ""):
        return lambda v, _: _num(v)
    lo, hi = ax.get_ylim()
    ticks = [t for t in ax.get_yticks() if lo <= t <= hi]
    step = min((b - a for a, b in zip(ticks, ticks[1:])), default=1.0) or 1.0
    dec = max(0, min(3, -int(math.floor(math.log10(step)))))
    return lambda v, _: f"{v:,.{dec}f}{suffix}"


def histogram(values, unit: str = "", half: bool = True) -> bytes | None:
    s = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if len(s) < 5:
        return None
    p = theme.current()
    fig, ax = _fig(2.0, half)
    distinct = s.nunique()
    if distinct <= 15:
        counts = s.value_counts().sort_index()
        ax.bar(counts.index.astype(float), counts.values, color=p.light, width=0.7)
        med = float(s.median())
        ax.bar([med], [counts.get(med, 0)], color=p.accent, width=0.7)
    else:
        ax.hist(s, bins=20, color=p.light)
    med = float(s.median())
    ax.axvline(med, color=INK_40, linewidth=0.8, linestyle=(0, (3, 2)))
    ax.text(med, ax.get_ylim()[1] * 0.95, f" median {med:g}{(' ' + unit) if unit else ''}", fontsize=6.5, color=INK_60, va="top")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: _num(v)))
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
    ax.grid(axis="y", color=INK_20, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=6.5)
    return _png(fig)


def cover_graphic(values: list[float], width_in: float = 8.27, height_in: float = 2.4) -> bytes | None:
    """A quiet, data-drawn shape for the cover: the main measure over time as a white line and area on a transparent
    background. No axes or labels; it decorates with the report's own data instead of stock art."""
    v = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    if len(v) < 6:
        return None
    plt.rcParams["font.family"] = chart_family()
    fig = plt.figure(figsize=(width_in, height_in), dpi=150)
    fig.patch.set_alpha(0)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_axis_off()
    ax.patch.set_alpha(0)
    x = list(range(len(v)))
    lo, hi = min(v), max(v)
    base = lo - (hi - lo) * 0.6 if hi > lo else lo - 1
    ax.fill_between(x, v, base, color="white", alpha=0.10, linewidth=0)
    ax.plot(x, v, color="white", alpha=0.55, linewidth=1.6)
    ax.set_xlim(0, len(v) - 1)
    ax.set_ylim(base, hi + (hi - lo) * 0.25 + 1e-9)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True)
    plt.close(fig)
    return buf.getvalue()

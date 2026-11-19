"""Static report charts (Matplotlib, PNG bytes). Every builder returns None when there is nothing honest to draw.

Style: paper background, ink bars, one accent mark per chart, no gridline clutter, direct value labels.
"""
from __future__ import annotations

import io
import math

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from reporting.theme import ACCENT, INK, INK_20, INK_40, INK_60, PAPER  # noqa: E402

DPI = 200
_W = 6.6          # inches; fits the PDF text column and the HTML page


def _fig(height: float):
    fig, ax = plt.subplots(figsize=(_W, height), dpi=DPI)
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(INK_20)
    ax.tick_params(colors=INK_60, labelsize=7.5, length=0)
    return fig, ax


def _png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.12)
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


def hbar(labels: list[str], values: list[float], fmt=_num, highlight: int | None = 0, unit: str = "") -> bytes | None:
    """Horizontal bars, largest on top. One bar (the highlight) takes the accent colour."""
    pairs = [(l, float(v)) for l, v in zip(labels, values) if v is not None and math.isfinite(float(v))]
    if not pairs:
        return None
    labels, values = [p[0] for p in pairs], [p[1] for p in pairs]
    fig, ax = _fig(0.32 * len(values) + 0.55)
    y = list(range(len(values)))[::-1]
    colors = [ACCENT if i == highlight else INK for i in range(len(values))]
    ax.barh(y, values, color=colors, height=0.56 if len(values) > 2 else 0.4)
    ax.set_yticks(y)
    ax.set_yticklabels([_short(l) for l in labels], color=INK, fontsize=8)
    ax.set_xticks([])
    span = max(abs(v) for v in values) or 1.0
    for yi, v in zip(y, values):
        ax.text(v + span * 0.012, yi, fmt(v) + unit, va="center", ha="left", fontsize=7.5, color=INK_60)
    ax.set_xlim(min(0, min(values)), max(0, max(values)) + span * 0.16)
    ax.spines["bottom"].set_visible(False)
    return _png(fig)


def forecast_chart(history: pd.DataFrame, forecast: pd.DataFrame, value_label: str = "") -> bytes | None:
    if history is None or forecast is None or len(history) < 2 or len(forecast) == 0:
        return None
    fig, ax = _fig(3.0)
    ax.plot(history["period"], history["value"], color=INK, linewidth=1.5, label="Actual")
    if {"lower", "upper"} <= set(forecast.columns) and forecast[["lower", "upper"]].notna().all().all():
        ax.fill_between(forecast["period"], forecast["lower"], forecast["upper"], color=INK, alpha=0.12, linewidth=0,
                        label="Prediction interval")
    ax.plot(forecast["period"], forecast["forecast"], color=ACCENT, linewidth=1.8, label="Forecast")
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
    fig, ax = _fig(2.6)
    ax.plot(s["x"], s["y"], color=INK, linewidth=1.5)
    ax.scatter([s["x"].iloc[-1]], [s["y"].iloc[-1]], color=ACCENT, s=22, zorder=3)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: _num(v)))
    ax.grid(axis="y", color=INK_20, linewidth=0.5)
    ax.set_axisbelow(True)
    if label:
        ax.set_ylabel(_short(label, 40), color=INK_60, fontsize=7.5)
    return _png(fig)

"""Report palette, shared by the HTML and PDF outputs and the charts.

A conventional business-report look: white paper, charcoal text, one accent colour (navy by default; the Reports page
offers a few other restrained accents), and muted red / amber / green used only for status (a KPI against its
reference level, the priority of a finding). Greys and tints are mixes of these colours, so the report keeps one
restrained look and prints cleanly. The accent options deliberately avoid red, amber and green so status colours
never clash with the brand colour.
"""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass

PAPER = "#FFFFFF"
INK = "#1B2430"          # text and headings
ACCENT = "#1F4E89"       # navy: section numbers, rules, chart bars, the cover band

GOOD = "#2E7D4F"
WARN = "#B7791F"
BAD = "#B42318"

ACCENTS = {"Navy": "#1F4E89", "Teal": "#0E6466", "Indigo": "#3F3D8F", "Graphite": "#36404D"}


def _mix(a: str, b: str, t: float) -> str:
    """Blend colour a towards colour b by fraction t (0 = a, 1 = b)."""
    ca = [int(a[i:i + 2], 16) for i in (1, 3, 5)]
    cb = [int(b[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02X}" for x, y in zip(ca, cb))


INK_80 = _mix(INK, PAPER, 0.15)       # body text
INK_60 = _mix(INK, PAPER, 0.38)       # secondary text
INK_40 = _mix(INK, PAPER, 0.58)       # captions, axis labels
INK_20 = _mix(INK, PAPER, 0.80)       # hairlines, light bars
PAPER_DEEP = _mix(PAPER, INK, 0.035)  # table banding
PAPER_RULE = _mix(PAPER, INK, 0.13)   # borders
TINT = _mix(PAPER, ACCENT, 0.06)      # panels (summary box, cards)
TINT_2 = _mix(PAPER, ACCENT, 0.16)    # panel borders, light accent bars
ACCENT_LIGHT = _mix(ACCENT, PAPER, 0.55)  # secondary chart bars
GOOD_SOFT, WARN_SOFT, BAD_SOFT = _mix(PAPER, GOOD, 0.10), _mix(PAPER, WARN, 0.12), _mix(PAPER, BAD, 0.09)

STATUS = {"good": GOOD, "warn": WARN, "bad": BAD, "neutral": INK_60}
STATUS_SOFT = {"good": GOOD_SOFT, "warn": WARN_SOFT, "bad": BAD_SOFT, "neutral": PAPER_DEEP}
SEVERITY_STATUS = {"high": "bad", "medium": "warn", "low": "neutral", "info": "neutral"}
SEVERITY_LABEL = {"high": "High priority", "medium": "Medium priority", "low": "Low priority", "info": "For information"}


@dataclass(frozen=True)
class Palette:
    accent: str
    dark: str            # accent darkened: cover block, headline text on tints
    light: str           # secondary chart bars
    tint: str            # panels
    tint2: str           # panel rules


def palette(accent: str = ACCENT) -> Palette:
    a = accent if isinstance(accent, str) and len(accent) == 7 and accent.startswith("#") else ACCENT
    a = a.upper()
    return Palette(a, _mix(a, INK, 0.35), _mix(a, PAPER, 0.55), _mix(PAPER, a, 0.06), _mix(PAPER, a, 0.16))


_CURRENT: ContextVar[Palette] = ContextVar("report_palette", default=palette(ACCENT))


def current() -> Palette:
    """The palette charts are being drawn with (set by the report builder for one report)."""
    return _CURRENT.get()


def use(p: Palette):
    """Set the chart palette; returns a token for `reset`."""
    return _CURRENT.set(p)


def reset(token) -> None:
    _CURRENT.reset(token)

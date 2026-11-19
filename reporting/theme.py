"""Report palette. Three colours in a 70 / 25 / 5 ratio, shared by the HTML and PDF outputs and the charts.

  70 %  PAPER   warm off-white   page background, table bands, whitespace
  25 %  INK     deep navy        headings, cover block, text, chart bars, table headers
   5 %  ACCENT  signal orange    one thing per page: the key number, the selected model, the rule under a heading

Greys are mixes of INK and PAPER (never a fourth hue), so the report keeps a single restrained look.
"""
from __future__ import annotations

PAPER = "#F6F3EC"
INK = "#13263B"
ACCENT = "#E4572E"


def _mix(a: str, b: str, t: float) -> str:
    """Blend colour a towards colour b by fraction t (0 = a, 1 = b)."""
    ca = [int(a[i:i + 2], 16) for i in (1, 3, 5)]
    cb = [int(b[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02X}" for x, y in zip(ca, cb))


INK_80 = _mix(INK, PAPER, 0.20)       # body text
INK_60 = _mix(INK, PAPER, 0.40)       # secondary text
INK_40 = _mix(INK, PAPER, 0.60)       # captions, axis labels
INK_20 = _mix(INK, PAPER, 0.80)       # hairlines, light bars
PAPER_DEEP = _mix(PAPER, INK, 0.06)   # table banding, card fill
PAPER_RULE = _mix(PAPER, INK, 0.16)   # borders

SEVERITY_LABEL = {"high": "HIGH", "medium": "MEDIUM", "low": "LOW", "info": "INFO"}

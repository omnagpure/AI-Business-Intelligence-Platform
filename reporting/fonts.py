"""Report typeface: IBM Plex Sans (SIL Open Font License, shipped in assets/fonts), with DejaVu Sans from Matplotlib
for the few symbols Plex does not have (■ ▲ ▼). Falls back to DejaVu, then Helvetica, if the files are missing, so a
report is always produced.
"""
from __future__ import annotations

import base64
import functools
import os
from pathlib import Path

import matplotlib

FONT_DIR = Path(__file__).resolve().parent.parent / "assets" / "fonts"
PLEX = {"regular": "IBMPlexSans-Regular.ttf", "semibold": "IBMPlexSans-SemiBold.ttf", "bold": "IBMPlexSans-Bold.ttf",
        "light": "IBMPlexSans-Light.ttf"}
_DEJAVU = Path(matplotlib.get_data_path()) / "fonts" / "ttf"


def plex_path(weight: str) -> Path | None:
    p = FONT_DIR / PLEX[weight]
    return p if p.exists() else None


@functools.lru_cache(maxsize=1)
def pdf_fonts() -> dict[str, str]:
    """Register the fonts with ReportLab once. Returns names for: regular, semibold, bold, light, symbol."""
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    names: dict[str, str] = {}
    sym = _DEJAVU / "DejaVuSans.ttf"
    if sym.exists():
        pdfmetrics.registerFont(TTFont("RSym", str(sym)))
        names["symbol"] = "RSym"
    if all(plex_path(w) for w in PLEX):
        for w in PLEX:
            pdfmetrics.registerFont(TTFont(f"Plex-{w}", str(plex_path(w))))
        pdfmetrics.registerFontFamily("Plex-regular", normal="Plex-regular", bold="Plex-bold", italic="Plex-regular", boldItalic="Plex-bold")
        names.update(regular="Plex-regular", semibold="Plex-semibold", bold="Plex-bold", light="Plex-light")
    elif sym.exists() and (_DEJAVU / "DejaVuSans-Bold.ttf").exists():
        pdfmetrics.registerFont(TTFont("RSans-B", str(_DEJAVU / "DejaVuSans-Bold.ttf")))
        pdfmetrics.registerFontFamily("RSym", normal="RSym", bold="RSans-B", italic="RSym", boldItalic="RSans-B")
        names.update(regular="RSym", semibold="RSans-B", bold="RSans-B", light="RSym")
    else:
        names.update(regular="Helvetica", semibold="Helvetica-Bold", bold="Helvetica-Bold", light="Helvetica", symbol="Helvetica")
    names.setdefault("symbol", names["regular"])
    return names


@functools.lru_cache(maxsize=1)
def chart_family() -> str:
    """Make Plex available to Matplotlib and return the family name to use (DejaVu Sans when Plex is missing)."""
    from matplotlib import font_manager
    try:
        for w in ("regular", "semibold", "bold"):
            p = plex_path(w)
            if p is None:
                return "DejaVu Sans"
            font_manager.fontManager.addfont(str(p))
        return "IBM Plex Sans"
    except Exception:
        return "DejaVu Sans"


@functools.lru_cache(maxsize=1)
def html_font_faces() -> str:
    """@font-face rules with the regular and bold files embedded, so the HTML report needs no network."""
    rules = []
    for w, css_w in (("regular", 400), ("semibold", 600), ("bold", 700)):
        p = plex_path(w)
        if p is None:
            return ""
        data = base64.b64encode(p.read_bytes()).decode()
        rules.append(f"@font-face{{font-family:'Plex';font-weight:{css_w};font-style:normal;"
                     f"src:url(data:font/ttf;base64,{data}) format('truetype')}}")
    return "".join(rules)


def available() -> bool:
    return all(plex_path(w) for w in PLEX) and os.path.isdir(FONT_DIR)

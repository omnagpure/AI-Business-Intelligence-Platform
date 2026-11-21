"""PDF report (ReportLab), laid out like a business report.

Cover (title from the data, key message, headline figures over a data-drawn graphic, report details) · Report at a
glance (contents with a one-line description of each section, how to read the report, about the data) · Executive
summary · numbered sections, each with a short label, a headline that states its point, big-number callouts, charts
(two-up where they fit) with a sentence on what each shows · appendix divider · lettered appendix · methodology.

White pages, charcoal text, one accent colour; red / amber / green only for status. IBM Plex Sans throughout (shipped
in assets/fonts), DejaVu for the few symbols Plex lacks. Running header, footer with "Page n of N".
All user-derived text goes through `_x` (XML-escaped) before it reaches a Paragraph.
"""
from __future__ import annotations

import hashlib
import io
import re
from xml.sax.saxutils import escape

from reportlab.lib.colors import Color, HexColor
from reportlab.lib.enums import TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.platypus import (BaseDocTemplate, CondPageBreak, Frame, Image, KeepTogether, NextPageTemplate, PageBreak, PageTemplate, Paragraph,
                                Spacer, Table as RLTable, TableStyle)
from reportlab.platypus.tableofcontents import TableOfContents

from analytics.recommendations import KINDS
from reporting.fonts import pdf_fonts
from reporting.report_builder import Card, Chart, ReportData, Section, Table
from reporting.theme import (INK, INK_60, INK_80, PAPER, PAPER_DEEP, PAPER_RULE, SEVERITY_LABEL, SEVERITY_STATUS, STATUS, STATUS_SOFT,
                             palette)

C_PAPER, C_INK = HexColor(PAPER), HexColor(INK)
C_80, C_60 = HexColor(INK_80), HexColor(INK_60)
C_DEEP, C_RULE = HexColor(PAPER_DEEP), HexColor(PAPER_RULE)
WHITE = Color(1, 1, 1)
STAR = "#C2410C"                                   # the Vesper mark's star

PAGE_W, PAGE_H = A4
MX, MT, MB = 20 * mm, 24 * mm, 20 * mm
CW = PAGE_W - 2 * MX
GAP = 6 * mm
_BAD_XML = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_SYMS = "■▲▼●"
KIND_STATUS = {"risk": "bad", "opportunity": "good", "pattern": "neutral", "data": "neutral"}


def _x(text: object) -> str:
    s = escape(_BAD_XML.sub("", "" if text is None else str(text)))
    sym = pdf_fonts()["symbol"]
    for ch in _SYMS:
        if ch in s:
            s = s.replace(ch, f'<font name="{sym}">{ch}</font>')
    return s


def _tone(t: str) -> str:
    return STATUS.get(t, STATUS["neutral"])


# ---- styles -------------------------------------------------------------------------------------
def _styles(d: ReportData):
    F = pdf_fonts()
    P = palette(d.accent)
    reg, semi, bold = F["regular"], F["semibold"], F["bold"]
    s = {"F": F, "P": P, "acc": HexColor(P.accent), "dark": HexColor(P.dark), "tint": HexColor(P.tint), "tint2": HexColor(P.tint2)}
    s["body"] = ParagraphStyle("body", fontName=reg, fontSize=9.2, leading=14, textColor=C_80)
    s["lead"] = ParagraphStyle("lead", parent=s["body"], fontSize=10.4, leading=16, textColor=C_INK, spaceAfter=8)
    s["para"] = ParagraphStyle("para", parent=s["body"], fontSize=9.8, leading=15.2, textColor=C_80, spaceAfter=8)
    s["exec"] = ParagraphStyle("exec", parent=s["body"], fontSize=10.6, leading=16.6, textColor=C_INK, spaceAfter=8)
    s["small"] = ParagraphStyle("small", parent=s["body"], fontSize=7.9, leading=11.2, textColor=C_60)
    s["eyebrow"] = ParagraphStyle("eyebrow", fontName=bold, fontSize=8.2, leading=11, textColor=s["acc"], spaceAfter=4)
    s["h1"] = ParagraphStyle("h1", fontName=bold, fontSize=18, leading=23, textColor=C_INK)
    s["h1s"] = ParagraphStyle("h1s", parent=s["h1"], fontSize=20, leading=25)
    s["h3"] = ParagraphStyle("h3", fontName=semi, fontSize=10.4, leading=13.5, textColor=C_INK, spaceBefore=12, spaceAfter=6, keepWithNext=1)
    s["h3s"] = ParagraphStyle("h3s", parent=s["h3"], keepWithNext=0)
    s["th"] = ParagraphStyle("th", fontName=semi, fontSize=7.8, leading=9.8, textColor=C_INK)
    s["thr"] = ParagraphStyle("thr", parent=s["th"], alignment=TA_RIGHT)
    s["td"] = ParagraphStyle("td", fontName=reg, fontSize=8.2, leading=11, textColor=C_80)
    s["tdr"] = ParagraphStyle("tdr", parent=s["td"], alignment=TA_RIGHT)
    s["tds"] = ParagraphStyle("tds", parent=s["td"], fontSize=7.5, leading=10, textColor=C_60)
    s["fl"] = ParagraphStyle("fl", fontName=reg, fontSize=8.2, leading=11.2, textColor=C_60)
    s["fv"] = ParagraphStyle("fv", fontName=semi, fontSize=8.8, leading=11.8, textColor=C_INK)
    s["bul"] = ParagraphStyle("bul", parent=s["body"], fontSize=9.6, leading=14.2, textColor=C_INK, leftIndent=12, firstLineIndent=-12, spaceAfter=5)
    s["step"] = ParagraphStyle("step", parent=s["body"], fontSize=8.8, leading=12.6, leftIndent=13, bulletIndent=1, spaceAfter=2)
    s["bigstep"] = ParagraphStyle("bigstep", parent=s["body"], fontSize=9.8, leading=14.6, textColor=C_INK, leftIndent=16, bulletIndent=0,
                                  spaceAfter=6, bulletFontName=bold, bulletColor=s["acc"])
    s["note"] = ParagraphStyle("note", parent=s["small"], leftIndent=10, bulletIndent=0, spaceAfter=2.5)
    s["cardt"] = ParagraphStyle("cardt", fontName=semi, fontSize=11, leading=14.5, textColor=C_INK, spaceAfter=3)
    s["tag"] = ParagraphStyle("tag", fontName=bold, fontSize=7.2, leading=9.5, textColor=C_60, spaceAfter=4)
    s["lab"] = ParagraphStyle("lab", fontName=semi, fontSize=8, leading=10.5, textColor=C_INK, spaceBefore=6, spaceAfter=1)
    s["cht"] = ParagraphStyle("cht", fontName=semi, fontSize=9.4, leading=12, textColor=C_INK, spaceAfter=4)
    s["chn"] = ParagraphStyle("chn", fontName=reg, fontSize=8.2, leading=11.6, textColor=C_60, spaceBefore=3)
    s["cap"] = ParagraphStyle("cap", parent=s["small"], spaceBefore=3, spaceAfter=10)
    s["cv"] = ParagraphStyle("cv", fontName=bold, fontSize=20, leading=24, textColor=C_INK)
    s["cl"] = ParagraphStyle("cl", fontName=reg, fontSize=8, leading=10.5, textColor=C_60)
    s["tilel"] = ParagraphStyle("tilel", fontName=reg, fontSize=7.8, leading=10, textColor=C_60)
    s["tilev"] = ParagraphStyle("tilev", fontName=bold, fontSize=18, leading=22, textColor=C_INK, spaceBefore=3, spaceAfter=2)
    s["tilen"] = ParagraphStyle("tilen", fontName=semi, fontSize=7.4, leading=9.8, textColor=C_60)
    s["tocg"] = ParagraphStyle("tocg", fontName=bold, fontSize=7.8, leading=10, textColor=C_60)
    s["toct"] = ParagraphStyle("toct", fontName=semi, fontSize=9.6, leading=12.2, textColor=C_INK)
    s["tocd"] = ParagraphStyle("tocd", fontName=reg, fontSize=7.8, leading=10.2, textColor=C_60)
    s["tocn"] = ParagraphStyle("tocn", fontName=bold, fontSize=9.6, leading=12.2, textColor=s["acc"])
    s["tocp"] = ParagraphStyle("tocp", parent=s["toct"], alignment=TA_RIGHT)
    s["boxh"] = ParagraphStyle("boxh", fontName=semi, fontSize=9.6, leading=12.5, textColor=C_INK, spaceAfter=5)
    s["boxt"] = ParagraphStyle("boxt", fontName=reg, fontSize=7.8, leading=10.6, textColor=C_80, spaceAfter=2)
    return s


# ---- the Vesper mark ----------------------------------------------------------------------------
def _quad(p, x0, y0, qx, qy, x1, y1):
    p.curveTo(x0 + 2 / 3 * (qx - x0), y0 + 2 / 3 * (qy - y0), x1 + 2 / 3 * (qx - x1), y1 + 2 / 3 * (qy - y1), x1, y1)


def _mark(c, x: float, y: float, size: float, stroke=C_INK, star=STAR) -> None:
    """The app's logo mark (assets/mark-*.svg) drawn natively: a check stroke and a four-point star. (x, y) is the
    bottom-left corner of a size x size box."""
    k = size / 24.0
    X = lambda v: x + v * k                                                # noqa: E731
    Y = lambda v: y + (24 - v) * k                                         # noqa: E731
    c.saveState()
    c.setStrokeColor(stroke)
    c.setLineWidth(3.4 * k)
    c.setLineCap(1)
    c.setLineJoin(1)
    p = c.beginPath()
    p.moveTo(X(3.2), Y(4.6))
    p.lineTo(X(10), Y(19.6))
    p.lineTo(X(15.4), Y(11.4))
    c.drawPath(p, stroke=1, fill=0)
    c.setFillColor(HexColor(star) if isinstance(star, str) else star)
    p = c.beginPath()
    pts = [(20, 0.6), (20.55, 3.95), (23.9, 4.5), (20.55, 5.05), (20, 8.4), (19.45, 5.05), (16.1, 4.5), (19.45, 3.95), (20, 0.6)]
    p.moveTo(X(pts[0][0]), Y(pts[0][1]))
    for i in range(1, len(pts), 2):
        _quad(p, X(pts[i - 1][0]), Y(pts[i - 1][1]), X(pts[i][0]), Y(pts[i][1]), X(pts[i + 1][0]), Y(pts[i + 1][1]))
    p.close()
    c.drawPath(p, stroke=0, fill=1)
    c.restoreState()


def _wrap(text: str, font: str, size: float, width: float) -> list[str]:
    words, lines, cur = str(text).split(), [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if pdfmetrics.stringWidth(trial, font, size) <= width or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def _mixed(c, x: float, y: float, text: str, font: str, size: float) -> None:
    """drawString that switches to the symbol font for ■ ▲ ▼."""
    sym = pdf_fonts()["symbol"]
    parts, cur, cur_sym = [], "", None
    for ch in text:
        is_sym = ch in _SYMS
        if cur and is_sym != cur_sym:
            parts.append((cur, cur_sym))
            cur = ""
        cur, cur_sym = cur + ch, is_sym
    if cur:
        parts.append((cur, cur_sym))
    for t, s in parts:
        f, sz = (sym, size * 0.8) if s else (font, size)
        c.setFont(f, sz)
        c.drawString(x, y, t)
        x += pdfmetrics.stringWidth(t, f, sz)


def _logo(c, d: ReportData, x: float, y: float, h: float, max_w: float, on_dark: bool = False) -> float:
    """Draw the user's logo, or the Vesper mark and name. Returns the width used."""
    F = pdf_fonts()
    if d.logo:
        try:
            img = ImageReader(io.BytesIO(d.logo))
            iw, ih = img.getSize()
            w = min(max_w, h * 1.6 * iw / ih)
            hh = w * ih / iw
            c.drawImage(img, x, y + (h - hh) / 2, width=w, height=hh, mask="auto")
            return w
        except Exception:
            pass
    _mark(c, x, y, h, stroke=WHITE if on_dark else C_INK)
    c.setFillColor(WHITE if on_dark else C_INK)
    c.setFont(F["bold"], h * 0.62)
    name = d.app_name.upper()
    c.drawString(x + h * 1.15, y + h * 0.22, name)
    return h * 1.15 + pdfmetrics.stringWidth(name, F["bold"], h * 0.62)


# ---- page canvas with "Page n of N" -------------------------------------------------------------
def _canvas_maker():
    F = pdf_fonts()

    class NumberedCanvas(rl_canvas.Canvas):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self._pages = []

        def showPage(self):
            self._pages.append(dict(self.__dict__))
            self._startPage()

        def save(self):
            total = len(self._pages)
            for state in self._pages:
                self.__dict__.update(state)
                n = self._pageNumber
                if n > 1:
                    self.setFillColor(C_INK)
                    self.setFont(F["semibold"], 7.6)
                    self.drawRightString(PAGE_W - MX, 8.5 * mm, f"Page {n} of {total}")
                super().showPage()
            super().save()
    return NumberedCanvas


# ---- document -----------------------------------------------------------------------------------
class _Doc(BaseDocTemplate):
    def __init__(self, buf, data: ReportData, st):
        super().__init__(buf, pagesize=A4, leftMargin=MX, rightMargin=MX, topMargin=MT, bottomMargin=MB, title=data.title,
                         author=data.organisation or data.prepared_by or data.app_name, subject=f"Analysis of {data.dataset_name}",
                         creator=data.app_name)
        self.data, self.st = data, st
        frame = lambda i: Frame(MX, MB, CW, PAGE_H - MB - MT, id=i, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)  # noqa: E731
        div = Frame(MX + 34 * mm, MB, CW - 34 * mm, PAGE_H - MB - MT - 76 * mm, id="d", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        self.addPageTemplates([PageTemplate("cover", [frame("c")], onPage=self._cover),
                               PageTemplate("body", [frame("b")], onPage=self._page),
                               PageTemplate("divider", [div], onPage=self._divider)])

    # ---- cover
    def _cover(self, c, doc):
        st, d = self.st, self.data
        F = st["F"]
        acc = st["acc"]
        c.saveState()
        c.setFillColor(C_PAPER)
        c.rect(0, 0, PAGE_W, PAGE_H, stroke=0, fill=1)
        _logo(c, d, MX, PAGE_H - 26 * mm, 8.5 * mm, 60 * mm)
        cls = (d.classification or "Internal").upper()
        c.setFont(F["bold"], 7.2)
        w = pdfmetrics.stringWidth(cls, F["bold"], 7.2) + 7 * mm
        c.setStrokeColor(acc)
        c.setLineWidth(0.8)
        c.roundRect(PAGE_W - MX - w, PAGE_H - 24.6 * mm, w, 6 * mm, 3 * mm, stroke=1, fill=0)
        c.setFillColor(acc)
        c.drawCentredString(PAGE_W - MX - w / 2, PAGE_H - 22.5 * mm, cls)
        y = PAGE_H - 72 * mm
        if d.eyebrow:
            c.setFillColor(acc)
            c.setFont(F["bold"], 9)
            c.drawString(MX, y, d.eyebrow.upper())
            y -= 15 * mm
        size = 34 if len(d.title) <= 32 else 28
        c.setFillColor(C_INK)
        c.setFont(F["bold"], size)
        for ln in _wrap(d.title, F["bold"], size, CW * 0.92)[:3]:
            c.drawString(MX, y, ln)
            y -= size * 1.12
        c.setFillColor(acc)
        c.rect(MX, y + size * 0.42, 22 * mm, 1.4 * mm, stroke=0, fill=1)
        y -= 7 * mm
        c.setFillColor(C_60)
        c.setFont(F["regular"], 11)
        for ln in _wrap(d.subtitle, F["regular"], 11, CW)[:2]:
            c.drawString(MX, y, ln)
            y -= 5.6 * mm
        block_top = 112 * mm
        if d.key_message:
            lines = _wrap(d.key_message, F["semibold"], 13.5, CW - 10 * mm)[:3]
            lh = 6.4 * mm
            top = min(y - 16 * mm, block_top + 24 * mm + len(lines) * lh + 30 * mm)
            top = max(top, block_top + 22 * mm + len(lines) * lh)
            c.setFillColor(acc)
            c.rect(MX, top - len(lines) * lh - 2 * mm, 1.3 * mm, len(lines) * lh + 10 * mm, stroke=0, fill=1)
            c.setFont(F["bold"], 7.6)
            c.drawString(MX + 6 * mm, top + 3.5 * mm, "KEY MESSAGE")
            c.setFillColor(C_INK)
            c.setFont(F["semibold"], 13.5)
            for i, ln in enumerate(lines):
                c.drawString(MX + 6 * mm, top - 3.2 * mm - i * lh, ln)
        if d.is_sample:
            c.setFillColor(HexColor(STATUS_SOFT["warn"]))
            c.rect(0, block_top, PAGE_W, 8 * mm, stroke=0, fill=1)
            c.setFillColor(HexColor(STATUS["warn"]))
            c.setFont(F["semibold"], 8.2)
            c.drawString(MX, block_top + 2.8 * mm, "Synthetic sample data. The figures do not describe a real business.")
        c.setFillColor(acc)
        c.rect(0, 0, PAGE_W, block_top, stroke=0, fill=1)
        if d.cover_png:
            try:
                c.drawImage(ImageReader(io.BytesIO(d.cover_png)), 0, 52 * mm, width=PAGE_W, height=60 * mm, mask="auto")
            except Exception:
                pass
        tiles = d.tiles[:3]
        if tiles:
            colw = CW / len(tiles)
            for k, t in enumerate(tiles):
                x = MX + k * colw
                c.setFillColor(WHITE)
                c.setFont(F["bold"], 24)
                c.drawString(x, 82 * mm, t.value[:14])
                c.setFillAlpha(0.85)
                c.setFont(F["regular"], 8.6)
                c.drawString(x, 76 * mm, t.label[:34])
                if t.note:
                    c.setFillAlpha(0.65)
                    _mixed(c, x, 71.5 * mm, t.note[:40], F["regular"], 7.4)
                c.setFillAlpha(1)
        c.setStrokeColor(WHITE)
        c.setStrokeAlpha(0.3)
        c.setLineWidth(0.6)
        c.line(MX, 52 * mm, PAGE_W - MX, 52 * mm)
        c.setStrokeAlpha(1)
        items = ([("Prepared for", d.prepared_by)] if d.prepared_by else []) + ([("Prepared by", d.organisation)] if d.organisation else []) + \
                ([("Data covers", d.period)] if d.period else []) + [("Date", d.generated_at.split(",")[0]), ("Report ID", d.report_id)]
        colw = CW / len(items)
        for k, (label, val) in enumerate(items):
            x = MX + k * colw
            c.setFillColor(WHITE)
            c.setFillAlpha(0.65)
            c.setFont(F["regular"], 7.4)
            c.drawString(x, 43 * mm, label)
            c.setFillAlpha(1)
            c.setFont(F["semibold"], 9.4)
            for j, ln in enumerate(_wrap(val, F["semibold"], 9.4, colw - 4 * mm)[:2]):
                c.drawString(x, 37.5 * mm - j * 4.4 * mm, ln)
        c.setFillAlpha(0.6)
        c.setFont(F["regular"], 7.2)
        c.drawString(MX, 14 * mm, f"Prepared with {d.app_name}. Every figure is computed from the uploaded data; no value is typed in by hand.")
        c.setFillAlpha(1)
        if d.logo:
            _mark(c, PAGE_W - MX - 6 * mm, 12.6 * mm, 6 * mm, stroke=WHITE)
        c.restoreState()

    # ---- running header and footer
    def _page(self, c, doc):
        st, d = self.st, self.data
        F = st["F"]
        c.saveState()
        c.setFillColor(st["acc"])
        c.rect(0, PAGE_H - 3 * mm, PAGE_W, 3 * mm, stroke=0, fill=1)
        x = MX
        if d.organisation:
            name = d.organisation[:40].upper()
            c.setFont(F["bold"], 7.6)
            c.setFillColor(st["acc"])
            c.drawString(x, PAGE_H - 13 * mm, name)
            x += pdfmetrics.stringWidth(name, F["bold"], 7.6) + 3 * mm
        else:
            _mark(c, x, PAGE_H - 14.1 * mm, 4.2 * mm)
            c.setFont(F["bold"], 7.6)
            c.setFillColor(C_INK)
            c.drawString(x + 5.4 * mm, PAGE_H - 13 * mm, d.app_name.upper())
            x += 5.4 * mm + pdfmetrics.stringWidth(d.app_name.upper(), F["bold"], 7.6) + 3 * mm
        c.setFont(F["regular"], 7.6)
        c.setFillColor(C_60)
        c.drawString(x, PAGE_H - 13 * mm, f"|   {d.title[:70]}")
        c.drawRightString(PAGE_W - MX, PAGE_H - 13 * mm, d.classification or "Internal")
        c.setStrokeColor(C_RULE)
        c.setLineWidth(0.5)
        c.line(MX, PAGE_H - 16 * mm, PAGE_W - MX, PAGE_H - 16 * mm)
        c.line(MX, 13 * mm, PAGE_W - MX, 13 * mm)
        c.setFont(F["regular"], 7)
        c.drawString(MX, 8.5 * mm, f"Report {d.report_id}  ·  {d.generated_at}  ·  {d.dataset_name[:50]}")
        c.restoreState()

    def _divider(self, c, doc):
        st = self.st
        F = st["F"]
        c.saveState()
        c.setFillColor(st["acc"])
        c.rect(0, 0, 22 * mm, PAGE_H, stroke=0, fill=1)
        c.setFillColor(st["tint"])
        c.rect(22 * mm, 0, PAGE_W - 22 * mm, PAGE_H, stroke=0, fill=1)
        c.setFillColor(C_INK)
        c.setFont(F["bold"], 34)
        c.drawString(MX + 34 * mm, PAGE_H - 70 * mm, "Appendix")
        c.setFillColor(st["acc"])
        c.rect(MX + 34 * mm, PAGE_H - 78 * mm, 22 * mm, 1.4 * mm, stroke=0, fill=1)
        c.setFillColor(C_60)
        c.setFont(F["regular"], 10.5)
        c.drawString(MX + 34 * mm, PAGE_H - 88 * mm, "Supporting detail: the data, its quality and how each figure is calculated.")
        c.restoreState()

    def afterFlowable(self, f):
        toc = getattr(f, "_toc", None)
        if toc:
            key, text = toc
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(text, key, 0, 0)
            self.notify("TOCEntry", (0, text, self.page, key))


def _key(text: str) -> str:
    return "sec_" + hashlib.md5(text.encode("utf-8", "ignore")).hexdigest()[:10]


class _Glance(TableOfContents):
    """Contents grouped into Summary, Analysis and Appendix, each entry with a one-line description and its page."""

    def __init__(self, rows, st):
        super().__init__()
        self.rows, self.st = rows, st          # rows: (group, label, title, desc, key)

    def wrap(self, availWidth, availHeight):
        st = self.st
        pages = {e[3]: e[2] for e in (self._lastEntries or [])}
        data, style = [], [("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0)]
        last = None
        for group, label, title, desc, key in self.rows:
            if group != last:
                r = len(data)
                data.append(["", Paragraph(_x(group.upper()), st["tocg"]), ""])
                style += [("TOPPADDING", (0, r), (-1, r), 8 if r else 0), ("BOTTOMPADDING", (0, r), (-1, r), 3),
                          ("LINEBELOW", (0, r), (-1, r), 0.8, st["acc"])]
                last = group
            r = len(data)
            link = f'<link destination="{key}">{_x(title)}</link>'
            data.append([Paragraph(_x(label), st["tocn"]), [Paragraph(link, st["toct"]), Paragraph(_x(desc), st["tocd"])] if desc else
                         Paragraph(link, st["toct"]), Paragraph(str(pages.get(key, "")), st["tocp"])])
            style += [("TOPPADDING", (0, r), (-1, r), 3.2), ("BOTTOMPADDING", (0, r), (-1, r), 3.6), ("LINEBELOW", (0, r), (-1, r), 0.4, C_RULE)]
        self._table = RLTable(data, colWidths=(10 * mm, availWidth - 24 * mm, 14 * mm), style=TableStyle(style))
        self.width, self.height = self._table.wrapOn(self.canv, availWidth, availHeight)
        return self.width, self.height


# ---- flowable builders --------------------------------------------------------------------------
def _box(content, bg, pad=(10, 9), width=CW, bar=None) -> RLTable:
    tb = RLTable([[content]], colWidths=[width])
    stl = [("BACKGROUND", (0, 0), (-1, -1), bg), ("LEFTPADDING", (0, 0), (-1, -1), pad[0]), ("RIGHTPADDING", (0, 0), (-1, -1), pad[0]),
           ("TOPPADDING", (0, 0), (-1, -1), pad[1]), ("BOTTOMPADDING", (0, 0), (-1, -1), pad[1]), ("VALIGN", (0, 0), (-1, -1), "TOP")]
    if bar is not None:
        stl.append(("LINEBEFORE", (0, 0), (0, -1), 2.4, bar))
    tb.setStyle(TableStyle(stl))
    return tb


def _heading(eyebrow: str, big: str, toc: tuple[str, str] | None, st, need: float = 80 * mm) -> list:
    eb = Paragraph(_x(eyebrow.upper()), st["eyebrow"])
    if toc:
        eb._toc = toc
    p = Paragraph(_x(big), st["h1"])
    rule = RLTable([[""]], colWidths=[CW], rowHeights=[1])
    rule.setStyle(TableStyle([("LINEABOVE", (0, 0), (-1, 0), 0.8, st["tint2"])]))
    return [CondPageBreak(need), eb, p, Spacer(1, 7), rule, Spacer(1, 9)]


def _weights(header, rows, width=CW) -> list[float]:
    """Column widths: proportional to content, but never narrower than the longest word of the header."""
    raw = []
    for i in range(len(header)):
        longest = max([len(str(header[i]))] + [min(len(str(r[i])), 60) for r in rows[:30]])
        raw.append(max(5, min(longest, 46)) ** 0.9)
    mins = []
    for i, h in enumerate(header):
        head_w = max(len(w) for w in str(h).split() or [""]) * 6.4 + 12
        cell_w = max([min(len(w), 24) for r in rows[:30] for w in str(r[i]).split()] + [0]) * 4.9 + 12
        mins.append(max(head_w, cell_w))
    widths, fixed = [0.0] * len(raw), set()
    for _ in range(len(raw)):
        free = [i for i in range(len(raw)) if i not in fixed]
        room = width - sum(widths[i] for i in fixed)
        tot = sum(raw[i] for i in free)
        grew = False
        for i in free:
            w = room * raw[i] / tot
            if w < mins[i]:
                widths[i], grew = mins[i], True
                fixed.add(i)
        if not grew:
            for i in free:
                widths[i] = room * raw[i] / tot
            break
    scale = width / sum(widths)
    return [w * scale for w in widths]


def _cell(text, t: Table, i: int, tone: str, sub: str, st):
    style = st["tdr"] if t.is_right(i) else st["td"]
    if tone and tone != "neutral" and i == t.tone_col:
        mark = "" if str(text)[:1] in "▲▼" else f'{_x("■")}&nbsp;'
        return Paragraph(f'<font color="{_tone(tone)}">{mark}{_x(text)}</font>', style)
    if i == 1 and sub:
        return [Paragraph(f"<font name=\"{st['F']['semibold']}\" color=\"{INK}\">{_x(text)}</font>", style), Paragraph(_x(sub), st["tds"])]
    return Paragraph(_x(text), style)


def _table(t: Table, st) -> list:
    hdr = [Paragraph(_x(h), st["thr"] if t.is_right(i) else st["th"]) for i, h in enumerate(t.header)]
    tones = t.tones or ("",) * len(t.rows)
    subs = t.sub or ("",) * len(t.rows)
    body = [[_cell(c, t, i, tones[r] if r < len(tones) else "", subs[r] if r < len(subs) else "", st) for i, c in enumerate(row)]
            for r, row in enumerate(t.rows)]
    sized = [tuple(c + (" " + subs[k] if i == 1 and k < len(subs) and subs[k] else "") for i, c in enumerate(r)) for k, r in enumerate(t.rows)]
    tb = RLTable([hdr] + body, colWidths=_weights(t.header, sized), repeatRows=1)
    style = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("BACKGROUND", (0, 0), (-1, 0), st["tint"]), ("LINEBELOW", (0, 0), (-1, 0), 0.9, st["acc"]),
             ("LINEBELOW", (0, 1), (-1, -1), 0.4, C_RULE), ("TOPPADDING", (0, 0), (-1, -1), 4.6), ("BOTTOMPADDING", (0, 0), (-1, -1), 4.6),
             ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5)]
    tb.setStyle(TableStyle(style))
    title = Paragraph(_x(t.title), st["h3"] if len(t.rows) <= 6 else st["h3s"])
    out = ([CondPageBreak(45 * mm)] if len(t.rows) > 6 else []) + [title, tb]
    out.append(Paragraph(_x(t.note), st["cap"]) if t.note else Spacer(1, 10))
    return out


def _facts(facts, st) -> list:
    cells = [[Paragraph(_x(k), st["fl"]), Paragraph(_x(v), st["fv"])] for k, v in facts]
    if len(cells) > 3:
        half = (len(cells) + 1) // 2
        left, right = cells[:half], cells[half:] + [[Paragraph("", st["fl"]), Paragraph("", st["fv"])]] * (half - len(cells[half:]))
        rows = [a + b for a, b in zip(left, right)]
        widths = [CW * 0.17, CW * 0.33, CW * 0.17, CW * 0.33]
    else:
        rows, widths = cells, [CW * 0.25, CW * 0.75]
    tb = RLTable(rows, colWidths=widths)
    tb.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                            ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                            ("LINEBELOW", (0, 0), (-1, -1), 0.4, C_RULE)]))
    return [tb, Spacer(1, 10)]


def _callouts(items, st) -> list:
    if not items:
        return []
    gap = 5
    w = (CW - gap * (len(items) - 1)) / len(items)
    row, widths, ts = [], [], [("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]
    for k, (val, label, tone) in enumerate(items):
        if k:
            row.append("")
            widths.append(gap)
        col = _tone(tone) if tone != "neutral" else INK
        row.append([Paragraph(f'<font color="{col}">{_x(val)}</font>', st["cv"]), Paragraph(_x(label), st["cl"])])
        widths.append(w)
        c = 2 * k
        ts += [("LINEBEFORE", (c, 0), (c, 0), 2.2, HexColor(_tone(tone)) if tone != "neutral" else st["acc"]),
               ("LEFTPADDING", (c, 0), (c, 0), 9), ("RIGHTPADDING", (c, 0), (c, 0), 4)]
    for k in range(1, len(items)):
        ts += [("LEFTPADDING", (2 * k - 1, 0), (2 * k - 1, 0), 0), ("RIGHTPADDING", (2 * k - 1, 0), (2 * k - 1, 0), 0)]
    tb = RLTable([row], colWidths=widths)
    tb.setStyle(TableStyle(ts))
    return [Spacer(1, 2), tb, Spacer(1, 12)]


def _img(png: bytes, width: float, max_h: float) -> Image:
    w, h = ImageReader(io.BytesIO(png)).getSize()
    height = width * h / w
    if height > max_h:
        height = max_h
        width = height * w / h
    return Image(io.BytesIO(png), width=width, height=height, hAlign="LEFT")


def _chart_block(ch: Chart, width: float, st) -> list:
    parts = [Paragraph(_x(ch.title), st["cht"]), _img(ch.png, width, 64 * mm if ch.half else 84 * mm)]
    if ch.note:
        parts.append(Paragraph(_x(ch.note), st["chn"]))
    return parts


def _charts(chs, st) -> list:
    out, i = [], 0
    chs = list(chs)
    while i < len(chs):
        ch = chs[i]
        if ch.half and i + 1 < len(chs) and chs[i + 1].half:
            hw = (CW - GAP) / 2
            tb = RLTable([[_chart_block(ch, hw, st), "", _chart_block(chs[i + 1], hw, st)]], colWidths=[hw, GAP, hw])
            tb.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                                    ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 0)]))
            out += [tb, Spacer(1, 14)]
            i += 2
        else:
            width = (CW - GAP) / 2 if ch.half else CW
            out += [KeepTogether(_chart_block(ch, width, st)), Spacer(1, 14)]
            i += 1
    return out


def _card(c: Card, st) -> list:
    tone = SEVERITY_STATUS.get(c.severity, "neutral")
    kind_tone = KIND_STATUS.get(c.kind, "neutral")
    kind, _, area = c.tag.partition(" · ")
    tag = (f'<font color="{_tone(kind_tone)}">{_x(kind.upper())}</font>&nbsp;&nbsp;·&nbsp;&nbsp;'
           f'<font color="{_tone(tone)}">{_x(SEVERITY_LABEL.get(c.severity, c.severity).upper())}</font>'
           + (f"&nbsp;&nbsp;·&nbsp;&nbsp;{_x(area.upper())}" if area else ""))
    inner = [Paragraph(tag, st["tag"]), Paragraph(_x(c.title), st["cardt"]), Paragraph(_x(c.body), st["body"])]
    if c.why:
        inner += [Paragraph("Why it matters", st["lab"]), Paragraph(_x(c.why), st["body"])]
    if c.clue:
        inner += [Paragraph("Related finding", st["lab"]), Paragraph(_x(c.clue), st["body"])]
    if c.action:
        inner += [Paragraph("Recommended action", st["lab"]), Paragraph(f"<font name=\"{st['F']['semibold']}\">{_x(c.action)}</font>", st["body"])]
        inner += [Paragraph(_x(s), st["step"], bulletText=f"{k}.") for k, s in enumerate(c.steps, 1)]
    if c.details:
        inner.append(Spacer(1, 4))
        inner.append(Paragraph("&nbsp;&nbsp;·&nbsp;&nbsp;".join(f'<font color="{INK_60}">{_x(k)}:</font> {_x(v)}' for k, v in c.details), st["small"]))
    tb = RLTable([[inner]], colWidths=[CW])
    tb.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.5, C_RULE), ("LINEBEFORE", (0, 0), (0, -1), 3, HexColor(_tone(tone))),
                            ("LEFTPADDING", (0, 0), (-1, -1), 12), ("RIGHTPADDING", (0, 0), (-1, -1), 11),
                            ("TOPPADDING", (0, 0), (-1, -1), 9), ("BOTTOMPADDING", (0, 0), (-1, -1), 10)]))
    return [KeepTogether([tb]), Spacer(1, 8)]


def _section(s: Section, st) -> list:
    part = "Appendix" if s.part == "appendix" else "Section"
    eyebrow = f"{part} {s.label}  ·  {s.title}"
    need = 120 * mm if (s.charts or s.cards or s.callouts) else 80 * mm
    out = _heading(eyebrow, s.headline or s.title, (_key(s.key), f"{s.label}  {s.title}"), st, need)
    if s.lead:
        out.append(Paragraph(_x(s.lead), st["lead"]))
    out += _callouts(s.callouts, st)
    conclusion = s.key == "conclusion"
    if not conclusion:
        out += [Paragraph(_x(p), st["para"]) for p in s.paragraphs]
    if s.facts:
        out += _facts(s.facts, st)
    first, later = (s.tables, ()) if s.key in ("kpis", "plan", "trends") else ((), s.tables)
    for t in first:
        out += _table(t, st)
    out += _charts(s.charts, st)
    if s.cards:
        out.append(Paragraph("Findings in detail", st["h3"]))
    for c in s.cards:
        out += _card(c, st)
    for t in later:
        out += _table(t, st)
    if conclusion:
        paras = list(s.paragraphs)
        closing = paras.pop() if paras else ""
        out += [Paragraph(_x(p), st["para"]) for p in paras]
        if s.steps:
            out.append(Paragraph("Next steps", st["h3"]))
            out += [Paragraph(_x(x), st["bigstep"], bulletText=f"{k}") for k, x in enumerate(s.steps, 1)]
        if closing:
            out += [Spacer(1, 6), _box([Paragraph(_x(closing), st["boxt"])], st["tint"], bar=st["acc"])]
    if s.notes:
        out.append(Spacer(1, 4))
        out += [Paragraph(_x(n), st["note"], bulletText="•") for n in s.notes]
    out.append(Spacer(1, 16))
    return out


def _tiles(d: ReportData, st) -> list:
    if not d.tiles:
        return []
    cells = []
    for t in d.tiles:
        note_color = _tone(t.status) if t.status != "neutral" else INK_60
        parts = [Paragraph(_x(t.label), st["tilel"]), Paragraph(_x(t.value), st["tilev"]),
                 Paragraph(f'<font color="{note_color}">{_x(t.note) or "&nbsp;"}</font>', st["tilen"])]
        if t.delta:
            col = _tone(t.delta_tone) if t.delta_tone != "neutral" else INK_60
            parts.append(Paragraph(f'<font color="{col}">{_x(t.delta)}</font>', st["tilen"]))
        cells.append(parts)
    gap = 5
    w = (CW - gap * (len(cells) - 1)) / len(cells)
    row, widths = [], []
    for k, c in enumerate(cells):
        if k:
            row.append("")
            widths.append(gap)
        row.append(c)
        widths.append(w)
    tb = RLTable([row], colWidths=widths)
    ts = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
          ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 9)]
    for k, t in enumerate(d.tiles):
        col = 2 * k
        ts += [("BACKGROUND", (col, 0), (col, 0), st["tint"]),
               ("LINEABOVE", (col, 0), (col, 0), 2.2, HexColor(_tone(t.status)) if t.status != "neutral" else st["acc"])]
    for k in range(1, len(cells)):
        ts += [("LEFTPADDING", (2 * k - 1, 0), (2 * k - 1, 0), 0), ("RIGHTPADDING", (2 * k - 1, 0), (2 * k - 1, 0), 0)]
    tb.setStyle(TableStyle(ts))
    return [tb, Spacer(1, 6)]


def _summary(d: ReportData, st) -> list:
    out = _heading("Executive summary", d.key_message.rstrip(".") if d.key_message else "Executive summary",
                   (_key("summary"), "Executive summary"), st)
    if d.lede:
        out.append(Paragraph(_x(d.lede), st["exec"]))
    out += [Paragraph(_x(p), st["exec"]) for p in d.summary_paras]
    if d.tiles:
        out.append(Paragraph("Scorecard", st["h3"]))
        out += _tiles(d, st)
    if d.takeaways:
        out.append(Paragraph("Key takeaways", st["h3"]))
        for kind, text in d.takeaways:
            col = _tone(KIND_STATUS.get(kind, "neutral"))
            out.append(Paragraph(f'<font color="{col}" size="7">{_x("■")}</font>&nbsp;&nbsp;<font color="{col}" name="{st["F"]["semibold"]}">'
                                 f'{_x(KINDS.get(kind, kind))}.</font> {_x(text)}', st["bul"]))
    if d.actions:
        out.append(Paragraph("Priority actions", st["h3"]))
        rows = [[Paragraph(h, st["th"]) for h in ("#", "Action", "Suggested owner", "When", "Priority")]]
        for k, a in enumerate(d.actions, 1):
            tone = SEVERITY_STATUS.get(a.severity, "neutral")
            pri = SEVERITY_LABEL.get(a.severity, a.severity).replace(" priority", "")
            rows.append([Paragraph(str(k), st["td"]),
                         [Paragraph(f"<font name=\"{st['F']['semibold']}\" color=\"{INK}\">{_x(a.action)}</font>", st["td"]),
                          Paragraph(_x(a.why or a.finding), st["tds"])],
                         Paragraph(_x(a.owner), st["td"]), Paragraph(_x(a.timeframe), st["td"]),
                         Paragraph(f'<font color="{_tone(tone)}">{_x("■")}&nbsp;{_x(pri)}</font>', st["td"])])
        tb = RLTable(rows, colWidths=[CW * 0.05, CW * 0.52, CW * 0.17, CW * 0.14, CW * 0.12], repeatRows=1)
        tb.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("BACKGROUND", (0, 0), (-1, 0), st["tint"]),
                                ("LINEBELOW", (0, 0), (-1, 0), 0.9, st["acc"]), ("LINEBELOW", (0, 1), (-1, -1), 0.4, C_RULE),
                                ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                                ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5)]))
        out += [tb, Spacer(1, 10)]
    notes = [n for n in (d.quality_line, *[s for s in d.summary if s.startswith("Not included")]) if n]
    if d.is_sample:
        notes.insert(0, "This report was built from synthetic sample data; the figures do not describe a real business.")
    if notes:
        out.append(_box([Paragraph(_x(n), st["small"]) for n in notes], C_DEEP, pad=(10, 7)))
    return out + [Spacer(1, 22)]


def _glance(d: ReportData, st) -> list:
    rows = [("Summary", "", "Executive summary", "Key message, scorecard, takeaways and priority actions", _key("summary"))]
    rows += [("Analysis", s.label, s.title, s.desc, _key(s.key)) for s in d.shown("main")]
    rows += [("Appendix", s.label, s.title, s.desc, _key(s.key)) for s in d.shown("appendix")]
    rows.append(("Appendix", d.method_label, "Methodology and limitations", "How the figures are produced and what they cannot tell you",
                 _key("method")))
    n_main, n_app = len(d.shown("main")), len(d.shown("appendix")) + 1
    out = [Paragraph("REPORT AT A GLANCE", st["eyebrow"]),
           Paragraph(_x(d.title) + f'&nbsp;&nbsp;<font size="8" color="{INK_60}" name="{st["F"]["regular"]}">'
                     f'{n_main} sections, {n_app} appendices</font>', st["h1s"]),
           Spacer(1, 8), _Glance(rows, st), Spacer(1, 10)]
    sw = lambda t: f'<font color="{_tone(t)}" size="8">{_x("■")}</font>'  # noqa: E731
    read = [Paragraph("How to read this report", st["boxh"]),
            Paragraph(f"{sw('good')}&nbsp; Meets its reference level, or improved", st["boxt"]),
            Paragraph(f"{sw('warn')}&nbsp; Outside its reference level, or medium priority", st["boxt"]),
            Paragraph(f"{sw('bad')}&nbsp; Well outside it, worse than before, or high priority", st["boxt"]),
            Paragraph("Reference levels are adjustable app defaults, not industry standards.", st["boxt"])]
    about = [Paragraph("About the data", st["boxh"])] + [Paragraph(f'<font color="{INK_60}">{_x(k)}:</font> {_x(v)}', st["boxt"]) for k, v in d.about]
    half = (CW - GAP) / 2
    tb = RLTable([[_box(read, st["tint"], pad=(9, 7), width=half), "", _box(about, st["tint"], pad=(9, 7), width=half)]],
                 colWidths=[half, GAP, half])
    tb.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                            ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 0)]))
    return out + [KeepTogether([tb]), PageBreak()]


def _divider_contents(d: ReportData, st) -> list:
    out = [Spacer(1, 6)]
    for s in d.shown("appendix"):
        out += [Paragraph(f'<font color="{d.accent}">{_x(s.label)}</font>&nbsp;&nbsp;&nbsp;{_x(s.title)}', st["toct"]),
                Paragraph(_x(s.desc), st["tocd"]), Spacer(1, 9)]
    out.append(Paragraph(f'<font color="{d.accent}">{_x(d.method_label)}</font>&nbsp;&nbsp;&nbsp;Methodology and limitations', st["toct"]))
    return out


def _method(d: ReportData, st) -> list:
    out = _heading(f"Appendix {d.method_label}  ·  Methodology", "Methodology and limitations",
                   (_key("method"), f"{d.method_label}  Methodology and limitations"), st)
    return out + [Paragraph(_x(m), st["note"], bulletText="•") for m in d.methodology]


def render_pdf(d: ReportData) -> bytes:
    st = _styles(d)
    buf = io.BytesIO()
    doc = _Doc(buf, d, st)
    story = [NextPageTemplate("body"), PageBreak()]
    story += _glance(d, st)
    story += _summary(d, st)
    for s in d.shown("main"):
        story += _section(s, st)
    story += [NextPageTemplate("divider"), PageBreak()] + _divider_contents(d, st) + [NextPageTemplate("body"), PageBreak()]
    for s in d.shown("appendix"):
        story += _section(s, st)
    story += _method(d, st)
    doc.multiBuild(story, canvasmaker=_canvas_maker())
    return buf.getvalue()

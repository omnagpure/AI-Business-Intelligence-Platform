"""PDF report (ReportLab). Cover page, contents with page numbers, running header and footer, paper-coloured pages.

Fonts: the DejaVu files that ship inside Matplotlib, so text in many scripts renders and nothing needs installing.
All user-derived text goes through `_x` (XML-escaped) before it reaches a Paragraph.
"""
from __future__ import annotations

import hashlib
import io
import os
import re
from xml.sax.saxutils import escape

import matplotlib
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, CondPageBreak, Frame, Image, KeepTogether, NextPageTemplate, PageBreak, PageTemplate, Paragraph,
                                Spacer, Table as RLTable, TableStyle)
from reportlab.platypus.tableofcontents import TableOfContents

from reporting.report_builder import Card, ReportData, Section, Table
from reporting.theme import ACCENT, INK, INK_20, INK_40, INK_60, INK_80, PAPER, PAPER_DEEP, PAPER_RULE

C_PAPER, C_INK, C_ACC = HexColor(PAPER), HexColor(INK), HexColor(ACCENT)
C_80, C_60, C_40, C_20 = HexColor(INK_80), HexColor(INK_60), HexColor(INK_40), HexColor(INK_20)
C_DEEP, C_RULE = HexColor(PAPER_DEEP), HexColor(PAPER_RULE)

PAGE_W, PAGE_H = A4
MX, MT, MB = 20 * mm, 26 * mm, 22 * mm
CW = PAGE_W - 2 * MX
_FONTS_DONE = False
_BAD_XML = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _register_fonts() -> tuple[str, str, str, str]:
    """Return (sans, sans_bold, serif, serif_bold) font names; fall back to built-in fonts if DejaVu is missing."""
    global _FONTS_DONE
    base = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
    files = {"RSans": "DejaVuSans.ttf", "RSans-B": "DejaVuSans-Bold.ttf", "RSerif": "DejaVuSerif.ttf", "RSerif-B": "DejaVuSerif-Bold.ttf"}
    if all(os.path.exists(os.path.join(base, f)) for f in files.values()):
        if not _FONTS_DONE:
            for name, f in files.items():
                pdfmetrics.registerFont(TTFont(name, os.path.join(base, f)))
            pdfmetrics.registerFontFamily("RSans", normal="RSans", bold="RSans-B", italic="RSans", boldItalic="RSans-B")
            pdfmetrics.registerFontFamily("RSerif", normal="RSerif", bold="RSerif-B", italic="RSerif", boldItalic="RSerif-B")
            _FONTS_DONE = True
        return "RSans", "RSans-B", "RSerif", "RSerif-B"
    return "Helvetica", "Helvetica-Bold", "Times-Roman", "Times-Bold"


def _x(text: object) -> str:
    return escape(_BAD_XML.sub("", "" if text is None else str(text)))


def _styles():
    sans, sansb, serif, serifb = _register_fonts()
    s = {}
    s["body"] = ParagraphStyle("body", fontName=sans, fontSize=9, leading=13.5, textColor=C_80)
    s["lead"] = ParagraphStyle("lead", parent=s["body"], fontSize=10.5, leading=15.5, textColor=C_INK, spaceAfter=8)
    s["small"] = ParagraphStyle("small", parent=s["body"], fontSize=7.8, leading=11, textColor=C_60)
    s["h2"] = ParagraphStyle("h2", fontName=serifb, fontSize=19, leading=23, textColor=C_INK, spaceBefore=0, spaceAfter=2)
    s["h3"] = ParagraphStyle("h3", fontName=sansb, fontSize=7.6, leading=10, textColor=C_60, spaceBefore=10, spaceAfter=4, keepWithNext=1)
    s["th"] = ParagraphStyle("th", fontName=sansb, fontSize=7, leading=9, textColor=C_INK)
    s["thr"] = ParagraphStyle("thr", parent=s["th"], alignment=TA_RIGHT)
    s["td"] = ParagraphStyle("td", fontName=sans, fontSize=7.8, leading=10.2, textColor=C_80)
    s["tdr"] = ParagraphStyle("tdr", parent=s["td"], alignment=TA_RIGHT)
    s["fl"] = ParagraphStyle("fl", fontName=sansb, fontSize=7, leading=10, textColor=C_60)
    s["fv"] = ParagraphStyle("fv", fontName=sans, fontSize=8.6, leading=12, textColor=C_INK)
    s["bul"] = ParagraphStyle("bul", parent=s["body"], fontSize=9.6, leading=14.5, textColor=C_INK, leftIndent=14, bulletIndent=0, spaceAfter=7)
    s["note"] = ParagraphStyle("note", parent=s["small"], leftIndent=10, bulletIndent=0, spaceAfter=2)
    s["cardt"] = ParagraphStyle("cardt", fontName=serifb, fontSize=10.8, leading=14, textColor=C_INK, spaceAfter=3)
    s["tag"] = ParagraphStyle("tag", fontName=sansb, fontSize=6.8, leading=9, textColor=C_60, spaceAfter=3)
    s["toc"] = ParagraphStyle("toc", fontName=sans, fontSize=9.4, leading=17, textColor=C_INK)
    s["cap"] = ParagraphStyle("cap", parent=s["small"], spaceBefore=3, spaceAfter=8)
    s["fonts"] = (sans, sansb, serif, serifb)
    return s


class _Doc(BaseDocTemplate):
    def __init__(self, buf, data: ReportData, fonts):
        super().__init__(buf, pagesize=A4, leftMargin=MX, rightMargin=MX, topMargin=MT, bottomMargin=MB, title=data.title,
                         author=data.prepared_by or "AI Business Intelligence", subject=f"Analysis of {data.dataset_name}", creator="AI Business Intelligence")
        self.data, self.fonts = data, fonts
        cover = PageTemplate("cover", [Frame(MX, MB, CW, PAGE_H - MB - MT, id="c", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)], onPage=self._cover)
        body = PageTemplate("body", [Frame(MX, MB, CW, PAGE_H - MB - MT, id="b", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)], onPage=self._page)
        self.addPageTemplates([cover, body])

    # ---- page furniture
    def _cover(self, c, doc):
        sans, sansb, serif, serifb = self.fonts
        d = self.data
        c.saveState()
        c.setFillColor(C_PAPER)
        c.rect(0, 0, PAGE_W, PAGE_H, stroke=0, fill=1)
        top = PAGE_H * 0.58                                   # ink block: the 25 % voice of the page
        c.setFillColor(C_INK)
        c.rect(0, PAGE_H - top, PAGE_W, top, stroke=0, fill=1)
        c.setFillColor(C_20)
        c.setFont(sansb, 7.5)
        c.drawString(MX, PAGE_H - 24 * mm, "AI BUSINESS INTELLIGENCE   ·   ANALYSIS REPORT")
        # title, wrapped by hand to keep the serif display look
        c.setFillColor(C_PAPER)
        size = 34 if len(d.title) <= 40 else 26
        lines = _wrap(d.title, serifb, size, CW)
        y = PAGE_H - 62 * mm
        c.setFont(serifb, size)
        for ln in lines[:4]:
            c.drawString(MX, y, ln)
            y -= size * 1.18
        c.setFillColor(C_ACC)                                  # the 5 % accent: one short bar
        c.rect(MX, y - 4 * mm, 18 * mm, 1.4 * mm, stroke=0, fill=1)
        c.setFillColor(C_PAPER)
        c.setFont(sans, 11)
        for ln in _wrap(d.subtitle, sans, 11, CW)[:2]:
            y -= 6.2 * mm
            c.drawString(MX, y - 8 * mm, ln)
        # meta block on paper
        mx = MX
        yy = 70 * mm
        items = ([("PREPARED FOR", d.prepared_by)] if d.prepared_by else []) + [("GENERATED", d.generated_at), ("REPORT ID", d.report_id)]
        for label, val in items:
            c.setFillColor(C_60)
            c.setFont(sansb, 6.8)
            c.drawString(mx, yy, label)
            c.setFillColor(C_INK)
            c.setFont(sans, 10)
            c.drawString(mx, yy - 5.2 * mm, val[:38])
            mx += 52 * mm
        if d.is_sample:
            c.setStrokeColor(C_ACC)
            c.setFillColor(C_ACC)
            c.setLineWidth(0.8)
            c.rect(MX, 44 * mm, 74 * mm, 7 * mm, stroke=1, fill=0)
            c.setFont(sansb, 7)
            c.drawString(MX + 3 * mm, 46.4 * mm, "SYNTHETIC SAMPLE DATA, NOT REAL")
        c.setFillColor(C_60)
        c.setFont(sans, 7.5)
        c.drawString(MX, 18 * mm, "Every figure in this report is computed from the uploaded data.")
        c.restoreState()

    def _page(self, c, doc):
        sans, sansb, _, _ = self.fonts
        d = self.data
        c.saveState()
        c.setFillColor(C_PAPER)
        c.rect(0, 0, PAGE_W, PAGE_H, stroke=0, fill=1)
        c.setFillColor(C_INK)
        c.rect(MX, PAGE_H - 13.5 * mm, 5 * mm, 1.1 * mm, stroke=0, fill=1)
        c.setFont(sansb, 6.8)
        c.setFillColor(C_60)
        c.drawString(MX + 8 * mm, PAGE_H - 13.8 * mm, "AI BUSINESS INTELLIGENCE")
        c.setFont(sans, 7)
        c.drawRightString(PAGE_W - MX, PAGE_H - 13.8 * mm, d.title[:70])
        c.setStrokeColor(C_RULE)
        c.setLineWidth(0.5)
        c.line(MX, PAGE_H - 17 * mm, PAGE_W - MX, PAGE_H - 17 * mm)
        c.line(MX, 15 * mm, PAGE_W - MX, 15 * mm)
        c.setFillColor(C_60)
        c.setFont(sans, 7)
        c.drawString(MX, 10.5 * mm, f"Report {d.report_id}  ·  {d.generated_at}")
        c.setFillColor(C_ACC)
        c.rect(PAGE_W - MX - 7 * mm, 10.2 * mm, 1.6 * mm, 1.6 * mm, stroke=0, fill=1)
        c.setFillColor(C_INK)
        c.setFont(sansb, 7.5)
        c.drawRightString(PAGE_W - MX, 10.5 * mm, str(doc.page))
        c.restoreState()

    def afterFlowable(self, f):
        if isinstance(f, Paragraph) and f.style.name == "h2":
            text = f.getPlainText()
            key = "sec_" + hashlib.md5(text.encode("utf-8", "ignore")).hexdigest()[:10]     # stable across the TOC passes
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(text, key, 0, 0)
            self.notify("TOCEntry", (0, text, self.page, key))


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


# ---- flowable builders --------------------------------------------------------------------------
def _P(text, style):
    return Paragraph(_x(text), style)


def _heading(num: int, title: str, st, need: float = 62 * mm) -> list:
    sans, sansb, serif, serifb = st["fonts"]
    p = Paragraph(f'<font name="{sansb}" size="9" color="{ACCENT}">{num:02d}</font>&nbsp;&nbsp;{_x(title)}', st["h2"])
    rule = RLTable([[""]], colWidths=[CW], rowHeights=[1])
    rule.setStyle(TableStyle([("LINEABOVE", (0, 0), (-1, 0), 0.6, C_INK)]))
    return [CondPageBreak(need), p, Spacer(1, 3), rule, Spacer(1, 9)]


def _weights(header, rows) -> list[float]:
    """Column widths: proportional to content, but never narrower than the longest word of the header."""
    raw = []
    for i in range(len(header)):
        longest = max([len(str(header[i]))] + [min(len(str(r[i])), 60) for r in rows[:30]])
        raw.append(max(5, min(longest, 46)) ** 0.9)
    mins = []
    for i, h in enumerate(header):
        head_w = max(len(w) for w in str(h).split() or [""]) * 6.6 + 10
        cell_w = max([min(len(w), 24) for r in rows[:30] for w in str(r[i]).split()] + [0]) * 4.6 + 10
        mins.append(max(head_w, cell_w))
    widths, fixed = [0.0] * len(raw), set()
    for _ in range(len(raw)):
        free = [i for i in range(len(raw)) if i not in fixed]
        room = CW - sum(widths[i] for i in fixed)
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
    scale = CW / sum(widths)
    return [w * scale for w in widths]


def _table(t: Table, st) -> list:
    hdr = [Paragraph(_x(h.upper()), st["thr"] if i >= t.numeric_from else st["th"]) for i, h in enumerate(t.header)]
    body = [[Paragraph(_x(c), st["tdr"] if i >= t.numeric_from else st["td"]) for i, c in enumerate(r)] for r in t.rows]
    tb = RLTable([hdr] + body, colWidths=_weights(t.header, t.rows), repeatRows=1)
    style = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEABOVE", (0, 0), (-1, 0), 1.3, C_INK), ("LINEBELOW", (0, 0), (-1, 0), 0.6, C_INK),
             ("LINEBELOW", (0, -1), (-1, -1), 0.6, C_INK), ("TOPPADDING", (0, 0), (-1, -1), 3.6), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.6),
             ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
    for r in range(2, len(body) + 1, 2):
        style.append(("BACKGROUND", (0, r), (-1, r), C_DEEP))
    tb.setStyle(TableStyle(style))
    out = [Paragraph(_x(t.title.upper()), st["h3"]), tb]
    if t.note:
        out.append(Paragraph(_x(t.note), st["cap"]))
    else:
        out.append(Spacer(1, 6))
    return out


def _facts(facts, st) -> list:
    rows = [[Paragraph(_x(k.upper()), st["fl"]), Paragraph(_x(v), st["fv"])] for k, v in facts]
    tb = RLTable(rows, colWidths=[CW * 0.26, CW * 0.74])
    tb.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                            ("LEFTPADDING", (0, 0), (-1, -1), 0), ("LINEBELOW", (0, 0), (-1, -1), 0.4, C_RULE)]))
    return [tb, Spacer(1, 8)]


def _image(png: bytes, caption: str, st) -> list:
    w, h = ImageReader(io.BytesIO(png)).getSize()
    width = CW
    height = width * h / w
    if height > 80 * mm:
        height = 80 * mm
        width = height * w / h
    return [KeepTogether([Image(io.BytesIO(png), width=width, height=height, hAlign="LEFT"), Paragraph(_x(caption), st["cap"])])]


def _card(c: Card, st) -> list:
    sev, _, grp = c.tag.partition(" · ")
    tagcolor = ACCENT if c.severity == "high" else INK_60
    tag = f'<font color="{tagcolor}">{_x(sev)}</font>' + (f"  ·  {_x(grp.upper())}" if grp else "")
    inner = [Paragraph(tag, st["tag"]), Paragraph(_x(c.title), st["cardt"]), Paragraph(_x(c.body), st["body"]), Spacer(1, 4)]
    for k, v in c.details:
        inner.append(Paragraph(f'<font name="{st["fonts"][1]}" color="{INK_60}">{_x(k)}</font>&nbsp;&nbsp;{_x(v)}', st["small"]))
    tb = RLTable([[inner]], colWidths=[CW])
    bar = C_ACC if c.severity == "high" else C_INK
    tb.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), C_DEEP), ("LINEBEFORE", (0, 0), (0, -1), 2.4, bar), ("LEFTPADDING", (0, 0), (-1, -1), 11),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 10), ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
    return [tb, Spacer(1, 7)]


def _section(s: Section, st) -> list:
    out = _heading(s.number, s.title, st, 105 * mm if (s.charts or s.cards) and s.status == "ok" else 62 * mm)
    if s.status != "ok":
        box = RLTable([[Paragraph(f'<b>Not included.</b> {_x(s.reason)}', st["small"])]], colWidths=[CW])
        box.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), C_DEEP), ("LINEBEFORE", (0, 0), (0, -1), 2.4, C_RULE), ("LEFTPADDING", (0, 0), (-1, -1), 10),
                                 ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
        return out + [box, Spacer(1, 14)]
    if s.lead:
        out.append(Paragraph(_x(s.lead), st["lead"]))
    if s.facts:
        out += _facts(s.facts, st)
    for cap, png in s.charts:
        out += _image(png, cap, st)
    for t in s.tables:
        out += _table(t, st)
    for c in s.cards:
        out += _card(c, st)
    for n in s.notes:
        out.append(Paragraph(_x(n), st["note"], bulletText="–"))
    out.append(Spacer(1, 16))
    return out


def _summary(d: ReportData, st) -> list:
    sans, sansb, serif, serifb = st["fonts"]
    out = _heading(1, "Executive summary", st)
    cells = []
    for i, (k, v) in enumerate(d.headline):
        col = ACCENT if i == 0 else INK
        cells.append([Paragraph(f'<font name="{serifb}" size="17" color="{col}">{_x(v)}</font>', ParagraphStyle("sv", leading=21)),
                      Paragraph(_x(k.upper()), ParagraphStyle("sl", fontName=sansb, fontSize=6.2, leading=8, textColor=C_60))])
    tb = RLTable([cells], colWidths=[CW / len(cells)] * len(cells))
    ts = [("LINEABOVE", (0, 0), (-1, 0), 1.6, C_INK), ("LINEBELOW", (0, 0), (-1, 0), 0.5, C_RULE), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8), ("LEFTPADDING", (0, 0), (-1, -1), 0)]
    for i in range(1, len(cells)):
        ts += [("LINEBEFORE", (i, 0), (i, 0), 0.5, C_RULE), ("LEFTPADDING", (i, 0), (i, 0), 9)]
    tb.setStyle(TableStyle(ts))
    out += [tb, Spacer(1, 12)]
    for p in d.summary:
        out.append(Paragraph(_x(p), st["bul"], bulletText="—"))
    return out + [Spacer(1, 14)]


def _method(d: ReportData, st) -> list:
    out = _heading(len(d.sections) + 2, "Methodology and limitations", st)
    return out + [Paragraph(_x(m), st["note"], bulletText="–") for m in d.methodology]


def render_pdf(d: ReportData) -> bytes:
    st = _styles()
    buf = io.BytesIO()
    doc = _Doc(buf, d, st["fonts"])
    toc = TableOfContents()
    toc.levelStyles = [ParagraphStyle("t0", fontName=st["fonts"][0], fontSize=9.4, leading=17, textColor=C_INK, leftIndent=0, rightIndent=14)]
    toc.dotsMinLevel = 0
    story = [NextPageTemplate("body"), PageBreak(),
             Paragraph("CONTENTS", st["h3"]), toc, Spacer(1, 16)]
    story += _summary(d, st)
    for s in d.sections:
        story += _section(s, st)
    story += _method(d, st)
    doc.multiBuild(story)
    return buf.getvalue()

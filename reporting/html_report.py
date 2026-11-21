"""Self-contained HTML report (one file: fonts, charts and logo embedded, no external requests). All text is escaped.

Same structure and look as the PDF: cover with key message and headline figures over a data-drawn graphic, report at
a glance, executive summary, numbered sections with headline titles, callouts and two-up charts, appendix, methodology.
"""
from __future__ import annotations

import base64
from html import escape

from analytics.recommendations import KINDS
from reporting.fonts import html_font_faces
from reporting.report_builder import Card, Chart, ReportData, Section, Table
from reporting.theme import (INK, INK_20, INK_40, INK_60, INK_80, PAPER, PAPER_DEEP, PAPER_RULE, SEVERITY_LABEL, SEVERITY_STATUS, STATUS,
                             STATUS_SOFT, palette)

KIND_STATUS = {"risk": "bad", "opportunity": "good", "pattern": "neutral", "data": "neutral"}


def _css(r: ReportData) -> str:
    p = palette(r.accent)
    return html_font_faces() + f"""
:root{{--paper:{PAPER};--ink:{INK};--accent:{p.accent};--dark:{p.dark};--tint:{p.tint};--tint2:{p.tint2};--ink80:{INK_80};--ink60:{INK_60};
--ink40:{INK_40};--ink20:{INK_20};--deep:{PAPER_DEEP};--rule:{PAPER_RULE};--good:{STATUS['good']};--warn:{STATUS['warn']};--bad:{STATUS['bad']};
--neutral:{STATUS['neutral']};--warnsoft:{STATUS_SOFT['warn']}}}
*{{box-sizing:border-box}}
html,body{{background:#EEF0F3}}
body{{margin:0;color:var(--ink80);font:14.5px/1.6 'Plex',"Segoe UI",system-ui,-apple-system,Arial,sans-serif;-webkit-font-smoothing:antialiased}}
.sheet{{max-width:920px;margin:28px auto;background:var(--paper);box-shadow:0 1px 3px rgba(0,0,0,.08),0 8px 30px rgba(0,0,0,.06)}}
.pad{{padding:0 56px}}
.cover{{position:relative}}
.cover .top{{padding:36px 56px 40px}}
.brand{{display:flex;justify-content:space-between;align-items:center}}
.brand .name{{display:flex;align-items:center;gap:10px;font-weight:700;letter-spacing:.06em;color:var(--ink);font-size:17px}}
.brand img{{max-height:40px;max-width:220px}}
.cls{{border:1px solid var(--accent);color:var(--accent);border-radius:99px;padding:2px 12px;font-size:11px;font-weight:700;letter-spacing:.08em}}
.eyebrow{{font-size:12px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:var(--accent)}}
.cover h1{{font-size:46px;line-height:1.1;font-weight:700;color:var(--ink);margin:14px 0 0;letter-spacing:-.01em;max-width:15em}}
.bar{{width:72px;height:5px;background:var(--accent);margin:22px 0 16px}}
.sub{{font-size:16px;color:var(--ink60);margin:0}}
.km{{margin-top:44px;border-left:5px solid var(--accent);padding:2px 0 2px 20px}}
.km .eyebrow{{font-size:11px}}
.km p{{font-size:20px;line-height:1.4;font-weight:600;color:var(--ink);margin:6px 0 0}}
.sample{{background:var(--warnsoft);color:var(--warn);font-weight:600;font-size:13px;padding:10px 56px}}
.block{{background:var(--accent);color:#fff;position:relative;overflow:hidden;padding:56px 56px 26px}}
.block .art{{position:absolute;left:0;right:0;top:0;height:62%;width:100%;object-fit:fill;pointer-events:none}}
.figs{{position:relative;display:grid;grid-template-columns:repeat(3,1fr);gap:24px}}
.figs .v{{font-size:34px;font-weight:700;line-height:1.1;font-variant-numeric:tabular-nums}}
.figs .l{{font-size:13px;opacity:.85;margin-top:4px}}
.figs .n{{font-size:11.5px;opacity:.65;margin-top:2px}}
.meta{{position:relative;display:flex;flex-wrap:wrap;gap:10px 44px;margin-top:40px;padding-top:16px;border-top:1px solid rgba(255,255,255,.3)}}
.meta div{{font-size:14px;font-weight:600}}
.meta span{{display:block;font-size:11.5px;font-weight:400;opacity:.65;margin-bottom:2px}}
.block small{{position:relative;display:block;margin-top:28px;font-size:11.5px;opacity:.6}}
section{{padding:44px 56px 8px}}
.eb{{font-size:11.5px;font-weight:700;letter-spacing:.07em;text-transform:uppercase;color:var(--accent)}}
h2{{font-size:27px;line-height:1.25;font-weight:700;margin:6px 0 0;color:var(--ink)}}
.rule{{height:1px;background:var(--tint2);margin:16px 0 18px}}
.lead{{font-size:16px;color:var(--ink);margin:0 0 14px}}
.para{{font-size:15px;margin:0 0 12px}}
.exec{{font-size:16.5px;line-height:1.65;color:var(--ink);margin:0 0 12px}}
h3{{font-size:15px;color:var(--ink);margin:28px 0 10px;font-weight:600}}
.callouts{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px;margin:6px 0 18px}}
.callout{{border-left:3px solid var(--c,var(--accent));padding:2px 0 2px 14px}}
.callout .v{{font-size:30px;font-weight:700;line-height:1.15;color:var(--cv,var(--ink))}}
.callout .l{{font-size:12.5px;color:var(--ink60)}}
.tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px}}
.tile{{background:var(--tint);border-top:3px solid var(--c,var(--accent));padding:12px 14px}}
.tile .l{{font-size:12px;color:var(--ink60)}}
.tile .v{{font-size:28px;line-height:1.2;font-weight:700;color:var(--ink);margin:4px 0 2px;font-variant-numeric:tabular-nums}}
.tile .s{{font-size:12px;font-weight:600;color:var(--c,var(--ink60))}}
.tile .d{{font-size:12px;font-weight:600;color:var(--dc,var(--ink60))}}
ul.take{{list-style:none;margin:0;padding:0}}
ul.take li{{padding:8px 0 8px 20px;border-bottom:1px solid var(--rule);position:relative;color:var(--ink)}}
ul.take li::before{{content:"";position:absolute;left:0;top:17px;width:9px;height:9px;background:var(--c,var(--neutral))}}
ul.take b{{color:var(--c,var(--neutral));font-weight:600}}
.dot{{display:inline-block;width:8px;height:8px;margin-right:6px;background:var(--c,var(--neutral))}}
.pill{{font-weight:600;white-space:nowrap;color:var(--c,var(--neutral))}}
.callbox{{background:var(--deep);padding:10px 14px;font-size:12.5px;color:var(--ink60);margin-top:18px}}
.callbox p{{margin:2px 0}}
dl.facts{{display:grid;grid-template-columns:max-content 1fr max-content 1fr;gap:0 18px;margin:0 0 10px}}
dl.facts dt,dl.facts dd{{padding:6px 0;border-bottom:1px solid var(--rule)}}
dl.facts dt{{font-size:12.5px;color:var(--ink60)}}
dl.facts dd{{margin:0;color:var(--ink);font-weight:600;font-size:13.5px}}
.tw{{overflow-x:auto}}
table{{width:100%;border-collapse:collapse;font-size:13px;margin:0 0 4px;font-variant-numeric:tabular-nums}}
th{{text-align:left;font-size:12px;color:var(--ink);font-weight:600;padding:8px 10px;background:var(--tint);border-bottom:1.5px solid var(--accent)}}
td{{padding:7px 10px;border-bottom:1px solid var(--rule);vertical-align:top;color:var(--ink80)}}
td .sub{{display:block;font-size:12px;color:var(--ink60)}}
td b{{color:var(--ink);font-weight:600}}
.r{{text-align:right}}
.tnote{{font-size:12px;color:var(--ink60);margin:4px 0 0}}
.charts{{display:grid;grid-template-columns:1fr 1fr;gap:20px 28px;margin:18px 0 6px}}
figure{{margin:0}}
figure.full{{grid-column:1/-1}}
figure img{{display:block;width:100%;height:auto}}
figcaption{{font-size:13.5px;font-weight:600;color:var(--ink);margin-bottom:6px}}
figure p{{font-size:12.5px;color:var(--ink60);margin:6px 0 0}}
.card{{border:1px solid var(--rule);border-left:4px solid var(--c,var(--neutral));padding:14px 18px;margin:0 0 12px;break-inside:avoid}}
.card .tag{{font-size:11px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;color:var(--ink60)}}
.card h4{{font-size:17px;line-height:1.35;color:var(--ink);margin:6px 0 6px;font-weight:600}}
.card p{{margin:0 0 6px}}
.card .lab{{font-size:12px;font-weight:600;color:var(--ink);margin-top:10px}}
.card ol{{margin:4px 0 0;padding-left:20px}}
.card .det{{font-size:12px;color:var(--ink60);margin-top:10px}}
ol.steps{{margin:6px 0 0;padding:0;list-style:none;counter-reset:s}}
ol.steps li{{counter-increment:s;position:relative;padding:0 0 10px 30px;color:var(--ink);font-size:15px}}
ol.steps li::before{{content:counter(s);position:absolute;left:0;top:0;font-weight:700;color:var(--accent)}}
.closing{{background:var(--tint);border-left:3px solid var(--accent);padding:12px 16px;font-size:13.5px;margin-top:12px}}
ul.notes{{margin:10px 0 0;padding-left:18px;font-size:12.5px;color:var(--ink60)}}
.glance{{padding-top:48px}}
.toc{{list-style:none;margin:16px 0 0;padding:0}}
.toc .g{{font-size:11.5px;font-weight:700;letter-spacing:.07em;color:var(--ink60);border-bottom:1.5px solid var(--accent);padding:14px 0 4px}}
.toc a{{display:grid;grid-template-columns:34px 1fr 40px;gap:0 6px;padding:8px 0;border-bottom:1px solid var(--rule);text-decoration:none;color:var(--ink)}}
.toc a:hover .t{{color:var(--accent);text-decoration:underline}}
.toc .n{{font-weight:700;color:var(--accent)}}
.toc .t{{font-weight:600}}
.toc .d{{display:block;font-size:12.5px;color:var(--ink60);font-weight:400}}
.boxes{{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:26px}}
.box{{background:var(--tint);padding:14px 18px;font-size:13px}}
.box h4{{margin:0 0 8px;font-size:14px;color:var(--ink);font-weight:600}}
.box p{{margin:3px 0}}
.box span{{color:var(--ink60)}}
.divider{{margin-top:48px;background:var(--tint);border-left:18px solid var(--accent);padding:44px 56px}}
.divider h2{{font-size:38px}}
.divider p{{color:var(--ink60);margin:10px 0 0}}
footer{{margin-top:48px;border-top:1px solid var(--rule);padding:16px 56px 36px;font-size:11.5px;color:var(--ink60);display:flex;justify-content:space-between;gap:16px;flex-wrap:wrap}}
@page{{size:A4;margin:14mm 12mm}}
@media print{{html,body{{background:#fff;-webkit-print-color-adjust:exact;print-color-adjust:exact}}.sheet{{box-shadow:none;margin:0;max-width:none}}
.cover{{break-after:page}}.glance{{break-after:page}}section{{padding-top:24px}}table,figure,.card,.tile,.callout{{break-inside:avoid}}}}
@media (max-width:700px){{.pad,section,.cover .top,.block{{padding-left:22px;padding-right:22px}}.cover h1{{font-size:32px}}
.charts,.boxes,.figs{{grid-template-columns:1fr}}dl.facts{{grid-template-columns:max-content 1fr}}}}
"""


def _e(x: object) -> str:
    return escape("" if x is None else str(x), quote=True)


def _c(status: str, var: str = "--c") -> str:
    return f"{var}:var(--{status if status in STATUS else 'neutral'})"


def _b64(png: bytes) -> str:
    return base64.b64encode(png).decode()


def _logo_src(data: bytes | None) -> str | None:
    if not data:
        return None
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "data:image/png;base64," + _b64(data)
    if data[:3] == b"\xff\xd8\xff":
        return "data:image/jpeg;base64," + _b64(data)
    return None


_MARK = ('<svg width="26" height="26" viewBox="0 0 24 24" aria-hidden="true"><path d="M3.2 4.6L10 19.6L15.4 11.4" fill="none" '
         'stroke="currentColor" stroke-width="3.4" stroke-linecap="round" stroke-linejoin="round"/><path d="M20 0.6Q20.55 3.95 23.9 '
         '4.5Q20.55 5.05 20 8.4Q19.45 5.05 16.1 4.5Q19.45 3.95 20 0.6Z" fill="#C2410C"/></svg>')


def _table(t: Table) -> str:
    head = "".join(f'<th class="{"r" if t.is_right(i) else ""}">{_e(h)}</th>' for i, h in enumerate(t.header))
    tones = t.tones or ("",) * len(t.rows)
    subs = t.sub or ("",) * len(t.rows)
    rows = []
    for k, r in enumerate(t.rows):
        tone = tones[k] if k < len(tones) else ""
        cells = []
        for i, c in enumerate(r):
            txt = _e(c)
            if i == t.tone_col and tone and tone != "neutral":
                dot = "" if str(c)[:1] in "▲▼" else '<span class="dot"></span>'
                txt = f'<span class="pill" style="{_c(tone)}">{dot}{txt}</span>'
            elif i == 1 and k < len(subs) and subs[k]:
                txt = f"<b>{txt}</b><span class=sub>{_e(subs[k])}</span>"
            cells.append(f'<td class="{"r" if t.is_right(i) else ""}">{txt}</td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")
    note = f'<p class="tnote">{_e(t.note)}</p>' if t.note else ""
    return f'<h3>{_e(t.title)}</h3><div class="tw"><table><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>{note}'


def _charts(chs) -> str:
    if not chs:
        return ""
    figs = []
    for ch in chs:
        note = f"<p>{_e(ch.note)}</p>" if ch.note else ""
        figs.append(f'<figure class="{"half" if ch.half else "full"}"><figcaption>{_e(ch.title)}</figcaption>'
                    f'<img alt="{_e(ch.title)}" src="data:image/png;base64,{_b64(ch.png)}">{note}</figure>')
    return f'<div class="charts">{"".join(figs)}</div>'


def _callouts(items) -> str:
    if not items:
        return ""
    out = []
    for v, l, tone in items:
        style = (_c(tone) + ";" + _c(tone, "--cv")) if tone != "neutral" else ""
        out.append(f'<div class="callout" style="{style}"><div class="v">{_e(v)}</div><div class="l">{_e(l)}</div></div>')
    return f'<div class="callouts">{"".join(out)}</div>'


def _card(c: Card) -> str:
    tone = SEVERITY_STATUS.get(c.severity, "neutral")
    kind, _, area = c.tag.partition(" · ")
    tag = (f'<span style="color:var(--{KIND_STATUS.get(c.kind, "neutral")})">{_e(kind)}</span> · '
           f'<span style="color:var(--{tone})">{_e(SEVERITY_LABEL.get(c.severity, c.severity))}</span>' + (f" · {_e(area)}" if area else ""))
    parts = [f'<div class="tag">{tag}</div><h4>{_e(c.title)}</h4><p>{_e(c.body)}</p>']
    if c.why:
        parts.append(f'<div class="lab">Why it matters</div><p>{_e(c.why)}</p>')
    if c.clue:
        parts.append(f'<div class="lab">Related finding</div><p>{_e(c.clue)}</p>')
    if c.action:
        parts.append(f'<div class="lab">Recommended action</div><p><b>{_e(c.action)}</b></p>')
        if c.steps:
            parts.append("<ol>" + "".join(f"<li>{_e(s)}</li>" for s in c.steps) + "</ol>")
    if c.details:
        parts.append('<div class="det">' + " · ".join(f"{_e(k)}: <b>{_e(v)}</b>" for k, v in c.details) + "</div>")
    return f'<div class="card" style="{_c(tone)}">{"".join(parts)}</div>'


def _section(s: Section) -> str:
    part = "Appendix" if s.part == "appendix" else "Section"
    out = [f'<div class="eb">{part} {_e(s.label)} · {_e(s.title)}</div><h2>{_e(s.headline or s.title)}</h2><div class="rule"></div>']
    if s.lead:
        out.append(f'<p class="lead">{_e(s.lead)}</p>')
    out.append(_callouts(s.callouts))
    conclusion = s.key == "conclusion"
    paras = list(s.paragraphs)
    closing = paras.pop() if conclusion and paras else ""
    out += [f'<p class="para">{_e(p)}</p>' for p in paras]
    if s.facts:
        out.append('<dl class="facts">' + "".join(f"<dt>{_e(k)}</dt><dd>{_e(v)}</dd>" for k, v in s.facts) + "</dl>")
    first, later = (s.tables, ()) if s.key in ("kpis", "plan", "trends") else ((), s.tables)
    out += [_table(t) for t in first]
    out.append(_charts(s.charts))
    if s.cards:
        out.append("<h3>Findings in detail</h3>" + "".join(_card(c) for c in s.cards))
    out += [_table(t) for t in later]
    if s.steps:
        out.append('<h3>Next steps</h3><ol class="steps">' + "".join(f"<li>{_e(x)}</li>" for x in s.steps) + "</ol>")
    if closing:
        out.append(f'<div class="closing">{_e(closing)}</div>')
    if s.notes:
        out.append('<ul class="notes">' + "".join(f"<li>{_e(n)}</li>" for n in s.notes) + "</ul>")
    return f'<section id="{_e(s.key)}">{"".join(out)}</section>'


def _summary(r: ReportData) -> str:
    out = [f'<section id="summary"><div class="eb">Executive summary</div><h2>{_e((r.key_message or "Executive summary").rstrip("."))}</h2>'
           '<div class="rule"></div>']
    out += [f'<p class="exec">{_e(p)}</p>' for p in ((r.lede,) if r.lede else ()) + r.summary_paras]
    if r.tiles:
        tiles = "".join(f'<div class="tile" style="{_c(t.status) if t.status != "neutral" else ""}"><div class="l">{_e(t.label)}</div>'
                        f'<div class="v">{_e(t.value)}</div><div class="s">{_e(t.note) or "&nbsp;"}</div>'
                        + (f'<div class="d" style="{_c(t.delta_tone, "--dc") if t.delta_tone != "neutral" else ""}">{_e(t.delta)}</div>' if t.delta else "")
                        + "</div>" for t in r.tiles)
        out.append(f'<h3>Scorecard</h3><div class="tiles">{tiles}</div>')
    if r.takeaways:
        out.append('<h3>Key takeaways</h3><ul class="take">' + "".join(
            f'<li style="{_c(KIND_STATUS.get(k, "neutral"))}"><b>{_e(KINDS.get(k, k))}.</b> {_e(t)}</li>' for k, t in r.takeaways) + "</ul>")
    if r.actions:
        rows = "".join(f'<tr><td>{k}</td><td><b>{_e(a.action)}</b><span class=sub>{_e(a.why or a.finding)}</span></td><td>{_e(a.owner)}</td>'
                       f'<td>{_e(a.timeframe)}</td><td><span class="pill" style="{_c(SEVERITY_STATUS.get(a.severity, "neutral"))}">'
                       f'<span class="dot"></span>{_e(SEVERITY_LABEL.get(a.severity, a.severity).replace(" priority", ""))}</span></td></tr>'
                       for k, a in enumerate(r.actions, 1))
        out.append('<h3>Priority actions</h3><div class="tw"><table><thead><tr><th>#</th><th>Action</th><th>Suggested owner</th><th>When</th>'
                   f'<th>Priority</th></tr></thead><tbody>{rows}</tbody></table></div>')
    notes = [n for n in (r.quality_line, *[s for s in r.summary if s.startswith("Not included")]) if n]
    if notes:
        out.append('<div class="callbox">' + "".join(f"<p>{_e(n)}</p>" for n in notes) + "</div>")
    return "".join(out) + "</section>"


def _cover(r: ReportData) -> str:
    logo = _logo_src(r.logo)
    brand = f'<img alt="{_e(r.organisation or "Logo")}" src="{logo}">' if logo else f'<span class="name">{_MARK}{_e(r.app_name.upper())}</span>'
    figs = "".join(f'<div><div class="v">{_e(t.value)}</div><div class="l">{_e(t.label)}</div><div class="n">{_e(t.note)}</div></div>'
                   for t in r.tiles[:3])
    meta = ([("Prepared for", r.prepared_by)] if r.prepared_by else []) + ([("Prepared by", r.organisation)] if r.organisation else []) + \
           ([("Data covers", r.period)] if r.period else []) + [("Date", r.generated_at.split(",")[0]), ("Report ID", r.report_id)]
    art = f'<img class="art" alt="" src="data:image/png;base64,{_b64(r.cover_png)}">' if r.cover_png else ""
    sample = '<div class="sample">Synthetic sample data. The figures do not describe a real business.</div>' if r.is_sample else ""
    km = f'<div class="km"><div class="eyebrow">Key message</div><p>{_e(r.key_message)}</p></div>' if r.key_message else ""
    return (f'<header class="cover"><div class="top"><div class="brand">{brand}<span class="cls">{_e((r.classification or "Internal").upper())}</span></div>'
            f'<div style="margin-top:84px" class="eyebrow">{_e(r.eyebrow)}</div><h1>{_e(r.title)}</h1><div class="bar"></div>'
            f'<p class="sub">{_e(r.subtitle)}</p>{km}</div>{sample}'
            f'<div class="block">{art}<div class="figs">{figs}</div><div class="meta">'
            + "".join(f"<div><span>{_e(k)}</span>{_e(v)}</div>" for k, v in meta)
            + f'</div><small>Prepared with {_e(r.app_name)}. Every figure is computed from the uploaded data; no value is typed in by hand.</small></div></header>')


def _glance(r: ReportData) -> str:
    groups = [("Summary", [("", "Executive summary", "Key message, scorecard, takeaways and priority actions", "summary")]),
              ("Analysis", [(s.label, s.title, s.desc, s.key) for s in r.shown("main")]),
              ("Appendix", [(s.label, s.title, s.desc, s.key) for s in r.shown("appendix")]
               + [(r.method_label, "Methodology and limitations", "How the figures are produced and what they cannot tell you", "method")])]
    toc = "".join(f'<li class="g">{_e(g.upper())}</li>' + "".join(
        f'<li><a href="#{_e(k)}"><span class="n">{_e(l)}</span><span><span class="t">{_e(t)}</span><span class="d">{_e(d)}</span></span></a></li>'
        for l, t, d, k in items) for g, items in groups)
    sw = lambda t: f'<span class="dot" style="{_c(t)}"></span>'   # noqa: E731
    read = (f"<h4>How to read this report</h4><p>{sw('good')}Meets its reference level, or improved</p>"
            f"<p>{sw('warn')}Outside its reference level, or medium priority</p><p>{sw('bad')}Well outside it, worse than before, or high priority</p>"
            "<p><span>Reference levels are adjustable app defaults, not industry standards.</span></p>")
    about = "<h4>About the data</h4>" + "".join(f"<p><span>{_e(k)}:</span> {_e(v)}</p>" for k, v in r.about)
    return (f'<section class="glance"><div class="eb">Report at a glance</div><h2>{_e(r.title)}</h2><ul class="toc">{toc}</ul>'
            f'<div class="boxes"><div class="box">{read}</div><div class="box">{about}</div></div></section>')


def render_html(r: ReportData) -> bytes:
    main, appx = r.shown("main"), r.shown("appendix")
    divider = ('<div class="divider"><h2>Appendix</h2><p>Supporting detail: the data, its quality and how each figure is calculated.</p></div>')
    method = (f'<section id="method"><div class="eb">Appendix {_e(r.method_label)} · Methodology</div><h2>Methodology and limitations</h2>'
              '<div class="rule"></div><ul class="notes" style="font-size:13.5px">' + "".join(f"<li>{_e(m)}</li>" for m in r.methodology) + "</ul></section>")
    body = _glance(r) + _summary(r) + "".join(_section(s) for s in main) + divider + "".join(_section(s) for s in appx) + method
    doc = (f'<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
           f'<title>{_e(r.title)}</title><style>{_css(r)}</style></head><body><div class="sheet">{_cover(r)}<main>{body}</main>'
           f'<footer><span>{_e(r.organisation or r.app_name)} · Report {_e(r.report_id)} · {_e(r.classification or "Internal")}</span>'
           f'<span>Every figure is computed from the uploaded data. Generated {_e(r.generated_at)}.</span></footer></div></body></html>')
    return doc.encode("utf-8")

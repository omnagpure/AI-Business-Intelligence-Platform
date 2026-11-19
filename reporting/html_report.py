"""Self-contained HTML report (one file, charts embedded, no external requests). All text is escaped."""
from __future__ import annotations

import base64
from html import escape

from reporting.report_builder import Card, ReportData, Section, Table
from reporting.theme import ACCENT, INK, INK_20, INK_40, INK_60, INK_80, PAPER, PAPER_DEEP, PAPER_RULE

_CSS = f"""
:root{{--paper:{PAPER};--ink:{INK};--accent:{ACCENT};--ink80:{INK_80};--ink60:{INK_60};--ink40:{INK_40};--ink20:{INK_20};--deep:{PAPER_DEEP};--rule:{PAPER_RULE};}}
*{{box-sizing:border-box}}
html{{background:var(--paper)}}
body{{margin:0;background:var(--paper);color:var(--ink80);font:15px/1.6 "Segoe UI",system-ui,-apple-system,"Helvetica Neue",Arial,sans-serif;-webkit-font-smoothing:antialiased}}
.cover{{background:var(--ink);color:var(--paper);padding:72px 0 56px}}
.wrap{{max-width:880px;margin:0 auto;padding:0 32px}}
.kicker{{font-size:11px;letter-spacing:.22em;text-transform:uppercase;color:var(--ink20)}}
.cover h1{{font:600 44px/1.12 Georgia,"Times New Roman",serif;letter-spacing:-.01em;margin:22px 0 0;color:var(--paper);max-width:16em}}
.cover .bar{{width:56px;height:4px;background:var(--accent);margin:28px 0 22px}}
.cover .sub{{font-size:17px;color:var(--paper);margin:0}}
.cover .meta{{display:flex;flex-wrap:wrap;gap:6px 32px;margin-top:34px;font-size:12px;color:var(--ink20)}}
.cover .meta b{{display:block;font-weight:500;letter-spacing:.12em;text-transform:uppercase;font-size:10px;color:var(--ink40);margin-bottom:2px}}
.stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));border-top:2px solid var(--ink);border-bottom:1px solid var(--rule);margin:40px 0 8px}}
.stat{{padding:16px 16px 14px 0}}
.stat+.stat{{padding-left:16px;border-left:1px solid var(--rule)}}
.stat .v{{font:600 28px/1.1 Georgia,serif;color:var(--ink);font-variant-numeric:tabular-nums}}
.stat:first-child .v{{color:var(--accent)}}
.stat .l{{font-size:10.5px;letter-spacing:.12em;text-transform:uppercase;color:var(--ink60);margin-top:6px}}
section{{padding:44px 0 4px}}
.sh{{display:flex;align-items:baseline;gap:14px;border-bottom:1px solid var(--rule);padding-bottom:10px;margin-bottom:18px}}
.sh .n{{font:600 13px Georgia,serif;color:var(--accent);letter-spacing:.08em}}
.sh h2{{font:600 26px/1.2 Georgia,"Times New Roman",serif;margin:0;color:var(--ink)}}
.lead{{font-size:16px;color:var(--ink);margin:0 0 16px}}
h3{{font-size:11px;letter-spacing:.14em;text-transform:uppercase;color:var(--ink60);margin:28px 0 8px;font-weight:600}}
ul.sum{{list-style:none;margin:0;padding:0}}
ul.sum li{{padding:12px 0 12px 22px;border-bottom:1px solid var(--rule);position:relative;color:var(--ink)}}
ul.sum li::before{{content:"";position:absolute;left:0;top:21px;width:8px;height:2px;background:var(--ink)}}
dl.facts{{display:grid;grid-template-columns:max-content 1fr;gap:6px 28px;margin:0 0 6px}}
dl.facts dt{{font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:var(--ink60);padding-top:3px}}
dl.facts dd{{margin:0;color:var(--ink)}}
.tw{{overflow-x:auto}}
table{{width:100%;border-collapse:collapse;font-size:13px;margin:0 0 4px;font-variant-numeric:tabular-nums}}
th{{text-align:left;font-size:10.5px;letter-spacing:.1em;text-transform:uppercase;color:var(--ink);font-weight:600;padding:8px 10px;border-top:2px solid var(--ink);border-bottom:1px solid var(--ink)}}
td{{padding:7px 10px;border-bottom:1px solid var(--rule);vertical-align:top;color:var(--ink80)}}
tbody tr:nth-child(even) td{{background:var(--deep)}}
.r{{text-align:right}}
.tnote{{font-size:12px;color:var(--ink60);margin:4px 0 0}}
figure{{margin:22px 0 4px}}
figure img{{display:block;width:100%;height:auto}}
figcaption{{font-size:12px;color:var(--ink60);margin-top:6px}}
.card{{background:var(--deep);border-left:3px solid var(--ink);padding:16px 20px;margin:0 0 14px;break-inside:avoid}}
.card.high{{border-left-color:var(--accent)}}
.card .tag{{font-size:10.5px;letter-spacing:.14em;text-transform:uppercase;color:var(--ink60);font-weight:600}}
.card.high .tag b{{color:var(--accent)}}
.card h4{{font:600 17px/1.3 Georgia,serif;color:var(--ink);margin:6px 0 6px}}
.card p{{margin:0 0 8px;color:var(--ink80)}}
.card dl{{display:grid;grid-template-columns:96px 1fr;gap:3px 14px;margin:8px 0 0;font-size:12.5px}}
.card dt{{color:var(--ink60);font-weight:600}}
.card dd{{margin:0;color:var(--ink80)}}
ul.notes{{margin:10px 0 0;padding-left:18px;font-size:13px;color:var(--ink60)}}
.skip{{background:var(--deep);padding:12px 16px;font-size:13.5px;color:var(--ink60);border-left:3px solid var(--rule)}}
.badge{{display:inline-block;border:1px solid var(--accent);color:var(--accent);font-size:10.5px;letter-spacing:.12em;text-transform:uppercase;padding:2px 8px;margin-top:20px;font-weight:600}}
footer{{margin-top:56px;border-top:2px solid var(--ink);padding:14px 0 40px;font-size:11.5px;color:var(--ink60);display:flex;justify-content:space-between;gap:16px;flex-wrap:wrap}}
@page{{size:A4;margin:16mm 14mm}}
@media print{{html,body{{background:var(--paper);-webkit-print-color-adjust:exact;print-color-adjust:exact}}
.cover{{padding:40px 0}}section{{padding-top:28px;break-inside:avoid-page}}table,figure,.card{{break-inside:avoid}}}}
@media (max-width:640px){{.cover h1{{font-size:32px}}dl.facts{{grid-template-columns:1fr}}.card dl{{grid-template-columns:1fr}}}}
"""


def _e(x: object) -> str:
    return escape("" if x is None else str(x), quote=True)


def _table(t: Table) -> str:
    head = "".join(f'<th class="{"r" if i >= t.numeric_from else ""}">{_e(h)}</th>' for i, h in enumerate(t.header))
    body = "".join("<tr>" + "".join(f'<td class="{"r" if i >= t.numeric_from else ""}">{_e(c)}</td>' for i, c in enumerate(r)) + "</tr>" for r in t.rows)
    note = f'<p class="tnote">{_e(t.note)}</p>' if t.note else ""
    return f'<h3>{_e(t.title)}</h3><div class="tw"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>{note}'


def _card(c: Card) -> str:
    sev, _, grp = c.tag.partition(" · ")
    tag = f"<b>{_e(sev)}</b> · {_e(grp)}" if grp else _e(c.tag)
    det = "".join(f"<dt>{_e(k)}</dt><dd>{_e(v)}</dd>" for k, v in c.details)
    return (f'<div class="card {"high" if c.severity == "high" else ""}"><div class="tag">{tag}</div><h4>{_e(c.title)}</h4>'
            f'<p>{_e(c.body)}</p><dl>{det}</dl></div>')


def _section(s: Section) -> str:
    head = f'<div class="sh"><span class="n">{s.number:02d}</span><h2>{_e(s.title)}</h2></div>'
    if s.status != "ok":
        return f'<section id="{_e(s.key)}">{head}<div class="skip"><b>Not included.</b> {_e(s.reason)}</div></section>'
    parts = [head]
    if s.lead:
        parts.append(f'<p class="lead">{_e(s.lead)}</p>')
    if s.facts:
        parts.append("<dl class=\"facts\">" + "".join(f"<dt>{_e(k)}</dt><dd>{_e(v)}</dd>" for k, v in s.facts) + "</dl>")
    for cap, png in s.charts:
        parts.append(f'<figure><img alt="{_e(cap)}" src="data:image/png;base64,{base64.b64encode(png).decode()}"><figcaption>{_e(cap)}</figcaption></figure>')
    parts += [_table(t) for t in s.tables]
    parts += [_card(c) for c in s.cards]
    if s.notes:
        parts.append("<ul class=\"notes\">" + "".join(f"<li>{_e(n)}</li>" for n in s.notes) + "</ul>")
    return f'<section id="{_e(s.key)}">{"".join(parts)}</section>'


def render_html(r: ReportData) -> bytes:
    meta = [("Prepared for", r.prepared_by)] if r.prepared_by else []
    meta += [("Dataset", r.dataset_name), ("Generated", r.generated_at), ("Report ID", r.report_id)]
    cover_meta = "".join(f"<div><b>{_e(k)}</b>{_e(v)}</div>" for k, v in meta)
    stats = "".join(f'<div class="stat"><div class="v">{_e(v)}</div><div class="l">{_e(k)}</div></div>' for k, v in r.headline)
    badge = '<div class="badge">Synthetic sample data, not real</div>' if r.is_sample else ""
    summary = "".join(f"<li>{_e(p)}</li>" for p in r.summary)
    method = ("<section id=\"method\"><div class=\"sh\"><span class=\"n\">" f"{len(r.sections) + 2:02d}</span><h2>Methodology and limitations</h2></div>"
              "<ul class=\"notes\" style=\"font-size:14px\">" + "".join(f"<li>{_e(m)}</li>" for m in r.methodology) + "</ul></section>")
    body = (f'<section id="summary"><div class="sh"><span class="n">01</span><h2>Executive summary</h2></div>'
            f'<div class="stats">{stats}</div><ul class="sum">{summary}</ul>{badge}</section>'
            + "".join(_section(s) for s in r.sections) + method)
    doc = (f'<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
           f'<title>{_e(r.title)}</title><style>{_CSS}</style></head><body>'
           f'<header class="cover"><div class="wrap"><div class="kicker">Tessaly · Analysis report</div><h1>{_e(r.title)}</h1>'
           f'<div class="bar"></div><p class="sub">{_e(r.subtitle)}</p><div class="meta">{cover_meta}</div></div></header>'
           f'<main class="wrap">{body}<footer><span>Tessaly · Report {_e(r.report_id)}</span>'
           f'<span>Every figure is computed from the uploaded data. Generated {_e(r.generated_at)}.</span></footer></main></body></html>')
    return doc.encode("utf-8")

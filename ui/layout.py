"""Design system: global CSS on top of the theme tokens, and the small set of shared UI helpers.

Direction (Vesper): warm neutral surfaces, hairline borders, one vermilion accent, Geist type, light and dark from
the same tokens (ui.theme). Pages use these helpers rather than ad-hoc CSS.
"""
from __future__ import annotations

from html import escape
from pathlib import Path

import streamlit as st

from config.settings import APP_NAME
from ui import theme

_FAVICON = str(Path(__file__).resolve().parent.parent / "assets" / "favicon.png")

_CSS = """
<style>
:root{--r:8px; --r-lg:12px;}
html, body, .stApp, [data-testid="stAppViewContainer"], button, input, textarea, select {
  font-family:'Geist',-apple-system,'Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif !important;
  -webkit-font-smoothing:antialiased;
}
.stApp {background:var(--bg);}
#MainMenu, footer, [data-testid="stDecoration"], [data-testid="stStatusWidget"] {display:none;}
header[data-testid="stHeader"] {background:transparent; height:2.25rem;}
.block-container {padding:1.25rem 2.5rem 4rem; max-width:1280px;}
::selection {background:var(--accent-soft);}

/* top bar: dataset switcher on the left, user menu on the right */
.st-key-topbar {padding:.15rem 0 .9rem; margin-bottom:1.4rem; border-bottom:1px solid var(--line);}
.st-key-topbar [data-testid="stSelectbox"] {min-width:240px; max-width:340px;}
.tb-meta {display:flex; align-items:center; gap:.5rem; flex-wrap:wrap; font-size:.8125rem; color:var(--muted); white-space:nowrap;}
.tb-meta .sep {color:var(--line-2);}
.tb-user {display:flex; flex-direction:column; gap:.1rem; padding:.15rem 0 .55rem; border-bottom:1px solid var(--line); margin-bottom:.35rem;}
.tb-user b {font-size:.9rem; font-weight:600; color:var(--text);}
.tb-user span {font-size:.8rem; color:var(--muted);}

/* page header */
.pg-head {margin:0 0 1.35rem;}
.pg-title {font-size:1.6rem !important; font-weight:600 !important; letter-spacing:-.025em; line-height:1.2 !important; margin:0 !important;
  padding:0 !important; color:var(--text);}
h2, h3 {font-size:1rem !important; font-weight:600 !important; letter-spacing:-.01em; margin-top:1.6rem !important; color:var(--text);}
.eyebrow {font-size:.875rem; font-weight:600; color:var(--text); margin:.2rem 0 .6rem; letter-spacing:-.005em;}
.card-k {font-size:.78rem; font-weight:500; color:var(--muted);}
.card-t {font-size:1rem; font-weight:600; letter-spacing:-.01em; color:var(--text); margin-top:.2rem; line-height:1.35; min-height:2.7em;}
.card-t.card-skip {color:var(--muted); font-weight:500;}
.card-d {font-size:.8125rem; color:var(--muted); margin:.35rem 0 .7rem; line-height:1.5; min-height:3em;}
.next-k {font-size:.78rem; font-weight:600; color:var(--accent);}
.next-t {font-size:1.1rem; font-weight:600; letter-spacing:-.01em; color:var(--text); margin-top:.15rem;}
.next-d {font-size:.875rem; color:var(--muted); margin-top:.25rem; line-height:1.5;}
.kv-title {font-size:1.125rem; font-weight:600; letter-spacing:-.01em; color:var(--text); display:flex; gap:.6rem; align-items:center; flex-wrap:wrap;}
.note-list {margin:.5rem 0 0; padding:0; list-style:none; font-size:.8125rem; color:var(--muted); line-height:1.6;}
.note-list li {padding-left:.9rem; position:relative;}
.note-list li::before {content:""; position:absolute; left:0; top:.62em; width:4px; height:1px; background:var(--faint);}
[data-testid="stCaptionContainer"] {color:var(--muted); font-size:.8125rem;}

/* metric rows: one bordered strip with hairline dividers */
div[data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] [data-testid="stMetric"]) {
  gap:0; border:1px solid var(--line); border-radius:var(--r-lg); background:var(--surface); overflow:hidden; box-shadow:var(--shadow);}
div[data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] [data-testid="stMetric"]) > [data-testid="stColumn"] {
  padding:.95rem 1.15rem; border-left:1px solid var(--line); min-width:0;}
div[data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] [data-testid="stMetric"]) > :first-child {border-left:none;}
[data-testid="stMetric"] {padding:0;}
[data-testid="stMetricLabel"] p {font-size:.78rem; font-weight:500; color:var(--muted);}
[data-testid="stMetricValue"] {font-size:1.5rem; font-weight:600; letter-spacing:-.02em; font-variant-numeric:tabular-nums; color:var(--text);}
[data-testid="stMetricDelta"] {font-size:.78rem; font-weight:500;}
/* text values ("0 high · 3 medium · 2 low", a domain name) wrap instead of being cut off */
[data-testid="stMetricValue"], [data-testid="stMetricValue"] * {white-space:normal !important; overflow:visible !important;
  text-overflow:clip !important; overflow-wrap:anywhere; line-height:1.2;}

/* KPI tiles: separate cards (value, change, sparkline) in a grid */
.st-key-kpi_grid div[data-testid="stHorizontalBlock"] {gap:.85rem !important; border:none !important; background:transparent !important;
  box-shadow:none !important; overflow:visible !important; margin-bottom:.85rem;}
.st-key-kpi_grid div[data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {padding:0 !important; border:none !important;}
.st-key-kpi_grid [data-testid="stMetric"] {background:var(--surface); border:1px solid var(--line); border-radius:var(--r-lg);
  padding:1rem 1.1rem .8rem; box-shadow:var(--shadow); height:100%; transition:border-color .15s ease;}
.st-key-kpi_grid [data-testid="stMetric"]:hover {border-color:var(--line-2);}

/* dashboard: header meta, section titles, data-check callout */
.dash-meta {display:flex; align-items:center; flex-wrap:wrap; gap:.45rem; font-size:.8125rem; color:var(--muted); margin:-.35rem 0 .35rem;}
.dash-sep {color:var(--line-2);}
.dash-faint {color:var(--faint);}
.dash-sec {display:flex; align-items:baseline; gap:.75rem; flex-wrap:wrap; margin:1.6rem 0 .7rem;}
.dash-sec-t {font-size:.75rem; font-weight:600; letter-spacing:.08em; text-transform:uppercase; color:var(--text);}
.dash-sec-note {font-size:.78rem; color:var(--faint);}
.dash-callout {border:1px solid var(--warn); border-left-width:3px; background:var(--warn-soft); border-radius:var(--r);
  padding:.7rem .95rem; font-size:.84rem; line-height:1.5; color:var(--text); margin:.6rem 0 .9rem;}
.dash-callout-t {font-weight:600; color:var(--warn); margin-right:.5rem;}
.dash-callout.info {border-color:var(--line-2); background:var(--surface-2);}
.dash-callout.info .dash-callout-t {color:var(--text);}
.dash-lede {font-size:1.05rem; line-height:1.55; color:var(--text); margin:.2rem 0 .9rem; max-width:62rem;}
.kpi.compact {min-height:0;}
.kpi.seg .kpi-value {font-size:1.15rem; letter-spacing:-.01em; line-height:1.3; margin:.3rem 0 .45rem;}
.kpi.seg .kpi-foot {line-height:1.45; font-size:.74rem; color:var(--muted);}
.seg-dot {display:inline-block; width:.6rem; height:.6rem; border-radius:50%; margin-right:.45rem; vertical-align:baseline;}
.kpi.compact .kpi-foot {margin-top:auto;}
.kpi.compact .kpi-delta {margin-bottom:.35rem;}

/* KPI cards: value, change chip, sparkline, footnote */
.kpi-grid {display:grid; grid-template-columns:repeat(4, minmax(0, 1fr)); gap:.85rem;}
.kpi {background:var(--surface); border:1px solid var(--line); border-radius:var(--r-lg); padding:.95rem 1.05rem .75rem;
  box-shadow:var(--shadow); display:flex; flex-direction:column; min-height:172px; transition:border-color .15s, transform .15s;}
.kpi:hover {border-color:var(--accent-line); transform:translateY(-1px);}
.kpi-label {font-size:.78rem; font-weight:500; color:var(--muted); line-height:1.3; min-height:2.05em;}
.kpi-value {font-size:1.7rem; font-weight:650; letter-spacing:-.02em; color:var(--text); line-height:1.15; margin:.2rem 0 .35rem;
  font-variant-numeric:tabular-nums; overflow-wrap:anywhere;}
.kpi-delta {display:flex; align-items:center; gap:.45rem; flex-wrap:wrap; min-height:1.35rem; font-size:.74rem;}
.kpi-vs {color:var(--faint);}
.chip {display:inline-flex; align-items:center; gap:.2rem; border-radius:999px; padding:.08rem .5rem; font-size:.72rem; font-weight:600;
  font-variant-numeric:tabular-nums; white-space:nowrap;}
.chip.good {color:var(--ok); background:var(--ok-soft);}
.chip.bad {color:var(--danger); background:var(--danger-soft);}
.chip.flat {color:var(--muted); background:var(--surface-2);}
.kpi-spark {position:relative; width:100%; height:40px; margin-top:auto;}
.kpi-spark-empty {margin-top:auto; height:40px;}
.kpi-area, .kpi-line {position:absolute; inset:0; -webkit-mask-size:100% 100%; mask-size:100% 100%;
  -webkit-mask-repeat:no-repeat; mask-repeat:no-repeat;}
.kpi-area {background:linear-gradient(to bottom, var(--accent-line), transparent); opacity:.55;}
.kpi-line {background:var(--accent);}
.kpi-foot {font-size:.7rem; color:var(--faint); margin-top:.4rem; border-top:1px dashed var(--line); padding-top:.4rem;}
.kpi-groups {display:grid; grid-template-columns:repeat(auto-fit, minmax(280px, 1fr)); gap:.85rem;}
.kpi-group {background:var(--surface); border:1px solid var(--line); border-radius:var(--r-lg); padding:.8rem 1rem;}
.kpi-group-t {font-size:.7rem; font-weight:600; letter-spacing:.07em; text-transform:uppercase; color:var(--faint); margin-bottom:.5rem;}
.kpi-mini-grid {display:grid; grid-template-columns:repeat(auto-fill, minmax(120px, 1fr)); gap:.6rem 1rem;}
.kpi-mini .kpi-label {min-height:0; font-size:.74rem;}
.kpi-mini-v {font-size:1.02rem; font-weight:600; color:var(--text); font-variant-numeric:tabular-nums; overflow-wrap:anywhere;}
.view-insight {font-size:.82rem; line-height:1.5; color:var(--muted); border-top:1px solid var(--line); padding-top:.55rem; margin-top:.1rem;}
[class*="st-key-view_"] {background:var(--surface); border-radius:var(--r-lg) !important;}
@media (max-width: 1100px) {.kpi-grid {grid-template-columns:repeat(2, minmax(0, 1fr));}}
@media (max-width: 560px) {.kpi-grid {grid-template-columns:1fr;}}

/* tabs */
button[data-baseweb="tab"] {font-size:.875rem; font-weight:500; color:var(--muted); padding:.55rem .95rem; height:auto;}
button[data-baseweb="tab"][aria-selected="true"] {color:var(--text); font-weight:600;}
[data-baseweb="tab-highlight"] {background:var(--accent) !important; height:2px !important;}
[data-baseweb="tab-border"] {background:var(--line) !important; height:1px !important;}

/* buttons and inputs */
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
  border:1px solid var(--line-2); background:var(--surface); color:var(--text); border-radius:var(--r);
  font-size:.875rem; font-weight:500; min-height:2.2rem; padding:0 .95rem; box-shadow:0 1px 0 rgba(0,0,0,.03);
  transition:border-color .12s, background-color .12s;}
.stButton > button:hover, .stDownloadButton > button:hover, .stFormSubmitButton > button:hover {border-color:var(--faint); background:var(--surface-2); color:var(--text);}
.stButton > button:focus-visible, .stDownloadButton > button:focus-visible {outline:2px solid var(--accent); outline-offset:2px;}
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"], .stFormSubmitButton > button[kind="primaryFormSubmit"] {
  background:var(--accent); border-color:var(--accent); color:var(--on-accent); font-weight:600;}
.stButton > button[kind="primary"]:hover, .stDownloadButton > button[kind="primary"]:hover,
.stFormSubmitButton > button[kind="primaryFormSubmit"]:hover {background:var(--accent-hi); border-color:var(--accent-hi); color:var(--on-accent);}
.stButton > button[kind="tertiary"] {border:none; background:transparent; box-shadow:none; color:var(--accent); padding:0; min-height:auto;}
.stButton > button[kind="tertiary"]:hover {background:transparent; color:var(--accent-hi); text-decoration:underline; text-underline-offset:3px;}
.stButton > button:disabled {opacity:.45;}
[data-baseweb="select"] > div, [data-baseweb="input"] > div, [data-baseweb="textarea"] {border-radius:var(--r) !important;}
[data-testid="stFileUploaderDropzone"] {border:1.5px dashed var(--line-2); background:var(--surface); border-radius:var(--r-lg); padding:1.4rem;}
[data-testid="stFileUploaderDropzone"]:hover {border-color:var(--accent-line); background:var(--accent-soft);}

/* data surfaces */
[data-testid="stDataFrame"], [data-testid="stDataEditor"] {border:1px solid var(--line); border-radius:var(--r); overflow:hidden;}
[data-testid="stAlert"] {border-radius:var(--r); font-size:.875rem;}
[data-testid="stExpander"] details {border:1px solid var(--line) !important; border-radius:var(--r) !important; background:var(--surface);}
[data-testid="stExpander"] summary {font-size:.875rem; color:var(--text); font-weight:500;}
[data-testid="stPlotlyChart"] {border:1px solid var(--line); border-radius:var(--r-lg); background:var(--surface); padding:.35rem .5rem .1rem;
  box-shadow:var(--shadow);}
hr {border-color:var(--line) !important; margin:1.5rem 0 !important;}

/* sidebar */
section[data-testid="stSidebar"][aria-expanded="true"] {width:248px !important; min-width:248px !important;}
[data-testid="stSidebarNavLink"] {border-radius:var(--r); padding:.38rem .65rem; font-size:.875rem; color:var(--muted); transition:background-color .12s, color .12s;}
[data-testid="stSidebarNavLink"]:hover {background:var(--surface-2); color:var(--text);}
[data-testid="stSidebarNavLink"][aria-current="page"] {background:var(--accent-soft); font-weight:600;}
[data-testid="stSidebarNavLink"][aria-current="page"] span {color:var(--accent) !important;}
[data-testid="stNavSectionHeader"] {font-size:.75rem !important; font-weight:600 !important; color:var(--faint) !important; padding-top:.7rem;}

/* badges and empty state */
.badge {display:inline-block; font-size:.6875rem; font-weight:600; line-height:1; padding:.3rem .55rem; border-radius:999px;
  border:1px solid var(--line-2); color:var(--muted); background:var(--surface); white-space:nowrap;}
.badge.accent {color:var(--accent); border-color:var(--accent-line); background:var(--accent-soft);}
.badge.warn {color:var(--warn); border-color:transparent; background:var(--warn-soft);}
.badge.danger {color:var(--danger); border-color:transparent; background:var(--danger-soft);}
.badge.ok {color:var(--ok); border-color:transparent; background:var(--ok-soft);}
.empty {border:1px solid var(--line); border-radius:var(--r-lg); padding:1.4rem 1.5rem; background:var(--surface); box-shadow:var(--shadow);}
.empty-t {font-size:.95rem; font-weight:600; color:var(--text);}
.empty-d {font-size:.875rem; color:var(--muted); margin:.2rem 0 .3rem; line-height:1.5;}

/* workflow steps and role chips */
.steps {display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:0; border:1px solid var(--line);
  border-radius:var(--r-lg); background:var(--surface); overflow:hidden; box-shadow:var(--shadow);}
.step {padding:.9rem 1.05rem; border-left:1px solid var(--line); min-width:0;}
.step:first-child {border-left:none;}
.step-n {font-size:.78rem; color:var(--faint); display:flex; gap:.45rem; align-items:center; font-variant-numeric:tabular-nums;}
.dot {width:7px; height:7px; border-radius:50%; background:var(--line-2); display:inline-block;}
.dot.done {background:var(--ok);} .dot.todo {background:var(--accent);} .dot.soon {background:var(--line-2);}
.step-t {font-size:.9rem; font-weight:600; color:var(--text); margin-top:.3rem;}
.step-d {font-size:.8rem; color:var(--muted); margin-top:.15rem; line-height:1.45;}
.chips {display:flex; flex-wrap:wrap; gap:.35rem; margin-top:.4rem;}
.chip {font-size:.75rem; padding:.24rem .55rem; border-radius:999px; background:var(--surface-2); border:1px solid var(--line); color:var(--text);}
.chip b {font-weight:500; color:var(--muted); margin-right:.35rem;}

@media (max-width: 900px) {
  .block-container {padding:1rem 1rem 3rem;}
  .pg-title {font-size:1.35rem !important;}
  .st-key-topbar [data-testid="stSelectbox"] {min-width:0; max-width:none;}
}
@media (max-width: 640px) {
  div[data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] [data-testid="stMetric"]) > [data-testid="stColumn"] {
    border-left:none; border-top:1px solid var(--line);}
  div[data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] [data-testid="stMetric"]) > :first-child {border-top:none;}
  .tb-meta {display:none;}
}
@media (prefers-reduced-motion: reduce) { * {transition:none !important;} }
</style>
"""


def configure_page(sidebar: bool = True) -> None:
    st.set_page_config(page_title=APP_NAME, page_icon=_FAVICON, layout="wide",
                       initial_sidebar_state="expanded" if sidebar else "collapsed")
    theme.handle_request()                      # a light/dark switch picked on the previous run
    st.html(f"<style>{theme.token_css()}</style>")
    st.markdown(_CSS, unsafe_allow_html=True)


def page_title(title: str, caption: str = "") -> None:
    """Page header: the title only. A description, when given, is shown on hover instead of as a subtitle line."""
    tip = f' title="{escape(caption)}"' if caption else ""
    st.markdown(f'<div class="pg-head"><div class="pg-title" role="heading" aria-level="1"{tip}>{escape(title)}</div></div>',
                unsafe_allow_html=True)


def dataset_meta_html() -> str:
    """Rows, columns and status of the active dataset, shown in the top bar. Empty when nothing is loaded."""
    from ui import state
    result = state.get_result()
    if result is None:
        return ""
    ds, prof = result.dataset, result.profile
    parts = [f"{fmt_int(prof.rows)} rows", f"{prof.columns} columns"]
    badges = ""
    if ds.is_sample:
        badges += " " + badge("Sample data", "warn")
    if state.get_cleaning() is not None:
        badges += " " + badge("Cleaned", "ok")
    sep = '<span class="sep">·</span>'
    return f'<div class="tb-meta ctx-line">{sep.join(f"<span>{p}</span>" for p in parts)}{badges}</div>'


def eyebrow(text: str) -> None:
    st.markdown(f'<div class="eyebrow">{escape(text)}</div>', unsafe_allow_html=True)


def badge(text: str, tone: str = "neutral") -> str:
    """Return badge HTML. tone: neutral | accent | warn | danger."""
    return f'<span class="badge {tone if tone != "neutral" else ""}">{escape(text)}</span>'


def note_list(items: list[str]) -> str:
    return '<ul class="note-list">' + "".join(f"<li>{escape(i)}</li>" for i in items) + "</ul>"


def empty_state(title: str, description: str) -> None:
    st.markdown(f'<div class="empty"><div class="empty-t">{escape(title)}</div>'
                f'<div class="empty-d">{escape(description)}</div></div>', unsafe_allow_html=True)


def no_dataset_notice() -> None:
    """One line and one action, instead of instructions."""
    from ui import nav
    with st.container(border=True):
        st.markdown('<div class="empty-t">No data yet</div><div class="empty-d">Upload a CSV or Excel file to get started.</div>',
                    unsafe_allow_html=True)
        if st.button("Upload a file", type="primary", key="nd_upload", icon=":material/upload:"):
            nav.go("upload")


def fmt_int(n: int) -> str:
    return f"{n:,}"


def steps(items: list[tuple[str, str, str, str]]) -> None:
    """Workflow strip. items: (label, title, detail, state) with state in done | todo | soon."""
    html = "".join(f'<div class="step"><div class="step-n"><span class="dot {escape(st_)}"></span>{escape(lab)}</div>'
                   f'<div class="step-t">{escape(t)}</div><div class="step-d">{escape(d)}</div></div>'
                   for lab, t, d, st_ in items)
    st.markdown(f'<div class="steps">{html}</div>', unsafe_allow_html=True)


def chips(pairs: list[tuple[str, str]]) -> None:
    """Small label/value chips. Values may be user column names, so everything is escaped."""
    html = "".join(f'<span class="chip"><b>{escape(k)}</b>{escape(v)}</span>' for k, v in pairs)
    st.markdown(f'<div class="chips">{html}</div>', unsafe_allow_html=True)


_MD_SPECIAL = set("\\`*_{}[]()#+-.!<>|~:$")


def md_escape(text: str) -> str:
    """Escape user-supplied text (file names, titles) for st.markdown, so it renders literally."""
    return "".join("\\" + ch if ch in _MD_SPECIAL else ch for ch in str(text))


# ---- analysis-page building blocks (dashboard, forecasting, anomalies) ----------------------------
def section(title: str, note: str = "") -> None:
    """Small uppercase section heading with an optional faint note."""
    extra = f'<span class="dash-sec-note">{escape(note)}</span>' if note else ""
    st.markdown(f'<div class="dash-sec"><span class="dash-sec-t">{escape(title)}</span>{extra}</div>', unsafe_allow_html=True)


def callout(title: str, text: str, tone: str = "warn") -> None:
    """Highlighted note. tone: warn | info."""
    st.markdown(f'<div class="dash-callout {tone}"><span class="dash-callout-t">{escape(title)}</span>{escape(text)}</div>',
                unsafe_allow_html=True)


def lede(text: str) -> None:
    """The one-sentence answer at the top of a result, in larger type."""
    st.markdown(f'<div class="dash-lede">{escape(text)}</div>', unsafe_allow_html=True)


def meta_line(parts: list[str], tags_html: str = "") -> None:
    """File · rows · coverage line under a page title. `parts` are plain text; `tags_html` is trusted badge HTML."""
    sep = '<span class="dash-sep">·</span>'
    st.markdown(f'<div class="dash-meta">{sep.join(f"<span>{escape(p)}</span>" for p in parts if p)} {tags_html}</div>',
                unsafe_allow_html=True)


def stat_cards(cards: list[dict]) -> None:
    """Uniform result cards in the KPI-card style. Each card: label, value, and optionally chip=(text, tone) with tone
    good | bad | flat, sub (text beside the chip) and foot (small text at the bottom)."""
    html = []
    for c in cards:
        chip = ""
        if c.get("chip") or c.get("sub"):
            text, tone = c.get("chip") or ("", "flat")
            chip_html = f'<span class="chip {escape(tone)}">{escape(text)}</span>' if text else ""
            chip = f'<div class="kpi-delta">{chip_html}<span class="kpi-vs">{escape(c.get("sub", ""))}</span></div>'
        foot = f'<div class="kpi-foot">{escape(c["foot"])}</div>' if c.get("foot") else ""
        html.append(f'<div class="kpi compact" title="{escape(c.get("tip", ""))}"><div class="kpi-label">{escape(c["label"])}</div>'
                    f'<div class="kpi-value">{escape(c["value"])}</div>{chip}{foot}</div>')
    st.html('<div class="kpi-grid">' + "".join(html) + "</div>")

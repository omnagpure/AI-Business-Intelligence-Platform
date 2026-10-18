"""Design system: tokens, global CSS and the small set of shared UI helpers.

Direction: neutral near-black surfaces, hairline borders, one muted teal accent, Inter type,
flat stat strips instead of boxed cards. Pages should use these helpers rather than ad-hoc CSS.
"""
from __future__ import annotations

from html import escape

import streamlit as st

from config.settings import APP_NAME

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;450;500;600&display=swap');
:root{
  --bg:#0B0D10; --surface:#0F1215; --surface-2:#14181D; --line:#1F242B; --line-2:#2A3038;
  --text:#E6E8EB; --muted:#8B929C; --faint:#5E6672;
  --accent:#3DA58A; --accent-hi:#4FBC9F; --warn:#D6A04A; --danger:#E5484D; --r:6px;
}
html, body, .stApp, [data-testid="stAppViewContainer"], button, input, textarea, select {
  font-family:'Inter',-apple-system,'Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif !important;
  -webkit-font-smoothing:antialiased; font-feature-settings:'cv11','ss03';
}
#MainMenu, footer, [data-testid="stDecoration"], [data-testid="stStatusWidget"] {display:none;}
header[data-testid="stHeader"] {background:transparent; height:2.5rem;}
.block-container {padding:2.25rem 2.5rem 4rem; max-width:1240px;}

/* page header */
.pg-head {padding-bottom:1.1rem; margin-bottom:1.4rem; border-bottom:1px solid var(--line);}
.pg-title {font-size:1.375rem; font-weight:600; letter-spacing:-.02em; line-height:1.25; margin:0; color:var(--text);}
.pg-sub {margin:.35rem 0 0; font-size:.875rem; color:var(--muted); line-height:1.5; max-width:62ch;}
h2, h3 {font-size:.95rem !important; font-weight:600 !important; letter-spacing:-.005em; margin-top:1.6rem !important;}
.eyebrow {font-size:.6875rem; font-weight:500; letter-spacing:.08em; text-transform:uppercase; color:var(--faint); margin-bottom:.35rem;}
.kv-title {font-size:1.125rem; font-weight:600; letter-spacing:-.01em; color:var(--text); display:flex; gap:.6rem; align-items:center; flex-wrap:wrap;}
.note-list {margin:.5rem 0 0; padding:0; list-style:none; font-size:.8125rem; color:var(--muted); line-height:1.6;}
.note-list li {padding-left:.9rem; position:relative;}
.note-list li::before {content:""; position:absolute; left:0; top:.62em; width:4px; height:1px; background:var(--faint);}
[data-testid="stCaptionContainer"] {color:var(--muted); font-size:.8125rem;}

/* stat strip: a row of metrics becomes one bordered strip with hairline dividers */
div[data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) {
  gap:0; border:1px solid var(--line); border-radius:var(--r); background:var(--surface); overflow:hidden;}
div[data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) > [data-testid="stColumn"],
div[data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) > [data-testid="column"] {
  padding:.85rem 1.1rem; border-left:1px solid var(--line); min-width:0;}
div[data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) > :first-child {border-left:none;}
[data-testid="stMetric"] {padding:0;}
[data-testid="stMetricLabel"] p {font-size:.75rem; font-weight:450; color:var(--muted);}
[data-testid="stMetricValue"] {font-size:1.35rem; font-weight:600; letter-spacing:-.015em; font-variant-numeric:tabular-nums;}

/* tabs */
button[data-baseweb="tab"] {font-size:.84rem; font-weight:450; color:var(--muted); padding:.55rem .95rem; height:auto;}
button[data-baseweb="tab"][aria-selected="true"] {color:var(--text); font-weight:500;}
[data-baseweb="tab-highlight"] {background:var(--accent) !important; height:2px !important;}
[data-baseweb="tab-border"] {background:var(--line) !important; height:1px !important;}

/* buttons and inputs */
.stButton > button, .stDownloadButton > button {
  border:1px solid var(--line-2); background:transparent; color:var(--text); border-radius:var(--r);
  font-size:.84rem; font-weight:500; min-height:2.1rem; padding:0 .9rem; box-shadow:none;
  transition:border-color .12s, background-color .12s;}
.stButton > button:hover, .stDownloadButton > button:hover {border-color:var(--faint); background:var(--surface-2); color:var(--text);}
.stButton > button:focus-visible, .stDownloadButton > button:focus-visible {outline:2px solid var(--accent); outline-offset:2px;}
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {
  background:var(--accent); border-color:var(--accent); color:#07120E;}
.stButton > button[kind="primary"]:hover {background:var(--accent-hi); border-color:var(--accent-hi); color:#07120E;}
.stButton > button:disabled {opacity:.45;}
[data-baseweb="select"] > div, [data-baseweb="input"] > div, [data-baseweb="textarea"] {border-radius:var(--r) !important;}
[data-testid="stFileUploaderDropzone"] {border:1px dashed var(--line-2); background:var(--surface); border-radius:8px; padding:1.4rem;}
[data-testid="stFileUploaderDropzone"]:hover {border-color:var(--faint);}

/* data surfaces */
[data-testid="stDataFrame"], [data-testid="stDataEditor"] {border:1px solid var(--line); border-radius:var(--r); overflow:hidden;}
[data-testid="stAlert"] {border-radius:var(--r); border:1px solid var(--line-2); font-size:.84rem;}
[data-testid="stExpander"] {border:1px solid var(--line); border-radius:var(--r);}
[data-testid="stExpander"] summary {font-size:.84rem; color:var(--muted);}
hr {border-color:var(--line) !important; margin:1.5rem 0 !important;}

/* sidebar */
section[data-testid="stSidebar"] {border-right:1px solid var(--line); background:var(--bg);}
section[data-testid="stSidebar"][aria-expanded="true"] {width:248px !important; min-width:248px !important;}
[data-testid="stSidebarNavLink"] {border-radius:var(--r); padding:.38rem .65rem; font-size:.84rem; color:var(--muted);}
[data-testid="stSidebarNavLink"]:hover {background:var(--surface-2); color:var(--text);}
[data-testid="stSidebarNavLink"][aria-current="page"] {background:var(--surface-2); color:var(--text); font-weight:500;}
[data-testid="stNavSectionHeader"] {font-size:.6875rem !important; letter-spacing:.08em; text-transform:uppercase; color:var(--faint) !important;}

/* badges and empty state */
.badge {display:inline-block; font-size:.6875rem; font-weight:500; line-height:1; padding:.28rem .5rem; border-radius:4px;
  border:1px solid var(--line-2); color:var(--muted); letter-spacing:.01em; white-space:nowrap;}
.badge.accent {color:var(--accent-hi); border-color:rgba(61,165,138,.4); background:rgba(61,165,138,.08);}
.badge.warn {color:var(--warn); border-color:rgba(214,160,74,.4); background:rgba(214,160,74,.08);}
.badge.danger {color:var(--danger); border-color:rgba(229,72,77,.4); background:rgba(229,72,77,.08);}
.empty {border:1px dashed var(--line-2); border-radius:8px; padding:1.6rem 1.5rem; background:var(--surface);}
.empty-t {font-size:.95rem; font-weight:600; color:var(--text);}
.empty-d {font-size:.84rem; color:var(--muted); margin-top:.25rem; line-height:1.5;}

/* workflow steps and role chips */
.steps {display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:0; border:1px solid var(--line);
  border-radius:var(--r); background:var(--surface); overflow:hidden;}
.step {padding:.85rem 1rem; border-left:1px solid var(--line); min-width:0;}
.step:first-child {border-left:none;}
.step-n {font-size:.6875rem; letter-spacing:.08em; text-transform:uppercase; color:var(--faint); display:flex; gap:.45rem; align-items:center;}
.dot {width:7px; height:7px; border-radius:50%; background:var(--line-2); display:inline-block;}
.dot.done {background:var(--accent);} .dot.todo {background:var(--warn);} .dot.soon {background:var(--line-2);}
.step-t {font-size:.9rem; font-weight:600; color:var(--text); margin-top:.3rem;}
.step-d {font-size:.78rem; color:var(--muted); margin-top:.15rem; line-height:1.45;}
.chips {display:flex; flex-wrap:wrap; gap:.35rem; margin-top:.4rem;}
.chip {font-size:.75rem; padding:.22rem .5rem; border-radius:4px; background:var(--surface-2); border:1px solid var(--line); color:var(--text);}
.chip b {font-weight:500; color:var(--muted); margin-right:.35rem;}
[data-testid="stVerticalBlockBorderWrapper"] {border-color:var(--line) !important; border-radius:var(--r) !important;}

@media (max-width: 900px) {
  .block-container {padding:1.25rem 1rem 3rem;}
  .pg-title {font-size:1.2rem;}
}
@media (max-width: 640px) {
  div[data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) > [data-testid="stColumn"],
  div[data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) > [data-testid="column"] {
    border-left:none; border-top:1px solid var(--line);}
  div[data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) > :first-child {border-top:none;}
}
@media (prefers-reduced-motion: reduce) { * {transition:none !important;} }
</style>
"""


def configure_page() -> None:
    st.set_page_config(page_title=APP_NAME, layout="wide", initial_sidebar_state="expanded")
    st.markdown(_CSS, unsafe_allow_html=True)


def page_title(title: str, caption: str = "") -> None:
    sub = f'<p class="pg-sub">{escape(caption)}</p>' if caption else ""
    st.markdown(f'<div class="pg-head"><h1 class="pg-title">{escape(title)}</h1>{sub}</div>',
                unsafe_allow_html=True)


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
    empty_state("No dataset loaded", "Open Upload in the sidebar and add a CSV or Excel file, or load the sample data.")


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

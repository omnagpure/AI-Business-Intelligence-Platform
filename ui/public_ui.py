"""Design layer for the public pages (homepage, sign in, registration).

It extends the product's own tokens from ui.layout (near-black surfaces, hairline borders, one teal accent, Inter), so
visitors see the same visual language they will work in after signing in. Icons are inline SVG line icons drawn on a
24px grid, rendered in currentColor, so no icon font or emoji is needed.
"""
from __future__ import annotations

from datetime import date
from urllib.parse import quote

import streamlit as st

from config.settings import APP_NAME, APP_VERSION

_I = {
    "analysis": '<path d="M3 3v18h18"/><path d="M7 16v-4M11 16V9M15 16v-3"/><circle cx="18.5" cy="6.5" r="2.5"/><path d="m20.3 8.3 1.7 1.7"/>',
    "clean": '<path d="M12 3.5 13.7 8l4.5 1.7-4.5 1.7L12 16l-1.7-4.6L5.8 9.7 10.3 8z"/><path d="m18.5 14.5.7 1.8 1.8.7-1.8.7-.7 1.8-.7-1.8-1.8-.7 1.8-.7z"/>',
    "kpi": '<path d="M4 18a8 8 0 1 1 16 0"/><path d="m12 18 3.5-4.5"/><circle cx="12" cy="18" r="1.2"/><path d="M7.2 12.8 6 12M12 8V6.5M16.8 12.8 18 12"/>',
    "forecast": '<path d="M3 17.5 9 11.5l4 4 8-8"/><path d="M15 7.5h6v6"/>',
    "anomaly": '<path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/><path d="M12 9.5v4M12 17h.01"/>',
    "truck": '<path d="M1.5 5.5h12.5v10.5H1.5z"/><path d="M14 9h4l3.5 3.5V16H14"/><circle cx="5.5" cy="18" r="2"/><circle cx="17.5" cy="18" r="2"/>',
    "insight": '<path d="M9 18h6M10 21.5h4"/><path d="M12 2.5a6.5 6.5 0 0 0-3.8 11.8c.6.5.8 1.1.8 1.7h6c0-.6.2-1.2.8-1.7A6.5 6.5 0 0 0 12 2.5z"/>',
    "report": '<path d="M14 2.5H6.5a2 2 0 0 0-2 2v15a2 2 0 0 0 2 2h11a2 2 0 0 0 2-2V8z"/><path d="M14 2.5V8h5.5M8.5 13h7M8.5 17h4.5"/>',
    "network": '<circle cx="5" cy="6" r="2.5"/><circle cx="19" cy="6" r="2.5"/><circle cx="12" cy="18" r="2.5"/><path d="M7.5 6h9M6.3 8.2l4.4 7.6M17.7 8.2l-4.4 7.6"/>',
    "boxes": '<path d="M21 8 12 3 3 8v8l9 5 9-5z"/><path d="m3 8 9 5 9-5M12 13v8"/>',
    "people": '<circle cx="9" cy="8" r="3.5"/><path d="M2.5 20a6.5 6.5 0 0 1 13 0"/><path d="M16 4.6a3.5 3.5 0 0 1 0 6.8M18 14.2a6.5 6.5 0 0 1 3.5 5.8"/>',
    "sliders": '<path d="M4 21v-7M4 10V3M12 21v-9M12 8V3M20 21v-5M20 12V3M1.5 14h5M9.5 8h5M17.5 16h5"/>',
    "pulse": '<path d="M22 12h-4l-3 8.5L9 3.5 6 12H2"/>',
    "upload": '<path d="M21 15v3.5a2.5 2.5 0 0 1-2.5 2.5h-13A2.5 2.5 0 0 1 3 18.5V15"/><path d="m17 8-5-5-5 5M12 3v12"/>',
    "cpu": '<rect x="5" y="5" width="14" height="14" rx="2"/><rect x="9" y="9" width="6" height="6" rx=".5"/><path d="M9 1.5V5M15 1.5V5M9 19v3.5M15 19v3.5M1.5 9H5M1.5 15H5M19 9h3.5M19 15h3.5"/>',
    "target": '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1.2"/>',
    "shield": '<path d="M12 21.5s8-3.8 8-10V5.2L12 2.5 4 5.2v6.3c0 6.2 8 10 8 10z"/><path d="m9 12 2 2 4-4"/>',
    "check": '<path d="M20 6.5 9.5 17 4 11.5"/>',
    "x": '<path d="M18 6 6 18M6 6l12 12"/>',
    "arrow": '<path d="M5 12h14M13 6l6 6-6 6"/>',
    "eye": '<path d="M2 12s3.6-7 10-7 10 7 10 7-3.6 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/>',
    "lock": '<rect x="4" y="10.5" width="16" height="11" rx="2"/><path d="M8 10.5V7a4 4 0 0 1 8 0v3.5"/>',
    "server": '<rect x="3" y="3.5" width="18" height="7" rx="1.5"/><rect x="3" y="13.5" width="18" height="7" rx="1.5"/><path d="M7 7h.01M7 17h.01"/>',
}


def _mask(svg: str, size_w: int, size_h: int, cls: str) -> str:
    """An SVG drawn as a CSS mask filled with currentColor. st.html sanitises with DOMPurify's HTML profile, which removes
    inline <svg>, so icons travel as data URIs inside a style attribute instead."""
    uri = "data:image/svg+xml;utf8," + quote(svg, safe="")
    return (f'<span class="{cls}" aria-hidden="true" style="width:{size_w}px;height:{size_h}px;'
            f"-webkit-mask-image:url('{uri}');mask-image:url('{uri}')\"></span>")


def icon(name: str, size: int = 20) -> str:
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="#000" stroke-width="1.6" '
           f'stroke-linecap="round" stroke-linejoin="round">{_I[name]}</svg>')
    return _mask(svg, size, size, "ico")


_MARK_SVG = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 22 22"><rect x="1" y="11" width="4.5" height="10" rx="1"/>'
             '<rect x="8.5" y="6" width="4.5" height="15" rx="1" opacity=".75"/>'
             '<rect x="16" y="1" width="4.5" height="20" rx="1" opacity=".5"/></svg>')
MARK = _mask(_MARK_SVG, 22, 22, "ico mark")


def brand(href: str = "./") -> str:
    return f'<a class="pb-brand" href="{href}" target="_self" aria-label="{APP_NAME} home">{MARK}<span>AI BI Platform</span></a>'


_CSS = """
<style>
/* ---- public shell: no sidebar, no toolbar, wider canvas ---- */
section[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="stExpandSidebarButton"],
header[data-testid="stHeader"] {display:none !important;}
.block-container {max-width:1200px; padding:1.1rem 2rem 0 !important;}
[data-testid="stMain"], [data-testid="stAppViewContainer"] {scroll-behavior:smooth;}
.stApp {background:var(--bg);}
.stButton > button, .stFormSubmitButton > button {min-height:2.6rem; border-radius:8px; font-size:.9rem; padding:0 1.1rem;}
.stFormSubmitButton > button[kind="primaryFormSubmit"] {background:var(--accent); border:1px solid var(--accent); color:#07120E; font-weight:600;}
.stFormSubmitButton > button[kind="primaryFormSubmit"]:hover {background:var(--accent-hi); border-color:var(--accent-hi); color:#07120E;}
.stButton > button[kind="tertiary"] {min-height:auto; padding:0; border:none; color:var(--accent-hi); font-size:.84rem; background:transparent;}
.stButton > button[kind="tertiary"]:hover {color:var(--text); background:transparent; text-decoration:underline; text-underline-offset:3px;}

.pb-brand {display:inline-flex; align-items:center; gap:.6rem; text-decoration:none !important; color:var(--text) !important;
  font-weight:600; font-size:1rem; letter-spacing:-.015em;}
.ico {display:inline-block; flex:none; background-color:currentColor; -webkit-mask-repeat:no-repeat; mask-repeat:no-repeat;
  -webkit-mask-size:contain; mask-size:contain; -webkit-mask-position:center; mask-position:center; vertical-align:middle;}
.ico.mark {color:var(--accent);}

/* links styled as buttons (public pages navigate with real links, so they work without any script) */
.pb-btn {display:inline-flex; align-items:center; justify-content:center; gap:.5rem; height:2.75rem; padding:0 1.25rem;
  border-radius:8px; font-size:.92rem; font-weight:600; text-decoration:none !important; white-space:nowrap;
  border:1px solid var(--line-2); color:var(--text) !important; background:transparent;
  transition:background-color .18s ease, border-color .18s ease, transform .18s ease;}
.pb-btn:hover {background:var(--surface-2); border-color:var(--faint);}
.pb-btn.primary {background:var(--accent); border-color:var(--accent); color:#07120E !important;}
.pb-btn.primary:hover {background:var(--accent-hi); border-color:var(--accent-hi);}
.pb-btn.sm {height:2.35rem; padding:0 1rem; font-size:.86rem;}
.pb-btn.ghost {border-color:transparent;}
.pb-btn.ghost:hover {background:var(--surface-2); border-color:transparent;}
.pb-btn .ico {transition:transform .18s ease;}
.pb-btn:hover .ico {transform:translateX(2px);}
.pb-btn:focus-visible, .lp-links a:focus-visible, .pb-brand:focus-visible, .lp-foot a:focus-visible {outline:2px solid var(--accent); outline-offset:3px;}

/* ---- top navigation ---- */
.lp-nav {display:flex; align-items:center; justify-content:space-between; gap:1.5rem; padding:.35rem 0 1rem;
  border-bottom:1px solid var(--line);}
.lp-links {display:flex; gap:2rem;}
.lp-links a {color:var(--muted) !important; text-decoration:none !important; font-size:.875rem; font-weight:500; transition:color .15s;}
.lp-links a:hover {color:var(--text) !important;}
.lp-cta {display:flex; gap:.5rem; align-items:center;}

/* ---- hero ---- */
.lp-hero {display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1.12fr); gap:4rem; align-items:center; padding:5rem 0 1rem; scroll-margin-top:2rem;}
.lp-pill {display:inline-flex; align-items:center; gap:.55rem; font-size:.78rem; font-weight:500; color:var(--muted);
  border:1px solid var(--line-2); border-radius:999px; padding:.28rem .8rem .28rem .3rem; background:var(--surface);}
.lp-pill b {background:rgba(61,165,138,.14); color:var(--accent-hi); font-weight:600; border-radius:999px; padding:.14rem .55rem; font-size:.72rem;}
.lp-h1 {font-size:clamp(2.35rem,4.4vw,3.5rem); line-height:1.06; letter-spacing:-.04em; font-weight:650; color:var(--text); margin:1.4rem 0 0;}
.lp-h1 span {color:var(--accent-hi);}
.lp-lead {font-size:1.075rem; line-height:1.7; color:var(--muted); margin:1.4rem 0 0; max-width:33rem;}
.lp-actions {display:flex; gap:.75rem; flex-wrap:wrap; margin-top:2.1rem;}
.lp-trust {display:flex; gap:1.4rem; flex-wrap:wrap; color:var(--faint); font-size:.8rem; margin-top:2.2rem;}
.lp-trust span {display:inline-flex; gap:.45rem; align-items:center;}
.lp-trust .ico {color:var(--accent);}

/* ---- product preview (rendered in the app's own visual language) ---- */
.pv-wrap {position:relative;}
.pv-wrap::before {content:""; position:absolute; inset:-12% -8% -10% -8%; z-index:0; pointer-events:none;
  background:radial-gradient(closest-side, rgba(61,165,138,.13), transparent 72%);}
.pv {position:relative; z-index:1; border:1px solid var(--line-2); border-radius:12px; background:var(--surface); overflow:hidden;
  box-shadow:0 40px 90px -30px rgba(0,0,0,.75), 0 0 0 1px rgba(255,255,255,.02);}
.pv-bar {display:flex; align-items:center; gap:.4rem; padding:.62rem .85rem; border-bottom:1px solid var(--line); background:var(--bg);}
.pv-bar i {width:9px; height:9px; border-radius:50%; background:var(--line-2);}
.pv-bar span {margin-left:.7rem; font-size:.7rem; color:var(--faint);}
.pv-body {display:grid; grid-template-columns:128px minmax(0,1fr);}
.pv-side {border-right:1px solid var(--line); padding:.7rem .55rem; background:var(--bg); display:flex; flex-direction:column; gap:.1rem;}
.pv-sec {font-size:.6rem; color:var(--faint); margin:.55rem .5rem .2rem; letter-spacing:.02em;}
.pv-sec:first-child {margin-top:.1rem;}
.pv-nav {font-size:.7rem; color:var(--faint); padding:.34rem .5rem; border-radius:5px;}
.pv-nav.on {background:var(--surface-2); color:var(--text); box-shadow:inset 2px 0 0 var(--accent); font-weight:500;}
.pv-main {padding:.9rem 1rem 1rem; display:flex; flex-direction:column; gap:.7rem; min-width:0;}
.pv-ctx {font-size:.68rem; color:var(--muted); display:flex; gap:.45rem; align-items:center; flex-wrap:wrap;}
.pv-ctx b {color:var(--text); font-weight:600;}
.pv-dot {width:6px; height:6px; border-radius:50%; background:var(--accent);}
.pv-stats {display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); border:1px solid var(--line); border-radius:6px; overflow:hidden; background:var(--bg);}
.pv-stat {padding:.55rem .7rem; border-left:1px solid var(--line); min-width:0;}
.pv-stat:first-child {border-left:none;}
.pv-k {font-size:.6rem; color:var(--muted); white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}
.pv-v {font-size:1.02rem; font-weight:600; color:var(--text); letter-spacing:-.015em; margin-top:.12rem; font-variant-numeric:tabular-nums;}
.pv-d {font-size:.58rem; margin-top:.1rem; color:var(--accent-hi);}
.pv-d.neg {color:var(--warn);}
.pv-card {border:1px solid var(--line); border-radius:6px; padding:.6rem .7rem; background:var(--bg); min-width:0;}
.pv-ct {font-size:.68rem; font-weight:600; color:var(--text); display:flex; justify-content:space-between; gap:.5rem;}
.pv-ct span {font-weight:400; color:var(--faint);}
.pv-chart {display:block; width:100%; height:auto; margin-top:.4rem;}
.pv-leg {display:flex; gap:.9rem; font-size:.58rem; color:var(--faint); margin-top:.2rem;}
.pv-leg i {display:inline-block; width:10px; height:2px; background:var(--accent); vertical-align:middle; margin-right:.3rem;}
.pv-leg i.f {background:none; border-top:2px dashed var(--accent-hi);}
.pv-row {display:grid; grid-template-columns:minmax(0,1.15fr) minmax(0,1fr); gap:.7rem;}
.pv-bars {display:flex; flex-direction:column; gap:.42rem; margin-top:.55rem;}
.pv-b {display:grid; grid-template-columns:58px minmax(0,1fr) 30px; gap:.45rem; align-items:center; font-size:.6rem; color:var(--muted);}
.pv-b em {font-style:normal; text-align:right; font-variant-numeric:tabular-nums; color:var(--text);}
.pv-track {height:6px; border-radius:3px; background:var(--surface-2); overflow:hidden;}
.pv-track i {display:block; height:100%; background:var(--accent); border-radius:3px;}
.pv-track i.w {background:var(--warn);}
.pv-ins {display:flex; flex-direction:column; gap:.5rem; margin-top:.55rem;}
.pv-in {font-size:.6rem; color:var(--muted); line-height:1.45; padding-left:.55rem; border-left:2px solid var(--warn);}
.pv-in.ok {border-left-color:var(--accent);}
.pv-in b {color:var(--text); font-weight:600; display:block; font-size:.64rem;}
.pv-cap {position:relative; z-index:1; font-size:.72rem; color:var(--faint); text-align:center; margin-top:.85rem;}

/* ---- sections ---- */
.lp-sec {padding:7rem 0 0; scroll-margin-top:1.5rem;}
.lp-eyebrow {font-size:.75rem; font-weight:600; color:var(--accent-hi); letter-spacing:.09em; text-transform:uppercase;}
h2.lp-h2 {font-size:clamp(1.75rem,3vw,2.35rem) !important; font-weight:650 !important; letter-spacing:-.032em; line-height:1.15;
  color:var(--text); margin:.8rem 0 0 !important; padding:0 !important; max-width:42rem;}
h1.lp-h1 {padding:0 !important;}
.lp-sub {font-size:1rem; color:var(--muted); line-height:1.7; margin:1rem 0 0; max-width:40rem;}
.lp-grid {display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:1px; background:var(--line); border:1px solid var(--line);
  border-radius:14px; overflow:hidden; margin-top:3rem;}
.lp-card {background:var(--surface); padding:1.6rem 1.45rem 1.75rem; transition:background-color .2s ease;}
.lp-card:hover {background:var(--surface-2);}
.lp-ic {width:40px; height:40px; border-radius:10px; display:grid; place-items:center; border:1px solid var(--line-2);
  background:var(--bg); color:var(--accent-hi); transition:border-color .2s ease;}
.lp-card:hover .lp-ic, .lp-sol:hover .lp-ic, .lp-step:hover .lp-ic {border-color:rgba(61,165,138,.55);}
.lp-ct {font-size:1rem; font-weight:600; color:var(--text); margin-top:1.15rem; letter-spacing:-.012em;}
.lp-cd {font-size:.875rem; color:var(--muted); line-height:1.62; margin-top:.5rem;}

.lp-split {display:grid; grid-template-columns:minmax(0,.85fr) minmax(0,1.15fr); gap:5rem; align-items:start;}
.lp-sols {border-bottom:1px solid var(--line);}
.lp-sol {display:grid; grid-template-columns:auto minmax(0,1fr); gap:1.25rem; padding:1.6rem 0; border-top:1px solid var(--line);}
.lp-sol .lp-ct {margin-top:.1rem;}
.lp-tags {display:flex; flex-wrap:wrap; gap:.4rem; margin-top:.8rem;}
.lp-tags span {font-size:.72rem; color:var(--muted); border:1px solid var(--line-2); border-radius:999px; padding:.18rem .6rem;}

.lp-steps {display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:1.25rem; margin-top:3rem; counter-reset:step;}
.lp-step {position:relative; border:1px solid var(--line); border-radius:14px; background:var(--surface); padding:1.75rem 1.6rem 1.9rem;
  transition:border-color .2s ease, transform .2s ease;}
.lp-step:hover {border-color:var(--line-2); transform:translateY(-2px);}
.lp-step-top {display:flex; justify-content:space-between; align-items:center;}
.lp-num {font-size:.8rem; font-weight:600; color:var(--faint); font-variant-numeric:tabular-nums; letter-spacing:.04em;}

.lp-princ {display:flex; flex-direction:column; gap:1.1rem;}
.lp-pr {display:grid; grid-template-columns:auto minmax(0,1fr); gap:1rem; padding:1.25rem 1.3rem; border:1px solid var(--line);
  border-radius:12px; background:var(--surface);}
.lp-pr .lp-ct {margin-top:.05rem;}
.lp-pr .lp-cd {margin-top:.3rem;}

.lp-band {margin-top:7rem; border:1px solid var(--line-2); border-radius:16px; padding:3.5rem 2rem; text-align:center; background:var(--surface);
  position:relative; overflow:hidden;}
.lp-band::before {content:""; position:absolute; inset:0; pointer-events:none;
  background:radial-gradient(60% 120% at 50% 0%, rgba(61,165,138,.10), transparent 70%);}
.lp-band > * {position:relative;}
.lp-band h2.lp-h2, .lp-band .lp-sub {margin-left:auto !important; margin-right:auto !important;}
.lp-band .lp-actions {justify-content:center;}

/* ---- footer ---- */
.lp-foot {margin-top:5rem; border-top:1px solid var(--line); padding:3rem 0 2.5rem;}
.lp-foot-grid {display:grid; grid-template-columns:minmax(0,1.6fr) repeat(3,minmax(0,1fr)); gap:2.5rem;}
.lp-foot p {font-size:.84rem; color:var(--muted); line-height:1.65; margin:.9rem 0 0; max-width:22rem;}
.lp-foot h4 {font-size:.78rem !important; font-weight:600 !important; color:var(--text); margin:0 0 .9rem !important; padding:0 !important; letter-spacing:.01em;}
.lp-foot ul {list-style:none; margin:0; padding:0; display:flex; flex-direction:column; gap:.6rem;}
.lp-foot li {margin:0; padding:0;}
.lp-foot a {font-size:.84rem; color:var(--muted) !important; text-decoration:none !important; transition:color .15s;}
.lp-foot a:hover {color:var(--text) !important;}
.lp-legal {display:flex; justify-content:space-between; gap:1rem; flex-wrap:wrap; margin-top:2.75rem; padding-top:1.5rem;
  border-top:1px solid var(--line); font-size:.78rem; color:var(--faint);}

/* ---- sign in / registration ---- */
.au-top {display:flex; justify-content:space-between; align-items:center; padding:.35rem 0 1rem;}
.au-top a.au-back {font-size:.84rem; color:var(--muted) !important; text-decoration:none !important;}
.au-top a.au-back:hover {color:var(--text) !important;}
.au-h {font-size:1.45rem; font-weight:650; letter-spacing:-.025em; color:var(--text); line-height:1.25;}
.au-s {font-size:.9rem; color:var(--muted); margin-top:.4rem; line-height:1.55;}
.st-key-auth_card {background:var(--surface); border:1px solid var(--line-2) !important; border-radius:14px !important;
  padding:2rem 2rem 1.6rem !important; box-shadow:0 30px 80px -40px rgba(0,0,0,.8);}
.st-key-auth_shell {padding-top:3.5rem;}
.au-meter {display:grid; grid-template-columns:repeat(4,1fr); gap:4px; margin:-.35rem 0 .35rem;}
.au-meter i {height:4px; border-radius:2px; background:var(--line-2); transition:background-color .2s;}
.au-meter.s1 i:nth-child(-n+1) {background:var(--danger);}
.au-meter.s2 i:nth-child(-n+2) {background:var(--warn);}
.au-meter.s3 i:nth-child(-n+3) {background:var(--accent);}
.au-meter.s4 i {background:var(--accent-hi);}
.au-mlabel {font-size:.75rem; color:var(--muted); display:flex; justify-content:space-between;}
.au-reqs {list-style:none; margin:.55rem 0 .2rem; padding:0; display:flex; flex-direction:column; gap:.3rem;}
.au-reqs li {display:flex; gap:.5rem; align-items:center; font-size:.78rem; color:var(--faint); margin:0;}
.au-reqs li.ok {color:var(--muted);}
.au-reqs li.ok .ico {color:var(--accent-hi);}
.au-foot {text-align:center; font-size:.78rem; color:var(--faint); margin-top:1.25rem;}
.auth-redirect {display:flex; flex-direction:column; align-items:center; gap:.9rem; padding:8rem 0; color:var(--muted); font-size:.9rem;}
.auth-redirect a {font-size:.8rem; color:var(--accent-hi) !important;}
.auth-spinner {width:22px; height:22px; border-radius:50%; border:2px solid var(--line-2); border-top-color:var(--accent); animation:au-spin .8s linear infinite;}
@keyframes au-spin {to {transform:rotate(360deg);}}

/* ---- responsive ---- */
@media (max-width: 1000px) {
  .lp-grid {grid-template-columns:repeat(2,minmax(0,1fr));}
  .lp-hero {grid-template-columns:minmax(0,1fr); gap:3.5rem; padding-top:3.5rem;}
  .lp-split {grid-template-columns:minmax(0,1fr); gap:2.5rem;}
  .lp-foot-grid {grid-template-columns:repeat(3,minmax(0,1fr));}
  .lp-foot-brand {grid-column:1 / -1;}
}
@media (max-width: 760px) {
  .block-container {padding:.9rem 1rem 0 !important;}
  .lp-links {display:none;}
  .lp-steps {grid-template-columns:minmax(0,1fr);}
  .lp-sec {padding-top:5rem;}
  .lp-band {margin-top:5rem; padding:2.5rem 1.25rem;}
  .st-key-auth_shell {padding-top:1.5rem;}
  .st-key-auth_card {padding:1.5rem 1.25rem 1.25rem !important;}
}
@media (max-width: 560px) {
  .lp-grid {grid-template-columns:minmax(0,1fr);}
  .lp-cta .pb-btn.ghost {display:none;}
  .pv-body {grid-template-columns:minmax(0,1fr);}
  .pv-side {display:none;}
  .pv-stats {grid-template-columns:repeat(2,minmax(0,1fr));}
  .pv-stat:nth-child(3) {border-left:none;}
  .pv-stat:nth-child(n+3) {border-top:1px solid var(--line);}
  .pv-row {grid-template-columns:minmax(0,1fr);}
  .lp-foot-grid {grid-template-columns:repeat(2,minmax(0,1fr));}
}
@media (prefers-reduced-motion: reduce) {
  [data-testid="stMain"], [data-testid="stAppViewContainer"] {scroll-behavior:auto;}
  .auth-spinner {animation:none;}
  .lp-step:hover {transform:none;}
}
</style>
"""


def apply_public_theme() -> None:
    st.html(_CSS)


def footer() -> None:
    year = date.today().year
    st.html(f"""
<footer class="lp-foot">
  <div class="lp-foot-grid">
    <div class="lp-foot-brand">{brand()}
      <p>Business intelligence for operational datasets: profiling, cleaning, KPIs, forecasting, anomaly detection and
      reporting, computed from the data you upload.</p></div>
    <div><h4>Platform</h4><ul><li><a href="./#features" target="_self">Features</a></li><li><a href="./#solutions" target="_self">Solutions</a></li>
      <li><a href="./#how" target="_self">How it works</a></li></ul></div>
    <div><h4>Company</h4><ul><li><a href="./#about" target="_self">About</a></li></ul></div>
    <div><h4>Account</h4><ul><li><a href="login" target="_self">Sign in</a></li><li><a href="register" target="_self">Create account</a></li></ul></div>
  </div>
  <div class="lp-legal"><span>&copy; {year} {APP_NAME}. All rights reserved.</span><span>Version {APP_VERSION}</span></div>
</footer>""")

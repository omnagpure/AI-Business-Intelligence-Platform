"""Design layer for the public pages (homepage, sign in, registration).

Uses the same colour tokens as the app (ui.theme), so the public pages follow the light/dark choice too. Icons are line
icons on a 24px grid drawn in currentColor; the logo is the Tessaly mark from assets/.
"""
from __future__ import annotations

import base64
from datetime import date
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote

import streamlit as st

from config.settings import APP_NAME, APP_VERSION
from ui import theme

ASSETS = Path(__file__).resolve().parent.parent / "assets"

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
    "sun": '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>',
    "moon": '<path d="M20.5 14.5A8.5 8.5 0 0 1 9.5 3.5a8.5 8.5 0 1 0 11 11z"/>',
    "monitor": '<rect x="2.5" y="4" width="19" height="13" rx="2"/><path d="M8 21h8M12 17v4"/>',
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


@lru_cache(maxsize=16)
def data_uri(name: str) -> str:
    """An asset as a data URI (st.html cannot load files from the app's folder)."""
    path = ASSETS / name
    kind = {".svg": "image/svg+xml", ".png": "image/png", ".webp": "image/webp", ".jpg": "image/jpeg"}[path.suffix]
    return f"data:{kind};base64," + base64.b64encode(path.read_bytes()).decode()


def themed_img(stem: str, ext: str, alt: str, cls: str = "", **attrs: str) -> str:
    """The light and dark variants of an asset; CSS shows the one matching the appearance in use."""
    extra = "".join(f' {k.replace("_", "-")}="{v}"' for k, v in attrs.items())
    return "".join(f'<img class="{cls} only-{tone}" src="{data_uri(f"{stem}-{tone}.{ext}")}" alt="{alt if tone == "light" else ""}"{extra}>'
                   for tone in ("light", "dark"))


def brand(href: str = "./") -> str:
    return (f'<a class="pb-brand" href="{href}" target="_self" aria-label="{APP_NAME} home">'
            f'{themed_img("mark", "svg", "", "pb-mark", width="22", height="22")}<span>{APP_NAME}</span></a>')


def theme_switch() -> str:
    """Three small buttons: light, dark, follow the system. Made live by theme.switch_script."""
    current = theme.mode()
    items = [("light", "sun", "Light"), ("dark", "moon", "Dark"), ("system", "monitor", "Use system setting")]
    return ('<div class="pb-theme" role="group" aria-label="Appearance">' + "".join(
        f'<a href="#" role="button" data-mode="{m}" class="{"on" if m == current else ""}" title="{t}" aria-label="{t}"'
        f' aria-pressed="{"true" if m == current else "false"}">{icon(i, 15)}</a>' for m, i, t in items) + "</div>")


def _visibility_css() -> str:
    """Show the light or dark variant of a themed image."""
    light, dark = ".only-dark{display:none !important;}", ".only-light{display:none !important;}"
    m = theme.mode()
    if m == "system":
        return light + "@media (prefers-color-scheme: dark){.only-dark{display:revert !important;}" + dark + "}"
    return dark if m == "dark" else light


_CSS = """
/* ---- public shell: no sidebar, no toolbar, wider canvas ---- */
section[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="stExpandSidebarButton"],
header[data-testid="stHeader"] {display:none !important;}
.block-container {max-width:1160px; padding:1.1rem 2rem 0 !important;}
[data-testid="stMain"], [data-testid="stAppViewContainer"] {scroll-behavior:smooth;}
.stButton > button, .stFormSubmitButton > button {min-height:2.6rem; font-size:.9rem; padding:0 1.1rem;}
.stButton > button[kind="tertiary"] {min-height:auto; padding:0; font-size:.84rem;}

.pb-brand {display:inline-flex; align-items:center; gap:.55rem; text-decoration:none !important; color:var(--text) !important;
  font-weight:600; font-size:1.05rem; letter-spacing:-.02em;}
.pb-mark {display:block; width:22px; height:22px;}
.ico {display:inline-block; flex:none; background-color:currentColor; -webkit-mask-repeat:no-repeat; mask-repeat:no-repeat;
  -webkit-mask-size:contain; mask-size:contain; -webkit-mask-position:center; mask-position:center; vertical-align:middle;}

/* links styled as buttons (public pages navigate with real links, so they work without any script) */
.pb-btn {display:inline-flex; align-items:center; justify-content:center; gap:.5rem; height:2.7rem; padding:0 1.2rem;
  border-radius:8px; font-size:.92rem; font-weight:600; text-decoration:none !important; white-space:nowrap;
  border:1px solid var(--line-2); color:var(--text) !important; background:var(--surface);
  transition:background-color .15s ease, border-color .15s ease;}
.pb-btn:hover {background:var(--surface-2); border-color:var(--faint);}
.pb-btn.primary {background:var(--accent); border-color:var(--accent); color:var(--on-accent) !important;}
.pb-btn.primary:hover {background:var(--accent-hi); border-color:var(--accent-hi);}
.pb-btn.sm {height:2.25rem; padding:0 .95rem; font-size:.86rem;}
.pb-btn.ghost {border-color:transparent; background:transparent;}
.pb-btn.ghost:hover {background:var(--surface-2);}
.pb-btn:focus-visible, .lp-links a:focus-visible, .pb-brand:focus-visible, .lp-foot a:focus-visible, .pb-theme a:focus-visible {
  outline:2px solid var(--accent); outline-offset:3px;}

/* appearance switch */
.pb-theme {display:inline-flex; gap:2px; padding:2px; border:1px solid var(--line); border-radius:8px; background:var(--surface);}
.pb-theme a {display:grid; place-items:center; width:28px; height:26px; border-radius:6px; color:var(--faint) !important;}
.pb-theme a:hover {color:var(--text) !important;}
.pb-theme a.on {background:var(--surface-2); color:var(--text) !important;}

/* ---- top navigation ---- */
.lp-nav {display:flex; align-items:center; justify-content:space-between; gap:1.5rem; padding:.35rem 0 1rem;}
.lp-links {display:flex; gap:1.75rem;}
.lp-links a {color:var(--muted) !important; text-decoration:none !important; font-size:.875rem; font-weight:500;}
.lp-links a:hover {color:var(--text) !important;}
.lp-cta {display:flex; gap:.5rem; align-items:center;}

/* ---- hero: copy on top, the real product below ---- */
.lp-hero {padding:4.5rem 0 0; scroll-margin-top:2rem;}
.lp-h1 {font-size:clamp(2.3rem,5vw,3.9rem); line-height:1.04; letter-spacing:-.045em; font-weight:650; color:var(--text);
  margin:0; max-width:15ch;}
h1.lp-h1 {padding:0 !important;}
.lp-hero-row {display:grid; grid-template-columns:minmax(0,1.25fr) minmax(0,1fr); gap:3rem; align-items:end;}
.lp-lead {font-size:1.06rem; line-height:1.65; color:var(--muted); margin:0; max-width:31rem;}
.lp-actions {display:flex; gap:.6rem; flex-wrap:wrap; margin-top:1.6rem;}
.lp-shot {margin-top:3.5rem; border:1px solid var(--line-2); border-radius:12px; overflow:hidden; background:var(--surface);
  box-shadow:var(--shadow-lg);}
.lp-shot img {display:block; width:100%; height:auto;}
.lp-cap {font-size:.78rem; color:var(--faint); margin-top:.8rem;}

/* ---- sections: a heading column and a content column ---- */
.lp-sec {display:grid; grid-template-columns:minmax(0,.8fr) minmax(0,1.6fr); gap:3rem; padding:5.5rem 0 0; scroll-margin-top:1.5rem;}
h2.lp-h2 {font-size:clamp(1.5rem,2.6vw,2rem) !important; font-weight:650 !important; letter-spacing:-.03em; line-height:1.15;
  color:var(--text); margin:0 !important; padding:0 !important;}
.lp-sub {font-size:.95rem; color:var(--muted); line-height:1.65; margin:.9rem 0 0;}
.lp-list {display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); column-gap:2.5rem; border-top:1px solid var(--line);}
.lp-item {display:grid; grid-template-columns:auto minmax(0,1fr); gap:.9rem; padding:1.15rem 0; border-bottom:1px solid var(--line);}
.lp-item .ico {color:var(--accent); margin-top:.1rem;}
.lp-ct {font-size:.95rem; font-weight:600; color:var(--text); letter-spacing:-.01em;}
.lp-cd {font-size:.86rem; color:var(--muted); line-height:1.58; margin-top:.25rem;}
.lp-tags {font-size:.78rem; color:var(--faint); margin-top:.45rem;}
.lp-steps {display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:2rem; border-top:1px solid var(--line); padding-top:1.3rem;}
.lp-num {font-size:.82rem; font-weight:600; color:var(--accent); font-variant-numeric:tabular-nums;}
.lp-steps .lp-ct {margin-top:.45rem;}
.lp-about p {font-size:.95rem; color:var(--muted); line-height:1.7; margin:0 0 1rem; max-width:40rem;}
.lp-about p b {color:var(--text); font-weight:600;}

/* ---- footer ---- */
.lp-foot {margin-top:6rem; border-top:1px solid var(--line); padding:2.5rem 0 2.25rem;}
.lp-foot-grid {display:grid; grid-template-columns:minmax(0,1.6fr) repeat(3,minmax(0,1fr)); gap:2.5rem;}
.lp-foot p {font-size:.84rem; color:var(--muted); line-height:1.65; margin:.8rem 0 0; max-width:22rem;}
.lp-foot h4 {font-size:.8rem !important; font-weight:600 !important; color:var(--text); margin:0 0 .8rem !important; padding:0 !important;}
.lp-foot ul {list-style:none; margin:0; padding:0; display:flex; flex-direction:column; gap:.55rem;}
.lp-foot li {margin:0; padding:0;}
.lp-foot a {font-size:.84rem; color:var(--muted) !important; text-decoration:none !important;}
.lp-foot a:hover {color:var(--text) !important;}
.lp-legal {display:flex; justify-content:space-between; align-items:center; gap:1rem; flex-wrap:wrap; margin-top:2.5rem; padding-top:1.4rem;
  border-top:1px solid var(--line); font-size:.78rem; color:var(--faint);}

/* ---- sign in / registration ---- */
.au-top {display:flex; justify-content:space-between; align-items:center; gap:1rem; padding:.35rem 0 1rem;}
.au-top-r {display:flex; align-items:center; gap:1rem;}
.au-top a.au-back {font-size:.84rem; color:var(--muted) !important; text-decoration:none !important;}
.au-top a.au-back:hover {color:var(--text) !important;}
.au-h {font-size:1.4rem; font-weight:650; letter-spacing:-.025em; color:var(--text); line-height:1.25;}
.au-s {font-size:.9rem; color:var(--muted); margin-top:.35rem; line-height:1.55;}
.st-key-auth_card {background:var(--surface); border:1px solid var(--line) !important; border-radius:14px !important;
  padding:2rem 2rem 1.6rem !important; box-shadow:var(--shadow-lg);}
.st-key-auth_shell {padding-top:3rem;}
.au-meter {display:grid; grid-template-columns:repeat(4,1fr); gap:4px; margin:-.35rem 0 .35rem;}
.au-meter i {height:4px; border-radius:2px; background:var(--line-2); transition:background-color .2s;}
.au-meter.s1 i:nth-child(-n+1) {background:var(--danger);}
.au-meter.s2 i:nth-child(-n+2) {background:var(--warn);}
.au-meter.s3 i:nth-child(-n+3) {background:var(--ok);}
.au-meter.s4 i {background:var(--ok);}
.au-mlabel {font-size:.75rem; color:var(--muted); display:flex; justify-content:space-between;}
.au-reqs {list-style:none; margin:.55rem 0 .2rem; padding:0; display:flex; flex-direction:column; gap:.3rem;}
.au-reqs li {display:flex; gap:.5rem; align-items:center; font-size:.78rem; color:var(--faint); margin:0;}
.au-reqs li.ok {color:var(--muted);}
.au-reqs li.ok .ico {color:var(--ok);}
.au-foot {text-align:center; font-size:.78rem; color:var(--faint); margin-top:1.25rem;}
.auth-redirect {display:flex; flex-direction:column; align-items:center; gap:.9rem; padding:8rem 0; color:var(--muted); font-size:.9rem;}
.auth-redirect a {font-size:.8rem; color:var(--accent) !important;}
.auth-spinner {width:22px; height:22px; border-radius:50%; border:2px solid var(--line-2); border-top-color:var(--accent); animation:au-spin .8s linear infinite;}
@keyframes au-spin {to {transform:rotate(360deg);}}

/* ---- responsive ---- */
@media (max-width: 1000px) {
  .lp-hero-row, .lp-sec {grid-template-columns:minmax(0,1fr); gap:1.5rem;}
  .lp-hero {padding-top:3rem;}
  .lp-sec {padding-top:4rem;}
  .lp-foot-grid {grid-template-columns:repeat(3,minmax(0,1fr));}
  .lp-foot-brand {grid-column:1 / -1;}
}
@media (max-width: 760px) {
  .block-container {padding:.9rem 1rem 0 !important;}
  .lp-links {display:none;}
  .lp-list, .lp-steps {grid-template-columns:minmax(0,1fr);}
  .lp-steps {gap:1.25rem;}
  .lp-shot {margin-top:2.25rem;}
  .st-key-auth_shell {padding-top:1.25rem;}
  .st-key-auth_card {padding:1.5rem 1.25rem 1.25rem !important;}
}
@media (max-width: 560px) {
  .lp-cta .pb-btn.ghost, .lp-nav .pb-theme {display:none;}
  .lp-foot-grid {grid-template-columns:repeat(2,minmax(0,1fr));}
}
@media (prefers-reduced-motion: reduce) {
  [data-testid="stMain"], [data-testid="stAppViewContainer"] {scroll-behavior:auto;}
  .auth-spinner {animation:none;}
}
"""


def apply_public_theme() -> None:
    st.html(f"<style>{_CSS}{_visibility_css()}</style>")
    st.html(theme.switch_script(".pb-theme a[data-mode]"), unsafe_allow_javascript=True)


def footer() -> None:
    year = date.today().year
    st.html(f"""
<footer class="lp-foot">
  <div class="lp-foot-grid">
    <div class="lp-foot-brand">{brand()}
      <p>Business intelligence for operational datasets: profiling, cleaning, KPIs, forecasting, anomaly detection and
      reporting, computed from the data you upload.</p></div>
    <div><h4>Product</h4><ul><li><a href="#lp-features">Features</a></li><li><a href="#lp-solutions">Solutions</a></li>
      <li><a href="#lp-how">How it works</a></li></ul></div>
    <div><h4>Company</h4><ul><li><a href="#lp-about">About</a></li></ul></div>
    <div><h4>Account</h4><ul><li><a href="login" target="_self">Sign in</a></li><li><a href="register" target="_self">Create account</a></li></ul></div>
  </div>
  <div class="lp-legal"><span>&copy; {year} {APP_NAME}. All rights reserved. Version {APP_VERSION}</span>{theme_switch()}</div>
</footer>""")

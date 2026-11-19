"""Light / dark / system appearance.

Two layers have to agree:
- Streamlit's own widgets (tables, inputs, menus) follow [theme.light] / [theme.dark] in .streamlit/config.toml. The
  browser remembers the choice in localStorage under "stActiveTheme-<page path>-v2" as "Light", "Dark" or "System".
- Our CSS reads colour tokens (CSS variables) that Python emits. Python learns the choice from the `bi_theme` cookie.

The switch writes both (every page path, since Streamlit keys the choice by the path the app was opened on) and
reloads, so the next page load starts in the chosen mode everywhere. "System" emits both token sets behind a
prefers-color-scheme media query, which is exactly what Streamlit does in its own System mode.
"""
from __future__ import annotations

import json

import streamlit as st

COOKIE = "bi_theme"
_SESSION = "_theme_mode"         # the choice made in this browser session (cookies are only read when a session starts)
RESUME_PARAM = "resume"          # ?resume=sample: reopen the sample dataset after the reload that switches theme
MODES = ("light", "dark", "system")
MODE_LABELS = {"light": "Light", "dark": "Dark", "system": "System"}

# Page paths the app can be opened on. Streamlit stores the theme per opening path, so all are written.
_PATHS = ["/", "/home", "/login", "/register", "/overview", "/upload", "/profile", "/quality", "/semantics", "/dashboard",
          "/explorer", "/forecasting", "/anomalies", "/segmentation", "/drivers", "/insights", "/reports", "/settings", "/account"]

LIGHT = {
    "bg": "#FBFAF8", "surface": "#FFFFFF", "surface-2": "#F4F2EE", "line": "#E8E5DF", "line-2": "#D9D5CD",
    "text": "#1C1B19", "muted": "#625F58", "faint": "#8C887F",
    "accent": "#C2410C", "accent-hi": "#9A3412", "accent-soft": "rgba(194,65,12,.08)", "accent-line": "rgba(194,65,12,.32)",
    "on-accent": "#FFFFFF", "warn": "#8F6400", "warn-soft": "rgba(176,128,0,.10)", "danger": "#C92A2A",
    "danger-soft": "rgba(201,42,42,.08)", "ok": "#2B8A3E", "ok-soft": "rgba(43,138,62,.10)",
    "shadow": "0 1px 2px rgba(28,27,25,.05), 0 6px 20px -12px rgba(28,27,25,.16)",
    "shadow-lg": "0 2px 4px rgba(28,27,25,.04), 0 24px 60px -24px rgba(28,27,25,.28)",
}
DARK = {
    "bg": "#121110", "surface": "#1A1917", "surface-2": "#23211F", "line": "#2D2B28", "line-2": "#3B3834",
    "text": "#EDEAE4", "muted": "#A8A39A", "faint": "#7F7A72",
    "accent": "#F2692A", "accent-hi": "#FF8A4F", "accent-soft": "rgba(242,105,42,.12)", "accent-line": "rgba(242,105,42,.42)",
    "on-accent": "#1B0D05", "warn": "#E3B341", "warn-soft": "rgba(227,179,65,.12)", "danger": "#F16063",
    "danger-soft": "rgba(241,96,99,.12)", "ok": "#5CC27A", "ok-soft": "rgba(92,194,122,.12)",
    "shadow": "0 1px 2px rgba(0,0,0,.4), 0 8px 24px -12px rgba(0,0,0,.6)",
    "shadow-lg": "0 2px 4px rgba(0,0,0,.3), 0 30px 70px -28px rgba(0,0,0,.8)",
}

# Charts: same hues in both modes, tuned to read on either background.
PALETTE = ["#D9541E", "#2F6FDB", "#2F9E6B", "#8257D8", "#C9962B", "#5B6B7D", "#C2477A", "#2A9BB5", "#7C8B3A", "#9C6B4E"]


def mode() -> str:
    """The visitor's choice: light, dark or system (the default)."""
    chosen = st.session_state.get(_SESSION)
    if chosen in MODES:
        return chosen
    try:
        value = st.context.cookies.get(COOKIE)
    except Exception:
        value = None
    return value if value in MODES else "system"


def appearance() -> str:
    """The appearance actually showing: light or dark. For "system", Streamlit reports what the browser resolved."""
    m = mode()
    if m != "system":
        return m
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except Exception:
        return "light"


def tokens(name: str) -> dict[str, str]:
    return DARK if name == "dark" else LIGHT


def _block(selector: str, values: dict[str, str]) -> str:
    return selector + "{" + "".join(f"--{k}:{v};" for k, v in values.items()) + "color-scheme:" + \
        ("dark" if values is DARK else "light") + ";}"


def token_css() -> str:
    """CSS variables for the chosen mode; for "system", both sets switched by the browser's preference."""
    m = mode()
    if m == "system":
        return _block(":root", LIGHT) + "@media (prefers-color-scheme: dark){" + _block(":root", DARK) + "}"
    return _block(":root", tokens(m))


def switch_script(selector: str) -> str:
    """JavaScript that makes any element matching `selector` (with a data-mode attribute) a theme switch. Uses one
    delegated listener, so switches rendered after the script, or re-rendered by Streamlit, work too."""
    return f"""<script>
(() => {{
  if (window.__tessalyThemeSwitch) return;
  window.__tessalyThemeSwitch = true;
  const paths = {json.dumps(_PATHS)};
  const names = {{light: "Light", dark: "Dark", system: "System"}};
  document.addEventListener("click", (ev) => {{
    const el = ev.target.closest({json.dumps(selector)});
    if (!el || !names[el.dataset.mode]) return;
    ev.preventDefault();
    const m = el.dataset.mode;
    try {{ paths.forEach((p) => window.localStorage.setItem(`stActiveTheme-${{p}}-v2`, JSON.stringify(names[m]))); }} catch (e) {{}}
    document.cookie = `{COOKIE}=${{m}}; path=/; max-age=31536000; samesite=lax`;
    window.location.reload();
  }});
}})();
</script>"""


def _persist_js(new_mode: str) -> str:
    """Remember the choice in the browser: our cookie, and Streamlit's own theme setting for every page path."""
    label = json.dumps(json.dumps(MODE_LABELS[new_mode]))
    return (f"const paths = {json.dumps(_PATHS)};"
            f"try {{ paths.forEach((p) => window.localStorage.setItem(`stActiveTheme-${{p}}-v2`, {label})); }} catch (e) {{}}"
            f'document.cookie = "{COOKIE}={new_mode}; path=/; max-age=31536000; samesite=lax";')


_REQUEST = "_theme_request"
_CONTROLS = "_theme_controls"     # keys of every appearance control rendered, so they can be kept in step


def _on_pick(key: str) -> None:
    """Widget callback: queue the switch and show the same choice on every appearance control."""
    value = st.session_state.get(key)
    if value not in MODES or value == mode():
        return
    st.session_state[_REQUEST] = value
    for other in st.session_state.get(_CONTROLS, set()):
        st.session_state[other] = value


def appearance_control(key: str, label_visibility: str = "visible", width: str = "content") -> None:
    """Light / Dark / System. Any number of these can be on a page; they always agree."""
    st.session_state.setdefault(_CONTROLS, set()).add(key)
    if st.session_state.get(key) not in MODES:
        st.session_state[key] = mode()
    st.segmented_control("Appearance", list(MODES), format_func=MODE_LABELS.get, key=key, required=True,
                         on_change=_on_pick, args=(key,), label_visibility=label_visibility, width=width)


def handle_request() -> None:
    """Apply a queued switch. Called at the very start of the page render, before the colour tokens are written.

    Streamlit applies its own theme (sidebar, inputs, tables) only when the page loads, so the page reloads. That is
    safe only when sign-in survives a reload, i.e. when the cookie routes are mounted (`streamlit run app.py`). Without
    them a reload would sign the user out, so the app recolours in place instead, and Streamlit's own widgets follow
    the next time the app is opened.
    """
    from auth.web import routes_enabled
    from ui import state
    new_mode = st.session_state.pop(_REQUEST, None)
    if new_mode not in MODES:
        return
    st.session_state[_SESSION] = new_mode
    persist, leave = _persist_js(new_mode), ""
    if routes_enabled():
        result = state.get_result()
        resume = result is not None and result.dataset.is_sample and state.get_dataset_id() is None
        query = f" + '?{RESUME_PARAM}=sample'" if resume else ""
        leave = f"window.location.replace(window.location.pathname{query});"
    st.html(f"<script>(() => {{ {persist} {leave} }})();</script>", unsafe_allow_javascript=True)


def plotly_colors() -> dict[str, str]:
    t = tokens(appearance())
    return {"text": t["text"], "muted": t["muted"], "grid": t["line"], "bg": t["surface"]}

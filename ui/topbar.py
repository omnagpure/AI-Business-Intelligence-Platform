"""App shell top bar: switch between your saved datasets on the left, your account menu on the right.

Also reopens the dataset you used last when you come back, so the app starts where you left off.
"""
from __future__ import annotations

import streamlit as st

from config.settings import Settings
from core.logging_config import get_logger
from storage import repository
from ui import auth_state, nav, notify, state, theme
from ui.layout import dataset_meta_html

log = get_logger("ui.topbar")
_AUTO = "_auto_opened"
_SAMPLE = "__sample__"
_NONE = "__none__"


def _reopen_last(settings: Settings, records: list) -> None:
    """Once per session: if nothing is open yet, reopen the most recently used dataset."""
    if st.session_state.get(_AUTO):
        return
    st.session_state[_AUTO] = True
    if state.get_result() is not None or not records:
        return
    from ui.upload_page import open_saved
    if open_saved(settings, records[0].id, quiet=True):
        notify.toast(f"Reopened {records[0].filename}", icon=":material/history:")


def _switcher(settings: Settings, records: list) -> None:
    result = state.get_result()
    active = state.get_dataset_id()
    options: list[str] = [r.id for r in records]
    labels = {r.id: r.filename + (f" · {r.sheet}" if r.sheet else "") for r in records}
    if result is not None and active not in options:            # sample data or a file that could not be saved
        current = _SAMPLE
        options.insert(0, _SAMPLE)
        labels[_SAMPLE] = result.dataset.filename
    elif result is None:
        current = _NONE
        options.insert(0, _NONE)
        labels[_NONE] = "No dataset open"
    else:
        current = active
    if not records and result is None:
        st.markdown('<div class="tb-meta">No dataset open</div>', unsafe_allow_html=True)
        return
    choice = st.selectbox("Dataset", options, index=options.index(current), format_func=labels.get,
                          key=f"tb_ds_{current}", label_visibility="collapsed", width=320)
    if choice != current and choice not in (_SAMPLE, _NONE):
        from ui.upload_page import open_saved
        if open_saved(settings, choice):
            notify.toast(f"Opened {labels[choice]}")
            st.rerun()


def _user_menu() -> None:
    user = auth_state.current_user()
    if user is None:
        return
    initials = "".join(p[0] for p in user.full_name.split()[:2]).upper() or "?"
    with st.popover(initials, icon=":material/account_circle:", help=user.full_name):
        st.html(f'<div class="tb-user"><b>{_esc(user.full_name)}</b><span>{_esc(user.email)}</span></div>')
        if st.button("Account", key="tb_account", icon=":material/person:", type="tertiary"):
            nav.go("account")
        if st.button("Settings", key="tb_settings", icon=":material/settings:", type="tertiary"):
            nav.go("settings")
        theme.appearance_control("tb_theme", width="stretch")
        st.divider()
        if st.button("Sign out", key="tb_logout", icon=":material/logout:", width="stretch"):
            auth_state.sign_out()


def _esc(text: str) -> str:
    from html import escape
    return escape(text)


def render_topbar(settings: Settings) -> None:
    user = auth_state.current_user()
    try:
        records = repository.list_datasets(user.id) if user is not None else []
    except Exception:
        log.exception("Could not list datasets for the top bar")
        records = []
    _reopen_last(settings, records)
    with st.container(key="topbar", horizontal=True, vertical_alignment="center", gap="small"):
        _switcher(settings, records)
        st.markdown(dataset_meta_html() or "", unsafe_allow_html=True)
        st.space("stretch")
        _user_menu()
    notify.show_pending()

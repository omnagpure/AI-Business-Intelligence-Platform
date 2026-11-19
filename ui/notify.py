"""Short confirmations as toasts. A message queued before st.rerun() is shown on the next run, so it is never lost."""
from __future__ import annotations

import streamlit as st

_KEY = "_toasts"


def toast(message: str, icon: str | None = ":material/check_circle:") -> None:
    """Queue a toast for the next run (use before st.rerun or st.switch_page)."""
    st.session_state.setdefault(_KEY, []).append((message, icon))


def show_pending() -> None:
    for message, icon in st.session_state.pop(_KEY, []):
        st.toast(message, icon=icon)

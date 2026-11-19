"""Page registry, so any page can send the user to another one (st.switch_page needs the Page object)."""
from __future__ import annotations

import streamlit as st

_KEY = "_ui_pages"


def register(pages: dict[str, "st.Page"]) -> None:
    st.session_state[_KEY] = pages


def go(key: str) -> None:
    page = st.session_state.get(_KEY, {}).get(key)
    if page is not None:
        st.switch_page(page)

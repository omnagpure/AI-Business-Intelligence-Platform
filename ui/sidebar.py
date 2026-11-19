"""Sidebar footer: remove the active dataset, and the privacy note. Your account and sign out live in the top bar."""
from __future__ import annotations

import streamlit as st

from ui import notify, state


def render_sidebar_footer() -> None:
    with st.sidebar:
        if state.get_result() is not None:
            st.divider()
            if st.button("Close dataset", key="clear_ds", icon=":material/close:", width="stretch",
                         help="Close the open dataset. Saved datasets stay in your account."):
                name = state.get_result().dataset.filename
                state.clear_result()
                notify.toast(f"Closed {name}")
                st.rerun()
        st.caption("Processed locally. Nothing leaves this machine.")

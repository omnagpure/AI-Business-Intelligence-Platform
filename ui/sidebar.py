"""Sidebar footer: active dataset summary and clear action."""
from __future__ import annotations

import streamlit as st

from ui import state
from ui.layout import fmt_int


def render_sidebar_footer() -> None:
    result = state.get_result()
    with st.sidebar:
        st.divider()
        if result is None:
            st.caption("No dataset loaded")
        else:
            ds = result.dataset
            st.caption("Active dataset")
            st.markdown(f"**{ds.filename}**")
            st.caption(f"{fmt_int(result.profile.rows)} rows · {result.profile.columns} columns"
                       + (" · synthetic sample" if ds.is_sample else ""))
            cs = state.get_cleaning()
            if cs is not None:
                st.caption(f"Cleaned data active · {cs.result.applied_count} step(s) applied")
            if st.button("Remove dataset", key="clear_ds"):
                state.clear_result()
                st.rerun()
        st.caption("Processed locally. Nothing leaves this machine.")

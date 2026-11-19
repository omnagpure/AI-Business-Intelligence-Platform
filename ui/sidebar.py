"""Sidebar footer: the signed-in user, sign out, the remove-dataset action and the privacy note.
The active dataset is shown by ui.layout.context_line."""
from __future__ import annotations

from html import escape

import streamlit as st

from ui import auth_state, state


def render_sidebar_footer() -> None:
    result = state.get_result()
    user = auth_state.current_user()
    with st.sidebar:
        st.divider()
        if result is not None and st.button("Remove dataset", key="clear_ds", icon=":material/delete:", width="stretch"):
            state.clear_result()
            st.rerun()
        if user is not None:
            initials = "".join(p[0] for p in user.full_name.split()[:2]).upper() or "?"
            st.markdown(f'<div class="sb-user"><span class="sb-av">{escape(initials)}</span><span class="sb-id"><b>{escape(user.full_name)}</b>'
                        f'<span>{escape(user.email)}</span></span></div>', unsafe_allow_html=True)
            if st.button("Sign out", key="sb_logout", icon=":material/logout:", width="stretch"):
                auth_state.sign_out()
        st.caption("Processed locally. Nothing leaves this machine.")

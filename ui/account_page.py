"""Account: who is signed in, password change, and signing out (here or everywhere)."""
from __future__ import annotations

import streamlit as st

from auth import passwords, service
from auth.service import AuthError
from storage import repository
from ui import auth_state, theme
from ui.layout import eyebrow, page_title


def render_account_page() -> None:
    user = auth_state.require_user()
    page_title("Account", "Your profile, password and sessions.")
    n_ds, n_rep = len(repository.list_datasets(user.id)), len(repository.list_reports(user.id, limit=1000))
    c = st.columns(3)
    c[0].metric("Name", user.full_name)
    c[1].metric("Saved datasets", n_ds)
    c[2].metric("Saved reports", n_rep)
    st.caption(f"Signed in as {user.email}")

    st.write("")
    eyebrow("Appearance")
    theme.appearance_control("acct_theme", label_visibility="collapsed")
    st.caption("System follows your device's light or dark setting. Saved in this browser.")

    st.write("")
    eyebrow("Change password")
    with st.form("acct_pw", border=True, clear_on_submit=True):
        current = st.text_input("Current password", type="password", autocomplete="current-password", max_chars=passwords.MAX_LENGTH)
        new = st.text_input("New password", type="password", autocomplete="new-password", max_chars=passwords.MAX_LENGTH,
                            help=f"At least {passwords.MIN_LENGTH} characters, with a letter and a number.")
        confirm = st.text_input("Confirm new password", type="password", autocomplete="new-password", max_chars=passwords.MAX_LENGTH)
        if st.form_submit_button("Update password", type="primary"):
            try:
                service.change_password(user.id, current, new, confirm)
            except AuthError as exc:
                st.error(exc.message)
            else:
                auth_state.sign_out(flash="Password updated. Sign in with your new password.")

    st.write("")
    eyebrow("Sessions")
    st.caption("Changing your password also signs out every session.")
    with st.container(horizontal=True):
        if st.button("Sign out", icon=":material/logout:", key="acct_logout"):
            auth_state.sign_out()
        if st.button("Sign out of all devices", icon=":material/devices:", key="acct_logout_all"):
            auth_state.sign_out(everywhere=True)

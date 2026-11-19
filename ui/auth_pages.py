"""Sign-in and registration pages (public). All checks happen in auth.service; these pages only collect input."""
from __future__ import annotations

import re

import streamlit as st

from auth import passwords, service
from auth.service import AuthError
from ui import auth_state, nav
from ui.public_ui import apply_public_theme, brand, icon

_FORGOT = "_auth_show_forgot"


def _shell_top() -> None:
    st.html(f'<div class="au-top">{brand()}<a class="au-back" href="./" target="_self">Back to home</a></div>')


def _card():
    """A centred card. Returns the card container."""
    shell = st.container(key="auth_shell", horizontal=True, horizontal_alignment="center")
    return shell.container(key="auth_card", width=440)


def _notices() -> None:
    if (flash := auth_state.pop_flash()) is not None:
        st.info(flash)
    if st.query_params.get("notice") == "expired":
        st.warning("That sign-in step expired or was already used. Please sign in again.")
    if st.session_state.get("_auth_next"):
        st.info("Sign in to continue to the page you requested.")


def render_login_page() -> None:
    apply_public_theme()
    _shell_top()
    with _card():
        st.html('<div class="au-h">Sign in to your account</div><div class="au-s">Welcome back. Enter your details to continue.</div>')
        _notices()
        remember_ok = auth_state.remember_available()
        with st.form("login_form", border=False, enter_to_submit=True):
            email = st.text_input("Email address", key="li_email", max_chars=254, autocomplete="email", placeholder="name@company.com")
            password = st.text_input("Password", key="li_password", type="password", max_chars=passwords.MAX_LENGTH,
                                     autocomplete="current-password", help="Use the eye icon in the field to show or hide the password.")
            remember = st.checkbox("Remember me for 30 days", key="li_remember", disabled=not remember_ok,
                                   help=None if remember_ok else "Available when the app is started with `streamlit run server.py`.")
            submitted = st.form_submit_button("Sign in", type="primary", width="stretch")
        if submitted:
            try:
                user = service.authenticate(email, password)
            except AuthError as exc:
                st.error(exc.message)
            else:
                auth_state.sign_in(user, remember=bool(remember and remember_ok))

        with st.container(horizontal=True, horizontal_alignment="distribute", vertical_alignment="center"):
            if st.button("Forgot password?", type="tertiary", key="li_forgot"):
                st.session_state[_FORGOT] = not st.session_state.get(_FORGOT, False)
            if st.button("Create an account", type="tertiary", key="li_to_register"):
                nav.go("register")
        if st.session_state.get(_FORGOT):
            st.info("This installation has no email service, so it cannot send password reset links. Ask the person who "
                    "administers it to reset your password from the server with `python -m scripts.reset_password`. "
                    "Resetting signs out all of your sessions.", title="Self-service password reset is not configured")
    st.html('<div class="au-foot">Protected sign-in. Passwords are stored only as Argon2id hashes.</div>')


def _requirements(password: str, confirm: str) -> str:
    items = [(len(password) >= passwords.MIN_LENGTH, f"At least {passwords.MIN_LENGTH} characters"),
             (bool(re.search(r"[A-Za-z]", password) and re.search(r"\d", password)), "At least one letter and one number"),
             (bool(confirm) and password == confirm, "Passwords match")]
    lis = "".join(f'<li class="{"ok" if ok else ""}">{icon("check" if ok else "x", 14)}{text}</li>' for ok, text in items)
    return f'<ul class="au-reqs">{lis}</ul>'


def render_register_page() -> None:
    apply_public_theme()
    _shell_top()
    with _card():
        st.html('<div class="au-h">Create your account</div><div class="au-s">Your datasets and reports stay private to your account.</div>')
        name = st.text_input("Full name", key="rg_name", max_chars=80, autocomplete="name", placeholder="Jordan Lee")
        email = st.text_input("Email address", key="rg_email", max_chars=254, autocomplete="email", placeholder="name@company.com")
        password = st.text_input("Password", key="rg_password", type="password", max_chars=passwords.MAX_LENGTH, autocomplete="new-password")
        s = passwords.strength(password)
        st.html(f'<div class="au-meter s{s.score if password else 0}"><i></i><i></i><i></i><i></i></div>'
                f'<div class="au-mlabel"><span>Password strength</span><span>{s.label if password else ""}</span></div>')
        confirm = st.text_input("Confirm password", key="rg_confirm", type="password", max_chars=passwords.MAX_LENGTH,
                                autocomplete="new-password")
        st.html(_requirements(password, confirm))
        if st.button("Create account", type="primary", width="stretch", key="rg_submit"):
            try:
                user = service.register(name, email, password, confirm)
            except AuthError as exc:
                st.error(exc.message)
            else:
                auth_state.sign_in(user, remember=False)
        with st.container(horizontal=True, horizontal_alignment="center", vertical_alignment="center", gap="small"):
            st.html('<span class="au-s" style="margin:0">Already have an account?</span>', width="content")
            if st.button("Sign in", type="tertiary", key="rg_to_login"):
                nav.go("login")

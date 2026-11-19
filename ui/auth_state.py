"""Who is signed in to this Streamlit session, and the sign-in / sign-out transitions.

The session id (a hash; the raw token lives only in the HttpOnly cookie) is re-validated against the database on every
run, so logging out elsewhere, a password change or expiry takes effect on the next interaction. Signing in or out
clears the whole Streamlit session first, so nothing one user loaded can be seen by the next user of the same tab.
"""
from __future__ import annotations

import json
from html import escape

import streamlit as st

from auth import service
from auth.service import User
from auth.web import (
    COOKIE_NAME,
    LOGOUT_PATH,
    SESSION_PATH,
    browser_binding,
    routes_enabled,
    safe_next,
)

_SID = "_auth_sid"
_USER = "_auth_user"
_REDIRECT = "_auth_redirect"       # pending browser navigation (cookie handoff or cookie clearing)
_TARGET = "_auth_target"           # page to open after an in-tab sign-in
_NEXT = "_auth_next"               # protected page the visitor asked for before signing in
_FLASH = "_auth_flash"


def _cookie(name: str) -> str | None:
    try:
        value = st.context.cookies.get(name)
    except Exception:
        return None
    return value if isinstance(value, str) and value else None


def resolve_user() -> User | None:
    """The signed-in user for this run, from the session or the session cookie, or None."""
    sid = st.session_state.get(_SID)
    if sid:
        sess = service.validate_session(sid)
        if sess is not None:
            st.session_state[_USER] = sess.user
            return sess.user
        _reset(keep=())                                   # expired or revoked elsewhere: drop everything this user loaded
        st.session_state[_FLASH] = "Your session ended. Please sign in again."
    token = _cookie(COOKIE_NAME)
    if token and len(token) <= 128:
        sess = service.validate_token(token)
        if sess is not None:
            st.session_state[_SID] = sess.id
            st.session_state[_USER] = sess.user
            return sess.user
    return None


def current_user() -> User | None:
    """The user resolved at the start of this run. Pages call require_user() instead when they need one."""
    return st.session_state.get(_USER) if st.session_state.get(_SID) else None


def require_user() -> User:
    user = current_user()
    if user is None:                                      # unreachable through navigation; a guard for direct calls
        st.error("Sign in to continue.")
        st.stop()
    return user


def _reset(keep: tuple[str, ...]) -> None:
    saved = {k: st.session_state[k] for k in keep if k in st.session_state}
    st.session_state.clear()
    st.session_state.update(saved)


def sign_in(user: User, remember: bool) -> None:
    """Start an authenticated session for a user whose password was just verified, then rerun."""
    target = safe_next(st.session_state.get(_NEXT))
    _reset(keep=())
    if routes_enabled():
        code = service.create_handoff(user.id, remember, browser_binding(_cookie("_streamlit_xsrf")))
        st.session_state[_REDIRECT] = f"{SESSION_PATH}?code={code}&next={target}"
    else:                                                 # plain `streamlit run app.py`: the session lives in this tab only
        token = service.create_session(user.id, remember=False)
        st.session_state[_SID] = service.session_id_for(token)
        st.session_state[_USER] = user
        st.session_state[_TARGET] = target
    st.rerun()


def sign_out(everywhere: bool = False, flash: str | None = None) -> None:
    user, sid = current_user(), st.session_state.get(_SID)
    if everywhere and user is not None:
        service.revoke_all_sessions(user.id)
    elif sid:
        service.revoke_session(sid)
    _reset(keep=())
    if flash:
        st.session_state[_FLASH] = flash
    if routes_enabled():
        st.session_state[_REDIRECT] = LOGOUT_PATH          # clears the cookie, then lands on the public homepage
    st.rerun()


def remember_available() -> bool:
    return routes_enabled()


def set_next(page_path: str | None) -> None:
    if page_path:
        st.session_state[_NEXT] = page_path


def pop_target() -> str | None:
    return st.session_state.pop(_TARGET, None)


def pop_flash() -> str | None:
    return st.session_state.pop(_FLASH, None)


def has_pending_redirect() -> bool:
    return bool(st.session_state.get(_REDIRECT))


def render_pending_redirect() -> bool:
    """If a browser navigation is pending, render it and return True (the caller stops the run)."""
    url = st.session_state.pop(_REDIRECT, None)
    if not url:
        return False
    st.html(f"<script>window.location.replace({json.dumps(url)});</script>", unsafe_allow_javascript=True)
    st.markdown(f'<div class="auth-redirect"><div class="auth-spinner"></div><div>One moment…</div>'
                f'<a href="{escape(url)}" target="_self">Continue</a></div>', unsafe_allow_html=True)
    return True

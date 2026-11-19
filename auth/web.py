"""HTTP routes that set and clear the session cookie, mounted by app.py (st.App).

Streamlit scripts can read cookies but cannot set them, and a cookie set from JavaScript could not be HttpOnly.
So after the password check the browser is sent to /auth/session with a single-use code; this route exchanges it for
an HttpOnly, SameSite=Lax session cookie and redirects into the app. /auth/logout revokes the session and clears it.
"""
from __future__ import annotations

import hashlib
import os
import re

from auth import service

COOKIE_NAME = "bi_session"
ROUTES_FLAG = "BI_AUTH_ROUTES"          # set by app.py, so the app knows the cookie routes exist
SESSION_PATH = "/auth/session"
LOGOUT_PATH = "/auth/logout"
_NEXT_RE = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")
_XSRF_COOKIE = "_streamlit_xsrf"


def routes_enabled() -> bool:
    return os.getenv(ROUTES_FLAG) == "1"


def safe_next(value: str | None, default: str = "dashboard") -> str:
    """Only a bare page path is allowed as a redirect target, never a URL (no open redirects)."""
    return value if value and _NEXT_RE.match(value) else default


def browser_binding(raw_xsrf_cookie: str | None) -> str | None:
    """A stable hash of this browser's Streamlit XSRF token, used to tie a handoff code to the browser that signed in.

    The cookie value is re-masked on every page load, so the underlying token is decoded first. If the decoder is
    unavailable in this Streamlit version, binding is skipped (the code is still single-use and expires in seconds).
    """
    if not raw_xsrf_cookie:
        return None
    try:
        from streamlit.web.server.starlette.starlette_app_utils import decode_xsrf_token_string
        token_bytes, _ = decode_xsrf_token_string(raw_xsrf_cookie)
    except Exception:
        return None
    if not token_bytes:
        return None
    return hashlib.sha256(bytes(token_bytes)).hexdigest()


def _no_store(response):
    response.headers["Cache-Control"] = "no-store"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


async def session_route(request):
    from starlette.responses import RedirectResponse

    code = request.query_params.get("code", "")
    target = safe_next(request.query_params.get("next"))
    exchanged = service.consume_handoff(code, browser_binding(request.cookies.get(_XSRF_COOKIE)))
    if exchanged is None:
        return _no_store(RedirectResponse("/login?notice=expired", status_code=303))
    user_id, remember = exchanged
    token = service.create_session(user_id, remember)
    # the Home page is the app's default page, which Streamlit serves at "/" only
    response = RedirectResponse("/" if target == "overview" else f"/{target}", status_code=303)
    response.set_cookie(COOKIE_NAME, token, max_age=service.REMEMBER_LIFETIME if remember else None, path="/",
                        httponly=True, samesite="lax", secure=request.url.scheme == "https")
    return _no_store(response)


async def logout_route(request):
    from starlette.responses import RedirectResponse

    token = request.cookies.get(COOKIE_NAME)
    if token:
        service.revoke_session(service.session_id_for(token))
    response = RedirectResponse("/", status_code=303)
    response.delete_cookie(COOKIE_NAME, path="/", httponly=True, samesite="lax", secure=request.url.scheme == "https")
    return _no_store(response)

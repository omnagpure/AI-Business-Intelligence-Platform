"""Entry point: the Streamlit app (vesper_app.py) plus the session-cookie routes and security headers.

Run:  streamlit run app.py        (server.py is kept as an alias)

The routes set and clear the HttpOnly session cookie, so sign-in survives page reloads (including the reload that
switches light/dark) and "Remember me" works. Running vesper_app.py directly still works, but then sign-in lasts only
for the open browser tab.
"""
from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.routing import Route

from auth.web import LOGOUT_PATH, ROUTES_FLAG, SESSION_PATH, logout_route, session_route

os.environ[ROUTES_FLAG] = "1"


class SecurityHeaders(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        return response


app = st.App(
    Path(__file__).resolve().parent / "vesper_app.py",
    routes=[Route(SESSION_PATH, session_route, methods=["GET"]), Route(LOGOUT_PATH, logout_route, methods=["GET"])],
    middleware=[Middleware(SecurityHeaders)],
)

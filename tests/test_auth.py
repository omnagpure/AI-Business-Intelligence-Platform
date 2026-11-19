"""Accounts, password storage, throttling, sessions, handoff codes and the cookie routes."""
from __future__ import annotations

import time
import uuid

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from auth import passwords, service
from auth.db import connect
from auth.service import AuthError
from auth.web import COOKIE_NAME, LOGOUT_PATH, SESSION_PATH, logout_route, safe_next, session_route

PW = "Valid-pass-2024"


def _email() -> str:
    return f"u{uuid.uuid4().hex[:10]}@example.com"


def _user(email=None, pw=PW):
    return service.register("Pat Morgan", email or _email(), pw, pw)


# ---- registration and storage -------------------------------------------------------------------------------------
def test_register_stores_argon2_hash_never_the_password():
    email = _email()
    user = service.register("  Pat   Morgan ", email.upper(), PW, PW)
    assert user.email == email and user.full_name == "Pat Morgan"
    with connect() as conn:
        stored = conn.execute("SELECT password_hash FROM users WHERE id = ?", (user.id,)).fetchone()[0]
    assert stored.startswith("$argon2id$") and PW not in stored


def test_duplicate_email_is_rejected_case_insensitively():
    email = _email()
    _user(email)
    with pytest.raises(AuthError) as exc:
        service.register("Someone Else", email.upper(), PW, PW)
    assert exc.value.field == "email" and "already exists" in exc.value.message


@pytest.mark.parametrize("name,email,pw,confirm,field", [
    ("", "a@example.com", PW, PW, "name"),
    ("Pat", "not-an-email", PW, PW, "email"),
    ("Pat", "a@example", PW, PW, "email"),
    ("Pat", "a@example.com", "short1", "short1", "password"),
    ("Pat", "a@example.com", "onlyletterslong", "onlyletterslong", "password"),
    ("Pat", "a@example.com", "password123", "password123", "password"),
    ("Pat", "a@example.com", PW, PW + "x", "confirm"),
])
def test_registration_validates_input(name, email, pw, confirm, field):
    with pytest.raises(AuthError) as exc:
        service.register(name, email, pw, confirm)
    assert exc.value.field == field


def test_strength_meter_orders_passwords_sensibly():
    assert passwords.strength("").score == 0
    assert passwords.strength("password123").score == 0
    assert passwords.strength("abcdefgh12").score < passwords.strength("Tr4il-Mix-Harbor-91").score == 4


# ---- sign in and throttling ---------------------------------------------------------------------------------------
def test_authenticate_success_and_uniform_failure_message():
    email = _email()
    user = _user(email)
    assert service.authenticate(email.upper(), PW).id == user.id
    with pytest.raises(AuthError) as wrong:
        service.authenticate(email, PW + "x")
    with pytest.raises(AuthError) as unknown:
        service.authenticate(_email(), PW)
    assert wrong.value.message == unknown.value.message == service.GENERIC_LOGIN_ERROR


def test_throttling_applies_to_known_and_unknown_emails_alike():
    known, unknown = _email(), _email()
    _user(known)
    for email in (known, unknown):
        for _ in range(service.MAX_FAILURES):
            with pytest.raises(AuthError):
                service.authenticate(email, "Wrong-pass-123")
        with pytest.raises(AuthError) as exc:
            service.authenticate(email, PW)             # even the right password is refused while throttled
        assert exc.value.message == service.THROTTLED_ERROR


def test_successful_sign_in_clears_failures():
    email = _email()
    _user(email)
    for _ in range(service.MAX_FAILURES - 1):
        with pytest.raises(AuthError):
            service.authenticate(email, "Wrong-pass-123")
    service.authenticate(email, PW)
    with pytest.raises(AuthError) as exc:
        service.authenticate(email, "Wrong-pass-123")
    assert exc.value.message == service.GENERIC_LOGIN_ERROR


# ---- sessions -----------------------------------------------------------------------------------------------------
def test_session_token_is_stored_only_as_a_hash_and_validates():
    user = _user()
    token = service.create_session(user.id, remember=True)
    with connect() as conn:
        ids = [r[0] for r in conn.execute("SELECT id FROM auth_sessions WHERE user_id = ?", (user.id,))]
    assert token not in ids and service.session_id_for(token) in ids
    sess = service.validate_token(token)
    assert sess.user.id == user.id and sess.remember and sess.expires_at > time.time() + 29 * 86400
    assert service.validate_token(token + "x") is None


def test_revoked_and_expired_sessions_are_rejected():
    user = _user()
    t1, t2 = service.create_session(user.id, False), service.create_session(user.id, False)
    service.revoke_session(service.session_id_for(t1))
    assert service.validate_token(t1) is None and service.validate_token(t2) is not None
    with connect() as conn:
        conn.execute("UPDATE auth_sessions SET expires_at = ? WHERE id = ?", (time.time() - 1, service.session_id_for(t2)))
    assert service.validate_token(t2) is None


def test_password_change_revokes_every_session():
    user = _user()
    tokens = [service.create_session(user.id, False) for _ in range(2)]
    with pytest.raises(AuthError):
        service.change_password(user.id, "not-my-password1", "New-pass-2025x", "New-pass-2025x")
    service.change_password(user.id, PW, "New-pass-2025x", "New-pass-2025x")
    assert all(service.validate_token(t) is None for t in tokens)
    assert service.authenticate(user.email, "New-pass-2025x").id == user.id


# ---- handoff codes ------------------------------------------------------------------------------------------------
def test_handoff_code_is_single_use():
    user = _user()
    code = service.create_handoff(user.id, True, None)
    assert service.consume_handoff(code, None) == (user.id, True)
    assert service.consume_handoff(code, None) is None


def test_handoff_code_expires_and_is_bound_to_the_browser():
    user = _user()
    expired = service.create_handoff(user.id, False, None)
    with connect() as conn:
        conn.execute("UPDATE auth_handoffs SET expires_at = ?", (time.time() - 1,))
    assert service.consume_handoff(expired, None) is None
    bound = service.create_handoff(user.id, False, "browser-a")
    assert service.consume_handoff(bound, "browser-b") is None
    bound = service.create_handoff(user.id, False, "browser-a")
    assert service.consume_handoff(bound, None) is None
    bound = service.create_handoff(user.id, False, "browser-a")
    assert service.consume_handoff(bound, "browser-a") == (user.id, False)


# ---- HTTP cookie routes -------------------------------------------------------------------------------------------
@pytest.fixture()
def client():
    app = Starlette(routes=[Route(SESSION_PATH, session_route), Route(LOGOUT_PATH, logout_route)])
    return TestClient(app, base_url="http://testserver", follow_redirects=False)


def test_session_route_sets_httponly_cookie_and_redirects(client):
    user = _user()
    code = service.create_handoff(user.id, True, None)
    r = client.get(SESSION_PATH, params={"code": code, "next": "dashboard"})
    assert r.status_code == 303 and r.headers["location"] == "/dashboard"
    cookie = r.headers["set-cookie"].lower()
    assert f"{COOKIE_NAME}=" in cookie and "httponly" in cookie and "samesite=lax" in cookie and "max-age=" in cookie
    assert r.headers["cache-control"] == "no-store"
    token = r.cookies[COOKIE_NAME]
    assert service.validate_token(token).user.id == user.id
    again = client.get(SESSION_PATH, params={"code": code})
    assert again.status_code == 303 and again.headers["location"] == "/login?notice=expired"


def test_session_cookie_without_remember_is_a_browser_session_cookie(client):
    user = _user()
    r = client.get(SESSION_PATH, params={"code": service.create_handoff(user.id, False, None)})
    assert "max-age" not in r.headers["set-cookie"].lower()


@pytest.mark.parametrize("value", ["//evil.example", "https://evil.example", "/dashboard", "../x", "", None, "a" * 40])
def test_redirect_target_cannot_leave_the_app(value):
    assert safe_next(value) == "dashboard"


def test_logout_route_revokes_and_clears_cookie(client):
    user = _user()
    token = service.create_session(user.id, True)
    client.cookies.set(COOKIE_NAME, token)
    r = client.get(LOGOUT_PATH)
    assert r.status_code == 303 and r.headers["location"] == "/"
    assert f'{COOKIE_NAME}=""' in r.headers["set-cookie"] or "max-age=0" in r.headers["set-cookie"].lower()
    assert service.validate_token(token) is None

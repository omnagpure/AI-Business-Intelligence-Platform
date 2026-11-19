"""Accounts and sessions. UI-free, so it is used by the Streamlit pages, the HTTP cookie routes, the admin CLI and tests.

Security model
- Passwords: Argon2id (auth.passwords). Plain-text passwords are never stored or logged.
- Login throttling: after MAX_FAILURES failed attempts for one email inside FAILURE_WINDOW, further attempts are refused
  for that window. Counted per email whether or not an account exists, and unknown emails cost a full hash
  verification, so neither the timing nor the messages reveal which emails are registered.
- Sessions: a random 256-bit token lives only in the user's HttpOnly cookie; the database keeps its SHA-256.
  Sessions expire (absolute lifetime) and can be revoked (logout, password change, "sign out everywhere").
- Handoff codes: after a password check inside the Streamlit session, the browser is sent once to /auth/session with
  a single-use code (valid HANDOFF_TTL seconds, stored hashed, optionally bound to the browser) that the HTTP route
  exchanges for the session cookie. This is how an HttpOnly cookie is set from a Streamlit app.
"""
from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import sqlite3
import time
import unicodedata
from dataclasses import dataclass

from auth import passwords
from auth.db import connect
from core.logging_config import get_logger

log = get_logger("auth")

MAX_FAILURES = 5
FAILURE_WINDOW = 15 * 60
REMEMBER_LIFETIME = 30 * 24 * 3600
SESSION_LIFETIME = 12 * 3600
HANDOFF_TTL = 60
_TOUCH_EVERY = 300

_EMAIL_RE = re.compile(r"^[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
                       r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+$")
GENERIC_LOGIN_ERROR = "The email or password is incorrect."
THROTTLED_ERROR = "Too many sign-in attempts for this email. Wait 15 minutes and try again."


class AuthError(Exception):
    """A user-facing failure. `field` names the form field it belongs to, when there is one."""

    def __init__(self, message: str, field: str | None = None):
        super().__init__(message)
        self.message, self.field = message, field


@dataclass(frozen=True)
class User:
    id: int
    email: str
    full_name: str


@dataclass(frozen=True)
class Session:
    id: str              # SHA-256 of the token
    user: User
    remember: bool
    expires_at: float


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


# ---- input validation --------------------------------------------------------------------------------------------
def normalize_email(email: str) -> str:
    return (email or "").strip().lower()


def email_error(email: str) -> str | None:
    if not email:
        return "Enter your email address."
    if len(email) > 254 or not _EMAIL_RE.match(email):
        return "Enter a valid email address, like name@company.com."
    return None


def clean_name(name: str) -> str:
    """Trimmed, single-spaced, with control characters removed."""
    name = "".join(ch for ch in (name or "") if unicodedata.category(ch)[0] != "C")
    return " ".join(name.split())


def name_error(name: str) -> str | None:
    if len(name) < 2:
        return "Enter your full name."
    if len(name) > 80:
        return "Use at most 80 characters for your name."
    if not any(ch.isalpha() for ch in name):
        return "Your name should contain letters."
    return None


# ---- accounts ----------------------------------------------------------------------------------------------------
def _row_user(row: sqlite3.Row) -> User:
    return User(int(row["id"]), row["email"], row["full_name"])


def register(full_name: str, email: str, password: str, confirm: str) -> User:
    """Create an account. Raises AuthError (with the offending field) on invalid input or a duplicate email."""
    name, email = clean_name(full_name), normalize_email(email)
    if err := name_error(name):
        raise AuthError(err, "name")
    if err := email_error(email):
        raise AuthError(err, "email")
    if errors := passwords.policy_errors(password, email, name):
        raise AuthError(" ".join(errors), "password")
    if not hmac.compare_digest(password.encode(), confirm.encode()):
        raise AuthError("The passwords do not match.", "confirm")
    pw_hash = passwords.hash_password(password)
    try:
        with connect() as conn:
            cur = conn.execute("INSERT INTO users (email, full_name, password_hash, created_at) VALUES (?, ?, ?, ?)",
                               (email, name, pw_hash, time.time()))
            user_id = int(cur.lastrowid)
    except sqlite3.IntegrityError:
        raise AuthError("An account with this email already exists. Sign in instead.", "email") from None
    log.info("Account created (user %s)", user_id)
    return User(user_id, email, name)


def _recent_failures(conn: sqlite3.Connection, email: str, now: float) -> int:
    return conn.execute("SELECT COUNT(*) FROM login_failures WHERE email = ? AND at > ?",
                        (email, now - FAILURE_WINDOW)).fetchone()[0]


def authenticate(email: str, password: str) -> User:
    """Check credentials. Raises AuthError with a message that does not reveal whether the email exists."""
    email, now = normalize_email(email), time.time()
    if not email or not password:
        raise AuthError("Enter your email and password.")
    if len(password) > passwords.MAX_LENGTH or email_error(email):
        passwords.burn_verify_time(password[: passwords.MAX_LENGTH])
        raise AuthError(GENERIC_LOGIN_ERROR)
    with connect() as conn:
        if _recent_failures(conn, email, now) >= MAX_FAILURES:
            raise AuthError(THROTTLED_ERROR)
        row = conn.execute("SELECT id, email, full_name, password_hash FROM users WHERE email = ?", (email,)).fetchone()
    if row is None:
        passwords.burn_verify_time(password)
        ok = False
    else:
        ok = passwords.verify_password(row["password_hash"], password)
    if not ok:
        with connect() as conn:                          # committed before raising: an error inside would roll it back
            conn.execute("INSERT INTO login_failures (email, at) VALUES (?, ?)", (email, now))
            conn.execute("DELETE FROM login_failures WHERE at < ?", (now - 24 * 3600,))
        log.info("Failed sign-in attempt")
        raise AuthError(GENERIC_LOGIN_ERROR)
    with connect() as conn:
        conn.execute("DELETE FROM login_failures WHERE email = ?", (email,))
        conn.execute("UPDATE users SET last_login_at = ? WHERE id = ?", (now, row["id"]))
        if passwords.needs_rehash(row["password_hash"]):
            conn.execute("UPDATE users SET password_hash = ? WHERE id = ?", (passwords.hash_password(password), row["id"]))
    log.info("Signed in (user %s)", row["id"])
    return _row_user(row)


def get_user(user_id: int) -> User | None:
    with connect() as conn:
        row = conn.execute("SELECT id, email, full_name FROM users WHERE id = ?", (user_id,)).fetchone()
    return _row_user(row) if row else None


def set_password(user_id: int, new_password: str) -> None:
    """Store a new password and revoke every session of the user (they sign in again with the new one)."""
    user = get_user(user_id)
    if user is None:
        raise AuthError("Account not found.")
    if errors := passwords.policy_errors(new_password, user.email, user.full_name):
        raise AuthError(" ".join(errors), "password")
    with connect() as conn:
        conn.execute("UPDATE users SET password_hash = ? WHERE id = ?", (passwords.hash_password(new_password), user_id))
        conn.execute("UPDATE auth_sessions SET revoked_at = ? WHERE user_id = ? AND revoked_at IS NULL", (time.time(), user_id))
    log.info("Password changed (user %s); sessions revoked", user_id)


def change_password(user_id: int, current: str, new_password: str, confirm: str) -> None:
    with connect() as conn:
        row = conn.execute("SELECT password_hash FROM users WHERE id = ?", (user_id,)).fetchone()
    if row is None or not passwords.verify_password(row["password_hash"], current):
        raise AuthError("Your current password is incorrect.", "current")
    if not hmac.compare_digest(new_password.encode(), confirm.encode()):
        raise AuthError("The new passwords do not match.", "confirm")
    if hmac.compare_digest(new_password.encode(), current.encode()):
        raise AuthError("Choose a password different from your current one.", "password")
    set_password(user_id, new_password)


def find_user_id(email: str) -> int | None:
    with connect() as conn:
        row = conn.execute("SELECT id FROM users WHERE email = ?", (normalize_email(email),)).fetchone()
    return int(row["id"]) if row else None


# ---- sessions ----------------------------------------------------------------------------------------------------
def create_session(user_id: int, remember: bool) -> str:
    """Start a session and return its raw token (to be placed in the cookie). Only the hash is stored."""
    token, now = secrets.token_urlsafe(32), time.time()
    lifetime = REMEMBER_LIFETIME if remember else SESSION_LIFETIME
    with connect() as conn:
        conn.execute("INSERT INTO auth_sessions (id, user_id, created_at, expires_at, last_seen_at, remember) VALUES (?, ?, ?, ?, ?, ?)",
                     (_sha256(token), user_id, now, now + lifetime, now, int(remember)))
        conn.execute("DELETE FROM auth_sessions WHERE expires_at < ? OR revoked_at < ?", (now - 86400, now - 86400))
    return token


def session_id_for(token: str) -> str:
    return _sha256(token)


def validate_session(session_id: str) -> Session | None:
    """The live session for this id, or None when it is unknown, expired or revoked."""
    if not session_id:
        return None
    now = time.time()
    with connect() as conn:
        row = conn.execute(
            "SELECT s.id, s.remember, s.expires_at, s.last_seen_at, u.id AS uid, u.email, u.full_name "
            "FROM auth_sessions s JOIN users u ON u.id = s.user_id "
            "WHERE s.id = ? AND s.revoked_at IS NULL AND s.expires_at > ?", (session_id, now)).fetchone()
        if row is None:
            return None
        if now - row["last_seen_at"] > _TOUCH_EVERY:
            conn.execute("UPDATE auth_sessions SET last_seen_at = ? WHERE id = ?", (now, session_id))
    user = User(int(row["uid"]), row["email"], row["full_name"])
    return Session(row["id"], user, bool(row["remember"]), float(row["expires_at"]))


def validate_token(token: str) -> Session | None:
    return validate_session(_sha256(token)) if isinstance(token, str) and token else None


def revoke_session(session_id: str) -> None:
    with connect() as conn:
        conn.execute("UPDATE auth_sessions SET revoked_at = ? WHERE id = ? AND revoked_at IS NULL", (time.time(), session_id))


def revoke_all_sessions(user_id: int) -> None:
    with connect() as conn:
        conn.execute("UPDATE auth_sessions SET revoked_at = ? WHERE user_id = ? AND revoked_at IS NULL", (time.time(), user_id))


# ---- login handoff (Streamlit session -> HttpOnly cookie) --------------------------------------------------------
def create_handoff(user_id: int, remember: bool, browser_binding: str | None) -> str:
    code, now = secrets.token_urlsafe(32), time.time()
    with connect() as conn:
        conn.execute("DELETE FROM auth_handoffs WHERE expires_at < ?", (now,))
        conn.execute("INSERT INTO auth_handoffs (code_hash, user_id, remember, browser_binding, expires_at) VALUES (?, ?, ?, ?, ?)",
                     (_sha256(code), user_id, int(remember), browser_binding, now + HANDOFF_TTL))
    return code


def consume_handoff(code: str, browser_binding: str | None) -> tuple[int, bool] | None:
    """Exchange a handoff code once. Returns (user_id, remember) or None if unknown, used, expired or from another browser."""
    if not code or len(code) > 128:
        return None
    with connect() as conn:
        row = conn.execute("DELETE FROM auth_handoffs WHERE code_hash = ? RETURNING user_id, remember, browser_binding, expires_at",
                           (_sha256(code),)).fetchone()
    if row is None or row["expires_at"] < time.time():
        return None
    bound = row["browser_binding"]
    if bound and not (browser_binding and hmac.compare_digest(bound, browser_binding)):
        log.warning("Handoff code presented by a different browser; refused")
        return None
    return int(row["user_id"]), bool(row["remember"])

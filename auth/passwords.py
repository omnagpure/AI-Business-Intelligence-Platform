"""Password hashing (Argon2id) and the password policy.

Hashes are self-describing Argon2 strings (algorithm, parameters and salt included), so parameters can be raised
later: `needs_rehash` tells the login code to upgrade a stored hash after a successful sign-in.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

MIN_LENGTH = 10
MAX_LENGTH = 128          # bounds hashing work per attempt

# OWASP-recommended Argon2id profile (memory 19 MiB, 2 iterations), raised to 3 iterations.
_HASHER = PasswordHasher(time_cost=3, memory_cost=19 * 1024, parallelism=1)

# A hash of a random password, verified against when the email is unknown, so a miss costs the same time as a hit.
_DUMMY_HASH = _HASHER.hash("timing-equaliser-not-a-real-password")

_COMMON = {
    "password", "password1", "password12", "password123", "1234567890", "qwertyuiop", "letmein123", "welcome123",
    "iloveyou12", "admin12345", "abc1234567", "passw0rd12", "1q2w3e4r5t", "qwerty1234", "123456789a",
}


def hash_password(password: str) -> str:
    return _HASHER.hash(password)


def verify_password(stored_hash: str, password: str) -> bool:
    try:
        return _HASHER.verify(stored_hash, password)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


def burn_verify_time(password: str) -> None:
    """Spend the same time as a real verification (for unknown emails), and always fail."""
    verify_password(_DUMMY_HASH, password)


def needs_rehash(stored_hash: str) -> bool:
    try:
        return _HASHER.check_needs_rehash(stored_hash)
    except InvalidHashError:
        return True


def policy_errors(password: str, email: str = "", full_name: str = "") -> list[str]:
    """Reasons the password is not acceptable; empty when it is."""
    errors = []
    if len(password) < MIN_LENGTH:
        errors.append(f"Use at least {MIN_LENGTH} characters.")
    if len(password) > MAX_LENGTH:
        errors.append(f"Use at most {MAX_LENGTH} characters.")
    if not re.search(r"[A-Za-z]", password) or not re.search(r"\d", password):
        errors.append("Include at least one letter and one number.")
    lowered = password.lower()
    local = email.split("@", 1)[0].lower()
    if lowered in _COMMON:
        errors.append("This password is too common.")
    elif len(local) >= 4 and local in lowered:
        errors.append("Do not include your email name in the password.")
    elif full_name and any(len(p) >= 4 and p.lower() in lowered for p in full_name.split()):
        errors.append("Do not include your name in the password.")
    if len(set(password)) <= 3 and password:
        errors.append("Use more varied characters.")
    return errors


@dataclass(frozen=True)
class Strength:
    score: int          # 0..4
    label: str


_LABELS = ("Very weak", "Weak", "Fair", "Strong", "Very strong")


def strength(password: str) -> Strength:
    """A simple guide for the meter. The policy above, not this score, decides what is accepted."""
    if not password:
        return Strength(0, _LABELS[0])
    classes = sum(bool(re.search(p, password)) for p in (r"[a-z]", r"[A-Z]", r"\d", r"[^A-Za-z0-9]"))
    score = 0
    if len(password) >= MIN_LENGTH:
        score += 1
    if len(password) >= 14:
        score += 1
    if classes >= 3:
        score += 1
    if len(password) >= 18 or (classes == 4 and len(password) >= 12):
        score += 1
    if password.lower() in _COMMON or len(set(password)) <= 3:
        score = 0
    score = min(score, 4)
    return Strength(score, _LABELS[score])

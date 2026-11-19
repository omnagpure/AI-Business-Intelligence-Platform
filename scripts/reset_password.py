"""Administrator password reset. Requires access to the server, which is what makes it safe without an email service.

    python -m scripts.reset_password user@example.com

Prompts for the new password twice (never echoed, never taken from the command line where shell history would keep it),
applies the normal password policy, stores the Argon2id hash, and signs the user out of every session.
"""
from __future__ import annotations

import argparse
import getpass
import sys

from auth import service
from config.settings import load_settings
from core.logging_config import setup_logging


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reset a user's password (administrators only).")
    parser.add_argument("email", help="Email address of the account")
    args = parser.parse_args(argv)
    setup_logging(load_settings())

    user_id = service.find_user_id(args.email)
    if user_id is None:
        print("No account uses that email address.", file=sys.stderr)
        return 1
    new = getpass.getpass("New password: ")
    if new != getpass.getpass("Repeat new password: "):
        print("The passwords do not match. Nothing was changed.", file=sys.stderr)
        return 1
    try:
        service.set_password(user_id, new)
    except service.AuthError as exc:
        print(f"{exc.message} Nothing was changed.", file=sys.stderr)
        return 1
    print("Password reset. All of this user's sessions were signed out.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

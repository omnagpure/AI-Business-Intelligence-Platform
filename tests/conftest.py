"""Test isolation for accounts and stored files.

- The whole test run uses a temporary data directory (BI_DATA_DIR), never the real data_store/.
- App tests drive the real app, which requires a signed-in user. Instead of any bypass in the app, a real account is
  created once and every AppTest starts with that account's real database session id, exactly as a browser session
  would after signing in. Tests that need a visitor set `at.session_state["_auth_sid"] = None`.
"""
from __future__ import annotations

import os

import pytest
from streamlit.testing.v1 import AppTest

TEST_EMAIL = "analyst@example.com"
TEST_PASSWORD = "correct-horse-42"


@pytest.fixture(scope="session", autouse=True)
def _isolated_data_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("bi_data")
    old = os.environ.get("BI_DATA_DIR")
    os.environ["BI_DATA_DIR"] = str(path)
    os.environ.pop("BI_AUTH_ROUTES", None)
    yield path
    if old is None:
        os.environ.pop("BI_DATA_DIR", None)
    else:
        os.environ["BI_DATA_DIR"] = old


@pytest.fixture(scope="session", autouse=True)
def signed_in_session(_isolated_data_dir):
    from auth import service

    user = service.register("Test Analyst", TEST_EMAIL, TEST_PASSWORD, TEST_PASSWORD)
    sid = service.session_id_for(service.create_session(user.id, remember=False))
    original = AppTest.from_file.__func__

    def from_file(cls, *args, **kwargs):
        at = original(cls, *args, **kwargs)
        at.session_state["_auth_sid"] = sid
        return at

    mp = pytest.MonkeyPatch()
    mp.setattr(AppTest, "from_file", classmethod(from_file))
    yield user
    mp.undo()

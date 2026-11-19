"""The visitor -> homepage -> sign in -> app journey, driven through the real Streamlit pages."""
from __future__ import annotations

import uuid
from pathlib import Path

from streamlit.testing.v1 import AppTest

from auth import service

APP = str(Path(__file__).resolve().parent.parent / "tessaly_app.py")
PW = "Valid-pass-2024"


def _walk(node):
    yield node
    for child in getattr(node, "children", {}).values():
        yield from _walk(child)


def _html(at) -> str:
    """Every markdown and st.html body in the rendered tree, nested containers included."""
    out = []
    for node in _walk(at._tree):
        if getattr(node, "type", "") == "markdown":
            out.append(node.value)
        elif getattr(node, "type", "") == "html":
            out.append(node.proto.body)
    return " ".join(out)


def _fresh_session():
    user = service.register("Flow Tester", f"f{uuid.uuid4().hex[:10]}@example.com", PW, PW)
    return user, service.session_id_for(service.create_session(user.id, remember=False))


def test_visitor_gets_the_public_homepage_and_no_app_pages():
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["_auth_sid"] = None
    at.run()
    assert not at.exception
    html = _html(at)
    assert "Turn Business Data Into" in html and "Explore Platform" in html and 'id="lp-features"' in html
    assert "Supplier performance analysis" in html and "Inventory optimization" in html
    assert not any(b.key in {"home_sample", "clear_ds", "tb_logout"} for b in at.button)   # no app UI rendered
    assert "ingestion_result" not in at.session_state


def test_signed_in_user_gets_the_app_with_their_name_in_the_account_menu():
    user, sid = _fresh_session()
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["_auth_sid"] = sid
    at.run()
    assert not at.exception
    assert any(b.key == "home_sample" for b in at.button)
    assert f"<b>{user.full_name}</b>" in _html(at) and any(b.key == "tb_logout" for b in at.button)


def test_revoked_session_drops_back_to_public_pages_and_forgets_data():
    _, sid = _fresh_session()
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["_auth_sid"] = sid
    at.run()
    at.button(key="home_sample").click().run()
    assert "ingestion_result" in at.session_state
    service.revoke_session(sid)                         # e.g. signed out on another device
    at.run()
    assert not at.exception
    assert "ingestion_result" not in at.session_state and "Turn Business Data Into" in _html(at)


def test_sign_out_revokes_session_and_clears_state():
    _, sid = _fresh_session()
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["_auth_sid"] = sid
    at.run()
    at.button(key="home_sample").click().run()
    at.button(key="tb_logout").click().run()
    assert not at.exception
    assert service.validate_session(sid) is None
    assert "ingestion_result" not in at.session_state and "_auth_sid" not in at.session_state
    assert "Turn Business Data Into" in _html(at)


def _login_script():
    from ui.auth_pages import render_login_page
    render_login_page()


def _register_script():
    from ui.auth_pages import render_register_page
    render_register_page()


def test_login_form_rejects_wrong_password_and_signs_in_with_right_one():
    user, _ = _fresh_session()
    at = AppTest.from_function(_login_script, default_timeout=30).run()
    assert [c.label for c in at.checkbox] == ["Remember me for 30 days"]
    assert at.checkbox[0].disabled                     # no cookie routes when the page script runs on its own
    at.text_input(key="li_email").input(user.email)
    at.text_input(key="li_password").input("Wrong-pass-999")
    at.button[0].click().run()
    assert [e.value for e in at.error] == [service.GENERIC_LOGIN_ERROR]
    assert "_auth_sid" not in at.session_state
    at.text_input(key="li_password").input(PW)
    at.button[0].click().run()
    assert not at.exception
    sess = service.validate_session(at.session_state["_auth_sid"])
    assert sess.user.id == user.id and at.session_state["_auth_target"] == "dashboard"


def test_forgot_password_says_plainly_that_reset_is_not_configured():
    at = AppTest.from_function(_login_script, default_timeout=30).run()
    at.button(key="li_forgot").click().run()
    assert any("not configured" in (i.proto.title or "") for i in at.info)


def test_registration_creates_account_and_signs_in():
    email = f"r{uuid.uuid4().hex[:10]}@example.com"
    at = AppTest.from_function(_register_script, default_timeout=30).run()
    at.text_input(key="rg_name").input("New Person")
    at.text_input(key="rg_email").input(email)
    at.text_input(key="rg_password").input(PW)
    at.text_input(key="rg_confirm").input(PW)
    at.run()
    html = _html(at)
    assert 'class="au-meter s4"' in html and "Very strong" in html and html.count('<li class="ok">') == 3
    at.button(key="rg_submit").click().run()
    assert not at.exception
    assert service.validate_session(at.session_state["_auth_sid"]).user.email == email


def test_registration_reports_duplicate_email():
    user, _ = _fresh_session()
    at = AppTest.from_function(_register_script, default_timeout=30).run()
    at.text_input(key="rg_name").input("Copy Cat")
    at.text_input(key="rg_email").input(user.email.upper())
    at.text_input(key="rg_password").input(PW)
    at.text_input(key="rg_confirm").input(PW)
    at.button(key="rg_submit").click().run()
    assert any("already exists" in e.value for e in at.error)
    assert "_auth_sid" not in at.session_state

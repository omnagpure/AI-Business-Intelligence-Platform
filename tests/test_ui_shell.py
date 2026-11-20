"""App shell: KPI trends, light/dark tokens, the top bar's dataset switcher and reopening the last dataset."""
from __future__ import annotations

import uuid
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from analytics.kpi_trend import kpi_trend, lower_is_better
from auth import service
from ingestion.file_loader import fingerprint_bytes
from storage import repository
from ui import theme

APP = str(Path(__file__).resolve().parent.parent / "vesper_app.py")
PW = "Valid-pass-2024"


# ---- KPI trend -------------------------------------------------------------------------------
def _monthly(values, start="2024-01-01"):
    """One row per day, each month's rows summing to the given value; the last month is complete too."""
    rows = []
    for i, v in enumerate(values):
        days = pd.date_range(pd.Timestamp(start) + pd.DateOffset(months=i), periods=1, freq="MS")
        month_days = pd.date_range(days[0], days[0] + pd.offsets.MonthEnd(0), freq="D")
        rows += [{"d": d, "revenue": v / len(month_days), "cust": f"c{j % 3}"} for j, d in enumerate(month_days)]
    return pd.DataFrame(rows)


def test_trend_uses_complete_periods_and_compares_last_two():
    df = _monthly([100, 120, 150, 120])
    t = kpi_trend(df, "SUM(revenue)", (), "d")
    assert [round(v) for v in t.points] == [100, 120, 150, 120]
    assert round(t.change_pct, 1) == -20.0 and t.period_label == "month"


def test_partial_last_month_is_left_out():
    df = _monthly([100, 120, 150, 120])
    df = df[df["d"] < "2024-04-15"]                         # April only half covered
    t = kpi_trend(df, "SUM(revenue)", (), "d")
    assert len(t.points) == 3 and round(t.change_pct) == 25


@pytest.mark.parametrize("formula, filters", [
    ("SUM(revenue) / COUNT DISTINCT(cust)", ()),            # a ratio: no comparable per-period trend
    ("SUM(revenue)", ("status = delivered",)),             # a filtered KPI: the trend would ignore the filter
    ("SUM(missing_col)", ()),
])
def test_no_trend_when_it_cannot_be_computed_honestly(formula, filters):
    assert kpi_trend(_monthly([1, 2, 3, 4]), formula, filters, "d") is None


def test_no_trend_without_a_date_or_with_too_few_periods():
    assert kpi_trend(_monthly([1, 2, 3, 4]), "SUM(revenue)", (), None) is None
    assert kpi_trend(_monthly([1, 2]), "SUM(revenue)", (), "d") is None


def test_distinct_count_and_row_count_trends():
    df = _monthly([10, 20, 30])
    assert kpi_trend(df, "COUNT DISTINCT(cust)", (), "d").points == [3.0, 3.0, 3.0]
    assert kpi_trend(df, "COUNT(rows after filters)", (), "d").points == [31.0, 29.0, 31.0]


def test_cost_like_kpis_colour_increases_as_bad():
    assert lower_is_better("avg_lead_time") and lower_is_better("stockout_rate")
    assert not lower_is_better("total_revenue")


# ---- theme tokens --------------------------------------------------------------------------------
def test_light_and_dark_define_the_same_tokens():
    assert set(theme.LIGHT) == set(theme.DARK)


@pytest.mark.parametrize("mode, expect", [("light", ["--bg:#FBFAF8"]), ("dark", ["--bg:#121110"]),
                                          ("system", ["--bg:#FBFAF8", "prefers-color-scheme: dark", "--bg:#121110"])])
def test_token_css_per_mode(monkeypatch, mode, expect):
    monkeypatch.setattr(theme, "mode", lambda: mode)
    css = theme.token_css()
    assert all(e in css for e in expect)
    if mode != "system":
        assert "prefers-color-scheme" not in css


def test_config_toml_colours_match_tokens():
    import tomllib
    cfg = tomllib.loads((Path(APP).parent / ".streamlit" / "config.toml").read_text(encoding="utf-8"))["theme"]
    for name, tokens in (("light", theme.LIGHT), ("dark", theme.DARK)):
        assert cfg[name]["backgroundColor"] == tokens["bg"]
        assert cfg[name]["primaryColor"] == tokens["accent"]
        assert cfg[name]["textColor"] == tokens["text"]


# ---- top bar ---------------------------------------------------------------------------------------
def _user_with_datasets(n):
    user = service.register("Shell Tester", f"s{uuid.uuid4().hex[:10]}@example.com", PW, PW)
    recs = []
    for i in range(n):
        data = f"order_date,revenue\n2024-01-0{i + 1},{10 * (i + 1)}\n2024-02-0{i + 1},5\n".encode()
        recs.append(repository.save_dataset(user.id, data, f"orders_{i}.csv", None, fingerprint_bytes(data), 2, 2))
    return user, service.session_id_for(service.create_session(user.id, remember=False)), recs


def test_last_used_dataset_reopens_after_sign_in():
    user, sid, recs = _user_with_datasets(2)
    repository.read_dataset(user.id, recs[0].id)              # opened most recently
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["_auth_sid"] = sid
    at.run()
    assert not at.exception
    assert at.session_state["active_dataset_id"] == recs[0].id
    assert at.session_state["ingestion_result"].dataset.filename == "orders_0.csv"


def test_dataset_switcher_lists_only_own_datasets_and_switches():
    _user_with_datasets(1)                                  # another user, whose files must not appear
    _, sid, recs = _user_with_datasets(2)
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["_auth_sid"] = sid
    at.run()
    box = next(s for s in at.selectbox if str(s.key).startswith("tb_ds_"))
    assert sorted(box.options) == sorted(["orders_0.csv", "orders_1.csv"])   # never another user's files
    target = next(r for r in recs if r.id != at.session_state["active_dataset_id"])
    box.set_value(target.id).run()
    assert not at.exception
    assert at.session_state["active_dataset_id"] == target.id


def test_closing_a_dataset_does_not_reopen_it():
    _, sid, _ = _user_with_datasets(1)
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["_auth_sid"] = sid
    at.run()
    assert "ingestion_result" in at.session_state
    at.button(key="clear_ds").click().run()
    at.run()
    assert not at.exception and "ingestion_result" not in at.session_state


# ---- appearance switch after sign-in ---------------------------------------------------------------
def _signed_in_app(page_key: str | None = None):
    _, sid, _ = _user_with_datasets(0)
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["_auth_sid"] = sid
    at.run()
    if page_key:
        at.session_state["_auth_target"] = page_key
        at.run()
    return at, sid


def _html_bodies(at) -> str:
    def walk(node):
        yield node
        for child in getattr(node, "children", {}).values():
            yield from walk(child)
    return " ".join(n.proto.body for n in walk(at._tree) if getattr(n, "type", "") == "html")


def test_switching_theme_never_signs_out_and_recolours_in_place_without_routes():
    at, sid = _signed_in_app()
    at.segmented_control(key="tb_theme").set_value("dark").run()
    assert not at.exception
    assert service.validate_session(sid) is not None and at.session_state["_auth_sid"] == sid    # still signed in
    html = _html_bodies(at)
    assert "--bg:#121110" in html and "prefers-color-scheme" not in html                      # dark tokens now
    assert "bi_theme=dark" in html and '"\\"Dark\\""' in html                              # remembered in the browser
    assert "location.replace" not in html and "location.reload" not in html                   # and no reload (it would sign out)


def test_theme_controls_stay_in_step_and_do_not_flip_back():
    at, _ = _signed_in_app("account")
    assert {c.key for c in at.segmented_control} >= {"tb_theme", "acct_theme"}
    at.segmented_control(key="acct_theme").set_value("light").run()
    assert at.segmented_control(key="tb_theme").value == "light"
    at.run()                                                     # a later run must not switch back
    at.run()
    assert at.session_state["_theme_mode"] == "light"
    assert "--bg:#FBFAF8" in _html_bodies(at)


def test_with_routes_the_switch_reloads_and_reopens_the_sample(monkeypatch):
    monkeypatch.setenv("BI_AUTH_ROUTES", "1")
    at, _ = _signed_in_app()
    at.button(key="home_sample").click().run()
    at.segmented_control(key="tb_theme").set_value("dark").run()
    html = _html_bodies(at)
    assert "location.replace(window.location.pathname + '?resume=sample')" in html
    # the reloaded page: a new session, signed in through the cookie, asked to reopen the sample
    _, sid, _ = _user_with_datasets(0)
    fresh = AppTest.from_file(APP, default_timeout=60)
    fresh.session_state["_auth_sid"] = sid
    fresh.query_params["resume"] = "sample"
    fresh.run()
    assert not fresh.exception
    assert fresh.session_state["ingestion_result"].dataset.is_sample
    assert "resume" not in fresh.query_params


def test_app_py_mounts_the_cookie_routes_around_the_pages():
    import ast
    tree = ast.parse((Path(APP).parent / "app.py").read_text(encoding="utf-8"))
    src = ast.unparse(tree)
    assert "st.App(" in src and "vesper_app.py" in src and "SESSION_PATH" in src and "LOGOUT_PATH" in src
    assert "from app import app" in (Path(APP).parent / "server.py").read_text(encoding="utf-8")

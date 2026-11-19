"""Phase 8 reporting: content comes from computed results, skipped analyses are stated, user text is escaped."""
import re
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from analytics.forecasting import ForecastResult, ModelScore
from analytics.kpis import compute_kpis
from analytics.formatting import format_value
from ingestion.pipeline import ingest_sample
from insights.insight_engine import generate_insights
from insights.insight_rules import InsightInputs
from reporting import theme
from reporting.html_report import render_html
from reporting.pdf_report import render_pdf
from reporting.report_builder import _flag_total, build_report, clean, safe_filename
from semantic.semantic_model import build_semantic_model

WHEN = datetime(2026, 1, 15, 9, 30)


def _forecast():
    hist = pd.DataFrame({"period": pd.date_range("2024-01-31", periods=12, freq="ME"), "value": [100.0] * 6 + [200.0] * 6})
    fut = pd.DataFrame({"period": pd.date_range("2025-01-31", periods=3, freq="ME"), "forecast": [300.0] * 3, "lower": [250.0] * 3, "upper": [350.0] * 3})
    sc = (ModelScore("naive", "Naive", 1, "ok", mae=10.0, rmse=12.0, mase=1.0, mape=5.0, n_errors=3),
          ModelScore("ses", "Exponential smoothing", 2, "ok", mae=8.0, rmse=9.0, mase=0.8, mape=4.0, n_errors=3))
    return ForecastResult("date", "revenue", "sum", "M", 3, hist, fut, pd.DataFrame(), sc, "ses", "naive", 20.0, 0.95, 12, True, 3, 3)


@pytest.fixture(scope="module")
def ctx():
    r = ingest_sample()
    df, p = r.dataset.df, r.profile
    m = build_semantic_model(df, p, {}, None, None)
    k = compute_kpis(df, m)
    rep = generate_insights(InsightInputs(kpis=k, profile=p))
    return r, df, p, m, k, rep


def _build(ctx, **over):
    r, df, p, m, k, rep = ctx
    kw = dict(filename=r.dataset.filename, is_sample=True, fingerprint=r.dataset.fingerprint, profile=p, quality=r.quality, analysis_rows=len(df),
              kpis=k, insight_report=rep, semantic_model=m, generated_at=WHEN)
    kw.update(over)
    return build_report(**kw)


def test_palette_is_three_distinct_colours_and_greys_are_mixes():
    base = {theme.PAPER, theme.INK, theme.ACCENT}
    assert len(base) == 3 and all(re.fullmatch(r"#[0-9A-F]{6}", c) for c in base)
    assert theme._mix("#000000", "#FFFFFF", 0.5) == "#808080"
    assert theme._mix(theme.INK, theme.PAPER, 0.0) == theme.INK


def test_report_id_is_deterministic_and_sections_numbered(ctx):
    a, b = _build(ctx), _build(ctx)
    assert a.report_id == b.report_id and re.fullmatch(r"[0-9A-F]{4}-[0-9A-F]{4}", a.report_id)
    assert [s.number for s in a.sections] == list(range(2, 2 + len(a.sections)))


def test_skipped_analyses_are_stated_with_reason(ctx):
    d = _build(ctx, unavailable={"forecast": "No date column."})
    f = d.section("forecast")
    assert f.status == "skipped" and f.reason == "No date column."
    assert ("Forecast", "No date column.") in d.skipped
    assert any("Not included" in p for p in d.summary)


def test_report_still_builds_with_only_profile_and_quality(ctx):
    r, df, p, *_ = ctx
    d = build_report(filename="x.csv", is_sample=False, fingerprint="f", profile=p, quality=r.quality, analysis_rows=len(df), generated_at=WHEN)
    assert d.section("kpis").status == "skipped" and d.section("dataset").status == "ok"
    assert render_pdf(d).startswith(b"%PDF") and b"</html>" in render_html(d)


def test_kpi_values_in_report_match_kpi_results_and_plain_pandas(ctx):
    r, df, p, m, k, rep = ctx
    d = _build(ctx)
    tbl = d.section("kpis").tables[0]
    shown = {row[1]: row[2] for row in tbl.rows}
    for res in k.ok():
        assert shown[res.name[:46]] == format_value(res.value, res.unit)
    rev = k.get("total_revenue")
    assert rev.value == pytest.approx(df["revenue"].sum())                          # independent pandas check
    assert shown[rev.name[:46]] == format_value(df["revenue"].sum(), rev.unit)


def test_forecast_lead_is_computed_by_hand(ctx):
    d = _build(ctx, forecast=_forecast())
    lead = d.section("forecast").lead
    assert "above" in lead and "50.0%" in lead                                    # forecast 300 vs last 3 observed mean 200
    assert "(200.00)" in lead or "(200)" in lead


def test_broken_result_skips_only_its_own_section(ctx):
    d = _build(ctx, forecast=object())
    assert "Not included" in " ".join(d.summary)
    assert d.section("forecast").status == "skipped"
    assert d.section("kpis").status == "ok"


def test_html_escapes_user_text_and_is_self_contained(ctx):
    r, df, p, m, k, rep = ctx
    d = _build(ctx, title="<script>alert(1)</script>", prepared_by='"><img src=x onerror=1>', forecast=_forecast())
    html = render_html(d).decode()
    assert "<script>alert" not in html and "&lt;script&gt;" in html
    assert "<img src=x" not in html
    assert not re.search(r'(src|href)="https?://', html)
    assert "data:image/png;base64," in html


def test_pdf_is_multipage_with_bookmarks(ctx):
    pdf = render_pdf(_build(ctx, forecast=_forecast()))
    assert pdf.startswith(b"%PDF") and len(re.findall(rb"/Type\s*/Page\b", pdf)) >= 4
    assert b"/Outlines" in pdf


def test_anomaly_total_does_not_double_count_methods():
    s = pd.DataFrame({"feature": ["a", "a", "b"], "flagged": [10, 10, 3]})
    assert _flag_total(s) == 13
    assert _flag_total(s.iloc[0:0]) == 0


def test_text_helpers():
    assert clean("a\x00b\nc", 3) == "a b…" or clean("a\x00b\nc", 3).endswith("…")
    assert safe_filename("../../etc/passwd") == "etc_passwd"
    assert safe_filename("???") == "report"


def test_methodology_states_which_settings_were_used(ctx):
    assert any("default settings" in m for m in _build(ctx).methodology)
    d = _build(ctx, user_settings=("Forecasting", "Drivers"))
    assert any("Forecasting, Drivers" in m for m in d.methodology)

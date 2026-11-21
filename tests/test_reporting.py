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
from reporting.report_builder import _flag_total, build_report, clean, kpi_status, safe_filename
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
    assert ("Outlook", "No date column.") in d.skipped
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


# ---- business-report structure (Phase 9, batch 9) ------------------------------------------------
def test_body_comes_before_appendix_and_labels_follow_parts(ctx):
    d = _build(ctx, forecast=_forecast())
    main, appx = d.shown("main"), d.shown("appendix")
    assert main[0].key == "insights" and [s.label for s in main] == [str(i) for i in range(1, len(main) + 1)]
    assert {s.key for s in appx} >= {"dataset", "quality", "definitions"}
    assert [s.label for s in appx] == [chr(ord("A") + i) for i in range(len(appx))]
    assert d.method_label == chr(ord("A") + len(appx))
    assert d.section("recs") is None                                  # actions live with their findings now
    assert all(s.label == "" for s in d.sections if s.status == "skipped")


def test_executive_summary_has_scorecard_takeaways_and_actions(ctx):
    d = _build(ctx)
    k = ctx[4]
    assert d.lede.startswith(f"This review covers {len(ctx[1]):,} records")
    assert 1 <= len(d.tiles) <= 4 and all(t.status in ("good", "warn", "bad", "neutral") for t in d.tiles)
    by_name = {r.name: format_value(r.value, r.unit) for r in k.results if r.status == "ok"}
    assert all(by_name.get(t.label) == t.value for t in d.tiles)      # every tile is a computed KPI, unchanged
    assert d.takeaways and len(d.actions) <= 5
    assert len({a.action for a in d.actions}) == len(d.actions)
    assert d.headline == tuple((t.label, t.value) for t in d.tiles)


def test_kpi_status_uses_the_insight_reference_levels():
    class R:
        def __init__(self, key, value):
            self.key, self.value = key, value
    assert kpi_status(R("on_time_rate", 95))[0] == "good"
    assert kpi_status(R("on_time_rate", 85))[0] == "warn"
    assert kpi_status(R("on_time_rate", 42.7)) == ("bad", "Below 90% reference")
    assert kpi_status(R("stockout_rate", 20))[0] == "bad"
    assert kpi_status(R("profit_margin", -1))[0] == "bad"
    assert kpi_status(R("total_revenue", 1e6)) == ("neutral", "")


def test_finding_cards_carry_why_and_action(ctx):
    d = _build(ctx)
    cards = d.section("insights").cards
    assert cards and all(c.kind in ("risk", "opportunity", "pattern", "data") for c in cards)
    assert any(c.action and c.why for c in cards)


def test_html_has_summary_contents_and_no_raw_markup(ctx):
    html = render_html(_build(ctx)).decode()
    assert 'id="summary"' in html and 'class="toc"' in html and "Appendix" in html
    assert "&lt;font" not in html and "<font" not in html


# ---- report design round 2: data-driven cover, at-a-glance, trends, action plan, glossary -------------
from reporting import fonts
from reporting.report_builder import AUTO_TITLES, readable


def test_title_comes_from_the_data_unless_typed(ctx):
    d = _build(ctx)
    assert d.title == AUTO_TITLES.get(ctx[3].domain.domain, AUTO_TITLES["general"])
    assert _build(ctx, title="Q3 board pack").title == "Q3 board pack"
    assert d.key_message and d.key_message[0].isupper() and d.key_message.endswith(".")


def test_readable_column_names():
    assert readable("Late_delivery_risk") == "Late delivery risk"
    assert readable("OrderDate") == "Order Date"
    assert readable("Order Item Total") == "Order Item Total"


def test_every_shown_section_has_a_headline_and_description(ctx):
    d = _build(ctx, forecast=_forecast(), df=ctx[1])
    for s in d.shown("main"):
        assert s.headline and s.desc, s.key
    assert all(s.desc for s in d.shown("appendix"))


def test_rows_enable_trends_comparison_and_breakdown(ctx):
    with_df = _build(ctx, df=ctx[1])
    without = _build(ctx)
    assert with_df.section("trends").status == "ok" and with_df.section("trends").charts
    assert without.section("trends").status == "skipped" and ("Performance over time", without.section("trends").reason) not in without.skipped
    assert with_df.cover_png and with_df.cover_png.startswith(b"\x89PNG")
    assert all(c.half for c in with_df.section("trends").charts)


def test_action_plan_has_owner_timeframe_and_measure(ctx):
    d = _build(ctx)
    plan_sec = d.section("plan")
    assert plan_sec.status == "ok"
    t = plan_sec.tables[0]
    assert t.header == ("#", "Action", "Suggested owner", "When", "Measure of success", "Priority")
    assert all(r[2] and r[3] and r[4] for r in t.rows)
    assert all(a.owner and a.timeframe for a in d.actions)


def test_conclusion_and_glossary(ctx):
    d = _build(ctx, forecast=_forecast())
    c = d.section("conclusion")
    assert c.status == "ok" and c.lead.startswith("Overall,") and c.steps
    g = d.section("glossary")
    assert g.status == "ok" and any(row[0] == "Forecast baseline" for row in g.tables[0].rows)


def test_branding_reaches_both_outputs(ctx):
    logo = (__import__("pathlib").Path(__file__).resolve().parent.parent / "assets" / "favicon.png").read_bytes()
    d = _build(ctx, organisation="Northwind", logo=logo, accent="#0E6466", classification="Confidential", df=ctx[1])
    assert d.accent == "#0E6466" and d.classification == "Confidential"
    html = render_html(d).decode()
    assert "Northwind" in html and "--accent:#0E6466" in html and "data:image/png;base64," in html and "CONFIDENTIAL" in html
    pdf = render_pdf(d)
    assert pdf.startswith(b"%PDF") and b"Page 2 of" not in pdf[:100]          # page numbers are drawn, not plain text at the top


def test_bad_logo_is_ignored_by_html_and_pdf(ctx):
    d = _build(ctx, logo=b"not an image")
    assert "data:image/" not in render_html(d).decode().split("<main>")[0].split('class="brand"')[1][:200]
    assert render_pdf(d).startswith(b"%PDF")


def test_plex_fonts_ship_and_register():
    assert fonts.available()
    names = fonts.pdf_fonts()
    assert names["regular"].startswith("Plex") and names["symbol"]
    assert "@font-face" in fonts.html_font_faces()

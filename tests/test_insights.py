"""Insight engine: hand-calculated rule outcomes, determinism, evidence and recommendation links, failure isolation."""
import numpy as np
import pandas as pd
import pytest

import insights.insight_engine as engine
from analytics.anomaly_detection import AnomalyReport
from analytics.drivers import DriverResult
from analytics.forecasting import ForecastResult, ModelScore
from analytics.kpis import NOT_CALCULABLE, OK, SKIPPED, KPIReport, KPIResult
from analytics.recommendations import recommend
from analytics.segmentation import SegmentationResult
from config.analytics_config import InsightConfig
from core.exceptions import InsufficientDataError
from ingestion.profiler import profile_dataframe
from insights.default_runs import default_anomalies, default_drivers, default_forecast, default_segmentation
from insights.evidence import Evidence, Insight
from insights.insight_engine import generate_insights
from insights.insight_rules import RULES, InsightInputs, Rule
from semantic.semantic_model import build_semantic_model


def kpi(key, value, unit="percent", status=OK, name=None, breakdown=(), period="2024-02 vs 2024-01", reason=""):
    return KPIResult(key, name or key, "General", status, value, unit, "formula", ("col",), (), (), "t", reason, period, breakdown)


def report(*results, rows=1000):
    return KPIReport(tuple(results), rows, rows, "general")


def run(*results, rows=1000, **kw):
    return generate_insights(InsightInputs(kpis=report(*results, rows=rows), **kw))


def titles(rep):
    return [i.title for i in rep.insights]


# ---- KPI rules ----------------------------------------------------------------------------------
def test_growth_thresholds():
    assert not run(kpi("revenue_growth", 5.0, name="Revenue growth")).insights                  # small change: nothing
    mid = run(kpi("revenue_growth", 15.0, name="Revenue growth")).insights[0]
    assert (mid.severity, mid.direction) == ("medium", "up")
    big = run(kpi("revenue_growth", -34.2, name="Revenue growth")).insights[0]
    assert big.severity == "high" and "fell 34.2%" in big.title and big.period == "2024-02 vs 2024-01"


def test_growth_ignores_not_calculable():
    rep = run(kpi("revenue_growth", None, status=NOT_CALCULABLE))
    assert not [i for i in rep.insights if i.rule == "growth"]                  # no growth claim without a growth value


def test_low_is_bad_levels():
    assert not run(kpi("on_time_rate", 95.0)).insights
    assert run(kpi("on_time_rate", 85.0)).insights[0].severity == "medium"      # below 90, not below 80
    assert run(kpi("on_time_rate", 75.0)).insights[0].severity == "high"        # below 80
    assert run(kpi("fill_rate", 96.0)).insights == ()


def test_high_is_bad_levels():
    assert not run(kpi("stockout_rate", 4.9)).insights
    assert run(kpi("stockout_rate", 5.0)).insights[0].severity == "medium"       # at the medium level
    i = run(kpi("stockout_rate", 20.0)).insights[0]
    assert i.severity == "high" and i.group == "Supply chain" and "adjustable" in i.thresholds


def test_category_share_names_the_top_named_category():
    r = kpi("revenue_by_category", 62.0, breakdown=(("Tools", 62.0), ("Toys", 20.0), ("Other", 18.0)))
    i = run(r).insights[0]
    assert i.severity == "high" and i.subject == "Tools"
    assert not run(kpi("revenue_by_category", 30.0)).insights


def test_margin():
    assert run(kpi("profit_margin", -3.0)).insights[0].severity == "high"
    assert run(kpi("profit_margin", 2.0)).insights[0].severity == "medium"
    assert not run(kpi("profit_margin", 12.0)).insights


def test_confidence_follows_rows_analysed():
    r = kpi("revenue_growth", 40.0)
    assert [run(r, rows=n).insights[0].confidence for n in (10, 100, 1000)] == ["low", "medium", "high"]


def test_unavailable_kpis_only_when_important():
    broken = kpi("profit_margin", None, status=NOT_CALCULABLE, reason="total revenue is zero")
    i = run(broken).insights[0]
    assert i.rule == "kpi_unavailable" and i.severity == "medium" and i.evidence[0].detail == "total revenue is zero"
    absent = kpi("on_time_rate", None, status=SKIPPED, reason="No date pair.")
    assert not run(absent).insights                                     # general data: a missing optional KPI is not a finding
    sc = KPIReport((absent, kpi("stockout_rate", 1.0), kpi("below_reorder_share", 1.0)), 1000, 1000, "supply_chain")
    got = generate_insights(InsightInputs(kpis=sc)).insights
    assert [(i.rule, i.severity) for i in got] == [("kpi_unavailable", "low")]


# ---- data-quality rules -------------------------------------------------------------------------
def _profile_insights(df):
    return {i.rule: i for i in generate_insights(InsightInputs(profile=profile_dataframe(df))).insights}


def test_missing_severity_from_profile():
    # 'b': 6 of 12 rows empty = 50.0% -> high (threshold 50); 'c': 3 of 12 = 25.0% -> listed but not the worst
    df = pd.DataFrame({"a": range(12), "b": [None] * 6 + [1.0] * 6, "c": [None] * 3 + [1.0] * 9})
    i = _profile_insights(df)["missing"]
    assert i.severity == "high" and i.subject == "b"
    assert [e.value for e in i.evidence] == ["50.0%", "25.0%"] and i.evidence[0].columns == ("b",)
    medium = pd.DataFrame({"a": range(10), "c": [None] * 3 + [1.0] * 7})           # 30% -> medium
    assert _profile_insights(medium)["missing"].severity == "medium"


def test_duplicates_and_small_dataset():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "b": list("abcdefghij")})
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)                           # 1 of 11 = 9.1% duplicates
    by = _profile_insights(df)
    assert by["duplicates"].severity == "medium" and "1 of 11" not in by["duplicates"].title
    assert by["small_dataset"].severity == "high"                                   # 11 rows < 30
    assert "duplicates" not in _profile_insights(df.drop_duplicates())


# ---- Phase 6 results (hand-built so the expected numbers are exact) -----------------------------
def forecast(chosen="holt", improvement=20.0, level=100.0, fc=130.0, n_hist=10, h=4):
    scores = (ModelScore("naive", "Naive", 0, "ok"), ModelScore("holt", "Holt", 2, "ok"))
    hist = pd.DataFrame({"period": pd.date_range("2024-01-01", periods=n_hist, freq="MS"), "value": [level] * n_hist})
    fut = pd.DataFrame({"period": pd.date_range("2024-11-01", periods=h, freq="MS"), "forecast": [fc] * h,
                        "lower": [fc - 5] * h, "upper": [fc + 5] * h})
    return ForecastResult("date", "units", "sum", "M", h, hist, fut, pd.DataFrame(), scores, chosen, "naive", improvement,
                          0.8, 12, True, h, 3)


def test_forecast_direction_hand_calculated():
    i = {x.rule: x for x in generate_insights(InsightInputs(forecast=forecast())).insights}["forecast_direction"]
    assert "rises about 30%" in i.title and i.severity == "high" and i.confidence == "high" and i.direction == "up"
    flat = generate_insights(InsightInputs(forecast=forecast(fc=105.0))).insights                # +5% < 10%: nothing
    assert not [x for x in flat if x.rule == "forecast_direction"]
    weak = generate_insights(InsightInputs(forecast=forecast(improvement=3.0))).insights[0]
    assert weak.confidence == "medium"


def test_baseline_forecast_gives_no_direction_but_a_reliability_note():
    rules = [x.rule for x in generate_insights(InsightInputs(forecast=forecast(chosen="naive", improvement=0.0, fc=200.0))).insights]
    assert rules == ["forecast_reliability"]


def anomaly(share, status="ok"):
    s = pd.DataFrame([{"method": "mad", "feature": "sales", "examined": 1000, "flagged": int(share * 1000), "share": share,
                       "threshold": 3.5, "status": status, "note": ""}])
    return AnomalyReport(pd.DataFrame(), s, 3)


def test_anomaly_share_levels():
    assert not generate_insights(InsightInputs(anomalies=anomaly(0.02))).insights
    assert generate_insights(InsightInputs(anomalies=anomaly(0.05))).insights[0].severity == "medium"
    hi = generate_insights(InsightInputs(anomalies=anomaly(0.12))).insights[0]
    assert hi.severity == "high" and hi.evidence[0].value == "12.0%" and "not necessarily an error" in " ".join(hi.limitations)
    assert not generate_insights(InsightInputs(anomalies=anomaly(0.5, status="skipped"))).insights


def segmentation(silhouette=0.6, cluster_mean=300.0):
    prof = pd.DataFrame({"entities": [2, 8], "share": [0.2, 0.8], "sum_value": [cluster_mean, 25.0],
                         "on_time": [50.0, 90.0]}, index=[1, 2])
    return SegmentationResult(pd.DataFrame(), prof, pd.Series({"sum_value": 100.0, "on_time": 82.0}), {1: "higher value"},
                              pd.DataFrame(), 2, "automatic", silhouette, "q", ("sum_value", "on_time"),
                              {"sum_value": "value", "on_time": "on-time"}, {"sum_value": "amount", "on_time": "percent"}, "robust")


def test_segment_value_concentration_hand_calculated():
    # cluster 1: 20% of entities, mean 300 vs overall 100 -> ratio 3.0, roughly 0.2 x 3 = 60% of the total (>= 50% -> high)
    i = generate_insights(InsightInputs(segmentation=segmentation())).insights[0]
    assert i.severity == "high" and i.confidence == "high" and "3.0x" in i.title and "60%" in i.description
    assert generate_insights(InsightInputs(segmentation=segmentation(cluster_mean=150.0))).insights == ()   # 1.5x < 2x
    assert generate_insights(InsightInputs(segmentation=segmentation(silhouette=0.2))).insights == ()        # clusters too weak


def drivers(reliable=True, share=0.6):
    imp = pd.DataFrame({"feature": ["promo", "region"], "kind": ["numeric", "categorical"], "importance": [3.0, 2.0],
                        "std": [0.1, 0.1], "share": [share, 1 - share], "direction": ["higher values go with higher target"] * 2})
    m = pd.DataFrame({"metric": ["R-squared (held-out)"], "model": [0.7], "baseline": [-0.1]})
    return DriverResult("sales", "regression", 500, 375, 125, "chronological", m, reliable, imp, pd.DataFrame(), {}, {})


def test_drivers_only_when_reliable_and_marked_as_association():
    i = generate_insights(InsightInputs(drivers=drivers())).insights[0]
    assert i.severity == "info" and i.confidence == "medium" and "not cause and effect" in " ".join(i.limitations)
    assert generate_insights(InsightInputs(drivers=drivers(reliable=False))).insights == ()
    assert generate_insights(InsightInputs(drivers=drivers(share=0.2))).insights == ()
    assert generate_insights(InsightInputs(drivers=drivers(share=0.95))).insights[0].confidence == "low"   # near-copy warning


# ---- engine ----------------------------------------------------------------------------------------
def test_ranking_puts_high_before_medium_and_is_deterministic():
    inp = InsightInputs(kpis=report(kpi("revenue_growth", 15.0), kpi("stockout_rate", 20.0), kpi("defect_rate", 3.0)))
    a, b = generate_insights(inp), generate_insights(inp)
    assert a == b
    assert [i.severity for i in a.insights] == ["high", "medium", "medium"]
    assert [i.priority for i in a.insights] == sorted((i.priority for i in a.insights), reverse=True)
    assert a.insights[0].priority == 33                                           # 10 x 3 (high) + 3 (high confidence)


def test_every_insight_has_evidence_and_every_recommendation_links_to_an_insight():
    rep = generate_insights(InsightInputs(kpis=report(kpi("revenue_growth", -30.0), kpi("on_time_rate", 70.0)),
                                          forecast=forecast(), anomalies=anomaly(0.12), segmentation=segmentation(),
                                          drivers=drivers()))
    ids = {i.id for i in rep.insights}
    assert len(rep.insights) >= 6 and all(i.evidence for i in rep.insights)
    assert rep.recommendations and all(r.insight_id in ids for r in rep.recommendations)
    assert all(i.next_step for i in rep.insights) and len({i.id for i in rep.insights}) == len(rep.insights)


def test_insight_without_evidence_is_rejected():
    with pytest.raises(ValueError):
        Insight("x", "r", "Performance", "t", "d", (), "low", "high")


def test_missing_inputs_are_reported_with_the_reason_not_invented():
    rep = generate_insights(InsightInputs(kpis=report(), unavailable=(("forecast", "No date column."),)))
    skipped = dict(rep.skipped)
    assert skipped["forecast_direction"] == "No date column."
    assert "results were not available" in skipped["anomalies"] and "profile" in skipped["missing"]
    assert "forecast" not in rep.inputs_used and "kpis" in rep.inputs_used


def test_a_failing_rule_does_not_stop_the_others(monkeypatch):
    def boom(inp, cfg):
        raise RuntimeError("bad")
    monkeypatch.setattr(engine, "RULES", (Rule("boom", "kpis", boom),) + tuple(RULES))
    rep = generate_insights(InsightInputs(kpis=report(kpi("stockout_rate", 20.0))))
    assert ("boom", "The rule could not be evaluated on this data.") in rep.skipped
    assert rep.insights and rep.insights[0].rule == "threshold"


def test_nothing_to_report_is_stated():
    rep = generate_insights(InsightInputs(kpis=report(kpi("on_time_rate", 99.0))))
    assert not rep.insights and "No finding crossed a threshold" in rep.notes[0]


def test_custom_thresholds_change_the_outcome():
    cfg = InsightConfig(growth_medium_pct=3.0)
    assert generate_insights(InsightInputs(kpis=report(kpi("revenue_growth", 5.0))), cfg).insights
    assert not generate_insights(InsightInputs(kpis=report(kpi("revenue_growth", 5.0)))).insights


def test_recommendation_templates_are_deterministic_text():
    i = generate_insights(InsightInputs(kpis=report(kpi("revenue_growth", -30.0)))).insights[0]
    assert recommend(i) == recommend(i) == i.next_step and "revenue decline" in i.next_step


# ---- default Phase 6 runs ---------------------------------------------------------------------------
def _daily(n=200):
    rng = np.random.default_rng(1)
    return pd.DataFrame({"date": pd.date_range("2024-01-01", periods=n),
                         "units_sold": np.array([10, 12, 14, 30, 50, 80, 20])[np.arange(n) % 7] + rng.normal(0, 2, n)})


def _model(df):
    return build_semantic_model(df, profile_dataframe(df), {})


def test_default_runs_do_not_mutate_and_raise_clear_errors_when_unsuitable():
    df = _daily()
    before = df.copy()
    f = default_forecast(df, _model(df))
    assert f.value_col == "units_sold" and len(f.forecast) > 0
    pd.testing.assert_frame_equal(df, before)
    with pytest.raises(InsufficientDataError):
        default_forecast(df.head(5), _model(df.head(5)))
    with pytest.raises(InsufficientDataError):
        default_segmentation(df, _model(df))                                        # no entity column
    with pytest.raises(InsufficientDataError):
        default_drivers(pd.DataFrame({"a": ["x", "y"]}), profile_dataframe(pd.DataFrame({"a": ["x", "y"]})),
                        _model(pd.DataFrame({"a": ["x", "y"]})))
    assert default_anomalies(df, profile_dataframe(df), _model(df)).summary is not None

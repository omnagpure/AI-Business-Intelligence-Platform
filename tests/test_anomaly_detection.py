"""Anomaly detection (Phase 6B): hand-calculated rules, sensitivity, Isolation Forest, time series, independence."""
import numpy as np
import pandas as pd
import pytest

from analytics import anomaly_detection as ad
from core.exceptions import AnomalyDetectionError, InsufficientDataError
from ingestion.profiler import profile_dataframe
from semantic.semantic_model import build_semantic_model


def _normalish(n=300, seed=1):
    return np.random.default_rng(seed).normal(100, 10, n)


# ---- rules by hand ----------------------------------------------------------------------------
def test_mad_scores_by_hand():
    x = np.array([10, 11, 9, 10, 12, 10, 11, 9, 10, 100], dtype=float)   # median 10, MAD 1
    z, med, mad = ad.mad_scores(x)
    assert (med, mad) == (10.0, 1.0)
    assert z[-1] == pytest.approx(0.6745 * 90) and z[4] == pytest.approx(0.6745 * 2)
    assert ad.mad_scores(np.array([5.0, 5.0, 5.0, 6.0])) is None          # MAD 0 -> not usable


def test_iqr_fences_by_hand():
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100], dtype=float)       # Q1 3.5, Q3 8.5, IQR 5
    lo, hi, q1, q3 = ad.iqr_fences(x, 1.5)
    assert (q1, q3) == (3.5, 8.5) and hi == 16.0 and lo == -4.0
    assert ad.iqr_fences(np.array([2.0, 2.0, 2.0, 2.0]), 1.5) is None


def test_zscore_only_where_roughly_normal():
    assert ad.zscore_applicable(_normalish())[0] is True
    skewed = np.random.default_rng(2).lognormal(3, 1.2, 300)
    ok, why = ad.zscore_applicable(skewed)
    assert ok is False and "skewed" in why
    assert ad.zscore_applicable(np.arange(10.0))[0] is False


# ---- row-level screening ----------------------------------------------------------------------
def _frame():
    x = _normalish()
    x[[10, 200]] = [400.0, -250.0]
    return pd.DataFrame({"amount": x, "flag": np.tile([0, 1], 150)})


def test_known_outliers_are_flagged_with_explanations_and_nothing_is_removed():
    df = _frame()
    before = df.copy()
    rep = ad.run_anomaly_detection(df, ["amount"], ("mad", "iqr", "zscore"), 3)
    pd.testing.assert_frame_equal(df, before)
    for m in ("mad", "iqr", "zscore"):
        f = rep.findings[rep.findings["method"] == m]
        assert {10, 200} <= set(f["row"]) and f.iloc[0]["score"] > 0
        assert set(f["direction"]) <= {"high", "low"}
    top = rep.findings[rep.findings["method"] == "mad"].iloc[0]
    assert top["row"] == 200 and "robust deviations below the median" in top["explanation"]   # |-250-100| > |400-100|
    assert "not necessarily an error" in top["explanation"] and "not necessarily an error" in rep.note
    assert (rep.summary["status"] == "ok").all()


def test_sensitivity_changes_how_many_values_are_flagged():
    df = pd.DataFrame({"amount": _normalish(2000, 7)})
    counts = [int(ad.run_anomaly_detection(df, ["amount"], ("mad",), s).summary["flagged"].iloc[0]) for s in (1, 3, 5)]
    assert counts[0] <= counts[1] <= counts[2] and counts[0] < counts[2]
    with pytest.raises(AnomalyDetectionError):
        ad.run_anomaly_detection(df, ["amount"], ("mad",), 9)
    with pytest.raises(AnomalyDetectionError):
        ad.run_anomaly_detection(df, ["amount"], ("nope",), 3)


def test_unsuitable_columns_are_skipped_with_a_reason():
    df = pd.DataFrame({"a": np.r_[np.zeros(80), [1.0] * 20], "skew": np.random.default_rng(3).lognormal(3, 1.2, 100)})
    rep = ad.run_anomaly_detection(df, ["a", "skew"], ("mad", "zscore"), 3)
    s = rep.summary.set_index(["method", "feature"])
    assert s.loc[("mad", "a"), "status"] == "skipped" and "identical" in s.loc[("mad", "a"), "note"]
    assert s.loc[("zscore", "skew"), "status"] == "skipped"
    with pytest.raises(InsufficientDataError):
        ad.run_anomaly_detection(df, ["missing_col"], ("mad",), 3)


def test_findings_are_capped_but_counts_stay_exact():
    from config.analytics_config import AnalyticsConfig
    cfg = AnalyticsConfig(anomaly_max_findings=5)
    df = pd.DataFrame({"x": np.r_[_normalish(500), np.full(30, 900.0) + np.arange(30)]})
    rep = ad.run_anomaly_detection(df, ["x"], ("mad",), 3, cfg)
    assert len(rep.findings) == 5 and int(rep.summary["flagged"].iloc[0]) >= 30
    assert any("strongest" in w for w in rep.warnings)


def test_skewed_column_gets_a_warning_when_many_values_flagged():
    df = pd.DataFrame({"x": np.random.default_rng(4).lognormal(3, 1.5, 1000)})
    rep = ad.run_anomaly_detection(df, ["x"], ("mad",), 3)
    assert any("skewed" in w for w in rep.warnings)


def test_eligible_columns_skip_flags_ids_and_short_columns():
    df = pd.DataFrame({"order_id": np.arange(100), "amount": _normalish(100), "flag": np.tile([0, 1], 50),
                       "rating": np.tile([1, 2, 3, 4, 5], 20)})
    ok, skipped = ad.eligible_columns(df, profile_dataframe(df), build_semantic_model(df, profile_dataframe(df)))
    assert ok == ["amount"]                                   # ids and flags are never screened
    assert "distinct values" in skipped["rating"]
    short, why = ad.eligible_columns(df.head(20), profile_dataframe(df.head(20)))
    assert "amount" not in short and "values" in why["amount"]


# ---- Isolation Forest -------------------------------------------------------------------------
def _cloud(seed=0):
    rng = np.random.default_rng(seed)
    a = pd.DataFrame({"units": rng.normal(50, 5, 500), "price": rng.normal(20, 2, 500)})
    out = pd.DataFrame({"units": [250.0, 260, -150, 240, 255], "price": [150.0, -90, 140, 160, 155]})
    return pd.concat([a, out], ignore_index=True)


def test_isolation_forest_finds_planted_multivariate_outliers_reproducibly():
    df = _cloud()
    a = ad.run_anomaly_detection(df, ["units", "price"], ("isolation_forest",), 3)
    b = ad.run_anomaly_detection(df, ["units", "price"], ("isolation_forest",), 3)
    pd.testing.assert_frame_equal(a.findings, b.findings)
    assert {500, 501, 502, 503, 504} <= set(a.findings["row"])
    assert a.findings["direction"].eq("multivariate").all() and "most different" in a.findings.iloc[0]["explanation"]


def test_isolation_forest_excludes_missing_rows_and_declines_small_or_single_feature_data():
    df = _cloud()
    df.loc[[0, 1, 2], "price"] = np.nan
    rep = ad.run_anomaly_detection(df, ["units", "price"], ("isolation_forest",), 3)
    assert any("3 row(s) with a missing value" in w for w in rep.warnings)
    assert not {0, 1, 2} & set(rep.findings["row"])
    small = ad.run_anomaly_detection(df.head(100), ["units", "price"], ("isolation_forest",), 3)
    assert small.summary["status"].iloc[0] == "skipped" and "complete rows" in small.summary["note"].iloc[0]
    one = ad.run_anomaly_detection(df, ["units"], ("isolation_forest",), 3)
    assert "at least 2" in one.summary["note"].iloc[0]


def test_one_method_failing_does_not_stop_the_others(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("boom")
    monkeypatch.setattr(ad, "_isolation_forest", boom)
    rep = ad.run_anomaly_detection(_cloud(), ["units", "price"], ("mad", "isolation_forest"), 3)
    st_ = dict(zip(rep.summary["method"], rep.summary["status"]))
    assert st_["isolation_forest"] == "failed" and "ok" in {s for m, s in zip(rep.summary["method"], rep.summary["status"]) if m == "mad"}


# ---- helpers ----------------------------------------------------------------------------------
def test_feature_comparison_by_hand_and_context_columns():
    df = pd.DataFrame({"sku": [f"S{i}" for i in range(11)], "date": pd.date_range("2024-01-01", periods=11),
                       "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100.0]})
    fc = ad.feature_comparison(df, 10, ["x"]).iloc[0]
    assert fc["Value"] == 100.0 and fc["Median"] == 6.0 and fc["Robust deviation"] == pytest.approx(0.6745 * 94 / 3.0)
    model = build_semantic_model(df, profile_dataframe(df))
    ctx = ad.context_columns(df, model)
    assert "date" in ctx
    out = ad.attach_context(pd.DataFrame({"row": [10, 3], "score": [1.0, 2.0]}), df, ["sku"])
    assert list(out["sku"]) == ["S10", "S3"]


# ---- unusual periods --------------------------------------------------------------------------
def _daily_sales(spike_at=100, seed=3):
    rng = np.random.default_rng(seed)
    n = 200
    y = np.array([10, 12, 14, 30, 50, 80, 20])[np.arange(n) % 7] + 100 + rng.normal(0, 2, n)
    y[spike_at] += 150
    return pd.DataFrame({"date": pd.date_range("2024-01-01", periods=n), "units": y})


def test_series_anomaly_flags_a_planted_spike_and_respects_the_weekly_pattern():
    df = _daily_sales()
    res = ad.series_anomalies(df, "date", "units", "D", "sum", 3)
    assert res.seasonal
    spike_day = pd.Timestamp("2024-01-01") + pd.Timedelta(days=100)
    assert spike_day in set(res.findings["row"]) and res.findings.iloc[0]["row"] == spike_day
    assert len(res.findings) <= 6                                          # busy weekend days are not flagged
    assert res.findings.iloc[0]["direction"] == "high" and "expected" in res.findings.iloc[0]["explanation"]
    assert res.series["flagged"].sum() == len(res.findings)


def test_series_anomaly_refuses_short_or_flawless_series():
    short = _daily_sales().head(20)
    with pytest.raises(InsufficientDataError):
        ad.series_anomalies(short, "date", "units", "D", "sum")
    perfect = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=100), "units": np.tile([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], 15)[:100]})
    with pytest.raises(InsufficientDataError, match="baseline"):
        ad.series_anomalies(perfect, "date", "units", "D", "sum")
    with pytest.raises(AnomalyDetectionError):
        ad.series_anomalies(_daily_sales(), "date", "nope", "D", "sum")


def test_anomaly_chart_builds_when_plotly_is_installed():
    pytest.importorskip("plotly")
    from visualization import charts
    res = ad.series_anomalies(_daily_sales(), "date", "units", "D", "sum", 3)
    fig = charts.anomaly_series_chart(res.series, "t", "y")
    assert fig is not None and len(fig.data) == 3

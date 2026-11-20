"""Forecasting (Phase 6A): hand-calculated metrics, chronological validation, refusal on unsuitable data."""
import numpy as np
import pandas as pd
import pytest

from analytics import forecasting as fc
from config.analytics_config import AnalyticsConfig
from core.exceptions import ForecastingError, InsufficientDataError
from ingestion.profiler import profile_dataframe
from semantic.semantic_model import build_semantic_model


def _daily(values, start="2024-01-01", col="units_sold"):
    return pd.DataFrame({"date": pd.date_range(start, periods=len(values), freq="D"), col: values})


def _weekly_pattern(n=140, seed=1):
    rng = np.random.default_rng(seed)
    pattern = np.array([10, 12, 14, 30, 50, 80, 20], dtype=float)
    return pattern[np.arange(n) % 7] + 100 + rng.normal(0, 2, n)


# ---- hand-calculated pieces -------------------------------------------------------------------
def test_baseline_models_by_hand():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    assert list(fc._naive(y, 3, 7, 3)) == [9, 9, 9]
    assert list(fc._seasonal_naive(y, 3, 7, 3)) == [3, 4, 5]          # last 7 values start at 3
    assert list(fc._moving_average(y, 2, 7, 3)) == [8, 8]             # mean(7, 8, 9)


def test_error_metrics_by_hand():
    m = fc.error_metrics(np.array([10.0, 20.0, 30.0]), np.array([12.0, 18.0, 33.0]), scale=2.0)
    assert m["mae"] == pytest.approx(7 / 3)
    assert m["rmse"] == pytest.approx(np.sqrt((4 + 4 + 9) / 3))
    assert m["mase"] == pytest.approx(7 / 3 / 2)
    assert m["mape"] == pytest.approx((0.2 + 0.1 + 0.1) / 3 * 100)


def test_mape_withheld_for_zero_or_tiny_actuals():
    assert fc.error_metrics(np.array([0.0, 5.0]), np.array([1.0, 4.0]), 1.0)["mape"] is None
    assert fc.error_metrics(np.array([0.1, 100.0]), np.array([1.0, 90.0]), 1.0)["mape"] is None
    assert fc.error_metrics(np.array([4.0, 6.0]), np.array([5.0, 6.0]), 0.0)["mase"] is None


def test_fold_origins_are_chronological_and_do_not_overlap():
    origins = fc.fold_origins(100, 10, 3)
    assert origins == [70, 80, 90]
    for o in origins:                       # each test block starts exactly where its training data ends
        assert o + 10 <= 100
    assert fc._plan_folds(100, 10, 3, 10) == (10, 3)
    assert fc._plan_folds(30, 14, 3, 10) == (5, 3)     # validation horizon shrinks to n // 6
    assert fc._plan_folds(30, 14, 3, 20) == (5, 2)     # folds are reduced, never fabricated
    assert fc._plan_folds(30, 14, 3, 40)[1] == 0


def test_choose_model_prefers_simpler_unless_clearly_better():
    s = lambda k, c, mae: fc.ModelScore(k, k, c, "ok", mae=mae)
    close = [s("naive", 0, 10.0), s("arima", 4, 9.8)]
    assert fc.choose_model(close, 0.05).key == "naive"                 # 2% better is not clearly better
    clear = [s("naive", 0, 10.0), s("arima", 4, 8.0)]
    assert fc.choose_model(clear, 0.05).key == "arima"
    with pytest.raises(ForecastingError):
        fc.choose_model([fc.ModelScore("x", "x", 0, "failed")], 0.05)


def test_seasonality_detection():
    assert fc.detect_seasonality(_weekly_pattern(), 7)[0] is True
    noise = np.random.default_rng(3).normal(100, 5, 140)
    assert fc.detect_seasonality(noise, 7)[0] is False
    assert fc.detect_seasonality(np.arange(10.0), 7)[0] is False        # fewer than two cycles


# ---- end to end -------------------------------------------------------------------------------
def test_weekly_pattern_selects_a_seasonal_model_that_beats_naive():
    df = _daily(_weekly_pattern())
    r = fc.forecast_series(df, "date", "units_sold", "D", "sum", 14)
    scores = {s.key: s for s in r.scores}
    assert r.seasonality_supported and r.chosen in {"seasonal_naive", "holt_winters", "sarima"}
    assert scores[r.chosen].mae < scores["naive"].mae / 3
    assert r.improvement_vs_baseline_pct is not None
    assert len(r.forecast) == 14 and r.forecast["period"].iloc[0] == pd.Timestamp("2024-01-01") + pd.Timedelta(days=140)
    assert (r.forecast["lower"] <= r.forecast["forecast"]).all() and (r.forecast["forecast"] <= r.forecast["upper"]).all()
    assert r.backtest["period"].is_monotonic_increasing and r.backtest["period"].min() > r.history["period"].iloc[50]


def test_noise_series_skips_seasonal_models_and_says_why():
    df = _daily(np.random.default_rng(5).normal(100, 5, 120))
    r = fc.forecast_series(df, "date", "units_sold", "D", "sum", 7)
    skipped = {s.key for s in r.scores if s.status == "skipped"}
    assert {"seasonal_naive", "holt_winters", "sarima"} <= skipped
    assert any("Seasonal models were not used" in t for t in r.limitations)


def test_reproducible_and_input_not_mutated():
    df = _daily(_weekly_pattern())
    before = df.copy()
    a = fc.forecast_series(df, "date", "units_sold", "D", "sum", 7)
    b = fc.forecast_series(df, "date", "units_sold", "D", "sum", 7)
    pd.testing.assert_frame_equal(a.forecast, b.forecast)
    pd.testing.assert_frame_equal(df, before)


def test_too_short_constant_or_gappy_series_are_refused():
    with pytest.raises(InsufficientDataError, match="at least"):
        fc.forecast_series(_daily(np.arange(20.0) + 1), "date", "units_sold", "D", "sum")
    with pytest.raises(InsufficientDataError, match="constant"):
        fc.forecast_series(_daily(np.full(100, 5.0)), "date", "units_sold", "D", "sum")
    gappy = _daily(np.arange(100.0) + 1)
    gappy = gappy[~gappy["date"].dt.day.between(1, 20)]              # most days missing
    with pytest.raises(InsufficientDataError, match="no data"):
        fc.forecast_series(gappy, "date", "units_sold", "D", "sum")


def test_invalid_requests_raise_forecasting_error():
    df = _daily(_weekly_pattern())
    with pytest.raises(ForecastingError):
        fc.forecast_series(df, "date", "units_sold", "D", "sum", 500)
    with pytest.raises(ForecastingError):
        fc.forecast_series(df, "date", "nope", "D", "sum")
    with pytest.raises(ForecastingError):
        fc.forecast_series(df, "date", "units_sold", "Q", "sum")


def test_count_target_never_goes_negative_and_partial_months_are_dropped():
    rng = np.random.default_rng(2)
    dates = pd.date_range("2022-01-15", "2024-12-10", freq="D")      # starts and ends mid-month
    df = pd.DataFrame({"date": np.repeat(dates, rng.integers(0, 4, len(dates)))})
    r = fc.forecast_series(df, "date", None, "M", "count", 6)
    assert r.history["period"].iloc[0] == pd.Timestamp("2022-02-01") and r.history["period"].iloc[-1] == pd.Timestamp("2024-11-01")
    assert (r.forecast[["forecast", "lower", "upper"]] >= 0).all().all()
    assert any("partly covered" in w for w in r.warnings)
    assert r.forecast["period"].iloc[0] == pd.Timestamp("2024-12-01")


def test_empty_days_treated_as_zero_for_sums_with_a_warning():
    df = _daily(_weekly_pattern(100))
    df = df.drop(index=[10, 11, 40])
    r = fc.forecast_series(df, "date", "units_sold", "D", "sum", 7)
    assert any("treated as zero" in w for w in r.warnings)
    assert len(r.history) == 100


def test_models_fail_independently(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("boom")
    monkeypatch.setattr(fc, "_MODELS", tuple((k, l, c, n, mt, boom if k == "arima" else f) for k, l, c, n, mt, f in fc._MODELS))
    r = fc.forecast_series(_daily(_weekly_pattern()), "date", "units_sold", "D", "sum", 7)
    assert {s.key: s.status for s in r.scores}["arima"] == "failed" and r.chosen != "arima"


# ---- setup helpers ----------------------------------------------------------------------------
def test_suggest_frequency():
    assert fc.suggest_frequency(_daily(np.ones(365)), "date") == "D"
    assert fc.suggest_frequency(_daily(np.ones(1500)), "date") == "W"
    assert fc.suggest_frequency(_daily(np.ones(10)), "date") == "D"


def test_suggest_setup_uses_semantic_roles():
    df = _daily(_weekly_pattern(200))
    df["unit_price"] = 9.5
    df["sku"] = "A1"
    model = build_semantic_model(df, profile_dataframe(df))
    setup = fc.suggest_setup(df, model)
    assert setup["date_col"] == "date" and setup["value_col"] == "units_sold" and setup["freq"] == "D"
    assert fc.target_columns(df, profile_dataframe(df), model) and "sku" not in fc.target_columns(df, profile_dataframe(df), model)


def test_forecast_chart_builds_when_plotly_is_installed():
    pytest.importorskip("plotly")
    from visualization import charts
    r = fc.forecast_series(_daily(_weekly_pattern()), "date", "units_sold", "D", "sum", 7)
    fig = charts.forecast_chart(r.history, r.forecast, "t", "y", r.backtest)
    names = {t.name for t in fig.data}
    assert fig is not None and {"Actual", "Forecast", "Validation predictions", "Prediction interval"} <= names
    fc_trace = next(t for t in fig.data if t.name == "Forecast")
    assert fc_trace.x[0] == r.history["period"].iloc[-1] and len(fc_trace.x) == len(r.forecast) + 1   # joined to the last actual
    assert charts.forecast_chart(r.history, r.forecast.iloc[0:0], "t") is None


# ---- flat forecasts and manual model choice ----------------------------------------------------
def _noise(n=365, seed=0):
    return _daily(np.random.default_rng(seed).normal(100, 10, n))


def test_flat_forecast_is_explained():
    r = fc.forecast_series(_noise(), "date", "units_sold", "D", "sum", 14)
    assert np.ptp(r.forecast["forecast"]) < 1e-9
    assert any("flat line" in w and "not a failure" in w for w in r.warnings)
    assert (r.forecast["upper"] - r.forecast["lower"]).min() > 0          # the range band is still shown


def test_seasonal_forecast_is_not_flat_and_has_no_flat_note():
    r = fc.forecast_series(_daily(_weekly_pattern()), "date", "units_sold", "D", "sum", 14)
    assert np.ptp(r.forecast["forecast"]) > 10 and not any("flat line" in w for w in r.warnings)


def test_manual_model_choice_is_honoured_and_reported():
    df = _noise()
    r = fc.forecast_series(df, "date", "units_sold", "D", "sum", 14, model="arima")
    assert r.chosen == "arima" and any("chosen manually" in w for w in r.warnings)
    assert {s.key for s in r.scores if s.status == "ok"} >= {"naive", "arima"}   # validation scores still shown
    auto = fc.forecast_series(df, "date", "units_sold", "D", "sum", 14)
    assert not any("chosen manually" in w for w in auto.warnings)


def test_manual_choice_of_unavailable_or_unknown_model_is_refused():
    df = _noise()
    with pytest.raises(ForecastingError, match="not available"):
        fc.forecast_series(df, "date", "units_sold", "D", "sum", 14, model="sarima")     # no seasonality here
    with pytest.raises(ForecastingError):
        fc.forecast_series(df, "date", "units_sold", "D", "sum", 14, model="magic")

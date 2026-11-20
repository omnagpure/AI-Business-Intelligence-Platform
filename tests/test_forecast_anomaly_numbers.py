"""Numbers behind the Forecasting and Anomaly pages: KPI targets per period, forecast summary, method agreement."""
import numpy as np
import pandas as pd
import pytest

from analytics import anomaly_detection as ad
from analytics.data_shifts import detect_shifts
from analytics.derived_targets import PREFIX, derived_targets, frame_with
from analytics.descriptive import aggregate_time
from analytics.forecasting import forecast_series, summarise_forecast
from analytics.kpis import compute_kpis
from ingestion.profiler import profile_dataframe
from semantic.semantic_model import build_semantic_model


def model_of(df, roles=None):
    return build_semantic_model(df, profile_dataframe(df), roles or {})


def _orders():
    """Two lines per order; on-time pattern differs by month; one cancelled shipment per month."""
    rows = []
    for m in range(1, 5):
        for o in range(10):
            late = o < m                                   # month m: m late orders of 10
            status = "Shipping canceled" if o == 9 else ("Late delivery" if late else "Shipping on time")
            for line in range(2):
                rows.append({"order_id": f"{m}-{o}", "order_date": pd.Timestamp(2024, m, 1 + o),
                             "days_for_shipping_real": 5 if late else 2, "days_for_shipment_scheduled": 3,
                             "delivery_status": status, "sales": 10.0})
    return pd.DataFrame(rows)


ROLES = {"order_id": "identifier", "order_date": "datetime", "days_for_shipping_real": "duration",
         "days_for_shipment_scheduled": "duration", "delivery_status": "category", "sales": "revenue"}


def test_kpi_targets_match_the_kpi_engine_month_by_month():
    df = _orders()
    m = model_of(df, ROLES)
    t = derived_targets(df, m)
    on = t[PREFIX + "On-time delivery rate (%)"]
    per = aggregate_time(frame_with(df, on), "order_date", on.label, "M", "mean")
    for month, value in zip(range(1, 5), per["value"]):
        sub = df[df["order_date"].dt.month == month]
        assert value == pytest.approx(compute_kpis(sub, m).get("on_time_rate").value)
    orders = t[PREFIX + "Orders"]
    counts = aggregate_time(frame_with(df, orders), "order_date", orders.label, "M", "sum")["value"]
    assert list(counts) == [10, 10, 10, 10]                       # distinct orders, not the 20 lines
    assert t[PREFIX + "Cancelled rate (%)"].values.mean() == pytest.approx(10.0)
    assert df.shape[1] == 6 and frame_with(df, orders).shape[1] == 7   # the original frame is not changed


def test_no_supply_chain_targets_for_other_data():
    df = pd.DataFrame({"invoice_id": range(50), "date": pd.date_range("2024-01-01", periods=50), "amount": 1.0})
    t = derived_targets(df, model_of(df, {"invoice_id": "identifier", "date": "datetime", "amount": "revenue"}))
    assert all("On-time" not in k and "Cancelled" not in k for k in t)


def _monthly(values):
    d = pd.date_range("2020-01-01", periods=len(values), freq="MS")
    return pd.DataFrame({"date": d, "rate": values})


def test_bounds_keep_a_rate_forecast_inside_0_100():
    y = np.clip(95 + np.arange(36) * 0.3 + np.random.default_rng(0).normal(0, 1.5, 36), 0, 100)
    r = forecast_series(_monthly(y), "date", "rate", "M", "mean", 6, bounds=(0.0, 100.0))
    assert (r.forecast[["forecast", "lower", "upper"]] <= 100).all().all()


def test_summary_compares_like_with_like():
    y = np.arange(1, 37, dtype=float) * 10                          # steady growth, monthly sums
    r = forecast_series(_monthly(y), "date", "rate", "M", "sum", 3)
    sm = summarise_forecast(r)
    assert len(r.history) == 35                                     # the last month (one day of data) is partial
    assert sm.previous_total == pytest.approx(y[-4:-1].sum())
    assert sm.horizon_total == pytest.approx(r.forecast["forecast"].sum())
    assert sm.change_pct == pytest.approx((sm.horizon_total - sm.previous_total) / sm.previous_total * 100)
    assert sm.next_lower <= sm.next_value <= sm.next_upper and sm.typical_error_pct is not None


def test_agreement_counts_every_flag_not_only_the_strongest_shown():
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"a": rng.normal(100, 5, 2000), "b": rng.normal(50, 2, 2000)})
    df.loc[:59, "a"] = 400                                          # 60 extreme rows
    cfg = ad.DEFAULT_ANALYTICS.__class__(**{**ad.DEFAULT_ANALYTICS.__dict__, "anomaly_max_findings": 10})
    rep = ad.run_anomaly_detection(df, ["a", "b"], ("mad", "iqr"), 3, cfg)
    assert len(rep.findings[rep.findings["method"] == "mad"]) <= 20   # the shown list is capped
    agreed = rep.agreed(2)
    assert set(range(60)) <= set(agreed["row"])                      # but all 60 count as agreed
    assert rep.methods_run == 2 and rep.examined_rows == 2000
    col = rep.column_votes.set_index("feature")
    assert col.loc["a", "agreed"] >= 60


def test_default_columns_skip_copies():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, 500)
    df = pd.DataFrame({"sales": x, "order_total": x.copy(), "scaled": x * 2 + 1, "other": rng.normal(0, 1, 500)})
    assert ad.default_columns(df, list(df.columns)) == ["sales", "other"]


def test_expected_band_matches_the_flag_rule():
    d = pd.date_range("2024-01-01", periods=200)
    v = 100 + np.random.default_rng(2).normal(0, 3, 200)
    v[150] = 160
    res = ad.series_anomalies(pd.DataFrame({"date": d, "units": v}), "date", "units", "D", "sum", 3)
    s = res.series
    assert ((s["value"] > s["upper"]) | (s["value"] < s["lower"])).eq(s["flagged"]).all()
    assert s.loc[s["flagged"], "period"].iloc[0] == d[150]


def test_weekly_shift_is_found_months_before_the_end():
    rows = []
    for i, day in enumerate(pd.date_range("2022-01-03", periods=420)):    # 60 weeks; lines per order drop at week 43
        lines = 3 if i < 300 else 1
        rows += [{"d": day, "order": f"o{i}"}] * lines
    shifts = detect_shifts(pd.DataFrame(rows), "d", "order", "W")
    per_order = [s for s in shifts if s.measure == "rows per order"]
    assert per_order and per_order[0].change_text() == "rows per order went from 3.0 to 1.0"

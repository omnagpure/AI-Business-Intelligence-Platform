"""Numbers behind the dashboard cards: cancelled shipments, shipping time, per-period KPI history, data-shape checks."""
import numpy as np
import pandas as pd
import pytest

from analytics.data_shifts import describe_shifts, detect_shifts
from analytics.formatting import format_value
from analytics.kpi_trend import kpi_history, lower_is_better
from analytics.kpis import compute_kpis
from analytics.supply_chain_views import suggest_views
from config.analytics_config import DEFAULT_ANALYTICS
from ingestion.profiler import profile_dataframe
from semantic.semantic_model import build_semantic_model


def model_of(df, roles=None):
    return build_semantic_model(df, profile_dataframe(df), roles or {})


def shipping_df():
    """DataCo-style rows: actual vs scheduled shipping days and a delivery status that includes cancellations."""
    return pd.DataFrame({
        "order_id": [1, 2, 3, 4, 5, 6],
        "order_date": pd.to_datetime(["2024-01-02"] * 6),
        "days_for_shipping_real": [2, 5, 3, 6, 4, 1],
        "days_for_shipment_scheduled": [2, 2, 4, 4, 2, 4],
        "delivery_status": ["Shipping on time", "Late delivery", "Advance shipping", "Late delivery",
                            "Shipping canceled", "Shipping canceled"],
        "shipping_mode": ["Standard", "First", "Standard", "First", "Standard", "First"],
        "sales": [10.0, 20, 30, 40, 50, 60],
    })


ROLES = {"order_id": "identifier", "order_date": "datetime", "days_for_shipping_real": "duration",
         "days_for_shipment_scheduled": "duration", "delivery_status": "category", "shipping_mode": "category",
         "sales": "revenue"}


def test_cancelled_shipments_are_left_out_of_on_time_and_delay():
    r = compute_kpis(shipping_df(), model_of(shipping_df(), ROLES))
    # known outcomes without the 2 cancelled rows: delays 0, +3, -1, +2 -> on time 2 of 4
    assert r.get("on_time_rate").value == pytest.approx(50.0)
    assert r.get("avg_delay").value == pytest.approx(1.0) and r.get("avg_delay").unit == "days"
    assert r.get("avg_late_delay").value == pytest.approx(2.5)             # late rows only: 3 and 2
    assert any("2 cancelled shipments" in n for n in r.get("on_time_rate").limitations)


def test_cancel_rate_matches_words_inside_status_values():
    r = compute_kpis(shipping_df(), model_of(shipping_df(), ROLES))
    k = r.get("cancel_rate")
    assert k.name == "Cancelled shipments" and k.value == pytest.approx(100 * 2 / 6)


def test_shipping_time_falls_back_to_the_actual_days_column():
    k = compute_kpis(shipping_df(), model_of(shipping_df(), ROLES)).get("avg_lead_time")
    assert k.name == "Average shipping time" and k.unit == "days"
    assert k.value == pytest.approx(np.mean([2, 5, 3, 6, 4, 1]))


def test_days_unit_and_direction():
    assert format_value(3.5, "days") == "3.5 days"
    assert lower_is_better("cancel_rate") and lower_is_better("below_reorder_rate") and lower_is_better("avg_late_delay")
    assert not lower_is_better("on_time_rate")


def _monthly_orders():
    """Six months; on-time share 50% in every month except the last (75%)."""
    rows = []
    for m, late_share in zip(range(1, 7), [.5, .5, .5, .5, .5, .25]):
        for i in range(8):
            late = i < 8 * late_share
            rows.append({"order_id": f"{m}-{i}", "order_date": pd.Timestamp(2024, m, 1 + i * 3),
                         "days_for_shipping_real": 5 if late else 2, "days_for_shipment_scheduled": 3,
                         "delivery_status": "Late delivery" if late else "Shipping on time",
                         "shipping_mode": "Standard", "sales": 100.0})
    rows.append({**rows[-1], "order_id": "end", "order_date": pd.Timestamp(2024, 6, 30)})
    return pd.DataFrame(rows)


def test_history_recomputes_each_period_and_rates_change_in_points():
    df = _monthly_orders()
    h = kpi_history(df, model_of(df, ROLES), DEFAULT_ANALYTICS, "order_date", "M", ("on_time_rate", "order_count", "revenue_growth"))
    assert "revenue_growth" not in h                                          # growth is never drawn as a history
    on = h["on_time_rate"]
    assert on.points[0] == pytest.approx(50.0) and on.change_unit == "pts"
    assert on.change == pytest.approx(on.points[-1] - on.points[-2]) and not on.flat
    assert h["order_count"].points[:5] == [8.0] * 5 and h["order_count"].change == pytest.approx(12.5)   # 9 orders in June (extra row)


def test_data_shift_is_reported_when_rows_per_order_drop():
    rows = []
    for m in range(1, 13):
        lines = 3 if m <= 9 else 1                                            # three lines per order, then one
        for o in range(20):
            for _ in range(lines):
                rows.append({"order": f"{m}-{o}", "d": pd.Timestamp(2023, m, 1 + o)})
    df = pd.DataFrame(rows)
    shifts = detect_shifts(df, "d", "order")
    per_order = [s for s in shifts if s.measure == "rows per order"]
    assert per_order and per_order[0].start == pd.Timestamp(2023, 10, 1)
    assert "Oct 2023" in describe_shifts(shifts)


def test_no_shift_on_steady_data():
    d = pd.date_range("2023-01-01", "2023-12-31", freq="D")
    df = pd.DataFrame({"d": d, "order": range(len(d))})
    assert detect_shifts(df, "d", "order") == []


def test_on_time_view_says_when_groups_hardly_differ():
    df = pd.concat([shipping_df()] * 5, ignore_index=True)
    df["shipping_mode"] = np.repeat(["A", "B", "A", "B", "A"], 6)              # whole copies: same mix in both groups
    v = {x.key: x for x in suggest_views(df, model_of(df, ROLES), max_views=6)}["on_time_by_group"]
    assert "hardly differ" in v.caption and v.ref == pytest.approx(50.0) and v.unit == "%"


def test_small_changes_are_flat():
    from analytics.kpi_trend import KpiHistory
    assert KpiHistory([1, 1, 1], ["a", "b", "c"], 0.3, "%", "month").flat
    assert KpiHistory([1, 1, 1], ["a", "b", "c"], 0.05, "pts", "month").flat
    assert not KpiHistory([1, 1, 1], ["a", "b", "c"], 0.2, "pts", "month").flat

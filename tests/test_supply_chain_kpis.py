"""Supply-chain KPIs against hand-calculated values, plus the data-validity guards."""
import pandas as pd
import pytest

from analytics.data_checks import parse_flag, sensitive_columns
from analytics.kpi_relevance import prioritise
from analytics.kpis import NOT_CALCULABLE, OK, SKIPPED, compute_kpis
from config.analytics_config import AnalyticsConfig
from ingestion.profiler import profile_dataframe
from semantic.semantic_model import build_semantic_model


def run(df, roles=None, **kw):
    model = build_semantic_model(df, profile_dataframe(df), roles or {})
    return compute_kpis(df, model, **kw)


def po_df():
    """Six purchase orders. Hand calculation in the comments of each test."""
    return pd.DataFrame({
        "purchase_order_id": [f"PO{i}" for i in range(1, 7)],
        "supplier_id": ["S1", "S1", "S2", "S2", "S3", "S3"],
        "ordered_at": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-05", "2024-01-05", "2024-01-10", "2024-01-10"]),
        "expected_at": pd.to_datetime(["2024-01-11", "2024-01-11", "2024-01-15", "2024-01-15", "2024-01-20", "2024-01-20"]),
        "received_at": pd.to_datetime(["2024-01-10", "2024-01-14", "2024-01-15", "2024-01-20", None, None]),
        "quantity_ordered": [100, 100, 50, 50, 80, 20],
        "quantity_received": [100, 80, 60, 50, 0, 0],
        "status": ["received", "partial", "received", "received", "open", "cancelled"],
        "order_value": [1000.0, 1000.0, 500.0, 500.0, 800.0, 200.0],
    })


PO_ROLES = {"purchase_order_id": "identifier", "supplier_id": "supplier", "ordered_at": "datetime",
            "expected_at": "datetime", "received_at": "datetime", "quantity_ordered": "quantity",
            "quantity_received": "quantity", "status": "category", "order_value": "revenue"}


def test_on_time_rate_and_delay_use_only_rows_with_a_known_outcome():
    r = run(po_df(), PO_ROLES)
    # received rows: delays vs expected = -1, +3, 0, +5 days -> on time when <= 0: 2 of 4
    assert r.get("on_time_rate").value == pytest.approx(50.0)
    assert r.get("avg_delay").value == pytest.approx((-1 + 3 + 0 + 5) / 4)
    assert any("outcome is unknown" in x for x in r.get("on_time_rate").limitations)


def test_lead_time_is_order_to_receipt_in_days():
    r = run(po_df(), PO_ROLES)
    # 9, 13, 10, 15 days -> 11.75
    assert r.get("avg_lead_time").value == pytest.approx(11.75)


def test_fill_rate_excludes_open_and_cancelled_and_caps_over_delivery():
    r = run(po_df(), PO_ROLES)
    # closed rows: (100,100) (100,80) (50,60) (50,50): received capped = 100+80+50+50 = 280 of 300
    k = r.get("fill_rate")
    assert k.value == pytest.approx(100 * 280 / 300)
    assert any("received more than ordered" in x for x in k.limitations)
    assert any("received more than ordered" in n for n in r.notes)            # surfaced at report level


def test_otif_needs_on_time_and_in_full_in_the_same_row():
    r = run(po_df(), PO_ROLES)
    # on time: rows 1 and 3. In full: rows 1, 3 (60>=50), 4. Both: rows 1 and 3 -> 2 of 4
    assert r.get("otif").value == pytest.approx(50.0)


def test_procurement_amount_is_spend_without_cancelled_orders():
    r = run(po_df(), PO_ROLES)
    assert r.get("total_revenue") is None and r.get("aov") is None and r.get("revenue_by_category") is None
    s = r.get("total_spend")
    assert s.status == OK and s.value == pytest.approx(3800.0)                # 4000 minus the 200 cancelled
    assert any("not sales revenue" in x for x in s.limitations)
    assert r.get("avg_po_value").value == pytest.approx(3800 / 5)


def test_supplier_concentration_excludes_cancelled_and_weighs_by_spend():
    r = run(po_df(), PO_ROLES)
    # spend: S1 2000, S2 1000, S3 800 -> top 2000/3800, top3 = 100%
    assert r.get("supplier_top_share").value == pytest.approx(100 * 2000 / 3800)
    assert r.get("supplier_top3_share").value == pytest.approx(100.0)
    assert dict(r.get("supplier_top_share").breakdown)["S1"] == 2000


def test_status_column_is_not_used_as_a_category_for_revenue_share():
    df = po_df().drop(columns=["supplier_id"])
    r = run(df, {k: v for k, v in PO_ROLES.items() if k != "supplier_id"})
    assert r.get("revenue_by_category").status == SKIPPED


def test_flag_based_on_time_rate_and_polarity():
    df = pd.DataFrame({"order": range(5), "on_time": ["Yes", "No", "yes", "Y", None]})
    assert run(df, {"order": "identifier", "on_time": "flag"}).get("on_time_rate").value == pytest.approx(75.0)
    late = pd.DataFrame({"order": range(4), "is_late": [1, 0, 0, 0]})
    assert run(late, {"order": "identifier", "is_late": "flag"}).get("on_time_rate").value == pytest.approx(75.0)


def test_placeholder_text_is_not_accepted_as_an_on_time_flag():
    df = pd.DataFrame({"order": range(3), "on_time_delivery": ["on_time_delivery_1", "on_time_delivery_2", "on_time_delivery_3"]})
    k = run(df, {"order": "identifier"}).get("on_time_rate")
    assert k.status == SKIPPED and k.value is None and "not Yes/No or 0/1" in k.reason


def test_actual_vs_scheduled_days_columns_are_used_when_no_dates_exist():
    df = pd.DataFrame({"days_real": [2, 5, 4, 1], "days_scheduled": [3, 4, 4, 2]})
    r = run(df, {"days_real": "duration", "days_scheduled": "duration"})
    assert r.get("on_time_rate").value == pytest.approx(75.0)                  # only the second row is late
    assert r.get("avg_delay").value == pytest.approx((-1 + 1 + 0 - 1) / 4)


def test_defect_rate_uses_totals_and_flags_impossible_rows():
    df = pd.DataFrame({"units_produced": [100, 100, 100], "defects": [10, 20, 150]})
    k = run(df, {"units_produced": "quantity", "defects": "quantity"}).get("defect_rate")
    assert k.status == OK and k.value == pytest.approx(100 * 180 / 300)
    assert any("more defects than units" in x for x in k.limitations)
    worse = pd.DataFrame({"units_produced": [10, 10], "defects": [30, 30]})
    w = run(worse, {"units_produced": "quantity", "defects": "quantity"}).get("defect_rate")
    assert w.status == NOT_CALCULABLE and "above 100%" in w.reason


def inventory_df():
    d = pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"])
    return pd.DataFrame({"date": d, "sku": ["A", "B", "A", "B"], "inventory_level": [10, 20, 0, 20],
                         "units_sold": [5, 5, 10, 0], "unit_cost": [2.0, 1.0, 2.0, 1.0],
                         "reorder_point": [5, 5, 5, 5], "stockout_flag": [0, 0, 1, 0]})


INV_ROLES = {"date": "datetime", "sku": "product", "inventory_level": "quantity", "units_sold": "quantity",
             "unit_cost": "cost", "reorder_point": "quantity", "stockout_flag": "flag"}


def test_stock_kpis_match_hand_calculation():
    r = run(inventory_df(), INV_ROLES)
    assert r.get("stockout_rate").value == pytest.approx(25.0)                 # 1 of 4 rows
    assert r.get("below_reorder_share").value == pytest.approx(25.0)           # only A on day 2 (0 < 5)
    # daily stock totals 30 and 20 -> avg 25; daily sales 10 and 10 -> avg 10; cover 2.5 days
    assert r.get("days_of_cover").value == pytest.approx(2.5)
    # COGS = 5*2 + 5*1 + 10*2 + 0 = 35; stock value day1 = 20+20 = 40, day2 = 0+20 = 20 -> avg 30
    # turnover = 35/30 over 2 days, annualised x 365/2
    assert r.get("inventory_turnover").value == pytest.approx(35 / 30 * 365 / 2)


def test_stockout_falls_back_to_zero_stock_and_a_dead_flag_is_called_out():
    df = inventory_df().drop(columns=["stockout_flag"])
    roles = {k: v for k, v in INV_ROLES.items() if k != "stockout_flag"}
    assert run(df, roles).get("stockout_rate").value == pytest.approx(25.0)    # inventory_level <= 0: 1 of 4
    flat = inventory_df().assign(stockout_flag=0)
    k = run(flat, INV_ROLES).get("stockout_rate")
    assert k.value == 0.0 and any("No stockout is recorded" in x for x in k.limitations)


def test_zero_denominators_are_not_calculable_never_invented():
    df = pd.DataFrame({"quantity_ordered": [0, 0], "quantity_received": [0, 0]})
    r = run(df, {"quantity_ordered": "quantity", "quantity_received": "quantity"})
    assert r.get("fill_rate").status == NOT_CALCULABLE and r.get("fill_rate").value is None


def test_delivered_before_ordered_is_excluded_and_reported():
    df = po_df()
    df.loc[0, "received_at"] = pd.Timestamp("2023-12-25")
    k = run(df, PO_ROLES).get("avg_lead_time")
    assert k.value == pytest.approx((13 + 10 + 15) / 3)
    assert any("before" in x for x in k.limitations)


def test_plain_sales_data_is_not_cluttered_with_supply_chain_kpis():
    df = pd.DataFrame({"order_id": ["A", "B", "C"], "revenue": [10.0, 20.0, 30.0], "customer_id": ["x", "y", "z"]})
    r = run(df, {"order_id": "identifier", "revenue": "revenue", "customer_id": "customer"})
    assert not [k for k in r.results if k.group in ("Supply chain", "Procurement")]


def test_growth_is_skipped_when_one_period_has_far_fewer_rows():
    days = list(pd.date_range("2024-01-01", periods=2)) + list(pd.date_range("2024-02-01", periods=28)) + [pd.Timestamp("2024-03-10")]
    df = pd.DataFrame({"order_date": days, "revenue": 10.0, "order_id": [f"O{i}" for i in range(len(days))]})
    roles = {"order_date": "datetime", "revenue": "revenue", "order_id": "identifier"}
    g = run(df, roles).get("revenue_growth")          # Jan has 2 rows, Feb has 28 (14x): Jan is probably incomplete
    assert g.status == SKIPPED and "very different numbers of rows" in g.reason
    assert run(df, roles, config=AnalyticsConfig(growth_max_row_ratio=20.0)).get("revenue_growth").status == OK


def test_parse_flag_and_sensitive_columns():
    f, why = parse_flag(pd.Series(["Yes", "no", None, 1, "0"]))
    assert why == "" and list(f.dropna().astype(bool)) == [True, False, True, False]
    assert parse_flag(pd.Series(["maybe", "yes"]))[0] is None
    assert sensitive_columns(["Order Id", "Customer Password", "email", "Customer Street"]) == ["Customer Password", "email", "Customer Street"]


def test_personal_data_columns_are_reported_but_not_used():
    df = po_df().assign(customer_email="a@b.c")
    r = run(df, PO_ROLES)
    assert any("personal data" in n and "customer_email" in n for n in r.notes)


def test_relevance_leads_with_supply_chain_kpis_and_hides_sales_only_gaps():
    r = run(po_df(), PO_ROLES)
    d = prioritise(r)
    assert d.supply_chain_mode
    keys = [k.key for k in d.headline]
    assert keys[:3] == ["on_time_rate", "otif", "fill_rate"] and len(keys) <= 6
    assert all(k.group not in ("Sales", "Customers", "Profit and cost") for k in d.unavailable_relevant)
    assert {k.key for k in d.unavailable_other}                                  # e.g. customers, profit, kept for transparency


def test_input_dataframe_is_not_modified():
    df = po_df()
    before = df.copy(deep=True)
    run(df, PO_ROLES)
    pd.testing.assert_frame_equal(df, before)

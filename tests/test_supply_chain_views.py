"""Suggested views (captions computed from data) and the primary-column helpers."""
import pandas as pd
import pytest

from analytics.kpis import DataFilter
from analytics.primary_columns import primary_candidates, with_primary
from analytics.supply_chain_views import suggest_views
from ingestion.profiler import profile_dataframe
from semantic.semantic_model import build_semantic_model


def model_for(df, roles):
    return build_semantic_model(df, profile_dataframe(df), roles)


def views(df, roles, **kw):
    return {v.key: v for v in suggest_views(df, model_for(df, roles), **kw)}


def delivery_df():
    """Supplier A: 12 deliveries, 3 late. Supplier B: 12 deliveries, 9 late. Lead time is 10 days for all."""
    n = 24
    ordered = pd.to_datetime(["2024-01-01"] * n)
    expected = ordered + pd.to_timedelta([12] * n, unit="D")
    late_a = [True] * 3 + [False] * 9
    late_b = [True] * 9 + [False] * 3
    late = late_a + late_b
    received = ordered + pd.to_timedelta([10] * n, unit="D")
    received = pd.Series([r + pd.Timedelta(days=5) if l else r for r, l in zip(received, late)])
    return pd.DataFrame({"supplier": ["A"] * 12 + ["B"] * 12, "ordered_at": ordered, "expected_at": expected,
                         "received_at": received})


DELIVERY_ROLES = {"supplier": "supplier", "ordered_at": "datetime", "expected_at": "datetime", "received_at": "datetime"}


def test_on_time_by_supplier_caption_is_computed_from_the_data():
    v = views(delivery_df(), DELIVERY_ROLES)["on_time_by_group"]
    # A: 9 of 12 on time = 75%; B: 3 of 12 = 25%; overall 12 of 24 = 50%
    assert dict(zip(v.data["label"], v.data["value"])) == {"B": pytest.approx(25.0), "A": pytest.approx(75.0)}
    assert v.data.iloc[0]["label"] == "B"                                    # lowest first
    assert v.caption.startswith("B has the lowest on-time rate (25.0% of 12 rows) against 50.0% overall")
    assert "best is A (75.0%)" in v.caption


def test_lead_time_view_reports_median_and_p90():
    v = views(delivery_df(), DELIVERY_ROLES)["lead_time"]
    # 12 rows are on time (10 days) and 12 are late (15 days): median 12.5, 90th percentile 15
    assert v.kind == "histogram" and len(v.data) == 24
    assert "Median lead time is 12.5 days" in v.caption and "within 15.0 days" in v.caption


def stock_df():
    d = pd.to_datetime(["2024-01-01"] * 2 + ["2024-01-02"] * 2 + ["2024-01-03"] * 2)
    return pd.DataFrame({"date": d, "sku": ["A", "B"] * 3, "inventory_level": [10, 10, 4, 4, 30, 30],
                         "reorder_point": [5, 5, 5, 5, 5, 5]})


STOCK_ROLES = {"date": "datetime", "sku": "product", "inventory_level": "quantity", "reorder_point": "quantity"}


def test_stock_vs_reorder_sums_items_per_date():
    v = views(stock_df(), STOCK_ROLES)["stock_vs_reorder"]
    # totals: 20/10, 8/10, 60/10 -> below on 1 of 3 dates; latest 60 against 10
    assert v.kind == "lines" and list(v.data["stock"]) == [20, 8, 60]
    assert "below the reorder points on 1 of 3 dates (33.3%)" in v.caption
    assert "60 against 10" in v.caption


def test_defect_rate_over_time_uses_totals_per_period():
    d = pd.to_datetime(["2024-01-05", "2024-01-20", "2024-02-05", "2024-02-20", "2024-03-05", "2024-03-20",
                        "2024-04-05", "2024-04-20", "2024-05-05", "2024-05-20", "2024-06-05", "2024-06-20"])
    df = pd.DataFrame({"date": d, "units_produced": [100] * 12, "defects": [1, 3, 10, 10, 2, 2, 5, 5, 4, 4, 3, 3]})
    v = views(df, {"date": "datetime", "units_produced": "quantity", "defects": "quantity"})["defects_over_time"]
    # monthly rate = defects / units: 2%, 10%, 2%, 5%, 4%, 3%; overall 52 / 1200 = 4.33%
    assert [round(x, 2) for x in v.data["rate"]] == [2.0, 10.0, 2.0, 5.0, 4.0, 3.0]
    assert "ranged from 2.0% to 10.0% by month" in v.caption and "highest in 2024-02-01" in v.caption
    assert "4.3% overall" in v.caption


def test_supplier_share_caption():
    df = pd.DataFrame({"supplier": ["A", "A", "B", "C", "D"], "quantity_ordered": [50, 30, 10, 5, 5]})
    v = views(df, {"supplier": "supplier", "quantity_ordered": "quantity"})["supplier_share"]
    assert v.caption.startswith("A accounts for 80.0% of ordered quantity; the top 3 of 4 suppliers account for 95.0%")


def test_views_are_absent_when_the_data_is_missing_and_never_invented():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    assert suggest_views(df, model_for(df, {})) == ()
    assert suggest_views(df.iloc[0:0], model_for(df, {})) == ()


def test_general_views_fill_in_for_data_without_supply_columns():
    df = pd.DataFrame({"order_date": pd.date_range("2024-01-01", periods=60, freq="D"),
                       "revenue": [10.0] * 60, "category": ["A", "B", "A", "A"] * 15})
    v = views(df, {"order_date": "datetime", "revenue": "revenue", "category": "category"})
    assert "trend" in v and "category_share" in v and "on_time_by_group" not in v
    assert "A is the largest category, with 75.0% of revenue" in v["category_share"].caption


def test_at_most_four_views_and_filters_apply():
    df = delivery_df()
    assert len(suggest_views(df, model_for(df, DELIVERY_ROLES))) <= 4
    flt = DataFilter((("supplier", ("A",)),))
    a = {v.key: v for v in suggest_views(df, model_for(df, DELIVERY_ROLES), data_filter=flt)}
    assert "on_time_by_group" not in a                                       # one supplier left: nothing to compare


def test_input_is_not_modified_by_views():
    df = delivery_df()
    before = df.copy(deep=True)
    suggest_views(df, model_for(df, DELIVERY_ROLES))
    pd.testing.assert_frame_equal(df, before)


def two_revenue_df():
    return pd.DataFrame({"Sales": [10.0, 20.0, 30.0], "Order Item Total": [9.0, 18.0, 27.0],
                         "Region": ["N", "S", "N"], "Segment": ["a", "b", "a"]})


def test_primary_candidates_list_the_current_choice_first():
    df = two_revenue_df()
    m = model_for(df, {"Sales": "revenue", "Order Item Total": "revenue", "Region": "category", "Segment": "category"})
    c = primary_candidates(df, m)
    cur, options = c["revenue"]
    assert options[0] == cur and set(options) == {"Sales", "Order Item Total"}
    assert set(c["category"][1]) == {"Region", "Segment"}


def test_with_primary_replaces_only_the_pickers_previous_override():
    roles = {"Segment": "category", "Sales": "revenue"}
    # the picker earlier chose Sales; now choose Order Item Total: Sales loses its override, Segment is untouched
    new = with_primary(roles, "Sales", "Order Item Total", "revenue")
    assert new == {"Segment": "category", "Order Item Total": "revenue"}
    assert roles == {"Segment": "category", "Sales": "revenue"}              # the input dict is not modified
    assert with_primary({}, None, "Sales", "revenue") == {"Sales": "revenue"}


def test_picking_a_primary_column_changes_the_revenue_kpi():
    from analytics.kpis import compute_kpis
    df = two_revenue_df()
    first = compute_kpis(df, model_for(df, {})).get("total_revenue").columns[0]
    other = "Order Item Total" if first == "Sales" else "Sales"
    picked = compute_kpis(df, model_for(df, with_primary({}, None, other, "revenue"))).get("total_revenue")
    assert picked.columns[0] == other and picked.value == pytest.approx(df[other].sum())


def test_multi_line_chart_draws_both_series():
    pytest.importorskip("plotly")
    from visualization import charts
    f = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=3), "stock": [1, 2, 3], "reorder_point": [2, 2, 2]})
    fig = charts.multi_line_chart(f, "date", ("stock", "reorder_point"), "t")
    assert len(fig.data) == 2
    assert charts.multi_line_chart(f.assign(stock=float("nan"), reorder_point=float("nan")), "date", ("stock", "reorder_point"), "t") is None

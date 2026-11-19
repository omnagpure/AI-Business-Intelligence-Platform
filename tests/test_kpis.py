"""KPI engine: formulas, zero denominators, growth rules, role overrides and filters."""
import numpy as np
import pandas as pd
import pytest

from analytics.kpis import (NOT_CALCULABLE, OK, SKIPPED, DataFilter, cache_key, compute_kpis, KPIMetric, KPIResult)
from config.analytics_config import AnalyticsConfig
from ingestion.profiler import profile_dataframe
from semantic.semantic_model import build_semantic_model

ROLES = {"order_id": "identifier", "customer_id": "customer", "order_date": "datetime", "revenue": "revenue",
         "profit": "profit", "qty": "quantity", "category": "category"}


def model_for(df, roles):
    return build_semantic_model(df, profile_dataframe(df), {c: r for c, r in roles.items() if c in df.columns})


def sales_df():
    return pd.DataFrame({
        "order_id": ["O1", "O2", "O3", "O4", "O5", "O6"],
        "customer_id": ["C1", "C1", "C2", "C3", "C1", "C3"],
        "order_date": pd.to_datetime(["2024-01-01", "2024-01-31", "2024-02-01", "2024-02-29", "2024-03-01", "2024-03-15"]),
        "revenue": [100.0, 100.0, 150.0, 150.0, 1000.0, 1000.0],
        "profit": [20.0, 20.0, 30.0, 30.0, 100.0, 100.0],
        "qty": [1, 2, 3, 4, 5, 6],
        "category": ["A", "A", "B", "B", "A", "C"],
    })


def kpis(df, roles=ROLES, **kw):
    return compute_kpis(df, model_for(df, roles), **kw)


def test_sales_kpis_match_hand_calculation():
    r = kpis(sales_df())
    assert r.get("total_revenue").value == 2500
    assert r.get("order_count").value == 6
    assert r.get("units").value == 21
    assert r.get("aov").value == pytest.approx(2500 / 6)
    top = r.get("revenue_by_category")                 # A=1200 of 2500
    assert top.value == pytest.approx(100 * 1200 / 2500) and dict(top.breakdown)["A"] == 1200
    assert r.get("distinct_customers").value == 3
    assert r.get("avg_spend").value == pytest.approx(2500 / 3)
    assert r.get("repeat_rate").value == pytest.approx(100 * 2 / 3)    # C1 has 3 orders, C3 has 2, C2 has 1


def test_every_kpi_records_its_metadata():
    k = kpis(sales_df()).get("total_revenue")
    assert isinstance(k, KPIResult) and KPIMetric is KPIResult
    assert k.formula == "SUM(revenue)" and k.columns == ("revenue",) and k.calculated_at and k.unit == "amount"
    assert k.status == OK and k.filters == ()


def test_profit_margin_uses_totals_not_average_of_row_margins():
    r = kpis(sales_df())
    assert r.get("total_profit").value == 300
    assert r.get("profit_margin").value == pytest.approx(100 * 300 / 2500)       # 12.0
    row_avg = (sales_df()["profit"] / sales_df()["revenue"]).mean() * 100
    assert r.get("profit_margin").value != pytest.approx(row_avg)               # 16.67 would be wrong


def test_zero_revenue_makes_margin_not_calculable():
    df = sales_df()
    df["revenue"] = 0.0
    m = kpis(df).get("profit_margin")
    assert m.status == NOT_CALCULABLE and m.value is None
    assert any("Not calculable" in x for x in m.limitations)
    aov = kpis(df).get("aov")
    assert aov.value == 0 and aov.status == OK               # zero revenue over real orders is a valid 0


def test_growth_uses_last_two_complete_months_and_ignores_partial_month():
    g = kpis(sales_df()).get("revenue_growth")     # Mar is partial (ends 15th): compare Feb (300) with Jan (200)
    assert g.status == OK and g.value == pytest.approx(50.0)
    assert g.period.startswith("2024-02 vs 2024-01")


def test_growth_skipped_when_previous_period_is_zero_or_missing():
    df = sales_df()
    df.loc[df["order_date"] < "2024-02-01", "revenue"] = 0.0
    g = kpis(df).get("revenue_growth")
    assert g.status == SKIPPED and g.value is None and "zero or negative" in g.reason
    one = sales_df().iloc[:2]
    assert kpis(one).get("revenue_growth").status == SKIPPED


def test_growth_year_over_year_needs_the_prior_year():
    r = kpis(sales_df(), config=AnalyticsConfig(comparison="year_over_year"))
    assert r.get("revenue_growth").status == SKIPPED and "one year earlier" in r.get("revenue_growth").reason


def test_missing_roles_skip_instead_of_inventing_values():
    df = sales_df()[["order_id", "qty"]]
    r = kpis(df, {"order_id": "identifier", "qty": "quantity"})
    assert r.get("total_revenue").status == SKIPPED and r.get("profit_margin").status == SKIPPED
    assert r.get("distinct_customers").status == SKIPPED
    assert all(k.value is None for k in r.results if k.status != OK)


def test_role_override_changes_the_kpis():
    df = sales_df()
    base = kpis(df)
    assert base.get("total_revenue").value == 2500
    swapped = kpis(df, {**ROLES, "revenue": "balance", "profit": "revenue"})     # user says profit is the revenue column
    assert swapped.get("total_revenue").value == 300 and swapped.get("total_revenue").columns == ("profit",)
    assert model_for(df, ROLES).fingerprint() != model_for(df, {**ROLES, "revenue": "balance"}).fingerprint()


def test_identifier_columns_are_never_aggregated():
    df = pd.DataFrame({"id": range(1, 41), "weight": np.arange(40) * 1.5 + 1, "lot": [i % 3 + 1 for i in range(40)]})
    r = kpis(df, {"id": "identifier"})
    used = {c for k in r.ok() for c in k.columns}
    assert "id" not in used and "weight" in used
    assert r.get("sum_weight").value == pytest.approx(df["weight"].sum())


def test_empty_data_and_empty_filter_show_nothing():
    model = model_for(sales_df(), ROLES)               # an empty frame cannot be profiled; the engine must still cope
    r = compute_kpis(sales_df().iloc[0:0], model)
    assert r.results == () and r.notes
    none = kpis(sales_df(), data_filter=DataFilter((("category", ("ZZZ",)),)))
    assert none.results == () and "filters" in none.notes[0]


def test_filters_change_results_and_are_recorded():
    f = DataFilter((("category", ("A",)),))
    r = kpis(sales_df(), data_filter=f)
    assert r.get("total_revenue").value == 1200 and r.rows_used == 3 and r.rows_total == 6
    assert r.get("total_revenue").filters and "category" in r.get("total_revenue").filters[0]
    d = kpis(sales_df(), data_filter=DataFilter(date_col="order_date", date_start="2024-02-01", date_end="2024-02-29"))
    assert d.get("total_revenue").value == 300           # end date is inclusive


def test_input_is_not_modified():
    df = sales_df()
    before = df.copy(deep=True)
    kpis(df, data_filter=DataFilter((("category", ("A",)),)))
    pd.testing.assert_frame_equal(df, before)


def test_cache_key_changes_with_every_input():
    f1, f2 = DataFilter(), DataFilter((("category", ("A",)),))
    base = cache_key("fp", "clean", "sem", f1.key())
    assert base == cache_key("fp", "clean", "sem", f1.key())
    assert len({base, cache_key("fp2", "clean", "sem", f1.key()), cache_key("fp", "clean2", "sem", f1.key()),
                cache_key("fp", "clean", "sem2", f1.key()), cache_key("fp", "clean", "sem", f2.key())}) == 5


def test_cost_only_derives_net_profit_and_unit_cost_is_scaled():
    df = pd.DataFrame({"revenue": [100.0, 200.0], "cost": [60.0, 120.0]})
    r = kpis(df, {"revenue": "revenue", "cost": "cost"})
    assert r.get("total_cost").value == 180 and r.get("total_profit").value == 120
    assert r.get("profit_margin").value == pytest.approx(100 * 120 / 300)
    u = pd.DataFrame({"units_sold": [10, 20], "unit_cost": [2.0, 3.0]})
    ru = kpis(u, {"units_sold": "quantity", "unit_cost": "cost"})
    assert ru.get("total_cost").value == 80 and "x" in ru.get("total_cost").formula
    assert ru.get("total_profit").status == SKIPPED


def test_inventory_uses_latest_snapshot_only():
    df = pd.DataFrame({"date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"]),
                       "sku": ["a", "b", "a", "b"], "stock_level": [10, 99, 5, 20], "unit_cost": [2.0, 1.0, 2.0, 1.0],
                       "reorder_point": [6, 6, 6, 10]})
    r = kpis(df, {"date": "datetime", "sku": "product", "stock_level": "quantity", "unit_cost": "cost",
                  "reorder_point": "quantity"})
    assert r.get("stock_units").value == 25
    assert r.get("stock_value").value == 5 * 2 + 20 * 1
    assert r.get("reorder_candidates").value == 1
    assert "Latest snapshot" in r.get("stock_units").limitations[-1]


def test_a_failing_builder_does_not_stop_the_others(monkeypatch):
    import analytics.kpis as k

    def boom(ctx):
        raise RuntimeError("x")
    monkeypatch.setattr(k, "_BUILDERS", (k._k_revenue, boom))
    r = compute_kpis(sales_df(), model_for(sales_df(), ROLES))
    assert r.get("total_revenue").value == 2500 and any(x.status == SKIPPED and "Could not" in x.reason for x in r.results)


def test_general_fallback_when_no_roles_apply():
    df = pd.DataFrame({"a": np.arange(30) * 1.0, "b": np.arange(30) * 2.0 + 5})
    r = kpis(df, {})
    assert r.get("row_count").value == 30
    assert r.get("mean_a").value == pytest.approx(14.5) and r.get("sum_b").value == pytest.approx(df["b"].sum())

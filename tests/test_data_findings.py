"""Data-pattern findings and their insight rules, against hand-calculated values."""
import numpy as np
import pandas as pd

from analytics.data_findings import compute_findings
from analytics.kpis import OK, KPIReport, KPIResult
from ingestion.profiler import profile_dataframe
from insights.insight_engine import generate_insights
from insights.insight_rules import InsightInputs
from semantic.semantic_model import build_semantic_model


def model(df, roles):
    return build_semantic_model(df, profile_dataframe(df), roles)


def insights(df, roles):
    f = compute_findings(df, model(df, roles))
    return f, {i.id.split(":")[0] + ":" + i.id.split(":")[1] if ":" in i.id else i.id: i
               for i in generate_insights(InsightInputs(findings=f)).insights}


def test_group_gap_hand_calculated():
    # supplier S1 averages 10, S2 averages 20, 30 rows each: overall 15, gap = (20 - 10) / 15 = 66.7% -> high
    df = pd.DataFrame({"supplier": ["S1"] * 30 + ["S2"] * 30, "units_sold": [10.0] * 30 + [20.0] * 30})
    f, got = insights(df, {"supplier": "supplier", "units_sold": "quantity"})
    g = f.group_gaps[0]
    assert (g.high_label, g.low_label, round(g.gap_pct, 1)) == ("S2", "S1", 66.7)
    i = got["group_gap:supplier"]
    assert i.severity == "high" and "67%" in i.title and "not tested for significance" in i.limitations[0]


def test_small_groups_are_not_compared():
    df = pd.DataFrame({"supplier": ["S1"] * 5 + ["S2"] * 5, "units_sold": [10.0] * 5 + [20.0] * 5})
    assert compute_findings(df, model(df, {"supplier": "supplier", "units_sold": "quantity"})).group_gaps == ()


def test_flag_effect_hand_calculated():
    # 40 of 200 rows flagged (20%): mean 30 with flag, 20 without -> +50%
    df = pd.DataFrame({"promo": [1] * 40 + [0] * 160, "units_sold": [30.0] * 40 + [20.0] * 160})
    f, got = insights(df, {"promo": "flag", "units_sold": "quantity"})
    assert round(f.flag_effects[0].lift_pct, 1) == 50.0
    i = got["flag_effect:promo"]
    assert i.severity == "high" and i.direction == "up" and "association" in i.limitations[0]


def test_flag_with_rare_outcome_is_ignored():
    df = pd.DataFrame({"promo": [1] * 4 + [0] * 196, "units_sold": [30.0] * 4 + [20.0] * 196})   # 2% flagged < 5%
    assert compute_findings(df, model(df, {"promo": "flag", "units_sold": "quantity"})).flag_effects == ()


def test_weekday_pattern_hand_calculated():
    # 10 weeks from Monday 2024-01-01; Saturday 30, other days 10 -> daily mean 90/7; spread = 20 / (90/7) = 155.6%
    d = pd.date_range("2024-01-01", periods=70)
    df = pd.DataFrame({"date": d, "units_sold": np.where(d.dayofweek == 5, 30.0, 10.0)})
    f, got = insights(df, {"date": "datetime", "units_sold": "quantity"})
    assert (f.weekday.peak_day, round(f.weekday.spread_pct, 1)) == ("Saturday", 155.6)
    assert got["weekday:units_sold"].severity == "high"


def test_flat_weekdays_give_no_finding():
    d = pd.date_range("2024-01-01", periods=70)
    df = pd.DataFrame({"date": d, "units_sold": 10.0})
    assert "weekday:units_sold" not in insights(df, {"date": "datetime", "units_sold": "quantity"})[1]


def test_trend_latest_complete_month_and_partial_month_excluded():
    d = pd.date_range("2024-01-01", "2024-03-31")
    q = np.where(d.month == 3, 15.0, 10.0)                       # Feb 29 x 10 = 290, Mar 31 x 15 = 465
    df = pd.DataFrame({"date": d, "units_sold": q})
    t = compute_findings(df, model(df, {"date": "datetime", "units_sold": "quantity"})).trend
    assert (t.latest, t.previous, round(t.change_pct, 2)) == (465.0, 290.0, 60.34)
    part = df[df["date"] <= "2024-03-15"]                        # March is partial: Feb 290 vs Jan 310 = -6.45%
    t2 = compute_findings(part, model(part, {"date": "datetime", "units_sold": "quantity"})).trend
    assert (t2.latest, t2.previous) == (290.0, 310.0) and round(t2.change_pct, 2) == -6.45


def test_concentration_hand_calculated():
    # 20 entities: two with 100 units, eighteen with 10 -> total 380; top 10% = 2 entities = 200 -> 52.6%
    df = pd.DataFrame({"sku": [f"E{i}" for i in range(20)], "units_sold": [100.0, 100.0] + [10.0] * 18})
    f, got = insights(df, {"sku": "product", "units_sold": "quantity"})
    c = f.concentrations[0]
    assert (c.top_n, round(c.share_pct, 1)) == (2, 52.6)
    assert got["concentration:sku"].severity == "medium"


def test_forecast_column_accuracy_hand_calculated():
    # forecast 12 vs actual 10 on every row: WAPE = 20%, bias = +20%
    df = pd.DataFrame({"units_sold": [10.0] * 40, "demand_forecast": [12.0] * 40})
    f, got = insights(df, {"units_sold": "quantity", "demand_forecast": "quantity"})
    a = f.forecast_accuracy
    assert (a.forecast_col, a.actual_col, round(a.wape_pct, 1), round(a.bias_pct, 1)) == ("demand_forecast", "units_sold", 20.0, 20.0)
    assert got["forecast_accuracy:demand_forecast"].severity == "medium"
    assert got["forecast_bias:demand_forecast"].direction == "over"


def test_accurate_forecast_column_is_not_flagged():
    df = pd.DataFrame({"units_sold": [10.0] * 40, "demand_forecast": [10.5] * 40})                # 5% error and bias
    assert not [k for k in insights(df, {"units_sold": "quantity", "demand_forecast": "quantity"})[1] if k.startswith("forecast_")]


def test_findings_do_not_mutate_and_one_section_failing_is_reported():
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=70), "units_sold": 10.0})
    before = df.copy()
    f = compute_findings(df, model(df, {"date": "datetime", "units_sold": "quantity"}))
    pd.testing.assert_frame_equal(df, before)
    assert f.measure == "units_sold" and f.notes == ()


def test_days_of_cover_levels():
    def cover(v):
        r = KPIResult("days_of_cover", "Days of cover", "Supply chain", OK, v, "number", "f", ("stock",), (), (), "t")
        return generate_insights(InsightInputs(kpis=KPIReport((r,), 500, 500, "supply_chain"))).insights
    assert cover(23.5) == ()
    assert (cover(5.0)[0].severity, cover(5.0)[0].direction) == ("high", "low")
    assert (cover(10.0)[0].severity, cover(120.0)[0].direction) == ("medium", "high")

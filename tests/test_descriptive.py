"""Descriptive analytics and correlation, checked against hand-calculated answers."""
import math

import numpy as np
import pandas as pd
import pytest

from analytics.correlation import correlation
from analytics.descriptive import (aggregate_time, analysable_numeric_columns, categorical_summary, crosstab,
                                   date_summary, grouped_summary, numeric_summary, top_n_with_other)
from config.analytics_config import AnalyticsConfig
from ingestion.profiler import profile_dataframe
from semantic.semantic_model import build_semantic_model


def test_numeric_summary_hand_calculated():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 100]})
    r = numeric_summary(df, ["x"]).loc["x"]
    assert r["count"] == 5 and r["missing"] == 0
    assert r["mean"] == pytest.approx(22.0)
    assert r["median"] == 3 and r["q1"] == 2 and r["q3"] == 4
    assert r["std"] == pytest.approx(math.sqrt(1902.5))      # sample std, ddof=1
    assert r["min"] == 1 and r["max"] == 100 and r["skew"] > 1


def test_numeric_summary_edge_cases_and_no_mutation():
    df = pd.DataFrame({"const": [5, 5, 5], "txt": ["1,000", "2,000", None], "none": [np.nan] * 3, "one": [7, None, None]})
    before = df.copy()
    s = numeric_summary(df, ["const", "txt", "none", "one"])
    assert np.isnan(s.loc["const", "skew"]) and s.loc["const", "std"] == 0
    assert s.loc["txt", "mean"] == 1500 and s.loc["txt", "missing"] == 1
    assert s.loc["none", "count"] == 0 and np.isnan(s.loc["none", "mean"])
    assert np.isnan(s.loc["one", "std"])
    pd.testing.assert_frame_equal(df, before)


def test_infinite_values_are_ignored():
    s = numeric_summary(pd.DataFrame({"x": [1.0, 2.0, np.inf, -np.inf]}), ["x"]).loc["x"]
    assert s["count"] == 2 and s["mean"] == 1.5


def test_categorical_summary_caps_top_n():
    df = pd.DataFrame({"c": list("aaaabbbccd") + [f"z{i}" for i in range(10)]})
    s = categorical_summary(df, ["c"], AnalyticsConfig(top_n=3))["c"]
    assert s.distinct == 14 and list(s.top["value"]) == ["a", "b", "c"]
    assert list(s.top["count"]) == [4, 3, 2] and s.other_count == 11 and "3 most frequent" in s.note
    assert s.top["pct"].iloc[0] == pytest.approx(100 * 4 / 20)


def test_date_summary_finds_missing_days():
    df = pd.DataFrame({"d": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-06", "2024-01-07", "bad"]})
    s = date_summary(df, "d")
    assert s.valid == 5 and s.invalid == 1 and s.span_days == 6
    assert s.inferred_freq == "daily" and s.missing_periods == 2
    assert s.gaps == (("2024-01-04", "2024-01-05"),)


def test_aggregate_time_flags_partial_periods_and_gaps():
    df = pd.DataFrame({"d": pd.to_datetime(["2024-01-15", "2024-01-20", "2024-03-01", "2024-03-31", "2024-04-10"]),
                       "v": [10, 20, 5, 5, 99]})
    t = aggregate_time(df, "d", "v", "M")
    assert list(t["value"]) == [30, 0, 10, 99] and list(t["rows"]) == [2, 0, 2, 1]
    assert list(t["complete"]) == [False, True, True, False]     # Jan starts on the 15th, Apr ends on the 10th


def test_aggregate_time_mean_leaves_empty_periods_missing():
    df = pd.DataFrame({"d": pd.to_datetime(["2024-01-01", "2024-03-01"]), "v": [4.0, 8.0]})
    t = aggregate_time(df, "d", "v", "M", "mean")
    assert t["value"].iloc[0] == 4 and np.isnan(t["value"].iloc[1])


def test_top_n_with_other_sums_the_rest():
    s = pd.Series({"a": 50, "b": 30, "c": 10, "d": 6, "e": 4})
    r = top_n_with_other(s, 3)
    assert list(r.index) == ["a", "b", "c", "Other"] and r["Other"] == 10 and r.sum() == s.sum()


def test_grouped_summary_other_and_share():
    df = pd.DataFrame({"g": ["a", "a", "b", "c", "d"], "v": [50, 10, 30, 6, 4]})
    r = grouped_summary(df, "g", "v", "sum", AnalyticsConfig(top_n=2))
    assert list(r["group"]) == ["a", "b", "Other"] and list(r["value"]) == [60, 30, 10]
    assert list(r["rows"]) == [2, 1, 2] and r["share_pct"].sum() == pytest.approx(100)


def test_grouped_summary_mean_has_no_other_bucket():
    df = pd.DataFrame({"g": list("aabbcc"), "v": [1, 3, 5, 7, 9, 11]})
    r = grouped_summary(df, "g", "v", "mean", AnalyticsConfig(top_n=2))
    assert list(r["group"]) == ["c", "b"] and list(r["value"]) == [10, 6]


def test_crosstab_counts_and_sums():
    df = pd.DataFrame({"r": ["x", "x", "y"], "c": ["p", "q", "p"], "v": [1, 2, 4]})
    assert crosstab(df, "r", "c").loc["x", "p"] == 1
    assert crosstab(df, "r", "c", "v").loc["y", "p"] == 4


def test_correlation_hand_checked_and_filters_constants():
    x = np.arange(1.0, 21.0)
    df = pd.DataFrame({"x": x, "y": 2 * x + 1, "inv": -x, "const": 3.0, "few": [1.0] * 3 + [np.nan] * 17,
                       "noise": np.tile([1.0, -1.0, 2.0, -2.0], 5)})
    r = correlation(df, list(df.columns))
    assert r.excluded["const"].startswith("constant") and "few" in r.excluded
    assert "const" not in r.matrix.columns
    assert r.matrix.loc["x", "y"] == pytest.approx(1.0) and r.matrix.loc["x", "inv"] == pytest.approx(-1.0)
    top = r.pairs.iloc[0]
    assert {top["a"], top["b"]} <= {"x", "y", "inv"} and top["abs_r"] == pytest.approx(1.0) and top["note"]
    assert not ((r.pairs["a"] == "x") & (r.pairs["b"] == "noise")).any()    # weak pair dropped
    assert (r.pairs["a"] != r.pairs["b"]).all()


def test_correlation_with_too_few_columns_is_explained():
    r = correlation(pd.DataFrame({"x": range(20)}), ["x"])
    assert r.matrix.empty and r.pairs.empty and "Fewer than two" in r.note


def test_analysable_columns_skip_ids_constants_and_coordinates():
    n = 60
    df = pd.DataFrame({"order_id": range(n), "customer_id": [i % 7 for i in range(n)], "latitude": np.linspace(10, 20, n),
                       "const": 1, "revenue": np.arange(n) * 2.5, "quantity": [i % 5 + 1 for i in range(n)]})
    m = build_semantic_model(df, profile_dataframe(df))
    cols = analysable_numeric_columns(df, profile_dataframe(df), m)
    assert "order_id" not in cols and "const" not in cols and "latitude" not in cols and "customer_id" not in cols
    assert "revenue" in cols and "quantity" in cols

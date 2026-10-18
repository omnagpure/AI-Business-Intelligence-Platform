import numpy as np
import pandas as pd
import pytest

from core.exceptions import DataValidationError
from ingestion.profiler import profile_dataframe
from ingestion.schema_detector import parse_datetime_series


@pytest.fixture
def df():
    return pd.DataFrame({
        "customer_id": range(1, 101),
        "amount": [1, 2, 3, 4, 100] * 20,
        "category": ["A", "B"] * 50,
        "const": ["x"] * 100,
        "note": [None] * 60 + ["n"] * 40,
        "day": pd.date_range("2024-01-01", periods=100).strftime("%Y-%m-%d"),
        "price_text": ["1,200", "35", "4.5", "1,000"] * 25,
    })


def test_numeric_stats_known_values(df):
    s = profile_dataframe(df).column("amount").numeric
    assert s["count"] == 100 and s["min"] == 1 and s["max"] == 100
    assert s["mean"] == pytest.approx(22.0) and s["median"] == 3 and s["q1"] == 2 and s["q3"] == 4


def test_missing_and_kinds(df):
    p = profile_dataframe(df)
    note = p.column("note")
    assert note.missing == 60 and note.missing_pct == 60.0
    assert p.column("customer_id").kind == "identifier" and p.column("customer_id").is_identifier
    assert p.column("category").kind == "categorical" and p.column("category").cardinality == "low"
    assert p.column("const").is_constant
    assert p.column("amount").kind == "numeric"


def test_text_stored_numbers_and_dates(df):
    p = profile_dataframe(df)
    pt = p.column("price_text")
    assert pt.kind == "numeric" and pt.stored_as_text and pt.numeric["max"] == 1200
    d = p.column("day")
    assert d.kind == "temporal" and d.stored_as_text
    assert d.temporal["min"].startswith("2024-01-01") and d.temporal["span_days"] == 99


def test_categorical_top_values(df):
    top = profile_dataframe(df).column("category").top_values
    assert top[0]["count"] == 50 and top[0]["pct"] == 50.0


def test_duplicates_and_dataset_level(df):
    d2 = pd.concat([df, df.iloc[:5]], ignore_index=True)
    p = profile_dataframe(d2)
    assert p.rows == 105 and p.duplicate_rows == 5 and p.duplicate_pct == pytest.approx(4.76, abs=0.01)
    assert p.total_missing == 65  # 60 in the original + the 5 repeated rows (all have a missing note)


def test_profile_does_not_mutate_input(df):
    before = df.copy(deep=True)
    profile_dataframe(df)
    pd.testing.assert_frame_equal(df, before)


def test_empty_dataframe_rejected():
    with pytest.raises(DataValidationError):
        profile_dataframe(pd.DataFrame())


def test_all_null_column_is_empty_kind():
    p = profile_dataframe(pd.DataFrame({"a": [1, 2], "b": [None, None]}))
    assert p.column("b").kind == "empty"


def test_infinite_values_flagged():
    p = profile_dataframe(pd.DataFrame({"x": [1.0, 2.0, np.inf, 4.0]}))
    assert any("infinite" in s for s in p.column("x").suspicious)
    assert p.column("x").numeric["max"] == 4.0


def test_possible_target_and_primary_key():
    d = pd.DataFrame({"order_id": [f"O{i}" for i in range(50)], "revenue": np.arange(50) * 1.5,
                      "region": ["N", "S"] * 25})
    p = profile_dataframe(d)
    assert "revenue" in p.possible_targets and p.potential_primary_keys == ["order_id"]


def test_parse_datetime_formats_and_invalid():
    us = parse_datetime_series(pd.Series(["1/31/2018 22:56", "2/1/2018 03:00"]))
    assert us.iloc[0] == pd.Timestamp("2018-01-31 22:56")
    bad = parse_datetime_series(pd.Series(["2024-01-01", "2024-01-02", "not a date"] * 10))
    assert bad.isna().sum() == 10

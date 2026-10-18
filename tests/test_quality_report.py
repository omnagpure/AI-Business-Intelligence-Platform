import pandas as pd

from ingestion.profiler import profile_dataframe
from ingestion.quality_report import build_quality_report


def report(df, warnings=None):
    return build_quality_report(df, profile_dataframe(df), warnings)


def codes(r, col=None):
    return {i.code for i in r.issues if col is None or i.column == col}


def base(n=100):
    return pd.DataFrame({"v": range(n), "g": ["a", "b"] * (n // 2)})


def test_clean_data_is_good():
    r = report(base())
    assert r.status == "good" and r.completeness_pct == 100.0 and not r.issues


def test_high_missing_and_status_poor():
    d = base(); d["m"] = [None] * 60 + list(range(40))
    r = report(d)
    assert "high_missing" in codes(r, "m") and r.status == "poor"


def test_medium_missing_threshold():
    d = base(); d["m"] = [None] * 30 + list(range(70))
    assert "missing" in codes(report(d), "m")


def test_duplicates_reported_with_count():
    d = pd.concat([base(), base().iloc[:10]], ignore_index=True)
    iss = next(i for i in report(d).issues if i.code == "duplicate_rows")
    assert iss.affected_rows == 10


def test_inconsistent_labels_and_whitespace():
    d = pd.DataFrame({"cat": ["Home", "home ", "Toys", "toys"] * 10, "v": range(40)})
    c = codes(report(d), "cat")
    assert "inconsistent_labels" in c and "whitespace" in c


def test_constant_and_empty_columns():
    d = base(); d["c"] = 1; d["e"] = None
    r = report(d)
    assert "constant_column" in codes(r, "c") and "empty_column" in codes(r, "e")


def test_pii_flagged_by_name():
    d = base(); d["Customer Email"] = [f"u{i}@x.com" for i in range(100)]
    assert "possible_pii" in codes(report(d), "Customer Email")


def test_numeric_text_and_negative_quantity():
    d = base(); d["amt"] = ["1,000", "2"] * 50; d["quantity"] = [-1] + [5] * 99
    r = report(d)
    assert "numeric_as_text" in codes(r, "amt") and "negative_values" in codes(r, "quantity")


def test_few_rows_warning_and_load_warnings_included():
    r = report(pd.DataFrame({"a": [1, 2, 3]}), ["note from loader"])
    assert "few_rows" in codes(r) and "load_warning" in codes(r)

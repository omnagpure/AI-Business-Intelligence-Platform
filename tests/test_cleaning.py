import numpy as np
import pandas as pd
import pytest

from cleaning.cleaner import (CleaningResult, actions_fingerprint, apply_actions, propose_actions,
                              select_with_dependencies)
from cleaning.labels import build_label_mapping, build_name_mapping
from cleaning.type_inference import to_boolean, to_numeric_clean
from core.exceptions import CleaningError
from ingestion.pipeline import ingest_sample
from ingestion.profiler import profile_dataframe


def messy() -> pd.DataFrame:
    n = 60
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "Order ID": [f"A{i:04d}" for i in range(n)],
        "Order Date": [f"2024-01-{(i % 28) + 1:02d}" for i in range(n)],
        "Revenue": [f"${i * 10:,}.50" for i in range(n)],
        "Category": ["Home"] * 30 + ["home "] * 5 + ["Toys"] * 25,
        "Units": rng.integers(1, 6, n).astype(float),
        "Region": ["North", "South"] * 30,
    })
    df.loc[[3, 4, 5], "Units"] = np.nan
    df.loc[10, "Units"] = 500.0               # extreme value
    df.loc[20, "Revenue"] = "n/a"             # cannot be parsed
    return df


def _prof(df):
    return profile_dataframe(df)


def by_id(actions):
    return {a.id: a for a in actions}


# --- parsing helpers -------------------------------------------------------------------------
def test_to_numeric_clean_handles_currency_commas_percent_parentheses():
    s = pd.Series(["$1,234.50", " 12% ", "(3)", "abc", None])
    out = to_numeric_clean(s)
    assert out.iloc[0] == 1234.5 and out.iloc[1] == 12 and out.iloc[2] == -3
    assert out.iloc[3:].isna().all()


def test_to_boolean():
    out = to_boolean(pd.Series(["Yes", "no", "maybe", None]))
    assert bool(out.iloc[0]) is True and bool(out.iloc[1]) is False
    assert out.isna().tolist() == [False, False, True, True]


def test_label_mapping_merges_to_most_frequent_form():
    s = pd.Series(["Home"] * 5 + ["home "] * 2 + ["HOME"] + ["Toys"])
    mapping, rows = build_label_mapping(s)
    assert mapping == {"home ": "Home", "HOME": "Home"} and rows == 3


def test_name_mapping_snake_case_and_collisions():
    m = build_name_mapping(["Order ID", "orderId", "order_id", "Total ($)", "ok"])
    assert m["Order ID"] == "order_id_2" and m["orderId"] == "order_id_3" and "order_id" not in m
    assert m["Total ($)"] == "total"
    assert "ok" not in m
    assert len(set(m.get(c, c) for c in ["Order ID", "orderId", "order_id", "Total ($)", "ok"])) == 5


# --- proposals -------------------------------------------------------------------------------
def test_proposals_describe_real_findings():
    df = messy()
    acts = by_id(propose_actions(df, _prof(df)))
    assert acts["convert_numeric:Revenue"].affected_rows == 60          # every non-null text value is touched
    assert "1 value(s) cannot be parsed" in acts["convert_numeric:Revenue"].consequence
    assert acts["convert_datetime:Order Date"].affected_rows == 60
    assert acts["standardize_labels:Category"].affected_rows == 5
    assert acts["fill_missing:Units"].affected_rows == 3
    assert acts["fill_missing:Units"].params == {"strategy": "median"}
    assert acts["clip_outliers:Units"].affected_rows == 1
    assert acts["normalize_columns"].params["mapping"]["Order ID"] == "order_id"


def test_nothing_risky_is_selected_by_default():
    for ds in (messy(), ingest_sample().dataset.df):
        for a in propose_actions(ds, _prof(ds)):
            if a.selected_by_default:
                assert a.op in ("convert_numeric", "convert_datetime", "standardize_labels",
                                "strip_whitespace") or (a.op == "drop_column" and "empty" in a.condition)


def test_no_actions_on_clean_data():
    df = pd.DataFrame({"a": range(50), "b": [i * 1.5 for i in range(50)], "c": ["x", "y"] * 25})
    assert propose_actions(df, _prof(df)) == []


# --- applying --------------------------------------------------------------------------------
def test_original_dataframe_is_never_modified():
    df = messy()
    before = df.copy(deep=True)
    acts = propose_actions(df, _prof(df))
    apply_actions(df, acts)
    pd.testing.assert_frame_equal(df, before)


def test_apply_conversions_labels_and_logs_lost_values():
    df = messy()
    acts = [a for a in propose_actions(df, _prof(df)) if a.selected_by_default]
    res = apply_actions(df, acts)
    out = res.df
    assert pd.api.types.is_numeric_dtype(out["Revenue"]) and out["Revenue"].isna().sum() == 1
    assert out["Revenue"].iloc[1] == 10.5
    assert pd.api.types.is_datetime64_any_dtype(out["Order Date"])
    assert set(out["Category"]) == {"Home", "Toys"}
    conv = next(e for e in res.log if e.action_id == "convert_numeric:Revenue")
    assert conv.status == "applied" and "could not be parsed" in conv.detail
    assert res.applied_count == len(res.log) and res.skipped_count == 0


def test_missing_values_are_not_zero_filled_and_median_is_correct():
    df = messy()
    expected = df["Units"].median()
    acts = [a for a in propose_actions(df, _prof(df)) if a.id == "fill_missing:Units"]
    res = apply_actions(df, acts)
    assert res.df["Units"].isna().sum() == 0
    assert abs(res.df.loc[3, "Units"] - expected) < 1e-9
    assert (res.df["Units"] == 0).sum() == 0


def test_exact_duplicates_removed_keep_first_and_index_preserved():
    base = pd.DataFrame({"k": [1, 2, 3, 1], "v": ["a", "b", "c", "a"]})
    df = pd.concat([base] * 15, ignore_index=True)
    exact = int(df.duplicated().sum())
    acts = [a for a in propose_actions(df, _prof(df)) if a.id == "drop_duplicates:exact"]
    assert acts and acts[0].affected_rows == exact
    res = apply_actions(df, acts)
    assert len(res.df) == len(df) - exact
    assert list(res.df.index) == [i for i in df.index if not df.duplicated().iloc[i]]


def test_near_duplicates_ignore_case_and_spacing():
    df = pd.DataFrame({"name": ["Ann Lee", "ann  lee ", "Bob"] * 12, "n": [1, 1, 2] * 12})
    df = pd.concat([df, pd.DataFrame({"name": ["Zed"] * 3, "n": [9, 8, 7]})], ignore_index=True)
    acts = by_id(propose_actions(df, _prof(df)))
    assert "drop_duplicates:near" in acts
    res = apply_actions(df, [acts["drop_duplicates:near"]])
    assert len(res.df) == 3 + 2    # Ann, Bob, Zed x3 -> Ann, Bob + 3 Zed rows


def test_outlier_bounds_clip_and_drop_are_exclusive():
    df = messy()
    acts = by_id(propose_actions(df, _prof(df)))
    clip, drop = acts["clip_outliers:Units"], acts["drop_outlier_rows:Units"]
    q = df["Units"].dropna()
    q1, q3 = q.quantile(.25), q.quantile(.75)
    assert abs(clip.params["upper"] - (q3 + 3 * (q3 - q1))) < 1e-9
    res = apply_actions(df, [clip, drop])
    assert res.df["Units"].max() == clip.params["upper"] and len(res.df) == len(df)
    skipped = [e for e in res.log if e.status == "skipped"]
    assert len(skipped) == 1 and "Conflicts" in skipped[0].detail
    only_drop = apply_actions(df, [drop])
    assert len(only_drop.df) == len(df) - 1


def test_negative_values_only_proposed_for_non_negative_columns():
    df = pd.DataFrame({"quantity": [-1] + [2] * 40, "balance": [-5] + [3] * 40})
    acts = by_id(propose_actions(df, _prof(df)))
    assert "negative_to_missing:quantity" in acts and "negative_to_missing:balance" not in acts
    res = apply_actions(df, [acts["negative_to_missing:quantity"]])
    assert res.df["quantity"].isna().sum() == 1


def test_dependency_resolution_runs_conversion_before_dependent_actions():
    df = messy()
    df["Revenue"] = df["Revenue"].where(df["Revenue"] != "n/a", None)
    df.loc[[1, 2], "Revenue"] = None
    acts = propose_actions(df, _prof(df))
    chosen = select_with_dependencies(acts, ["fill_missing:Revenue"])
    assert {a.id for a in chosen} == {"fill_missing:Revenue", "convert_numeric:Revenue"}
    res = apply_actions(df, chosen)
    assert res.df["Revenue"].isna().sum() == 0 and res.skipped_count == 0
    lone = apply_actions(df, [a for a in acts if a.id == "fill_missing:Revenue"])
    assert lone.skipped_count == 1 and "convert" in lone.log[0].detail


def test_column_rename_keeps_originals_and_runs_last():
    df = messy()
    acts = [a for a in propose_actions(df, _prof(df)) if a.id in ("normalize_columns", "convert_numeric:Revenue")]
    res = apply_actions(df, acts)
    assert "revenue" in res.df.columns and res.rename_map["revenue"] == "Revenue"
    assert res.log[-1].op == "normalize_columns"


def test_actions_on_missing_columns_are_skipped_not_crashed():
    df = messy()
    acts = [a for a in propose_actions(df, _prof(df)) if a.id == "convert_numeric:Revenue"]
    res = apply_actions(df.drop(columns=["Revenue"]), acts)
    assert res.skipped_count == 1 and "not present" in res.log[0].detail


def test_unknown_operation_raises_clear_error():
    df = messy()
    bad = propose_actions(df, _prof(df))[0]
    bad = type(bad)(**{**bad.to_record(), "op": "explode", "requires": ()})
    with pytest.raises(CleaningError):
        apply_actions(df, [bad])


def test_empty_selection_returns_identical_data():
    df = messy()
    res = apply_actions(df, [])
    pd.testing.assert_frame_equal(res.df, df)
    assert res.log == [] and isinstance(res, CleaningResult)


# --- reproducibility / cache key -------------------------------------------------------------
def test_apply_is_reproducible_and_fingerprint_is_order_independent():
    df = messy()
    acts = [a for a in propose_actions(df, _prof(df)) if a.selected_by_default]
    r1, r2 = apply_actions(df, acts), apply_actions(df, list(reversed(acts)))
    pd.testing.assert_frame_equal(r1.df, r2.df)
    assert r1.actions_fingerprint == r2.actions_fingerprint == actions_fingerprint(acts)
    assert actions_fingerprint(acts[:-1]) != actions_fingerprint(acts)


def test_replaying_params_from_log_gives_same_result():
    df = messy()
    acts = [a for a in propose_actions(df, _prof(df)) if a.selected_by_default]
    again = propose_actions(df, _prof(df))
    pd.testing.assert_frame_equal(apply_actions(df, acts).df,
                                  apply_actions(df, [a for a in again if a.id in {x.id for x in acts}]).df)


# --- integration with the shipped sample data --------------------------------------------------
def test_sample_data_cleaning_end_to_end():
    r = ingest_sample()
    df = r.dataset.df
    acts = by_id(propose_actions(df, r.profile))
    assert acts["drop_duplicates:exact"].affected_rows == 5
    assert "standardize_labels:product_category" in acts and "fill_missing:region" in acts
    chosen = [acts[k] for k in ("standardize_labels:product_category", "drop_duplicates:exact", "fill_missing:region")]
    res = apply_actions(df, chosen, source_fingerprint=r.dataset.fingerprint)
    assert res.rows_after == len(df) - 5
    assert res.df["region"].isna().sum() == 0 and res.df["product_category"].nunique() == 4
    assert res.source_fingerprint == r.dataset.fingerprint

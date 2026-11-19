"""Segmentation (6C): entity features, KMeans choice, missing values, reproducibility, guard rails."""
import numpy as np
import pandas as pd
import pytest

from analytics.segmentation import (FeatureTable, build_entity_features, entity_candidates, quality_label,
                                    run_segmentation)
from config.analytics_config import AnalyticsConfig
from core.exceptions import InsufficientDataError, SegmentationError
from ingestion.profiler import profile_dataframe
from semantic.semantic_model import build_semantic_model

ROLES = {"supplier": "supplier", "ordered_at": "datetime", "expected_at": "datetime", "received_at": "datetime",
         "quantity_ordered": "quantity", "quantity_received": "quantity", "order_value": "revenue"}


def supplier_df(extra_rows=()):
    """12 suppliers x 10 orders. S01-S06 always on time; S07-S12 late on 8 of 10 orders. Order value = 1000 + 50 x index."""
    rows = []
    for i in range(12):
        good, late_n = i < 6, (0 if i < 6 else 8)
        for j in range(10):
            o = pd.Timestamp("2024-01-01") + pd.Timedelta(days=7 * j)
            exp = o + pd.Timedelta(days=14)
            late = j < late_n
            rows.append(dict(supplier=f"S{i + 1:02d}", ordered_at=o, expected_at=exp,
                             received_at=exp + pd.Timedelta(days=3 if late else -2), quantity_ordered=100,
                             quantity_received=100 if (good or not late) else 90, order_value=1000.0 + 50 * i))
    rows += list(extra_rows)
    return pd.DataFrame(rows)


def model_for(df):
    return build_semantic_model(df, profile_dataframe(df), ROLES)


def test_entity_candidates_need_enough_entities_and_rows():
    df = supplier_df()
    c = entity_candidates(df, model_for(df))
    assert c == [{"column": "supplier", "role": "supplier", "entities": 12, "avg_rows": 10.0}]
    few = df[df["supplier"].isin(["S01", "S02", "S03"])]
    assert entity_candidates(few, model_for(few)) == []                        # 3 suppliers < 12


def test_entity_features_match_hand_calculation():
    df = supplier_df()
    t = build_entity_features(df, model_for(df), "supplier")
    # S01: 10 of 10 on time = 100%, spend 10 x 1000; S07: orders j >= 8 are on time = 2 of 10 = 20%, spend 10 x 1300
    assert t.frame.loc["S01", "on_time_rate"] == pytest.approx(100.0)
    assert t.frame.loc["S07", "on_time_rate"] == pytest.approx(20.0)
    assert t.frame.loc["S01", "total_value"] == pytest.approx(10000.0)
    assert t.frame.loc["S07", "total_value"] == pytest.approx(13000.0)
    assert "same value for every entity" in t.excluded_features["rows"]       # every supplier has 10 rows
    assert "duplicates" in t.excluded_features["fill_rate"]                    # moves exactly with on-time rate here
    assert "supplier" not in t.frame.columns                                   # the identifier is never a feature


def test_entities_with_too_few_rows_are_left_out_and_reported():
    one = dict(supplier="S13", ordered_at=pd.Timestamp("2024-01-01"), expected_at=pd.Timestamp("2024-01-15"),
               received_at=pd.Timestamp("2024-01-10"), quantity_ordered=100, quantity_received=100, order_value=500.0)
    df = supplier_df([one])
    t = build_entity_features(df, model_for(df), "supplier")
    assert "S13" not in t.frame.index and t.excluded_entities == 1
    assert any("fewer than 3 rows" in w for w in t.warnings)


def test_good_and_bad_suppliers_land_in_different_clusters():
    df = supplier_df()
    r = run_segmentation(build_entity_features(df, model_for(df), "supplier"))
    cl = dict(zip(r.assignments["supplier"], r.assignments["cluster"]))
    assert len({cl[f"S{i:02d}"] for i in range(1, 7)}) == 1 and len({cl[f"S{i:02d}"] for i in range(7, 13)}) == 1
    assert cl["S01"] != cl["S07"] and r.k == 2 and r.k_mode == "automatic"
    assert r.silhouette > 0.5 and r.quality.startswith("Clear")
    low = [t for t in r.traits.values() if "lower on-time rate" in t]
    assert low and "20.0% vs 60.0% overall" in low[0]                          # (6 x 100 + 6 x 20) / 12 = 60


def test_same_seed_gives_identical_clusters_and_input_is_unchanged():
    df = supplier_df()
    before = df.copy(deep=True)
    t = build_entity_features(df, model_for(df), "supplier")
    a, b = run_segmentation(t), run_segmentation(t)
    pd.testing.assert_frame_equal(a.assignments, b.assignments)
    pd.testing.assert_frame_equal(df, before)


def table_from(frame):
    return FeatureTable("supplier", frame, {c: c for c in frame}, {c: "number" for c in frame}, {}, 0, ())


def test_missing_values_are_filled_with_the_median_and_reported():
    rng = np.random.default_rng(0)
    f = pd.DataFrame({"a": np.r_[rng.normal(0, 1, 10), rng.normal(8, 1, 10)], "b": np.r_[rng.normal(0, 1, 10), rng.normal(8, 1, 10)]},
                     index=[f"E{i}" for i in range(20)])
    f.loc["E3", "a"] = np.nan
    r = run_segmentation(table_from(f))
    assert any("filled with the median" in w and "a (1 entities)" in w for w in r.warnings)


def test_guard_rails_raise_clear_errors():
    f = pd.DataFrame({"a": range(20), "b": [x % 5 for x in range(20)]}, index=[f"E{i}" for i in range(20)], dtype=float)
    with pytest.raises(SegmentationError, match="at least two features"):
        run_segmentation(table_from(f), ["a"])
    with pytest.raises(SegmentationError, match="Unknown feature"):
        run_segmentation(table_from(f), ["a", "zzz"])
    with pytest.raises(SegmentationError, match="K must be"):
        run_segmentation(table_from(f), k=30)
    with pytest.raises(SegmentationError, match="Scaler"):
        run_segmentation(table_from(f), scaler="minmax")
    with pytest.raises(InsufficientDataError, match="at least 12"):
        run_segmentation(table_from(f.iloc[:8]))


def test_chosen_k_is_used_and_both_scalers_work():
    rng = np.random.default_rng(1)
    f = pd.DataFrame({"a": np.r_[rng.normal(0, 1, 15), rng.normal(10, 1, 15), rng.normal(20, 1, 15)],
                      "b": np.r_[rng.normal(0, 1, 15), rng.normal(10, 1, 15), rng.normal(0, 1, 15)]}, index=[f"E{i}" for i in range(45)])
    for sc in ("robust", "standard"):
        r = run_segmentation(table_from(f), k=3, scaler=sc)
        assert r.k == 3 and r.k_mode == "chosen" and sorted(r.profile["entities"], reverse=True) == list(r.profile["entities"])
    assert run_segmentation(table_from(f)).k == 3                              # three clear groups: silhouette picks K=3


def test_quality_labels_and_no_claim_of_objective_truth():
    assert quality_label(0.7).startswith("Clear") and quality_label(0.3).startswith("Moderate") and quality_label(0.1).startswith("Weak")
    df = supplier_df()
    r = run_segmentation(build_entity_features(df, model_for(df), "supplier"))
    assert "not objectively correct" in r.note

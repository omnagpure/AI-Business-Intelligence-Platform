"""Segmentation honesty: one candidate per grouping, orders not rows, no tiny-gap traits, names and separation."""
import pandas as pd
import pytest

from analytics.segmentation import (_meaningful, build_entity_features, entity_candidates, run_segmentation,
                                    separation)
from ingestion.profiler import profile_dataframe
from semantic.semantic_model import build_semantic_model
from tests.test_segmentation import ROLES, supplier_df


def model_for(df, roles=ROLES):
    return build_semantic_model(df, profile_dataframe(df), roles)


def test_id_and_name_with_the_same_grouping_are_offered_once():
    df = supplier_df()
    df["supplier_name"] = "Name " + df["supplier"]
    c = entity_candidates(df, model_for(df, {**ROLES, "supplier_name": "supplier"}))
    assert len(c) == 1 and c[0]["same_as"]


def test_tiny_gaps_are_not_traits():
    assert not _meaningful(0.04 * 100, 100, "number")      # 4% of the overall value
    assert _meaningful(0.06 * 100, 100, "number")
    assert not _meaningful(1.5, 40.0, "percent") and _meaningful(2.5, 40.0, "percent")
    assert not _meaningful(0.03, 0.5, "days")               # a fraction of a day


def test_segments_are_named_by_their_real_difference_and_separation_points_to_it():
    df = supplier_df()
    r = run_segmentation(build_entity_features(df, model_for(df), "supplier"))
    assert sorted(r.names.values()) == ["Higher on-time rate", "Lower on-time rate"]
    assert r.separation.index[0] in ("on_time_rate", "avg_delay", "fill_rate")
    assert r.separation.iloc[0] > 0.9 and r.separation.between(0, 1).all()


def test_separation_is_share_of_variance_between_groups():
    X = pd.DataFrame({"a": [0.0, 0.0, 10.0, 10.0], "b": [1.0, 2.0, 1.0, 2.0]})
    s = separation(X, pd.Series([1, 1, 2, 2]))
    assert s["a"] == pytest.approx(1.0) and s["b"] == pytest.approx(0.0)


def test_orders_are_counted_once_when_an_order_has_several_lines():
    df = supplier_df()
    # S01-S06: two lines per order (5 orders); S07-S12: one line per order (10 orders). Every supplier has 10 rows.
    df["order_id"] = [f"O{i // 2}" if i < 60 else f"O{i}" for i in range(len(df))]
    t = build_entity_features(df, model_for(df, {**ROLES, "order_id": "identifier"}), "supplier")
    assert t.labels["rows"] == "Orders"
    assert t.frame.loc["S01", "rows"] == 5 and t.frame.loc["S07", "rows"] == 10

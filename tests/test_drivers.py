"""Driver analysis (6D): importance, leakage rules, honest reliability, validation split, reproducibility."""
import numpy as np
import pandas as pd
import pytest

from analytics.drivers import driver_targets, run_driver_analysis
from core.exceptions import DriverAnalysisError, InsufficientDataError
from ingestion.profiler import profile_dataframe
from semantic.semantic_model import build_semantic_model


def make(n=600, seed=0):
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(0, 1, n), rng.normal(0, 1, n)
    grp = rng.choice(["A", "B", "C"], n)
    y = 3 * x1 + 5 * (grp == "A") + rng.normal(0, 0.5, n)
    return pd.DataFrame({"date": pd.date_range("2024-01-01", periods=n), "row_id": [f"R{i}" for i in range(n)],
                         "x1": x1, "x2": x2, "grp": grp, "y": y, "z_scaled": 2 * y + 1, "y_per_unit": y / 3 + rng.normal(0, 0.01, n)})


ROLES = {"date": "datetime", "row_id": "identifier"}


def run(df, target="y", features=None, roles=None, **kw):
    prof = profile_dataframe(df)
    return run_driver_analysis(df, target, prof, build_semantic_model(df, prof, roles or ROLES), features, **kw)


def test_the_real_driver_ranks_first_and_the_model_beats_the_baseline():
    r = run(make())
    assert r.task == "regression" and r.reliable
    assert list(r.importance["feature"][:2]) == ["x1", "grp"] and r.importance.loc[r.importance["feature"] == "x2", "share"].iloc[0] < 0.02
    assert r.metrics.loc[0, "model"] > 0.9 and r.metrics.loc[0, "baseline"] < 0.1       # held-out R-squared
    assert "higher values go with higher target" in r.importance.set_index("feature").loc["x1", "direction"]
    assert "grp" in r.groups and r.groups["grp"].iloc[0]["group"] == "A"                  # group A sits about 5 above the rest


def test_columns_built_from_the_target_are_left_out_with_a_reason():
    r = run(make())
    assert "almost identical" in r.excluded["z_scaled"]                                 # 2 x y + 1: correlation 1.00
    assert "name overlaps" in r.excluded["y_per_unit"]
    assert r.excluded["row_id"] == "identifier"
    assert not {"z_scaled", "y_per_unit", "row_id"} & set(r.importance["feature"])


def test_a_categorical_copy_of_the_target_is_detected():
    df = make()
    df["band"] = pd.qcut(df["y"], 20, labels=[f"q{i:02d}" for i in range(20)]).astype(str)       # 20 bins of y: about 99% explained
    r = run(df)
    assert "alone explain" in r.excluded["band"]


def test_split_is_chronological_and_independent_of_row_order():
    df = make()
    a = run(df)
    b = run(df.sample(frac=1, random_state=3).reset_index(drop=True))
    assert a.split.startswith("chronological: trained on 2024-01-01") and "tested on the later" in a.split
    assert a.n_test == 150 and a.n_train == 450
    np.testing.assert_allclose(a.metrics[["model", "baseline"]].to_numpy(), b.metrics[["model", "baseline"]].to_numpy())


def test_without_a_date_the_split_is_random_with_a_fixed_seed_and_says_so():
    df = make().drop(columns=["date"])
    r = run(df, roles={"row_id": "identifier"})
    assert r.split.startswith("random split with fixed seed 42") and any("random" in w for w in r.warnings)


def test_pure_noise_is_flagged_as_unreliable():
    rng = np.random.default_rng(5)
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=500), "a": rng.normal(size=500), "b": rng.normal(size=500),
                       "y": rng.normal(size=500)})
    r = run(df, roles={"date": "datetime"})
    assert not r.reliable and any("not trustworthy" in w for w in r.warnings)


def test_one_dominant_column_triggers_a_definition_warning():
    rng = np.random.default_rng(2)
    n = 400
    a = rng.normal(size=n)
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=n), "a": a, "b": rng.normal(size=n), "y": 4 * a + rng.normal(0, 0.05, n)})
    r = run(df, roles={"date": "datetime"})
    assert any("part of how the target is defined" in w for w in r.warnings)


def test_yes_no_target_uses_classification_and_auc():
    rng = np.random.default_rng(4)
    n = 600
    dist = rng.uniform(0, 100, n)
    late = np.where(rng.uniform(size=n) < 1 / (1 + np.exp(-(dist - 50) / 8)), "Yes", "No")
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=n), "distance": dist, "noise": rng.normal(size=n), "late": late})
    prof = profile_dataframe(df)
    assert driver_targets(df, prof)["late"] == "classification"
    r = run(df, target="late", roles={"date": "datetime"})
    assert r.task == "classification" and r.metrics.loc[0, "metric"].startswith("ROC AUC") and r.metrics.loc[0, "model"] > 0.8
    assert r.importance.loc[0, "feature"] == "distance"


def test_unsuitable_targets_and_too_little_data_are_refused_clearly():
    df = make()
    with pytest.raises(DriverAnalysisError, match="cannot be used as a target"):
        run(df, target="row_id")
    with pytest.raises(DriverAnalysisError, match="not in the data"):
        run(df, target="nope")
    with pytest.raises(InsufficientDataError):
        run(make(60))
    with pytest.raises(InsufficientDataError, match="Fewer than two"):
        run(df, features=["x1"])


def test_same_seed_gives_identical_importance_and_input_is_unchanged():
    df = make()
    before = df.copy(deep=True)
    a, b = run(df), run(df)
    pd.testing.assert_frame_equal(a.importance, b.importance)
    pd.testing.assert_frame_equal(df, before)


def test_association_is_not_presented_as_cause():
    assert "not proof of cause" in run(make()).note

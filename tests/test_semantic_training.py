"""Tests for the optional trained name model: data generation, training, safe loading, fallbacks."""
import json
import shutil

import numpy as np
import pandas as pd
import pytest

from ingestion.profiler import profile_dataframe
from semantic.column_classifier import classify_columns
from semantic.semantic_model import build_semantic_model
from semantic.trained_name_model import META_FILE, MODEL_FILE, load_name_model
from training.make_data import REAL_HEADERS, build_datasets, normalise


def test_data_is_deterministic_and_split_by_base_name():
    a_train, a_test = build_datasets(7)
    b_train, b_test = build_datasets(7)
    pd.testing.assert_frame_equal(a_train, b_train)
    pd.testing.assert_frame_equal(a_test, b_test)
    assert not set(a_train["base"]) & set(a_test["base"])
    assert set(a_train["role"]) == set(a_test["role"])           # every role appears in both


def test_no_exact_name_appears_with_two_different_roles_in_training():
    train, _ = build_datasets(7)
    roles_per_name = train.groupby(train["name"].map(normalise))["role"].nunique()
    # a handful of neutral variants such as 'x' or 'id' can collide; none may be widespread
    assert (roles_per_name > 1).mean() < 0.02


def test_real_headers_are_never_in_the_training_names_by_construction():
    train, _ = build_datasets(7)
    assert len(REAL_HEADERS) > 50 and "base" in train.columns
    assert set(REAL_HEADERS.values()) <= set(train["role"])


def test_missing_model_means_rules_only_with_a_clear_reason(tmp_path):
    model, status = load_name_model(tmp_path)
    assert model is None and not status.available and "No trained model" in status.reason


def test_corrupt_or_tampered_artifact_is_refused(tmp_path):
    (tmp_path / MODEL_FILE).write_bytes(b"not a real model")
    (tmp_path / META_FILE).write_text(json.dumps({"schema_version": 1, "sha256": "0" * 64}))
    model, status = load_name_model(tmp_path)
    assert model is None and "checksum" in status.reason
    (tmp_path / META_FILE).write_text("{broken")
    model, status = load_name_model(tmp_path)
    assert model is None and not status.available


def test_train_save_load_predict_roundtrip(tmp_path):
    pytest.importorskip("sklearn")
    from training.train_semantic_model import train
    meta = train(tmp_path, seed=11)
    assert (tmp_path / MODEL_FILE).is_file() and meta["sha256"]
    model, status = load_name_model(tmp_path)
    assert model is not None and status.available and status.version == meta["model_version"]
    probs = model.predict_proba("unit_price")
    assert abs(sum(probs.values()) - 1.0) < 1e-6 and set(probs) >= {"price", "revenue", "unknown"}
    assert max(probs, key=probs.get) == "price"
    # tampering after training is detected
    with open(tmp_path / MODEL_FILE, "ab") as fh:
        fh.write(b"x")
    assert load_name_model(tmp_path)[0] is None


def test_evaluation_runs_on_held_out_data_and_ties_results_to_the_artifact(tmp_path):
    pytest.importorskip("sklearn")
    from training.evaluate_semantic_model import evaluate, rules_name_only
    from training.train_semantic_model import train
    meta = train(tmp_path, seed=42)
    out = evaluate(tmp_path)
    assert out["model_sha256"] == meta["sha256"]
    assert 0.0 <= out["headline"]["held_out_accuracy"] <= 1.0 and out["headline"]["held_out_n"] > 500
    assert rules_name_only("total_sales") == "revenue" and rules_name_only("zzzz") == "unknown"
    assert load_name_model(tmp_path)[1].metrics["held_out_n"] == out["headline"]["held_out_n"]


class _AlwaysRevenue:
    def predict_proba(self, name):
        return {"revenue": 0.95, "unknown": 0.01}


def _df():
    rng = np.random.default_rng(8)
    return pd.DataFrame({"mystery": rng.normal(100, 20, 60), "order_date": pd.date_range("2024-01-01", periods=60).astype(str)})


def test_model_changes_source_only_when_it_changes_the_outcome():
    df = _df()
    prof = profile_dataframe(df)
    plain, blended = classify_columns(df, prof, None), classify_columns(df, prof, _AlwaysRevenue())
    assert plain["mystery"].role == "unknown" and blended["mystery"].role == "revenue"
    assert blended["mystery"].source == "model"
    assert blended["order_date"].role == "datetime" and blended["order_date"].source == "rules"


def test_user_override_still_wins_over_the_model():
    df = _df()
    m = build_semantic_model(df, profile_dataframe(df), role_overrides={"mystery": "cost"}, name_model=_AlwaysRevenue())
    assert (m.role_of("mystery"), m.columns["mystery"].source) == ("cost", "user")


def test_model_scores_do_not_modify_the_dataframe_or_rules_only_result():
    df = _df()
    before = df.copy(deep=True)
    prof = profile_dataframe(df)
    first = classify_columns(df, prof, None)
    classify_columns(df, prof, _AlwaysRevenue())
    pd.testing.assert_frame_equal(df, before)
    assert classify_columns(df, prof, None) == first

"""The faster permutation importance gives scikit-learn's numbers; thread count never changes results."""
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from analytics.drivers import _permutation_importance
from core.parallel import worker_count


def _fit(task, n=500, seed=0, all_missing_numeric=False):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"n1": rng.normal(size=n), "n2": rng.normal(size=n), "c1": rng.choice(["a", "b", "c"], n).astype(object),
                      "c2": rng.choice(["u", "v"], n).astype(object)})
    X.loc[rng.choice(n, 30, replace=False), "n1"] = np.nan
    if all_missing_numeric:
        X["n3"] = np.nan
    num = [c for c in X.columns if c.startswith("n")]
    y = 2 * X["n1"].fillna(0) + (X["c1"] == "a") * 3 + rng.normal(0, 0.3, n)
    if task == "classification":
        y = (y > y.median()).astype(float)
    pre = ColumnTransformer([("num", SimpleImputer(strategy="median"), num),
                             ("cat", Pipeline([("fill", SimpleImputer(strategy="constant", fill_value="(missing)")),
                                               ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), ["c1", "c2"])])
    Est = RandomForestClassifier if task == "classification" else RandomForestRegressor
    pipe = Pipeline([("pre", pre), ("rf", Est(n_estimators=30, max_depth=6, random_state=1, n_jobs=1))]).fit(X[:350], y[:350])
    return pipe, X[350:], y[350:]


@pytest.mark.parametrize("task,scoring", [("regression", "r2"), ("classification", "roc_auc")])
def test_matches_scikit_learn_exactly(task, scoring):
    pipe, X, y = _fit(task)
    ref = permutation_importance(pipe, X, y, scoring=scoring, n_repeats=5, random_state=42, n_jobs=1)
    mean, std = _permutation_importance(pipe, X, y, scoring, 5, 42)
    assert np.array_equal(mean, ref.importances_mean) and np.array_equal(std, ref.importances_std)


def test_result_does_not_depend_on_the_number_of_threads():
    import os
    pipe, X, y = _fit("regression")
    old, out = os.environ.get("BI_WORKERS"), {}
    try:
        for w in ("1", "4"):
            os.environ["BI_WORKERS"] = w
            out[w] = _permutation_importance(pipe, X, y, "r2", 5, 7)
    finally:
        os.environ.pop("BI_WORKERS", None) if old is None else os.environ.__setitem__("BI_WORKERS", old)
    assert np.array_equal(out["1"][0], out["4"][0]) and np.array_equal(out["1"][1], out["4"][1])


def test_unexpected_layout_falls_back_to_scikit_learn():
    pipe, X, y = _fit("regression", all_missing_numeric=True)       # the imputer drops the all-missing column: layout differs
    X, y = X.copy(), y
    mean, _std = _permutation_importance(pipe, X, y, "r2", 3, 5)
    ref = permutation_importance(pipe, X, y, scoring="r2", n_repeats=3, random_state=5, n_jobs=1)
    assert np.array_equal(mean, ref.importances_mean)


def test_worker_count_is_capped_and_overridable():
    import os
    old = os.environ.get("BI_WORKERS")
    try:
        os.environ.pop("BI_WORKERS", None)
        assert 1 <= worker_count() <= 4
        os.environ["BI_WORKERS"] = "2"
        assert worker_count() == 2
        os.environ["BI_WORKERS"] = "junk"
        assert 1 <= worker_count() <= 4
    finally:
        os.environ.pop("BI_WORKERS", None) if old is None else os.environ.__setitem__("BI_WORKERS", old)

"""Driver analysis (Phase 6D): which columns help explain a chosen target. Association, never causation.

A Random Forest is fitted on a held-out split (chronological when the data has a date column, otherwise a seeded
random split), compared with a trivial baseline, and columns are ranked by permutation importance on the held-out rows.
Columns that look derived from the target are excluded and listed with the reason. When the model does not beat the
baseline the ranking is flagged as unreliable. Pure pandas/scikit-learn; no Streamlit. Inputs are never modified.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from analytics.data_checks import parse_flag
from analytics.descriptive import analysable_numeric_columns, datetime_series, groupable_columns, numeric_series
from analytics.kpis import primary_date_column
from config.analytics_config import DEFAULT_ANALYTICS, AnalyticsConfig
from core.exceptions import DriverAnalysisError, InsufficientDataError
from core.parallel import worker_count
from core.progress import ProgressFn, notify
from semantic.column_lexicon import token_set

NOTE = ("Importance shows which columns help predict the target in this data. It is an association, not proof of cause: "
        "changing a driver may not change the target, and a hidden factor may drive both.")
DATE_PARTS = {"month": "Month (from date)", "weekday": "Weekday (from date)"}


@dataclass(frozen=True)
class DriverResult:
    target: str
    task: str                              # "regression" or "classification"
    n_rows: int                            # rows used (after dropping a missing target and sampling)
    n_train: int
    n_test: int
    split: str                             # plain-words description of the validation split
    metrics: pd.DataFrame                  # metric, model, baseline (on held-out rows)
    reliable: bool
    importance: pd.DataFrame               # feature, kind, importance, std, share, direction
    correlations: pd.DataFrame             # numeric features: feature, spearman
    groups: dict[str, pd.DataFrame]        # categorical feature -> group means of the target
    excluded: dict[str, str]               # column -> why it was left out
    warnings: tuple[str, ...] = field(default_factory=tuple)
    note: str = NOTE


# ---- targets ----------------------------------------------------------------------------------
def _binary(series: pd.Series, min_share: float = 0.05):
    f, _ = parse_flag(series)
    if f is None:
        return None
    k = f.dropna()
    if k.empty or k.nunique() < 2 or min(k.mean(), 1 - k.mean()) < min_share:
        return None
    return f.astype("Float64").astype(float)


def driver_targets(df: pd.DataFrame, profile, model=None, config: AnalyticsConfig = DEFAULT_ANALYTICS) -> dict[str, str]:
    """{column: 'regression' | 'classification'} for columns that can be explained: measures with enough distinct
    values and yes/no (or 0/1) columns where both outcomes are common."""
    out: dict[str, str] = {}
    num = analysable_numeric_columns(df, profile, model)
    for c in profile.columns_profile:
        if c.name not in df.columns:
            continue
        if c.kind in ("boolean", "categorical", "numeric") and c.unique == 2 and _binary(df[c.name]) is not None:
            out[c.name] = "classification"
        elif c.name in num:
            x = numeric_series(df[c.name]).dropna()
            if len(x) >= config.driver_min_rows and x.nunique() >= config.driver_min_unique:
                out[c.name] = "regression"
    return out


# ---- features and leakage rules ---------------------------------------------------------------
def candidate_features(df: pd.DataFrame, profile, model, target: str,
                       config: AnalyticsConfig = DEFAULT_ANALYTICS) -> tuple[list[str], list[str], dict[str, str]]:
    """(numeric features, categorical features, excluded {column: reason}) before any user choice."""
    excluded: dict[str, str] = {}
    num = [c for c in analysable_numeric_columns(df, profile, model) if c != target]
    cat = []
    for c in groupable_columns(df, profile, model, config):
        if c == target:
            continue
        if df[c].nunique(dropna=True) > config.driver_max_categories:
            excluded[c] = f"more than {config.driver_max_categories} distinct values"
        else:
            cat.append(c)
    for c in profile.columns_profile:
        if c.name in df.columns and c.name != target and c.name not in num + cat and c.name not in excluded:
            if c.is_identifier:
                excluded[c.name] = "identifier"
            elif c.is_constant:
                excluded[c.name] = "same value in every row"
            elif model is not None and c.name in model.columns and model.role_of(c.name) in {"text", "identifier"}:
                excluded[c.name] = f"{model.role_of(c.name)} column"
    return num, cat, excluded


def _leakage(df: pd.DataFrame, model, target: str, y: pd.Series, num: list[str], cat: list[str],
             config: AnalyticsConfig) -> dict[str, str]:
    """Columns that are probably built from the target. They would make the model look good for the wrong reason."""
    out: dict[str, str] = {}
    tt = token_set(target)
    trole = model.role_of(target) if model is not None and target in model.columns else None
    for c in num + cat:
        ct = token_set(c)
        if tt and (tt <= ct or ct <= tt) and ct:
            out[c] = "its name overlaps the target's name, so it is probably derived from it"
            continue
        if model is not None and c in model.columns:
            role = model.role_of(c)
            if (trole == "profit" and role in {"revenue", "cost"}) or (trole == "revenue" and role == "profit"):
                out[c] = f"{role} is arithmetically tied to a {trole} target"
                continue
        if c in cat:
            g = pd.DataFrame({"g": df[c].astype("string"), "y": y}).dropna()
            sst = float(((g["y"] - g["y"].mean()) ** 2).sum())
            if g["g"].nunique() >= 2 and len(g) >= 5 * g["g"].nunique() and sst > 0:
                sse = float(((g["y"] - g.groupby("g")["y"].transform("mean")) ** 2).sum())
                if 1 - sse / sst >= config.driver_leak_r2:
                    out[c] = (f"its values alone explain {1 - sse / sst:.0%} of the target, so it is probably another "
                              "way of recording the target")
                    continue
        if c in num:
            x = numeric_series(df[c])
            both = x.notna() & y.notna()
            if both.sum() >= 30 and x[both].nunique() > 1 and y[both].nunique() > 1:
                r = abs(x[both].corr(y[both]))
                if r >= config.driver_leak_corr:
                    out[c] = f"almost identical to the target (correlation {r:.2f}); likely a copy or a part of it"
    return out


def _spearman(x: pd.Series, y: pd.Series) -> float | None:
    both = x.notna() & y.notna()
    if both.sum() < 10 or x[both].nunique() < 2 or y[both].nunique() < 2:
        return None
    return float(x[both].rank().corr(y[both].rank()))


def available_date_parts(df: pd.DataFrame, model, target: str | None = None) -> list[str]:
    """Labels of the month / weekday features that can be derived from the data's date column (empty when none)."""
    col = primary_date_column(df, model) if model is not None else None
    if col is None or col == target or col not in df.columns:
        return []
    d = datetime_series(df[col])
    if d.notna().mean() < 0.8:
        return []
    return [DATE_PARTS[k] for k, v in (("month", d.dt.month), ("weekday", d.dt.dayofweek)) if v.nunique() > 1]


def _feature_columns(pre: ColumnTransformer, columns: list[str]) -> list[np.ndarray] | None:
    """For each raw column (in `columns` order) the indices of the columns it produced in the transformed matrix, or None
    when the layout is not the expected numeric-then-one-hot one (the caller then falls back to scikit-learn's routine)."""
    try:
        blocks: dict[str, np.ndarray] = {}
        pos = 0
        for name, trans, cols in pre.transformers_:
            if name == "num":
                stats = getattr(trans, "statistics_", None)
                if len(cols) and (stats is None or len(stats) != len(cols) or np.isnan(np.asarray(stats, dtype=float)).any()):
                    return None                                # the imputer drops all-missing columns, so the layout would shift
                for c in cols:
                    blocks[c] = np.array([pos])
                    pos += 1
            elif name == "cat":
                oh = trans.named_steps["oh"]
                for c, cats in zip(cols, oh.categories_):
                    blocks[c] = np.arange(pos, pos + len(cats))
                    pos += len(cats)
            elif name != "remainder" or len(cols):
                return None
        return [blocks[c] for c in columns] if all(c in blocks for c in columns) else None
    except Exception:
        return None


def _permutation_importance(pipe: Pipeline, X: pd.DataFrame, y: pd.Series, scoring: str, n_repeats: int, seed: int,
                            progress: ProgressFn | None = None):
    """Same numbers as sklearn.inspection.permutation_importance(pipe, X, y, ...) (same random draws, same scoring), but the
    preprocessing runs once instead of once per scoring and the columns are scored on worker threads. Returns (mean, std)."""
    layout = _feature_columns(pipe.named_steps["pre"], list(X.columns))
    if layout is None or len(X) == 0:
        pi = permutation_importance(pipe, X, y, scoring=scoring, n_repeats=n_repeats, random_state=seed, n_jobs=1)
        return pi.importances_mean, pi.importances_std
    rf = pipe.named_steps["rf"]
    Xt = np.asarray(pipe.named_steps["pre"].transform(X), dtype=float)
    if max((int(c.max()) for c in layout if len(c)), default=-1) >= Xt.shape[1]:        # layout does not match the matrix: use scikit-learn
        pi = permutation_importance(pipe, X, y, scoring=scoring, n_repeats=n_repeats, random_state=seed, n_jobs=1)
        return pi.importances_mean, pi.importances_std
    yv = y.to_numpy()

    def score(M: np.ndarray) -> float:
        if scoring == "r2":
            return float(r2_score(yv, rf.predict(M)))
        return float(roc_auc_score(yv, rf.predict_proba(M)[:, 1]))

    base = score(Xt)
    draw = int(np.random.RandomState(seed).randint(np.iinfo(np.int32).max + 1))     # as in scikit-learn: one seed, reused per column

    def one(cols: np.ndarray) -> list[float]:
        rs = np.random.RandomState(draw)
        M, idx, out = Xt.copy(), np.arange(len(Xt)), []
        for _ in range(n_repeats):
            rs.shuffle(idx)                                                           # cumulative shuffle, as scikit-learn does
            M[:, cols] = M[np.ix_(idx, cols)]
            out.append(score(M))
        return out

    workers = min(worker_count(), len(layout))
    scores: list[list[float]] = []

    def tick() -> None:                                                               # always on the calling thread, never a worker
        notify(progress, 3, 4, f"Checking which columns matter, {len(scores)} of {len(layout)} done")

    tick()
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for r in ex.map(one, layout):                                             # results arrive in column order
                scores.append(r)
                tick()
    else:
        for c in layout:
            scores.append(one(c))
            tick()
    drops = base - np.array(scores)                                                   # (features, repeats)
    return drops.mean(axis=1), drops.std(axis=1)


# ---- main -------------------------------------------------------------------------------------
def run_driver_analysis(df: pd.DataFrame, target: str, profile, model=None, features: list[str] | None = None,
                        config: AnalyticsConfig = DEFAULT_ANALYTICS, progress: ProgressFn | None = None) -> DriverResult:
    notify(progress, 0, 4, "Preparing the data")
    if target not in df.columns:
        raise DriverAnalysisError("The chosen target column is not in the data.")
    if int(df[target].notna().sum()) < config.driver_min_rows:
        raise InsufficientDataError(f"Only {int(df[target].notna().sum())} rows have a value for '{target}'; "
                                    f"driver analysis needs at least {config.driver_min_rows}.")
    targets = driver_targets(df, profile, model, config)
    if target not in targets:
        raise DriverAnalysisError("This column cannot be used as a target: it needs enough distinct numeric values, "
                                  "or a Yes/No column where both answers are common.")
    task = targets[target]
    y_all = _binary(df[target]) if task == "classification" else numeric_series(df[target])
    known = y_all.notna()
    if int(known.sum()) < config.driver_min_rows:
        raise InsufficientDataError(f"Only {int(known.sum())} rows have a value for '{target}'; "
                                    f"driver analysis needs at least {config.driver_min_rows}.")
    num, cat, excluded = candidate_features(df, profile, model, target, config)
    if features is not None:
        chosen = set(features)
        for c in num + cat:
            if c not in chosen:
                excluded[c] = "not selected"
        num, cat = [c for c in num if c in chosen], [c for c in cat if c in chosen]
    warnings: list[str] = []
    leak = _leakage(df, model, target, y_all, num, cat, config)
    excluded.update(leak)
    num, cat = [c for c in num if c not in leak], [c for c in cat if c not in leak]
    for c in list(num + cat):
        miss = float(df[c].isna().mean())
        if miss > config.driver_max_missing:
            excluded[c] = f"missing in {miss:.0%} of rows"
    num, cat = [c for c in num if c not in excluded], [c for c in cat if c not in excluded]
    date_col = primary_date_column(df, model) if model is not None else None
    dts = datetime_series(df[date_col]) if date_col and date_col != target and date_col in df.columns else None
    use_date = dts is not None and dts.notna().mean() >= 0.8
    X = pd.DataFrame({c: numeric_series(df[c]) for c in num}, index=df.index)
    for c in cat:
        X[c] = df[c].astype("string").fillna("(missing)").astype(object)
    if use_date:
        for part, label in DATE_PARTS.items():
            if label in available_date_parts(df, model, target) and (features is None or label in features):
                vals = dts.dt.month if part == "month" else dts.dt.dayofweek
                X[label] = vals.astype("Int64").astype(str).where(dts.notna(), "(missing)")
                cat = cat + [label]
    if X.shape[1] < 2:
        raise InsufficientDataError("Fewer than two usable explanatory columns remain after leaving out identifiers, "
                                    "near-copies of the target and columns with too many gaps.")
    mask = known & (dts.notna() if use_date else True)
    if use_date and int((known & ~dts.notna()).sum()):
        warnings.append(f"{int((known & ~dts.notna()).sum()):,} rows without a valid date were left out so the split can follow time.")
    idx = df.index[mask]
    if len(idx) > config.driver_max_fit_rows:
        idx = pd.Index(np.random.default_rng(config.driver_random_seed).choice(idx, config.driver_max_fit_rows, replace=False))
        warnings.append(f"A random sample of {len(idx):,} rows was used to keep the analysis fast.")
    if len(idx) < config.driver_min_rows:
        raise InsufficientDataError(f"Only {len(idx)} usable rows remain; driver analysis needs at least {config.driver_min_rows}.")
    X, y = X.loc[idx], y_all.loc[idx]
    if use_date:
        order = dts.loc[idx].sort_values(kind="stable").index
        X, y = X.loc[order], y.loc[order]
        n_test = max(int(round(len(X) * config.driver_test_share)), 1)
        Xtr, Xte, ytr, yte = X.iloc[:-n_test], X.iloc[-n_test:], y.iloc[:-n_test], y.iloc[-n_test:]
        d = dts.loc[X.index]
        split = (f"chronological: trained on {d.iloc[0]:%Y-%m-%d} to {d.iloc[len(Xtr) - 1]:%Y-%m-%d}, "
                 f"tested on the later {d.iloc[len(Xtr)]:%Y-%m-%d} to {d.iloc[-1]:%Y-%m-%d}")
    else:
        strat = y if task == "classification" and y.value_counts().min() >= 2 else None
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=config.driver_test_share, random_state=config.driver_random_seed,
                                              stratify=strat)
        split = f"random split with fixed seed {config.driver_random_seed} (the data has no date column)"
        warnings.append("Without a date column the split is random; if rows are related over time the score can look better than it would on future data.")
    if task == "classification" and yte.nunique() < 2:
        raise InsufficientDataError("The held-out rows contain only one outcome, so the model cannot be scored.")
    notify(progress, 1, 4, "Fitting the model (one fit, no partial progress)")
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), [c for c in X.columns if c in num]),
        ("cat", Pipeline([("fill", SimpleImputer(strategy="constant", fill_value="(missing)")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), [c for c in X.columns if c in cat]),
    ])
    Est = RandomForestClassifier if task == "classification" else RandomForestRegressor
    pipe = Pipeline([("pre", pre), ("rf", Est(n_estimators=config.driver_trees, max_depth=config.driver_max_depth,
                                              min_samples_leaf=5, random_state=config.driver_random_seed, n_jobs=worker_count()))])
    pipe.fit(Xtr, ytr)
    pipe.named_steps["rf"].set_params(n_jobs=1)        # trees are identical for any thread count; predicting on one thread keeps sums in a fixed order
    notify(progress, 2, 4, "Scoring on the held-out rows")
    if task == "regression":
        pred = pipe.predict(Xte)
        base = np.full(len(yte), float(ytr.mean()))
        metrics = pd.DataFrame([
            {"metric": "R-squared (held-out)", "model": r2_score(yte, pred), "baseline": r2_score(yte, base)},
            {"metric": "Mean absolute error (held-out)", "model": mean_absolute_error(yte, pred), "baseline": mean_absolute_error(yte, base)}])
        reliable, scoring = metrics.loc[0, "model"] >= config.driver_min_r2, "r2"
    else:
        prob = pipe.predict_proba(Xte)[:, 1]
        major = 1.0 if ytr.mean() >= 0.5 else 0.0                    # the trivial guess: always the training majority
        metrics = pd.DataFrame([
            {"metric": "ROC AUC (held-out)", "model": roc_auc_score(yte, prob), "baseline": 0.5},
            {"metric": "Accuracy (held-out)", "model": float(((prob >= 0.5) == yte.astype(bool)).mean()),
             "baseline": float((yte == major).mean())}])
        reliable, scoring = metrics.loc[0, "model"] >= config.driver_min_auc, "roc_auc"
    if not reliable:
        warnings.append("The model predicts the held-out rows little better than a trivial guess, so the ranking below is "
                        "not trustworthy. The target may depend on columns that are not in this data.")
    sub = Xte if len(Xte) <= config.driver_perm_rows else Xte.sample(config.driver_perm_rows, random_state=config.driver_random_seed)
    pi_mean, pi_std = _permutation_importance(pipe, sub, yte.loc[sub.index], scoring, config.driver_perm_repeats,
                                              config.driver_random_seed, progress)
    imp = pd.DataFrame({"feature": list(sub.columns), "importance": pi_mean, "std": pi_std})
    imp["kind"] = np.where(imp["feature"].isin(cat), "category", "numeric")
    pos = imp["importance"].clip(lower=0)
    imp["share"] = pos / pos.sum() if pos.sum() > 0 else 0.0
    sp = {c: _spearman(X[c], y) for c in num}
    imp["direction"] = [("higher values go with " + ("higher" if sp[f] > 0 else "lower") + " target") if f in sp and sp[f] is not None
                        and abs(sp[f]) >= 0.05 else ("no clear direction" if f in sp else "see group table") for f in imp["feature"]]
    imp = imp.sort_values("importance", ascending=False).reset_index(drop=True)
    if len(imp) and imp.loc[0, "share"] >= config.driver_dominance:
        warnings.append(f"'{imp.loc[0, 'feature']}' holds {imp.loc[0, 'share']:.0%} of the importance. Check that it is not part "
                        "of how the target is defined or calculated; if it is, it is a definition, not a driver.")
    notify(progress, 4, 4, "Ranking the columns")
    corr = pd.DataFrame([{"feature": c, "spearman": v} for c, v in sp.items() if v is not None])
    if not corr.empty:
        corr = corr.reindex(corr["spearman"].abs().sort_values(ascending=False).index).reset_index(drop=True)
    groups: dict[str, pd.DataFrame] = {}
    overall = float(y.mean())
    for f in [f for f in imp["feature"] if f in cat][:3]:
        g = pd.DataFrame({"group": X[f], "v": y}).groupby("group")["v"].agg(["mean", "size"]).reset_index()
        g = g[g["size"] >= 10].sort_values("mean", ascending=False).head(10)
        if len(g) >= 2:
            g["vs_overall"] = g["mean"] - overall
            groups[f] = g.rename(columns={"mean": "target_mean", "size": "rows"}).reset_index(drop=True)
    return DriverResult(target, task, len(X), len(Xtr), len(Xte), split, metrics, bool(reliable), imp, corr, groups,
                        excluded, tuple(warnings))

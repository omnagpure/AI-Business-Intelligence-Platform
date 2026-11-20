"""Anomaly detection (Phase 6B). Pure pandas/NumPy/scikit-learn, no Streamlit. Inputs are never modified.

Nothing is deleted or corrected: an anomaly is an unusual value, not necessarily an error.
Methods: robust z-score (median / MAD), IQR fences, classic z-score (only where roughly normal) and
Isolation Forest (only on suitable numeric features, fixed seed). ``series_anomalies`` flags unusual periods
against a robust baseline (trend plus seasonal pattern when the data supports one).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from analytics.descriptive import FREQ_LABELS, analysable_numeric_columns, numeric_series
from analytics.forecasting import MA_WINDOW, SEASONAL_PERIOD, detect_seasonality, prepare_series
from config.analytics_config import DEFAULT_ANALYTICS, AnalyticsConfig
from core.exceptions import AnomalyDetectionError, InsufficientDataError
from core.progress import ProgressFn, notify

NOTE = "An anomaly is an unusual value, not necessarily an error; check it before acting on it."
METHODS = ("mad", "iqr", "zscore", "isolation_forest")
METHOD_LABELS = {"mad": "Robust z-score (median/MAD)", "iqr": "IQR fences", "zscore": "Z-score",
                 "isolation_forest": "Isolation Forest"}
# sensitivity 1..5 -> threshold (lower threshold / higher contamination = more flags)
_MAD_Z = {1: 5.0, 2: 4.0, 3: 3.5, 4: 3.0, 5: 2.5}
_IQR_K = {1: 3.0, 2: 2.2, 3: 1.5, 4: 1.2, 5: 1.0}
_Z = {1: 4.0, 2: 3.5, 3: 3.0, 4: 2.5, 5: 2.0}
_CONTAM = {1: 0.005, 2: 0.01, 3: 0.02, 4: 0.05, 5: 0.10}
FINDING_COLUMNS = ["row", "method", "feature", "value", "score", "threshold", "direction", "explanation"]
# Columns screened first when the user has not chosen: money, volumes and times say most about the business.
ROLE_PRIORITY = ("revenue", "quantity", "duration", "cost", "profit", "price", "balance", "percentage")


@dataclass(frozen=True)
class AnomalyReport:
    findings: pd.DataFrame            # FINDING_COLUMNS; capped per method/column, sorted by score
    summary: pd.DataFrame             # method, feature, examined, flagged, share, threshold, status, note
    sensitivity: int
    warnings: tuple[str, ...] = field(default_factory=tuple)
    note: str = NOTE
    votes: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=["row", "methods", "checks", "features"]))
    examined_rows: int = 0
    column_votes: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=["feature", "flagged", "agreed"]))
    methods_run: int = 0                  # methods that ran on at least one column (skipped ones do not count)

    def agreed(self, min_methods: int = 2) -> pd.DataFrame:
        """Records flagged by at least `min_methods` different methods, most agreement first (exact, not capped)."""
        v = self.votes
        return v[v["methods"] >= min_methods].reset_index(drop=True) if len(v) else v


@dataclass(frozen=True)
class SeriesAnomalyResult:
    series: pd.DataFrame              # period, value, baseline, lower, upper, score, flagged
    findings: pd.DataFrame            # FINDING_COLUMNS for flagged periods ("row" is the period start)
    freq: str
    threshold: float
    seasonal: bool
    warnings: tuple[str, ...] = field(default_factory=tuple)
    note: str = NOTE


def _check_sensitivity(s: int) -> int:
    if s not in _MAD_Z:
        raise AnomalyDetectionError("Sensitivity must be a whole number from 1 to 5.")
    return int(s)


def _empty_findings() -> pd.DataFrame:
    return pd.DataFrame(columns=FINDING_COLUMNS)


# ---- column suitability ------------------------------------------------------------------------
def eligible_columns(df: pd.DataFrame, profile, model=None,
                     config: AnalyticsConfig = DEFAULT_ANALYTICS) -> tuple[list[str], dict[str, str]]:
    """Numeric columns worth screening, and the reason each other numeric column was left out."""
    ok: list[str] = []
    skipped: dict[str, str] = {}
    base = set(analysable_numeric_columns(df, profile, model))
    for c in profile.columns_profile:
        if c.name not in df.columns or c.kind != "numeric":
            continue
        if c.name not in base:
            skipped[c.name] = "identifier, constant or non-measure column"
            continue
        x = numeric_series(df[c.name]).dropna()
        if len(x) < config.anomaly_min_obs:
            skipped[c.name] = f"only {len(x)} values (needs {config.anomaly_min_obs})"
        elif x.nunique() < config.anomaly_min_unique:
            skipped[c.name] = f"only {x.nunique()} distinct values (flag or code, not a measure)"
        else:
            ok.append(c.name)
    if model is not None:
        rank = {r: i for i, r in enumerate(ROLE_PRIORITY)}
        ok.sort(key=lambda c: rank.get(model.role_of(c) if c in model.columns else "", len(rank)))
    return ok, skipped


# ---- univariate rules --------------------------------------------------------------------------
def mad_scores(x: np.ndarray) -> tuple[np.ndarray, float, float] | None:
    """Modified z-scores 0.6745*(x-median)/MAD. None when MAD is 0 (too many identical values)."""
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    if mad == 0:
        return None
    return 0.6745 * (x - med) / mad, med, mad


def iqr_fences(x: np.ndarray, k: float) -> tuple[float, float, float, float] | None:
    q1, q3 = float(np.quantile(x, 0.25)), float(np.quantile(x, 0.75))
    iqr = q3 - q1
    if iqr == 0:
        return None
    return q1 - k * iqr, q3 + k * iqr, q1, q3


def zscore_applicable(x: np.ndarray) -> tuple[bool, str]:
    """The classic z-score assumes a roughly symmetric, light-tailed distribution.

    Shape is judged on the data without its most extreme 1% per tail, so a few genuine outliers do not
    by themselves disable the rule; consistent skew or heavy tails still do.
    """
    if len(x) < 30:
        return False, "fewer than 30 values"
    k = max(1, int(0.01 * len(x)))
    core = np.sort(x)[k:-k]
    sd = float(np.std(core))
    if sd == 0:
        return False, "no variation"
    z = (core - core.mean()) / sd
    skew, kurt = float(np.mean(z ** 3)), float(np.mean(z ** 4) - 3)
    if abs(skew) > 1.0 or kurt > 7.0:
        return False, f"distribution is skewed or heavy-tailed (skew {skew:.1f}, excess kurtosis {kurt:.1f})"
    return True, ""


def _fmt(v: float) -> str:
    return f"{v:,.2f}"


def _finding(row, method, feature, value, score, thr, direction, text) -> dict:
    return {"row": row, "method": method, "feature": feature, "value": value, "score": float(score),
            "threshold": float(thr), "direction": direction, "explanation": f"{text} {NOTE}"}


def _top(mask: np.ndarray, score: np.ndarray, cap: int) -> tuple[np.ndarray, int]:
    """Positions of the `cap` strongest flagged values and the exact number flagged."""
    pos = np.flatnonzero(mask)
    return pos[np.argsort(-score[pos], kind="stable")][:cap], len(pos)


def _univariate(col: str, s: pd.Series, method: str, sensitivity: int, cap: int = 500,
                all_rows: list | None = None) -> tuple[list[dict], dict]:
    """Return (findings, summary row) for one column and one rule. Never raises for data reasons.
    When `all_rows` is given, the labels of every flagged row (not only the `cap` strongest) are appended to it."""
    x_all = numeric_series(s).dropna()
    x = x_all.to_numpy(dtype="float64")
    row = {"method": method, "feature": col, "examined": len(x), "flagged": 0, "share": 0.0, "threshold": None,
           "status": "ok", "note": ""}
    rows: list[dict] = []
    if method == "mad":
        res = mad_scores(x)
        thr = _MAD_Z[sensitivity]
        row["threshold"] = thr
        if res is None:
            return [], {**row, "status": "skipped", "note": "median absolute deviation is 0 (too many identical values)"}
        z, med, _ = res
        mask = np.abs(z) > thr
        top, n_flag = _top(mask, np.abs(z), cap)
        for i in top:
            idx = x_all.index[i]
            side = "above" if z[i] > 0 else "below"
            rows.append(_finding(idx, method, col, x[i], abs(z[i]), thr, "high" if z[i] > 0 else "low",
                                 f"{col} = {_fmt(x[i])} is {abs(z[i]):.1f} robust deviations {side} the median "
                                 f"({_fmt(med)}); the rule flags above {thr}."))
    elif method == "iqr":
        thr = _IQR_K[sensitivity]
        row["threshold"] = thr
        f = iqr_fences(x, thr)
        if f is None:
            return [], {**row, "status": "skipped", "note": "interquartile range is 0 (too many identical values)"}
        lo, hi, q1, q3 = f
        width = q3 - q1
        mask = (x < lo) | (x > hi)
        dists = np.where(x > hi, (x - hi) / width, (lo - x) / width)
        top, n_flag = _top(mask, dists, cap)
        for i in top:
            idx = x_all.index[i]
            high = x[i] > hi
            dist = dists[i]
            rows.append(_finding(idx, method, col, x[i], dist, thr, "high" if high else "low",
                                 f"{col} = {_fmt(x[i])} lies outside the usual range {_fmt(lo)} to {_fmt(hi)} "
                                 f"({thr} x IQR beyond the quartiles)."))
    elif method == "zscore":
        thr = _Z[sensitivity]
        row["threshold"] = thr
        ok, why = zscore_applicable(x)
        if not ok:
            return [], {**row, "status": "skipped", "note": f"z-score not used: {why}"}
        z = (x - x.mean()) / x.std()
        mask = np.abs(z) > thr
        top, n_flag = _top(mask, np.abs(z), cap)
        for i in top:
            idx = x_all.index[i]
            rows.append(_finding(idx, method, col, x[i], abs(z[i]), thr, "high" if z[i] > 0 else "low",
                                 f"{col} = {_fmt(x[i])} is {abs(z[i]):.1f} standard deviations "
                                 f"{'above' if z[i] > 0 else 'below'} the mean ({_fmt(x.mean())})."))
    else:
        raise AnomalyDetectionError(f"Unknown method: {method}")
    row["flagged"] = n_flag
    row["share"] = n_flag / len(x) if len(x) else 0.0
    if all_rows is not None:
        all_rows.extend(x_all.index[mask].tolist())
    return rows, row


# ---- Isolation Forest --------------------------------------------------------------------------
def _isolation_forest(df: pd.DataFrame, features: list[str], sensitivity: int,
                      config: AnalyticsConfig) -> tuple[list[dict], dict, list[str]]:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import RobustScaler

    notes: list[str] = []
    cont = _CONTAM[sensitivity]
    base = {"method": "isolation_forest", "feature": ", ".join(features), "examined": 0, "flagged": 0, "share": 0.0,
            "threshold": cont, "status": "skipped", "note": ""}
    if len(features) < 2:
        return [], {**base, "note": "needs at least 2 suitable numeric features"}, notes
    if len(features) > config.anomaly_max_features:
        features = features[:config.anomaly_max_features]
        notes.append(f"Only the first {config.anomaly_max_features} selected features were used.")
    X = pd.DataFrame({c: numeric_series(df[c]) for c in features})
    complete = X.dropna()
    dropped = len(X) - len(complete)
    if dropped:
        notes.append(f"{dropped:,} row(s) with a missing value in the selected features were left out of Isolation Forest "
                     "(not filled in).")
    if len(complete) < config.anomaly_if_min_rows:
        return [], {**base, "examined": len(complete),
                    "note": f"only {len(complete)} complete rows (needs {config.anomaly_if_min_rows})"}, notes
    scaled = RobustScaler().fit_transform(complete.to_numpy())
    rng = np.random.default_rng(config.anomaly_random_seed)
    fit_rows = scaled
    if len(scaled) > config.anomaly_if_max_fit_rows:
        fit_rows = scaled[rng.choice(len(scaled), config.anomaly_if_max_fit_rows, replace=False)]
        notes.append(f"Isolation Forest was fitted on a random sample of {config.anomaly_if_max_fit_rows:,} rows; all rows were scored.")
    forest = IsolationForest(n_estimators=200, contamination=cont, random_state=config.anomaly_random_seed, n_jobs=1)
    forest.fit(fit_rows)
    score = -forest.score_samples(scaled)                       # higher = more unusual
    cutoff = float(-forest.offset_)                             # score above this is flagged
    flagged = np.flatnonzero(score > cutoff)
    med = complete.median()
    mad = (complete - med).abs().median().replace(0, np.nan)
    rows: list[dict] = []
    for i in flagged:
        r = complete.iloc[i]
        dev = (0.6745 * (r - med) / mad).abs().dropna().sort_values(ascending=False)
        top = [f"{c} = {_fmt(r[c])} (median {_fmt(med[c])})" for c in dev.index[:3]]
        text = ("Unusual combination of values across the selected features" +
                (f"; most different: {'; '.join(top)}." if top else "."))
        rows.append(_finding(complete.index[i], "isolation_forest", "multiple features", np.nan, score[i], cutoff,
                             "multivariate", text))
    return rows, {**base, "examined": len(complete), "flagged": len(rows), "share": len(rows) / len(complete),
                  "status": "ok", "note": "", "_rows": complete.index[flagged].tolist()}, notes


# ---- orchestration -----------------------------------------------------------------------------
def run_anomaly_detection(df: pd.DataFrame, columns: list[str], methods: tuple[str, ...] = METHODS,
                          sensitivity: int | None = None,
                          config: AnalyticsConfig = DEFAULT_ANALYTICS, progress: ProgressFn | None = None) -> AnomalyReport:
    """Screen `columns` with each requested method. Each method/column fails independently."""
    sens = _check_sensitivity(config.anomaly_sensitivity if sensitivity is None else sensitivity)
    bad = [m for m in methods if m not in METHODS]
    if bad:
        raise AnomalyDetectionError(f"Unknown method: {bad[0]}")
    cols = [c for c in columns if c in df.columns]
    if not cols:
        raise InsufficientDataError("No numeric columns were selected for anomaly detection.")
    findings: list[dict] = []
    summary: list[dict] = []
    notes: list[str] = []
    hits: list[tuple] = []                      # (row, method, feature) for every flag, before any cap
    units = sum(1 if m == "isolation_forest" else len(cols) for m in methods)      # one unit = one method on one column
    unit = 0
    for method in methods:
        if method == "isolation_forest":
            notify(progress, unit, units, f"Screening ({METHOD_LABELS[method]}), {unit + 1} of {units}")
            unit += 1
            try:
                f, s, n = _isolation_forest(df, cols, sens, config)
            except Exception:
                f, s, n = [], {"method": method, "feature": ", ".join(cols), "examined": 0, "flagged": 0, "share": 0.0,
                               "threshold": None, "status": "failed", "note": "could not be run on this data"}, []
            findings += f[:config.anomaly_max_findings] if len(f) <= config.anomaly_max_findings else \
                sorted(f, key=lambda d: -d["score"])[:config.anomaly_max_findings]
            hits += [(r, method, "multiple features") for r in s.pop("_rows", [])]
            summary.append(s)
            notes += n
            continue
        for col in cols:
            notify(progress, unit, units, f"Screening {col} ({METHOD_LABELS[method]}), {unit + 1} of {units}")
            unit += 1
            flagged_rows: list = []
            try:
                f, s = _univariate(col, df[col], method, sens, config.anomaly_max_findings, flagged_rows)
                hits += [(r, method, col) for r in flagged_rows]
            except Exception:
                f, s = [], {"method": method, "feature": col, "examined": 0, "flagged": 0, "share": 0.0,
                            "threshold": None, "status": "failed", "note": "could not be run on this column"}
            if s["flagged"] > len(f):
                notes.append(f"{col} ({METHOD_LABELS[method]}): showing the {config.anomaly_max_findings} strongest of "
                             f"{s['flagged']:,} flagged values.")
            findings += f
            summary.append(s)
            if s["status"] == "ok" and s["share"] > 0.10:
                notes.append(f"{col} ({METHOD_LABELS[method]}) flagged {s['share']:.0%} of values; the column is probably "
                             "skewed. Lower the sensitivity, or treat these as the natural heavy tail.")
    out = pd.DataFrame(findings, columns=FINDING_COLUMNS)
    if len(out):
        out = out.sort_values(["method", "score"], ascending=[True, False]).reset_index(drop=True)
    summ = pd.DataFrame(summary)
    ran = int(summ.loc[summ["status"] == "ok", "method"].nunique()) if len(summ) else 0
    return AnomalyReport(out, summ, sens, tuple(dict.fromkeys(notes)), votes=_votes(hits),
                         examined_rows=int(df[cols].notna().any(axis=1).sum()), column_votes=_column_votes(hits, cols),
                         methods_run=ran)


def _column_votes(hits: list[tuple], cols: list[str]) -> pd.DataFrame:
    """Per column: records flagged by any single-column method, and by at least two of them (exact counts)."""
    rows = []
    h = pd.DataFrame(hits, columns=["row", "method", "feature"]) if hits else pd.DataFrame(columns=["row", "method", "feature"])
    h = h[h["feature"] != "multiple features"]
    per = h.groupby(["feature", "row"])["method"].nunique() if len(h) else pd.Series(dtype=float)
    for c in cols:
        v = per.loc[c] if len(per) and c in per.index.get_level_values(0) else pd.Series(dtype=float)
        rows.append({"feature": c, "flagged": int(len(v)), "agreed": int((v >= 2).sum())})
    return pd.DataFrame(rows, columns=["feature", "flagged", "agreed"])


def default_columns(df: pd.DataFrame, cols: list[str], limit: int = 6, sample: int = 5000) -> list[str]:
    """The first `limit` columns, skipping any that repeats one already chosen (same values, or correlation above
    0.999 on a sample): screening a copy twice would make one unusual record look flagged twice as often."""
    chosen: list[str] = []
    frame = df[cols].head(sample).apply(numeric_series)
    for c in cols:
        x = frame[c]
        dup = False
        for k in chosen:
            pair = pd.concat([x, frame[k]], axis=1).dropna()
            if len(pair) < 10:
                continue
            a, b = pair.iloc[:, 0], pair.iloc[:, 1]
            if a.equals(b) or (a.std() > 0 and b.std() > 0 and abs(float(a.corr(b))) > 0.999):
                dup = True
                break
        if not dup:
            chosen.append(c)
        if len(chosen) >= limit:
            break
    return chosen


def _votes(hits: list[tuple]) -> pd.DataFrame:
    """One row per flagged record: how many methods and checks flagged it, and in which columns."""
    if not hits:
        return pd.DataFrame(columns=["row", "methods", "checks", "features"])
    h = pd.DataFrame(hits, columns=["row", "method", "feature"])

    def cols(f):
        named = list(dict.fromkeys(x for x in f if x != "multiple features"))
        return ", ".join(named) if named else "several columns together"
    g = h.groupby("row", sort=False).agg(methods=("method", "nunique"), checks=("method", "size"), features=("feature", cols))
    return g.reset_index().sort_values(["methods", "checks"], ascending=False, kind="stable").reset_index(drop=True)


def feature_comparison(df: pd.DataFrame, row, columns: list[str]) -> pd.DataFrame:
    """One row's values next to each column's median and its robust deviation (for the feature comparison view)."""
    out = []
    for c in columns:
        if c not in df.columns or row not in df.index:
            continue
        x = numeric_series(df[c])
        v = x.loc[row] if not isinstance(x.loc[row], pd.Series) else x.loc[row].iloc[0]
        xs = x.dropna()
        if xs.empty:
            continue
        med = float(xs.median())
        mad = float((xs - med).abs().median())
        z = None if (mad == 0 or pd.isna(v)) else float(0.6745 * (v - med) / mad)
        out.append({"Feature": c, "Value": None if pd.isna(v) else float(v), "Median": med, "Robust deviation": z})
    return pd.DataFrame(out)


def context_columns(df: pd.DataFrame, model, limit: int = 3) -> list[str]:
    """Columns that tell you which entity/date a row is (identifier, date, product, customer)."""
    out: list[str] = []
    for role in ("datetime", "identifier", "product", "customer", "supplier"):
        for c in model.columns_with_role(role) if model is not None else []:
            if c in df.columns and c not in out:
                out.append(c)
                break
    return out[:limit]


def attach_context(findings: pd.DataFrame, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Add the context columns of each flagged row (looked up by row label); the data itself is untouched."""
    if findings.empty or not cols:
        return findings
    ctx = df.loc[findings["row"].to_numpy(), cols].reset_index(drop=True)
    return pd.concat([findings.reset_index(drop=True), ctx], axis=1)


# ---- unusual periods in a time series ----------------------------------------------------------
def series_anomalies(df: pd.DataFrame, date_col: str, value_col: str | None, freq: str = "D", agg: str = "sum",
                     sensitivity: int | None = None,
                     config: AnalyticsConfig = DEFAULT_ANALYTICS) -> SeriesAnomalyResult:
    """Flag periods whose value is far from a robust baseline (rolling-median trend + seasonal pattern if supported)."""
    sens = _check_sensitivity(config.anomaly_sensitivity if sensitivity is None else sensitivity)
    if freq not in SEASONAL_PERIOD or agg not in {"sum", "mean", "median", "count"}:
        raise AnomalyDetectionError("Unsupported period or aggregation.")
    if date_col not in df.columns or (value_col is not None and agg != "count" and value_col not in df.columns):
        raise AnomalyDetectionError("The selected column is not in the dataset.")
    table, notes = prepare_series(df, date_col, value_col, freq, agg, config)
    y = pd.Series(table["value"].to_numpy(dtype="float64"))
    m = SEASONAL_PERIOD[freq]
    seasonal, why = detect_seasonality(y.to_numpy(), m)
    win = m if seasonal else MA_WINDOW[freq]
    trend = y.rolling(win, center=True, min_periods=max(2, win // 2)).median()
    baseline = trend
    if seasonal:
        pos = np.arange(len(y)) % m
        baseline = trend + (y - trend).groupby(pos).transform("median")
    resid = y - baseline
    med = float(resid.median())
    mad = float((resid - med).abs().median())
    if mad == 0:
        raise InsufficientDataError("The series follows its baseline almost exactly, so no period stands out.")
    score = 0.6745 * (resid - med) / mad
    thr = _MAD_Z[sens]
    flagged = score.abs() > thr
    reach = thr * mad / 0.6745                       # outside baseline + med +/- reach is flagged
    series = pd.DataFrame({"period": table["period"].to_numpy(), "value": y, "baseline": baseline,
                           "lower": baseline + med - reach, "upper": baseline + med + reach,
                           "score": score, "flagged": flagged})
    rows = []
    for i in np.flatnonzero(flagged.to_numpy()):
        v, b, z = y.iloc[i], baseline.iloc[i], score.iloc[i]
        rows.append(_finding(table["period"].iloc[i], "series_baseline", value_col or "rows", v, abs(z), thr,
                             "high" if z > 0 else "low",
                             f"The {FREQ_LABELS[freq]} starting {pd.Timestamp(table['period'].iloc[i]).date()} had {_fmt(v)}, "
                             f"versus about {_fmt(b)} expected from nearby periods"
                             f"{' and the usual pattern' if seasonal else ''} ({abs(z):.1f} robust deviations)."))
    findings = pd.DataFrame(rows, columns=FINDING_COLUMNS).sort_values("score", ascending=False).reset_index(drop=True) \
        if rows else _empty_findings()
    if seasonal:
        notes.append(f"A repeating pattern of {m} {FREQ_LABELS[freq]}s was taken into account ({why}).")
    return SeriesAnomalyResult(series, findings, freq, thr, seasonal, tuple(notes))

"""Forecasting (Phase 6A). Pure pandas/NumPy/statsmodels, no Streamlit. Inputs are never modified.

Method (all deterministic, no random numbers are used):
1. Aggregate to day / week / month with ``aggregate_time``; incomplete first/last periods are dropped.
2. Refuse (InsufficientDataError, with a reason) when the series is too short, constant or too gappy.
3. Compare candidate models by rolling-origin (expanding window) validation: every fold trains only on
   periods before its origin and is scored on the periods right after it. No random splits.
4. Seasonality is tested on the data *before the first validation origin* (no look-ahead).
5. The simplest model within ``forecast_simpler_margin`` of the best MAE is chosen.
6. Intervals are empirical: quantiles of the chosen model's out-of-sample validation errors.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from analytics.descriptive import FREQ_LABELS, aggregate_time, analysable_numeric_columns, datetime_series
from analytics.kpis import primary_column, primary_date_column
from config.analytics_config import DEFAULT_ANALYTICS, AnalyticsConfig
from core.exceptions import ForecastingError, InsufficientDataError
from core.progress import ProgressFn, notify

SEASONAL_PERIOD = {"D": 7, "W": 52, "M": 12}
MIN_OBS = {"D": 56, "W": 26, "M": 24}              # complete periods needed before any model is tried
DEFAULT_HORIZON = {"D": 14, "W": 8, "M": 6}
MA_WINDOW = {"D": 7, "W": 4, "M": 3}
_STEP = {"D": "D", "W": "7D", "M": "MS"}
MAPE_MIN_SHARE = 0.05                               # MAPE needs every actual >= 5% of the mean actual


# ---- result types ------------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelScore:
    key: str
    label: str
    complexity: int
    status: str                      # ok | skipped | failed
    reason: str = ""
    mae: float | None = None
    rmse: float | None = None
    mase: float | None = None
    mape: float | None = None        # None when zeros / near-zero actuals make it misleading
    n_errors: int = 0


@dataclass(frozen=True)
class ForecastResult:
    date_col: str
    value_col: str | None
    agg: str
    freq: str
    horizon: int
    history: pd.DataFrame            # period, value
    forecast: pd.DataFrame           # period, forecast, lower, upper
    backtest: pd.DataFrame           # period, actual, predicted, fold  (chosen model, out-of-sample)
    scores: tuple[ModelScore, ...]
    chosen: str
    baseline: str
    improvement_vs_baseline_pct: float | None
    interval_level: float
    seasonal_period: int
    seasonality_supported: bool
    validation_horizon: int
    folds: int
    warnings: tuple[str, ...] = field(default_factory=tuple)
    limitations: tuple[str, ...] = field(default_factory=tuple)

    def scores_frame(self) -> pd.DataFrame:
        return pd.DataFrame([{"Model": s.label, "Status": s.status, "MAE": s.mae, "RMSE": s.rmse, "MASE": s.mase,
                              "MAPE %": s.mape, "Note": s.reason, "Selected": s.key == self.chosen}
                             for s in sorted(self.scores, key=lambda x: (x.mae is None, x.mae or 0.0))])


# ---- setup helpers -----------------------------------------------------------------------------
def suggest_frequency(df: pd.DataFrame, date_col: str) -> str:
    """Finest of D/W/M that gives comfortably more periods than the minimum (daily only up to ~3 years)."""
    d = datetime_series(df[date_col]).dropna()
    if d.empty:
        return "D"
    span = (d.max() - d.min()).days + 1
    for f, days in (("D", 1.0), ("W", 7.0), ("M", 30.4)):
        if span / days >= MIN_OBS[f] * 1.5 and (f != "D" or span <= 1100):
            return f
    return "D"


def suggest_setup(df: pd.DataFrame, model) -> dict:
    """Default date column, target, aggregation and frequency from the semantic roles (all changeable)."""
    date_col = primary_date_column(df, model)
    value = primary_column(df, model, "quantity") or primary_column(df, model, "revenue")
    return {"date_col": date_col, "value_col": value, "agg": "sum",
            "freq": suggest_frequency(df, date_col) if date_col else "D"}


def target_columns(df: pd.DataFrame, profile, model) -> list[str]:
    return analysable_numeric_columns(df, profile, model)


# ---- series preparation ------------------------------------------------------------------------
def prepare_series(df: pd.DataFrame, date_col: str, value_col: str | None, freq: str, agg: str,
                   config: AnalyticsConfig = DEFAULT_ANALYTICS) -> tuple[pd.DataFrame, list[str]]:
    """Return (period, value) for complete periods only, plus warnings. Raises InsufficientDataError."""
    notes: list[str] = []
    t = aggregate_time(df, date_col, value_col, freq, agg)
    if not t["complete"].all():
        notes.append(f"{int((~t['complete']).sum())} partly covered {FREQ_LABELS[freq]} period(s) at the edges were left out.")
        t = t[t["complete"]]
    if t.empty:
        raise InsufficientDataError("No complete periods are available.")
    gap = (t["rows"] == 0) | t["value"].isna()
    share = float(gap.mean())
    if share > config.forecast_max_gap_share:
        raise InsufficientDataError(f"{share:.0%} of the {FREQ_LABELS[freq]}s have no data; a forecast would be unreliable. "
                                    "Try a coarser period.")
    values = t["value"].copy()
    if gap.any():
        if agg in ("sum", "count"):
            notes.append(f"{int(gap.sum())} empty {FREQ_LABELS[freq]}(s) are treated as zero.")
        else:
            values = values.where(~gap).interpolate(limit_direction="both")
            notes.append(f"{int(gap.sum())} empty {FREQ_LABELS[freq]}(s) were filled by linear interpolation.")
    out = pd.DataFrame({"period": t["period"].to_numpy(), "value": values.to_numpy(dtype="float64")})
    if len(out) > config.forecast_max_history:
        notes.append(f"Only the latest {config.forecast_max_history} periods are modelled.")
        out = out.iloc[-config.forecast_max_history:].reset_index(drop=True)
    if len(out) < MIN_OBS[freq]:
        raise InsufficientDataError(f"Only {len(out)} complete {FREQ_LABELS[freq]}(s) of data; at least {MIN_OBS[freq]} "
                                    "are needed for a validated forecast.")
    if float(out["value"].std(ddof=0)) == 0.0:
        raise InsufficientDataError("The series is constant, so there is nothing to forecast.")
    return out, notes


def detect_seasonality(y: np.ndarray, m: int) -> tuple[bool, str]:
    """Seasonal if >= 2 full cycles exist and the detrended autocorrelation at lag m is clearly positive."""
    n = len(y)
    if m < 2 or n < 2 * m:
        return False, f"needs at least 2 full cycles of {m} periods (have {n})"
    x = np.arange(n, dtype=float)
    d = y - np.polyval(np.polyfit(x, y, 1), x)
    d = d - d.mean()
    denom = float((d ** 2).sum())
    if denom == 0:
        return False, "no variation after removing the trend"
    r = float((d[m:] * d[:-m]).sum() / denom)
    thr = max(0.2, 2.0 / np.sqrt(n))
    return r >= thr, f"autocorrelation at lag {m} is {r:.2f} (threshold {thr:.2f})"


# ---- candidate models --------------------------------------------------------------------------
def _quiet(fn):
    def wrapper(*a, **k):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return fn(*a, **k)
    return wrapper


def _naive(y, h, m, w):
    return np.repeat(y[-1], h)


def _seasonal_naive(y, h, m, w):
    return y[-m:][np.arange(h) % m]


def _moving_average(y, h, m, w):
    return np.repeat(y[-w:].mean(), h)


@_quiet
def _ses(y, h, m, w):
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing
    return np.asarray(SimpleExpSmoothing(y, initialization_method="estimated").fit().forecast(h))


@_quiet
def _holt(y, h, m, w):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    return np.asarray(ExponentialSmoothing(y, trend="add", damped_trend=True,
                                           initialization_method="estimated").fit().forecast(h))


@_quiet
def _holt_winters(y, h, m, w):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    return np.asarray(ExponentialSmoothing(y, trend="add", damped_trend=True, seasonal="add", seasonal_periods=m,
                                           initialization_method="estimated").fit().forecast(h))


@_quiet
def _arima(y, h, m, w):
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
    d = 1 if adfuller(y, autolag="AIC")[1] > 0.05 else 0     # unit-root test on the training window only
    return np.asarray(ARIMA(y, order=(1, d, 1), trend="n" if d else "c").fit().forecast(h))


@_quiet
def _sarima(y, h, m, w):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    fit = SARIMAX(y, order=(0, 1, 1), seasonal_order=(0, 1, 1, m)).fit(disp=False)
    return np.asarray(fit.forecast(h))


# key, label, complexity (lower = simpler), needs seasonality, minimum training length, function
_MODELS = (
    ("naive", "Naive (last value)", 0, False, lambda m, w: 2, _naive),
    ("seasonal_naive", "Seasonal naive", 1, True, lambda m, w: m, _seasonal_naive),
    ("moving_average", "Moving average", 1, False, lambda m, w: w + 1, _moving_average),
    ("ses", "Simple exponential smoothing", 2, False, lambda m, w: 10, _ses),
    ("holt", "Holt (damped trend)", 3, False, lambda m, w: 10, _holt),
    ("holt_winters", "Holt-Winters (damped trend, additive seasonality)", 3, True, lambda m, w: 2 * m + 2, _holt_winters),
    ("arima", "ARIMA(1,d,1)", 4, False, lambda m, w: 20, _arima),
    ("sarima", "SARIMA(0,1,1)(0,1,1)", 5, True, lambda m, w: 3 * m + 2, _sarima),
)
_BY_KEY = {k: (label, c, needs, mt, fn) for k, label, c, needs, mt, fn in _MODELS}


# ---- validation --------------------------------------------------------------------------------
def fold_origins(n: int, horizon: int, folds: int) -> list[int]:
    """Training length of each fold, oldest first. Fold j trains on y[:origin] and is tested on the next `horizon`."""
    return [n - horizon * (folds - j) for j in range(folds)]


def _plan_folds(n: int, horizon: int, folds: int, min_train: int) -> tuple[int, int]:
    hv = int(min(horizon, max(1, n // 6)))
    k = folds
    while k >= 1 and n - hv * k < min_train:
        k -= 1
    return hv, k


def error_metrics(actual: np.ndarray, pred: np.ndarray, scale: float) -> dict:
    """MAE, RMSE, MASE and MAPE (MAPE only if no actual is zero or near zero)."""
    e = actual - pred
    mae = float(np.mean(np.abs(e)))
    out = {"mae": mae, "rmse": float(np.sqrt(np.mean(e ** 2))), "mase": mae / scale if scale > 0 else None, "mape": None}
    floor = MAPE_MIN_SHARE * float(np.mean(np.abs(actual)))
    if len(actual) and float(np.min(np.abs(actual))) > 0 and float(np.min(np.abs(actual))) >= floor:
        out["mape"] = float(np.mean(np.abs(e) / np.abs(actual)) * 100)
    return out


def choose_model(scores: list[ModelScore], margin: float) -> ModelScore:
    """Simplest model whose MAE is within `margin` of the best MAE."""
    ok = [s for s in scores if s.status == "ok" and s.mae is not None]
    if not ok:
        raise ForecastingError("No forecasting model could be validated on this series.")
    best = min(ok, key=lambda s: (s.mae, s.complexity))
    eligible = [s for s in ok if s.mae <= best.mae * (1 + margin) + 1e-12]
    return min(eligible, key=lambda s: (s.complexity, s.mae))


def _future_periods(last: pd.Timestamp, freq: str, h: int) -> pd.DatetimeIndex:
    return pd.date_range(last, periods=h + 1, freq=_STEP[freq])[1:]


# ---- main entry point --------------------------------------------------------------------------
def forecast_series(df: pd.DataFrame, date_col: str, value_col: str | None, freq: str = "D", agg: str = "sum",
                    horizon: int | None = None, config: AnalyticsConfig = DEFAULT_ANALYTICS,
                    model: str | None = None, progress: ProgressFn | None = None) -> ForecastResult:
    """Validate several models chronologically and forecast with the selected one.

    `model` forces one of the candidate models (its validation scores are still reported); None selects automatically.
    Raises InsufficientDataError (user-safe reason) when the data cannot support a forecast.
    """
    if freq not in SEASONAL_PERIOD:
        raise ForecastingError("Period must be day, week or month.")
    if agg not in {"sum", "mean", "median", "count"}:
        raise ForecastingError("Aggregation must be sum, mean, median or count.")
    if date_col not in df.columns or (value_col is not None and agg != "count" and value_col not in df.columns):
        raise ForecastingError("The selected column is not in the dataset.")
    horizon = int(horizon or DEFAULT_HORIZON[freq])
    if not 1 <= horizon <= config.forecast_max_horizon:
        raise ForecastingError(f"Horizon must be between 1 and {config.forecast_max_horizon} periods.")
    table, notes = prepare_series(df, date_col, value_col, freq, agg, config)
    y = table["value"].to_numpy(dtype="float64")
    n = len(y)
    m, w = SEASONAL_PERIOD[freq], MA_WINDOW[freq]

    hv, k = _plan_folds(n, horizon, config.forecast_folds, 10)
    if k < 1:
        raise InsufficientDataError(f"{n} periods are too few to validate a forecast {horizon} periods ahead.")
    origins = fold_origins(n, hv, k)
    seasonal, why = detect_seasonality(y[:origins[0]], m)       # decided without looking at validation data
    if horizon > n / 3:
        notes.append("The horizon is long relative to the history; treat the far end of the forecast with caution.")
    scale = float(np.mean(np.abs(np.diff(y[:origins[0]])))) if origins[0] > 1 else 0.0

    scores: list[ModelScore] = []
    preds: dict[str, list[np.ndarray]] = {}
    for i, (key, label, cx, needs_seasonal, min_train, fn) in enumerate(_MODELS):
        notify(progress, i, len(_MODELS), f"Comparing models, {i + 1} of {len(_MODELS)}")
        if needs_seasonal and not seasonal:
            scores.append(ModelScore(key, label, cx, "skipped", f"seasonality not supported ({why})"))
            continue
        need = min_train(m, w)
        if origins[0] < need:
            scores.append(ModelScore(key, label, cx, "skipped", f"needs {need} periods before the first validation fold"))
            continue
        try:
            fold_preds = [np.asarray(fn(y[:o], hv, m, w), dtype=float) for o in origins]
            if any(len(p) != hv or not np.isfinite(p).all() for p in fold_preds):
                raise ValueError("non-finite forecast")
        except Exception as exc:                                # one model failing must not stop the others
            scores.append(ModelScore(key, label, cx, "failed", f"could not be fitted ({type(exc).__name__})"))
            continue
        actual = np.concatenate([y[o:o + hv] for o in origins])
        pred = np.concatenate(fold_preds)
        met = error_metrics(actual, pred, scale)
        preds[key] = fold_preds
        scores.append(ModelScore(key, label, cx, "ok", "", met["mae"], met["rmse"], met["mase"], met["mape"], len(actual)))

    auto = choose_model(scores, config.forecast_simpler_margin)
    chosen = auto
    if model is not None:
        if model not in _BY_KEY:
            raise ForecastingError("Unknown forecasting model.")
        forced = next(x for x in scores if x.key == model)
        if forced.status != "ok":
            raise ForecastingError(f"{forced.label} is not available for this series: {forced.reason or 'it could not be fitted'}.")
        chosen = forced
        if forced.key != auto.key:
            notes.append(f"{forced.label} was chosen manually (validation MAE {forced.mae:,.2f}); the automatic choice "
                         f"would be {auto.label} (MAE {auto.mae:,.2f}).")
    base_candidates = [s for s in scores if s.status == "ok" and s.key in ("naive", "seasonal_naive")]
    baseline = min(base_candidates, key=lambda s: s.mae)
    improvement = None
    if baseline.mae and baseline.mae > 0:
        improvement = float((baseline.mae - chosen.mae) / baseline.mae * 100)

    # final fit on all history; fall back to the baseline rather than showing nothing
    used = chosen
    try:
        point = np.asarray(_BY_KEY[chosen.key][4](y, horizon, m, w), dtype=float)
        if len(point) != horizon or not np.isfinite(point).all():
            raise ValueError("non-finite forecast")
    except Exception:
        used = baseline
        point = np.asarray(_BY_KEY[baseline.key][4](y, horizon, m, w), dtype=float)
        notes.append(f"{chosen.label} could not be fitted on the full history; the baseline ({baseline.label}) is shown.")

    # empirical interval from the used model's out-of-sample errors
    errs = np.concatenate([y[o:o + hv] - p for o, p in zip(origins, preds[used.key])])
    level = config.forecast_interval
    lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
    if len(errs) >= 10:
        qlo, qhi = float(np.quantile(errs, lo_q)), float(np.quantile(errs, hi_q))
    else:
        s = float(np.std(errs)) if len(errs) > 1 else float(np.mean(np.abs(errs)))
        qlo, qhi = -1.645 * s, 1.645 * s
        notes.append("Few validation errors were available, so the interval is approximate.")
    steps = np.arange(1, horizon + 1)
    widen = np.sqrt(np.maximum(1.0, steps / hv))               # beyond the validated horizon the band is widened
    lower = point + min(qlo, 0.0) * widen
    upper = point + max(qhi, 0.0) * widen
    if agg in ("sum", "count") and float(np.min(y)) >= 0:
        point, lower, upper = np.maximum(point, 0), np.maximum(lower, 0), np.maximum(upper, 0)
    if float(np.ptp(point)) <= 1e-9 * max(1.0, abs(float(np.mean(point)))):
        notes.append(f"The forecast is a flat line. {used.label} predicts a constant level because no trend or repeating "
                     "pattern was reliable enough to beat it in validation. That is a genuine result, not a failure: the band shows "
                     "the expected range. A different period (for example week) or target may show more structure.")
    future = _future_periods(pd.Timestamp(table["period"].iloc[-1]), freq, horizon)
    forecast = pd.DataFrame({"period": future, "forecast": point, "lower": lower, "upper": upper})

    bt = pd.DataFrame({
        "period": np.concatenate([table["period"].to_numpy()[o:o + hv] for o in origins]),
        "actual": np.concatenate([y[o:o + hv] for o in origins]),
        "predicted": np.concatenate(preds[used.key]),
        "fold": np.repeat(np.arange(1, k + 1), hv)})

    limits = [
        "A forecast extends past patterns in the data; it cannot anticipate promotions, stock-outs, price changes or other events not in the history.",
        f"The {level:.0%} interval comes from out-of-sample validation errors over {len(errs)} points and is approximate.",
        f"Validation: {k} rolling-origin fold(s) of {hv} {FREQ_LABELS[freq]}(s); no random splits, no future data used.",
    ]
    if not seasonal:
        limits.append(f"Seasonal models were not used: {why}.")
    if improvement is not None and used.key not in ("naive", "seasonal_naive") and improvement <= 0:
        notes.append("The selected model does not beat the baseline; the forecast is no better than repeating the past.")
    return ForecastResult(date_col, value_col, agg, freq, horizon, table, forecast, bt, tuple(scores), used.key,
                          baseline.key, improvement, level, m, seasonal, hv, k, tuple(notes), tuple(limits))

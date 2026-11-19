"""Default-setting runs of the Phase 6 analyses, used to feed the insight engine without user input.

Each function raises InsufficientDataError (user-safe reason) when the data cannot support the analysis."""
from __future__ import annotations

from analytics.anomaly_detection import eligible_columns, run_anomaly_detection
from analytics.drivers import driver_targets, run_driver_analysis
from analytics.forecasting import forecast_series, suggest_setup
from analytics.kpis import primary_column
from analytics.segmentation import build_entity_features, entity_candidates, run_segmentation
from config.analytics_config import DEFAULT_ANALYTICS, AnalyticsConfig
from core.exceptions import InsufficientDataError
from core.progress import ProgressFn, notify

MAX_ANOMALY_COLUMNS = 8
PAGE_NAMES = {"forecast": "Forecasting", "anomalies": "Anomalies", "segmentation": "Segmentation", "drivers": "Drivers"}


def default_forecast(df, model, config: AnalyticsConfig = DEFAULT_ANALYTICS, progress: ProgressFn | None = None):
    s = suggest_setup(df, model)
    if not s["date_col"] or not s["value_col"]:
        raise InsufficientDataError("Forecasting needs a date column and a quantity or revenue column.")
    return forecast_series(df, s["date_col"], s["value_col"], s["freq"], s["agg"], None, config, progress=progress)


def default_anomalies(df, profile, model, config: AnalyticsConfig = DEFAULT_ANALYTICS, progress: ProgressFn | None = None):
    result = eligible_columns(df, profile, model, config)
    cols = result[0] if isinstance(result, tuple) else result
    if not cols:
        raise InsufficientDataError("No numeric column is suitable for anomaly screening.")
    return run_anomaly_detection(df, list(cols)[:MAX_ANOMALY_COLUMNS], ("mad", "iqr", "zscore"), None, config, progress)


def default_segmentation(df, model, config: AnalyticsConfig = DEFAULT_ANALYTICS, progress: ProgressFn | None = None):
    cands = entity_candidates(df, model, config)
    if not cands:
        raise InsufficientDataError("No supplier, customer, product or warehouse column has enough entities to segment.")
    notify(progress, 0, 1, "Building one row per " + str(cands[0]["column"]))
    table = build_entity_features(df, model, cands[0]["column"], config)
    return run_segmentation(table, None, None, "robust", config, progress)


def default_driver_target(df, profile, model, config: AnalyticsConfig = DEFAULT_ANALYTICS) -> str | None:
    targets = driver_targets(df, profile, model, config)
    for role in ("profit", "revenue", "quantity"):
        c = primary_column(df, model, role)
        if c in targets:
            return c
    return next(iter(targets), None)


def default_drivers(df, profile, model, config: AnalyticsConfig = DEFAULT_ANALYTICS, progress: ProgressFn | None = None):
    target = default_driver_target(df, profile, model, config)
    if target is None:
        raise InsufficientDataError("No column is suitable as a driver-analysis target.")
    return run_driver_analysis(df, target, profile, model, None, config, progress)

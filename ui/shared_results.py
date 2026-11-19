"""Cached default-setting results shared by the Dashboard, Insights and Reports pages: same cache names and parameters,
so a result computed on one page is reused on the others instead of being computed again."""
from __future__ import annotations

from analytics.data_findings import compute_findings
from analytics.kpis import compute_kpis
from config.analytics_config import DEFAULT_ANALYTICS, DEFAULT_INSIGHTS
from insights.default_runs import default_anomalies, default_drivers, default_forecast, default_segmentation
from ui.analysis_runner import safe_cached

KPI_NAME, FINDINGS_NAME = "insight_kpis", "insight_findings"


def kpi_source(df, model, config=DEFAULT_ANALYTICS, failure="The metrics could not be calculated."):
    return lambda: safe_cached(KPI_NAME, config, lambda: compute_kpis(df, model, config), failure)


def findings_source(df, model):
    return lambda: safe_cached(FINDINGS_NAME, (DEFAULT_ANALYTICS, DEFAULT_INSIGHTS),
                               lambda: compute_findings(df, model, DEFAULT_INSIGHTS), "Data patterns could not be calculated.")


def advanced_sources(df, profile, model) -> dict:
    """{analysis: fn(progress) -> (status, payload)}, cached as insight_<analysis> with DEFAULT_ANALYTICS."""
    cfg = DEFAULT_ANALYTICS
    runs = {"forecast": lambda p: default_forecast(df, model, cfg, p), "anomalies": lambda p: default_anomalies(df, profile, model, cfg, p),
            "segmentation": lambda p: default_segmentation(df, model, cfg, p), "drivers": lambda p: default_drivers(df, profile, model, cfg, p)}
    return {name: (lambda progress=None, n=name, f=fn: safe_cached(f"insight_{n}", cfg, lambda: f(progress),
                                                                     f"The {n} analysis could not be completed."))
            for name, fn in runs.items()}

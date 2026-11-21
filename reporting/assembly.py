"""Builds a report from computed results in named steps, with progress. No Streamlit.

The UI supplies `Sources`: callables that return (status, payload) like the cached analysis wrappers, so anything
already computed on the Insights page or an analysis page is reused instead of recomputed. Steps, in this order:
Metrics, Insights (data patterns), Forecast, Anomalies, Segments, Drivers, PDF, then HTML only when asked for.
A step that fails or cannot run is listed as skipped with the reason and the run carries on; one failing format
never blocks the other. Charts are drawn once, inside build_report, and both renderers reuse those images.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from config.analytics_config import DEFAULT_INSIGHTS
from core.logging_config import get_logger
from core.progress import ProgressFn, StepRunner, notify
from insights.default_runs import PAGE_NAMES
from insights.insight_engine import generate_insights
from insights.insight_rules import InsightInputs
from reporting.html_report import render_html
from reporting.pdf_report import render_pdf
from reporting.report_builder import ReportData, build_report

log = get_logger("reporting.assembly")

Source = Callable[[], "tuple[str, Any]"]
AdvancedSource = Callable[["ProgressFn | None"], "tuple[str, Any]"]
SECTION_KEY = {"forecast": "forecast", "anomalies": "anomalies", "segmentation": "segments", "drivers": "drivers"}
STEP_LABEL = {"forecast": "Forecast", "anomalies": "Anomalies", "segmentation": "Segments", "drivers": "Drivers"}
SWITCHED_OFF = "Advanced analyses were switched off for this report."
FORMAT_LABEL = {"pdf": "PDF", "html": "HTML"}


@dataclass(frozen=True)
class ReportContext:
    filename: str
    is_sample: bool
    fingerprint: str
    profile: Any
    quality: Any
    analysis_rows: int
    semantic_model: Any = None
    cleaning: Any = None
    title: str = ""                      # empty: a title is chosen from the data's business area
    prepared_by: str = ""
    df: Any = None                       # the analysed rows: enables trends, period comparison and breakdowns
    organisation: str = ""
    logo: bytes | None = None
    accent: str = "#1F4E89"
    classification: str = "Internal"


@dataclass(frozen=True)
class Sources:
    kpis: Source
    findings: Source
    advanced: Mapping[str, AdvancedSource] = field(default_factory=dict)      # forecast / anomalies / segmentation / drivers


@dataclass
class Assembled:
    data: ReportData | None
    files: dict[str, bytes] = field(default_factory=dict)          # "pdf" and, if asked for, "html"
    errors: dict[str, str] = field(default_factory=dict)           # format -> user-safe reason
    skipped: list[tuple[str, str]] = field(default_factory=list)   # (step, reason)
    used_user_settings: tuple[str, ...] = ()


def step_names(advanced: bool, with_html: bool) -> list[str]:
    return ["Metrics", "Insights", *STEP_LABEL.values(), "PDF", *(["HTML"] if with_html else [])]


def assemble_report(ctx: ReportContext, sources: Sources, *, user_results: Mapping[str, Any] | None = None, advanced: bool = True,
                    with_html: bool = False, progress: ProgressFn | None = None,
                    renderers: Mapping[str, Callable[[ReportData], bytes]] | None = None) -> Assembled:
    render = {"pdf": render_pdf, "html": render_html, **(renderers or {})}
    mine = dict(user_results or {})
    total = len(step_names(advanced, with_html))
    run = StepRunner(total, progress)
    unavailable: dict[str, str] = {}           # report section key -> reason
    reasons: dict[str, str] = {}               # analysis name -> reason, for the insight engine

    ok, kpis, why = run.run_status("Metrics", sources.kpis)
    if not ok:
        unavailable["kpis"], kpis = why, None
    ok, findings, _why = run.run_status("Insights", sources.findings)
    findings = findings if ok else None

    adv: dict[str, Any] = {}
    used: list[str] = []
    for name, label in STEP_LABEL.items():
        key = SECTION_KEY[name]
        if not advanced:
            run.skip(label, SWITCHED_OFF)
            unavailable[key] = reasons[name] = SWITCHED_OFF
        elif name in mine:                                           # the user ran it on its own page: their settings win
            run.reuse(label, f"your result from the {PAGE_NAMES[name]} page")
            adv[name] = mine[name]
            used.append(PAGE_NAMES[name])
        elif name not in sources.advanced:
            run.skip(label, "no way to run this analysis was provided")
            unavailable[key] = reasons[name] = "This analysis was not run for this report."
        else:
            inner = _inner(progress, run, label)
            ok, payload, why = run.run_status(label, lambda n=name, f=inner: sources.advanced[n](f))
            if ok:
                adv[name] = payload
            else:
                unavailable[key] = reasons[name] = why

    out = Assembled(None, used_user_settings=tuple(used))
    state: dict[str, Any] = {}

    def prepare() -> ReportData:                                     # built once; PDF and HTML both reuse it (and its charts)
        if "data" in state:
            return state["data"]
        insight_report = None
        try:
            insight_report = generate_insights(InsightInputs(kpis=kpis, profile=ctx.profile, findings=findings,
                                                             unavailable=tuple(reasons.items()), **adv),
                                               DEFAULT_INSIGHTS)
        except Exception:
            log.exception("Insight generation failed for report")
            unavailable["insights"] = "The insights could not be generated for this dataset."
        state["data"] = build_report(
            filename=ctx.filename, is_sample=ctx.is_sample, fingerprint=ctx.fingerprint, profile=ctx.profile, quality=ctx.quality,
            analysis_rows=ctx.analysis_rows, semantic_model=ctx.semantic_model, unavailable=unavailable, title=ctx.title,
            prepared_by=ctx.prepared_by, user_settings=tuple(used), kpis=kpis, insight_report=insight_report, forecast=adv.get("forecast"),
            anomalies=adv.get("anomalies"), segmentation=adv.get("segmentation"), drivers=adv.get("drivers"), cleaning=ctx.cleaning,
            df=getattr(ctx, "df", None), organisation=getattr(ctx, "organisation", ""), logo=getattr(ctx, "logo", None),
            accent=getattr(ctx, "accent", "#1F4E89"), classification=getattr(ctx, "classification", "Internal"))
        return state["data"]

    for fmt in ("pdf", *(["html"] if with_html else [])):
        label = FORMAT_LABEL[fmt]
        ok, payload, why = run.run(label, lambda f=fmt: render[f](prepare()), f"The {label} file could not be created.")
        if ok:
            out.files[fmt] = payload
        else:
            out.errors[fmt] = why
    out.data = state.get("data")
    out.skipped = list(run.skipped)
    run.done()
    return out


def _inner(progress: ProgressFn | None, run: StepRunner, label: str) -> ProgressFn | None:
    """Progress from inside one analysis, shown under that step's name without moving the step count."""
    if progress is None:
        return None
    return lambda _i, _n, message: notify(progress, run.finished, run.total, f"{label}: {message}")

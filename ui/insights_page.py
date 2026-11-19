"""Insights page (Phase 7): rule-based findings with evidence, thresholds, limitations and a next step."""
from __future__ import annotations

import re

import pandas as pd
import streamlit as st

from config.analytics_config import DEFAULT_INSIGHTS
from core.estimates import describe_estimate
from core.logging_config import get_logger
from core.progress import StepRunner
from insights.default_runs import PAGE_NAMES
from insights.evidence import GROUPS, SEVERITIES, Insight
from insights.insight_engine import generate_insights
from insights.insight_rules import InsightInputs
from ui import state
from ui.analytics_state import base_key
from ui.progress_bar import ProgressBar
from ui.shared_results import advanced_sources, findings_source, kpi_source
from ui.user_runs import user_runs
from ui.layout import empty_state, eyebrow, no_dataset_notice, page_title

log = get_logger("ui.insights")
_ADV = "ins_advanced_for"
STEP_LABEL = {"forecast": "Forecast", "anomalies": "Anomalies", "segmentation": "Segments", "drivers": "Drivers"}
_SEV_LABEL = {"high": ":red[High]", "medium": ":orange[Medium]", "low": ":blue[Low]", "info": ":gray[Info]"}


def _md(text: str) -> str:
    """Escape markdown so column names and values from the uploaded file are shown as plain text."""
    return re.sub(r"([\\`*_{}\[\]()#+!|<>~$])", r"\\\1", str(text))


def _advanced(df, profile, model) -> tuple[dict, list[tuple[str, str]]]:
    """Run the four Phase 6 analyses with default settings, one named step each. Each is cached and fails on its own."""
    sources = advanced_sources(df, profile, model)
    mine = user_runs()                                   # settings the user chose on the analysis pages win over defaults
    got, why = {}, []
    with ProgressBar("Running the advanced analyses", numbered=True) as bar:
        run = StepRunner(len(PAGE_NAMES), bar)
        for name in PAGE_NAMES:
            label = STEP_LABEL[name]
            if name in mine:
                run.reuse(label, f"your result from the {PAGE_NAMES[name]} page")
                got[name] = mine[name]
                continue
            inner = lambda i, n, m, lb=label: bar(run.finished, run.total, f"{lb}: {m}")        # noqa: E731
            ok, payload, reason = run.run_status(label, lambda n=name, f=inner: sources[n](f))
            if ok:
                got[name] = payload
            else:
                why.append((name, reason))
        run.done()
    return got, why


def render_insights_page() -> None:
    result = state.get_result()
    if result is None:
        page_title("Insights", "Findings computed from your data, each with its evidence.")
        no_dataset_notice()
        return
    ds = result.dataset
    page_title("Insights", ds.filename + (" · synthetic sample data" if ds.is_sample else ""))
    try:
        df, profile, model = state.get_analysis_df(), state.get_analysis_profile(), state.get_semantic_model()
    except Exception:
        log.exception("Insight inputs unavailable")
        st.error("The dataset could not be prepared for analysis. Check the Data quality and Column roles pages.")
        return
    if df is None or profile is None or model is None or len(df) == 0:
        empty_state("Nothing to analyse", "The active dataset has no rows.")
        return

    status, kpis = kpi_source(df, model, failure="The KPIs could not be calculated.")()
    if status != "ok":
        st.warning(str(kpis))
        kpis = None

    fstatus, findings = findings_source(df, model)()
    findings = findings if fstatus == "ok" else None

    include = st.session_state.get(_ADV) == base_key()
    if not include:
        st.caption("Insights below use KPIs and data quality. Add forecast, anomaly, segment and driver findings: analyses you already ran on "
                   "their own page are reused with your settings; the others run with default settings (a few seconds on large files).")
        hint = describe_estimate("insights", len(df))
        if hint:
            st.caption(hint)
        if st.button("Include advanced analyses", key="ins_adv_btn"):
            st.session_state[_ADV] = base_key()
            st.rerun()
    adv, unavailable = ({}, [])
    if include:
        adv, unavailable = _advanced(df, profile, model)

    report = generate_insights(InsightInputs(kpis=kpis, profile=profile, findings=findings, unavailable=tuple(unavailable), **adv),
                               DEFAULT_INSIGHTS)
    _show(report, include)


def _show(report, advanced: bool) -> None:
    counts = {s: sum(1 for i in report.insights if i.severity == s) for s in SEVERITIES}
    c = st.columns(4)
    for col, s in zip(c, SEVERITIES):
        col.metric(s.title(), counts[s])
    f1, f2 = st.columns(2)
    sev = f1.multiselect("Severity", list(SEVERITIES), default=list(SEVERITIES), key="ins_sev")
    grp = f2.multiselect("Group", list(GROUPS), default=list(GROUPS), key="ins_grp")
    shown = [i for i in report.insights if i.severity in sev and i.group in grp]
    for note in report.notes:
        st.info(note)
    if not shown and report.insights:
        st.caption("No insight matches the selected filters.")
    for i in shown:
        _card(i)
    with st.expander("Method, thresholds and rules that did not run"):
        st.markdown("Insights come from fixed rules applied to computed results; nothing is generated by a language model. "
                    "Priority = 10 x severity (high 3, medium 2, low 1, info 0) + confidence (high 3, medium 2, low 1). "
                    "Confidence for KPI findings reflects how many rows were analysed. Thresholds are this app's adjustable "
                    "reference levels, not industry standards, and drivers show association, not cause.")
        if report.skipped:
            st.dataframe(pd.DataFrame(report.skipped, columns=["Rule", "Why it did not run"]), hide_index=True)
        if not advanced:
            st.caption("Forecast, anomaly, segment and driver rules need the advanced analyses (button above).")


def _card(i: Insight) -> None:
    with st.container(border=True):
        eyebrow(f"{i.group} · confidence {i.confidence}")
        st.markdown(f"{_SEV_LABEL[i.severity]} &nbsp; **{_md(i.title)}**")
        st.markdown(_md(i.description))
        if i.next_step:
            st.markdown(f"**Next step:** {_md(i.next_step)}")
        with st.expander("Evidence, threshold and limitations"):
            ev = pd.DataFrame([{"Metric": e.metric, "Value": e.value, "Calculation": e.detail,
                                "Columns": ", ".join(e.columns), "Period": e.period} for e in i.evidence])
            st.dataframe(ev, hide_index=True)
            if i.thresholds:
                st.caption(f"Threshold: {i.thresholds}")
            for t in i.limitations:
                st.markdown(f"- {_md(t)}")

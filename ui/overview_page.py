"""Overview: where the active dataset stands in the workflow. Shows only values computed from the data."""
from __future__ import annotations

from html import escape

import streamlit as st

from core.logging_config import get_logger
from ingestion.pipeline import ingest_sample
from semantic.column_lexicon import DOMAIN_LABELS, ROLE_LABELS
from ui import state
from ui.layout import badge, chips, empty_state, eyebrow, fmt_int, page_title, steps

log = get_logger("ui.overview")
_QUALITY = {"good": ("Good", "accent"), "review": ("Needs review", "warn"), "poor": ("Poor", "danger")}
_SEV_TONE = {"high": "danger", "medium": "warn", "low": "neutral"}


def _empty() -> None:
    page_title("Overview", "Upload business data to profile it, clean it and understand what its columns mean.")
    empty_state("No dataset loaded", "Start by uploading a CSV or Excel file, or try the synthetic sample data.")
    st.write("")
    left, right, _ = st.columns([1, 1, 3])
    if left.button("Load sample data", type="primary", help="Synthetic sales data for trying the app"):
        state.set_result(ingest_sample())
        st.rerun()
    right.caption("Use Upload in the sidebar for your own file.")
    st.write("")
    steps([("1", "Upload", "CSV or Excel, validated and previewed", "soon"),
           ("2", "Profile and clean", "Types, missing values, duplicates", "soon"),
           ("3", "Understand", "Column roles and business domain", "soon"),
           ("4", "Analyse", "KPIs, forecasts, insights (coming)", "soon")])


def render_overview_page() -> None:
    result = state.get_result()
    if result is None:
        _empty()
        return
    ds, prof, q = result.dataset, result.profile, result.quality
    page_title("Overview", f"{ds.filename}" + (" · synthetic sample data" if ds.is_sample else ""))
    if ds.is_sample:
        st.warning("This is synthetic sample data generated for demonstration. It is not real.")
    label, _tone = _QUALITY.get(q.status, (q.status.title(), "neutral"))

    model = None
    try:
        model = state.get_semantic_model()
    except Exception:
        log.exception("Semantic model unavailable on overview")

    c = st.columns(5)
    c[0].metric("Rows", fmt_int(prof.rows))
    c[1].metric("Columns", prof.columns)
    c[2].metric("Completeness", f"{q.completeness_pct:.1f}%")
    c[3].metric("Data quality", label)
    c[4].metric("Domain", DOMAIN_LABELS[model.domain.domain] if model else "n/a")

    cleaning, n_props = state.get_cleaning(), len(state.get_proposals())
    n_review = len(model.needs_review()) if model else 0
    n_over = len(model.overridden()) if model else 0
    st.write("")
    steps([
        ("1", "Upload", f"{fmt_int(prof.rows)} rows loaded", "done"),
        ("2", "Profile", f"{len(q.issues)} quality note(s)", "done"),
        ("3", "Clean", (f"{cleaning.result.applied_count} step(s) applied" if cleaning
                        else f"{n_props} suggestion(s) to review"), "done" if cleaning else "todo"),
        ("4", "Understand", ("Confirmed" if model and not n_review else f"{n_review} column(s) to review")
         if model else "Unavailable", "done" if model and not n_review else "todo"),
        ("5", "Analyse", "KPIs, forecasts and insights come in later phases", "soon"),
    ])

    left, right = st.columns(2, gap="large")
    with left:
        eyebrow("Data quality")
        issues = [i for i in q.issues if i.severity in ("high", "medium")][:5]
        if not issues:
            st.caption("No high or medium priority issues found.")
        for i in issues:
            col = f" · {i.column}" if i.column else ""
            st.markdown(f"{badge(i.severity.title(), _SEV_TONE.get(i.severity, 'neutral'))}"
                        f'<span style="font-size:.84rem;margin-left:.5rem">{_esc(i.message)}{_esc(col)}</span>',
                        unsafe_allow_html=True)
        st.caption("Details on the Data quality page.")
    with right:
        eyebrow("What the data appears to contain")
        if model is None:
            st.caption("Column analysis unavailable for this dataset.")
        else:
            for role in ("datetime", "revenue", "cost", "profit", "quantity", "customer", "product"):
                cols = model.columns_with_role(role)
                if cols:
                    chips([(ROLE_LABELS[role], ", ".join(cols[:3]) + (f" +{len(cols) - 3}" if len(cols) > 3 else ""))])
            st.caption(f"{n_over} override(s) by you · suggestions are heuristics, check the Semantics page.")


def _esc(text: str) -> str:
    return escape(str(text))

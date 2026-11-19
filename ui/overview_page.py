"""Overview: where the active dataset stands in the workflow. Shows only values computed from the data."""
from __future__ import annotations

from html import escape

import streamlit as st

from core.logging_config import get_logger
from ingestion.pipeline import ingest_sample
from semantic.column_lexicon import DOMAIN_LABELS, ROLE_LABELS
from ui import nav, state
from ui.home_cards import render_finding_cards
from ui.layout import badge, chips, empty_state, eyebrow, fmt_int, page_title, steps

log = get_logger("ui.overview")
_QUALITY = {"good": ("Good", "accent"), "review": ("Needs review", "warn"), "poor": ("Poor", "danger")}
_SEV_TONE = {"high": "danger", "medium": "warn", "low": "neutral"}


def _empty() -> None:
    page_title("Home", "Upload business data, check it, and see what it says. Everything is processed on this machine.")
    left, right = st.columns(2, gap="large")
    with left, st.container(border=True):
        st.markdown('<div class="empty-t">Try the sample data</div>'
                    '<div class="empty-d">A synthetic sales dataset, so you can see every feature in under a minute.</div>',
                    unsafe_allow_html=True)
        st.write("")
        if st.button("Load sample data", type="primary", key="home_sample", help="Synthetic sales data for trying the app"):
            state.set_result(ingest_sample())
            st.rerun()
    with right, st.container(border=True):
        st.markdown('<div class="empty-t">Use your own data</div>'
                    '<div class="empty-d">A CSV or Excel file. It is checked on upload, and your original file is never changed.</div>',
                    unsafe_allow_html=True)
        st.write("")
        if st.button("Upload a file", key="home_upload"):
            nav.go("upload")
    st.write("")
    eyebrow("How it works")
    steps([("1", "Add your data", "CSV or Excel, validated and previewed", "soon"),
           ("2", "Check and clean", "Types, missing values, duplicates", "soon"),
           ("3", "Confirm column roles", "What each column means for your business", "soon"),
           ("4", "Explore", "Dashboard, forecasts, anomalies, insights", "soon")])


def _next_step(has_cleaning: bool, n_props: int, n_review: int) -> tuple[str, str, str, str]:
    """(heading, detail, page key, button label), chosen from the real state of the dataset."""
    if not has_cleaning and n_props:
        return (f"Review {n_props} cleaning suggestion(s)", "The checks found things you can fix. Review them before analysing.",
                "quality", "Review suggestions")
    if n_review:
        return (f"Confirm {n_review} column role(s)", "Roles decide which KPIs and charts the app can build, so a quick check pays off.",
                "roles", "Review roles")
    return ("Open the dashboard", "Your data is ready. KPIs and charts are computed from your column roles.",
            "dashboard", "Open dashboard")


def render_overview_page() -> None:
    result = state.get_result()
    if result is None:
        _empty()
        return
    ds, prof, q = result.dataset, result.profile, result.quality
    page_title("Home", "Where your data stands, and what to do next.")
    if ds.is_sample:
        st.warning("This is synthetic sample data generated for demonstration. It is not real.")
    label, _tone = _QUALITY.get(q.status, (q.status.title(), "neutral"))

    model = None
    try:
        model = state.get_semantic_model()
    except Exception:
        log.exception("Semantic model unavailable on overview")

    cleaning, n_props = state.get_cleaning(), len(state.get_proposals())
    n_review = len(model.needs_review()) if model else 0
    heading, detail, target, button = _next_step(cleaning is not None, n_props, n_review)
    with st.container(border=True):
        text, action = st.columns([5, 1.6], vertical_alignment="center")
        text.markdown(f'<div class="next-k">Next step</div><div class="next-t">{_esc(heading)}</div>'
                      f'<div class="next-d">{_esc(detail)}</div>', unsafe_allow_html=True)
        if action.button(button, type="primary", key="home_next", width="stretch"):
            nav.go(target)
    st.write("")

    c = st.columns(5)
    c[0].metric("Rows", fmt_int(prof.rows))
    c[1].metric("Columns", prof.columns)
    c[2].metric("Completeness", f"{q.completeness_pct:.1f}%")
    c[3].metric("Data quality", label)
    c[4].metric("Domain", DOMAIN_LABELS[model.domain.domain] if model else "n/a")

    n_over = len(model.overridden()) if model else 0
    st.write("")
    cards_area = st.container()
    st.write("")
    steps([
        ("1", "Upload", f"{fmt_int(prof.rows)} rows loaded", "done"),
        ("2", "Profile", f"{len(q.issues)} quality note(s)", "done"),
        ("3", "Clean", (f"{cleaning.result.applied_count} step(s) applied" if cleaning
                        else f"{n_props} suggestion(s) to review"), "done" if cleaning else "todo"),
        ("4", "Understand", ("Confirmed" if model and not n_review else f"{n_review} column(s) to review")
         if model else "Unavailable", "done" if model and not n_review else "todo"),
        ("5", "Analyse", "Open the Dashboard for KPIs and charts" if model else "Unavailable", "done" if model else "soon"),
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
            st.caption(f"{n_over} override(s) by you · suggestions are heuristics, check the Column roles page.")

    _fill_cards(cards_area)


def _fill_cards(area) -> None:
    """Drawn after the rest of the page, so the first load shows everything else while the cards compute."""
    try:
        df, profile, model = state.get_analysis_df(), state.get_analysis_profile(), state.get_semantic_model()
    except Exception:
        log.exception("Finding cards unavailable")
        df = profile = model = None
    with area:
        if df is None or profile is None or model is None or len(df) == 0:
            empty_state("Findings are not available yet", "The data could not be prepared for analysis. "
                        "Check the Data quality and Column roles pages.")
        else:
            render_finding_cards(df, profile, model)


def _esc(text: str) -> str:
    return escape(str(text))

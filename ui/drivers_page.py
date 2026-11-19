"""Driver analysis page (Phase 6D): which columns help explain a chosen target. Association, not causation."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from analytics.drivers import NOTE, available_date_parts, candidate_features, driver_targets, run_driver_analysis
from config.analytics_config import DEFAULT_ANALYTICS
from core.logging_config import get_logger
from ui import state
from ui.progress_bar import ProgressBar
from ui.analysis_runner import safe_cached
from ui.analytics_state import base_key, show_figure
from ui.filters import filtered, no_rows_notice, render_filter_bar
from ui.layout import empty_state, no_dataset_notice, page_title
from visualization import charts

log = get_logger("ui.drivers")


def _explain(df, profile, model, p):
    with ProgressBar("Driver analysis") as bar:
        return run_driver_analysis(df, p[1], profile, model, list(p[2]), progress=bar)


def render_drivers_page() -> None:
    result = state.get_result()
    if result is None:
        page_title("Driver analysis", "Which columns help explain a result.")
        no_dataset_notice()
        return
    ds = result.dataset
    page_title("Driver analysis", ds.filename + (" · synthetic sample data" if ds.is_sample else ""))
    if ds.is_sample:
        st.warning("This is synthetic sample data generated for demonstration. It is not real.")
    try:
        df, profile, model = state.get_analysis_df(), state.get_analysis_profile(), state.get_semantic_model()
    except Exception:
        log.exception("Driver inputs unavailable")
        st.error("The dataset could not be prepared for analysis. Check the Data quality and Column roles pages.")
        return
    if df is None or profile is None or model is None or len(df) == 0:
        empty_state("Nothing to analyse", "The active dataset has no rows.")
        return
    render_filter_bar(df, model)
    df = filtered(df)
    if len(df) == 0:
        no_rows_notice()
        return
    targets = driver_targets(df, profile, model)
    if not targets:
        empty_state("No suitable target", "A target needs a numeric measure with enough distinct values, or a Yes/No column "
                                          "where both answers are common.")
        return
    st.caption(NOTE)
    t1, t2 = st.columns(2)
    target = t1.selectbox("Target to explain", list(targets), format_func=lambda c: f"{c} ({'yes/no' if targets[c] == 'classification' else 'number'})",
                          key="dr_target")
    num, cat, _ = candidate_features(df, profile, model, target)
    allf = num + cat + available_date_parts(df, model, target)
    feats = t2.multiselect("Columns to consider", allf, default=allf, key="dr_feats")
    st.caption("Columns that look derived from the target (copies, parts of it, or another way of recording it) are left out "
               "automatically and listed with the reason.")
    if st.button("Run driver analysis", type="primary", key="dr_run"):
        st.session_state["dr_params"] = (base_key(), target, tuple(feats))
    p = st.session_state.get("dr_params")
    if not p or p[0] != base_key():
        return
    status, res = safe_cached("driver_result", p[1:], lambda: _explain(df, profile, model, p),
                              "Driver analysis could not be completed for this selection.")
    if status != "ok":
        empty_state("Analysis not possible", res) if status == "insufficient" else st.warning(res)
        return
    st.markdown(f"**Model check on held-out rows** · {res.n_train:,} rows to learn from, {res.n_test:,} to test. Split: {res.split}.")
    m = res.metrics.copy()
    m[["model", "baseline"]] = m[["model", "baseline"]].round(3)
    st.dataframe(m.rename(columns={"metric": "Measure", "model": "Model", "baseline": "Trivial guess"}), hide_index=True)
    if res.reliable:
        st.success("The model predicts held-out rows clearly better than a trivial guess, so the ranking below is informative.")
    for w in res.warnings:
        st.warning(w)
    top = res.importance.head(15)
    fig = charts.bar_chart(top, "feature", "importance", "Permutation importance (drop in score when the column is shuffled)", "Importance")
    if fig is not None:
        show_figure(fig)
    out = res.importance.assign(share=lambda d: (100 * d["share"]).round(1).astype(str) + "%").round(4)
    st.dataframe(out.rename(columns={"feature": "Column", "kind": "Type", "importance": "Importance", "std": "Spread",
                                     "share": "Share", "direction": "Direction (association only)"}), hide_index=True)
    if not res.correlations.empty:
        with st.expander("Rank correlation with the target (numeric columns)"):
            st.dataframe(res.correlations.round(3), hide_index=True)
    for feature, g in res.groups.items():
        with st.expander(f"{feature}: average target by group"):
            st.dataframe(g.round(3), hide_index=True)
    if res.excluded:
        with st.expander(f"{len(res.excluded)} column(s) left out and why"):
            st.dataframe(pd.DataFrame({"Column": list(res.excluded), "Reason": list(res.excluded.values())}), hide_index=True)
    st.caption(res.note)

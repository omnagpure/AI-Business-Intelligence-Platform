"""Anomaly page (Phase 6B): unusual values and unusual periods. Nothing is removed or changed."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from analytics.anomaly_detection import (METHOD_LABELS, METHODS, NOTE, attach_context, context_columns, eligible_columns,
                                         feature_comparison, run_anomaly_detection, series_anomalies)
from analytics.descriptive import FREQ_LABELS
from analytics.forecasting import suggest_setup
from config.analytics_config import DEFAULT_ANALYTICS
from core.exceptions import BIError, InsufficientDataError
from core.logging_config import get_logger
from ui import state
from ui.progress_bar import ProgressBar
from ui.analytics_state import base_key, cached, show_figure
from ui.filters import filtered, no_rows_notice, render_filter_bar
from ui.layout import empty_state, no_dataset_notice, page_title
from visualization import charts

log = get_logger("ui.anomaly")
_SENS = {1: "1 - strict (few flags)", 2: "2", 3: "3 - balanced", 4: "4", 5: "5 - sensitive (many flags)"}
_FREQ = {"Day": "D", "Week": "W", "Month": "M"}
_AGG = {"Sum": "sum", "Average": "mean", "Row count": "count"}
_ROW_COUNT = "(row count)"


def _screen(df, p):
    with ProgressBar("Screening for unusual values") as bar:
        return run_anomaly_detection(df, list(p[1]), tuple(p[2]), p[3], progress=bar)


def _safe(name, params, fn):
    """Cached; failures become ('insufficient'|'error', message) so one module never breaks the page."""
    def compute():
        try:
            return "ok", fn()
        except InsufficientDataError as exc:
            return "insufficient", exc.user_message
        except BIError as exc:
            return "error", exc.user_message
        except Exception:
            log.exception("Anomaly detection failed")
            return "error", "Anomaly detection could not be completed for this selection."
    return cached(name, params, compute)


def render_anomaly_page() -> None:
    result = state.get_result()
    if result is None:
        page_title("Anomaly detection", "Unusual values and periods computed from your data.")
        no_dataset_notice()
        return
    ds = result.dataset
    page_title("Anomaly detection", ds.filename + (" · synthetic sample data" if ds.is_sample else ""))
    if ds.is_sample:
        st.warning("This is synthetic sample data generated for demonstration. It is not real.")
    try:
        df, profile, model = state.get_analysis_df(), state.get_analysis_profile(), state.get_semantic_model()
    except Exception:
        log.exception("Anomaly inputs unavailable")
        st.error("The dataset could not be prepared for analysis. Check the Data quality and Column roles pages.")
        return
    if df is None or profile is None or model is None or len(df) == 0:
        empty_state("Nothing to examine", "The active dataset has no rows.")
        return
    render_filter_bar(df, model)
    df = filtered(df)
    if len(df) == 0:
        no_rows_notice()
        return
    st.caption(NOTE + " Flagged records are never deleted or changed.")
    sens = st.select_slider("Sensitivity", options=list(_SENS), value=DEFAULT_ANALYTICS.anomaly_sensitivity,
                            format_func=_SENS.get, key="an_sens")
    t1, t2 = st.tabs(["Unusual values", "Unusual periods"])
    with t1:
        _values_tab(df, profile, model, sens)
    with t2:
        _periods_tab(df, model, sens)


def _values_tab(df, profile, model, sens) -> None:
    cols, skipped = eligible_columns(df, profile, model)
    if not cols:
        empty_state("No suitable numeric columns", "Anomaly screening needs measure columns with enough distinct values.")
        return
    c1, c2 = st.columns(2)
    chosen = c1.multiselect("Columns to screen", cols, default=cols[:6], key="an_cols")
    methods = c2.multiselect("Methods", list(METHODS), default=list(METHODS), format_func=METHOD_LABELS.get, key="an_methods")
    st.caption("Isolation Forest looks at the selected columns together (numeric only, rows with missing values are left out).")
    if st.button("Run detection", type="primary", key="an_run"):
        st.session_state["an_params"] = (base_key(), tuple(chosen), tuple(methods), sens)
    p = st.session_state.get("an_params")
    if not p or p[0] != base_key():
        if skipped:
            with st.expander("Numeric columns not screened"):
                st.dataframe(pd.DataFrame({"Column": list(skipped), "Reason": list(skipped.values())}), hide_index=True)
        return
    status, rep = _safe("anomaly_values", p[1:], lambda: _screen(df, p))
    if status != "ok":
        empty_state("Detection not possible", rep) if status == "insufficient" else st.warning(rep)
        return
    st.markdown(f"**{len(rep.findings):,} flagged value(s) shown** across {len(rep.summary)} check(s).")
    for w in rep.warnings:
        st.warning(w)
    s = rep.summary.rename(columns={"method": "Method", "feature": "Feature", "examined": "Examined", "flagged": "Flagged",
                                    "share": "Share", "threshold": "Threshold", "status": "Status", "note": "Note"}).copy()
    s["Method"] = s["Method"].map(METHOD_LABELS)
    s["Share"] = (s["Share"] * 100).round(1).astype(str) + "%"
    st.dataframe(s, hide_index=True)
    if rep.findings.empty:
        st.info("Nothing was flagged at this sensitivity.")
        return
    f1, f2 = st.columns(2)
    m_pick = f1.selectbox("Show method", ["All"] + sorted(rep.findings["method"].unique()), key="an_f_m",
                          format_func=lambda m: m if m == "All" else METHOD_LABELS.get(m, m))
    shown = rep.findings if m_pick == "All" else rep.findings[rep.findings["method"] == m_pick]
    d_pick = f2.selectbox("Direction", ["All"] + sorted(shown["direction"].unique()), key="an_f_d")
    if d_pick != "All":
        shown = shown[shown["direction"] == d_pick]
    ctx = context_columns(df, model)
    table = attach_context(shown.head(DEFAULT_ANALYTICS.preview_rows), df, ctx)
    table = table.assign(method=table["method"].map(METHOD_LABELS))
    st.dataframe(table, hide_index=True)
    st.caption(f"Showing up to {DEFAULT_ANALYTICS.preview_rows} rows, strongest first.")
    rows = list(dict.fromkeys(shown["row"].head(50).tolist()))
    if rows:
        pick = st.selectbox("Compare one flagged row with typical values (row label)", rows, key="an_row")
        fc = feature_comparison(df, pick, list(p[1]))
        if len(fc):
            st.dataframe(fc, hide_index=True)


def _periods_tab(df, model, sens) -> None:
    dates = [c for c in df.columns if c in model.columns and model.role_of(c) == "datetime"]
    if not dates:
        empty_state("No date column", "This check needs a column with the Date / time role. Set one on the Column roles page.")
        return
    from analytics.descriptive import analysable_numeric_columns
    setup = suggest_setup(df, model)
    targets = [_ROW_COUNT] + analysable_numeric_columns(df, state.get_analysis_profile(), model)
    c1, c2, c3, c4 = st.columns(4)
    date_col = c1.selectbox("Date column", dates, index=dates.index(setup["date_col"]) if setup["date_col"] in dates else 0,
                            key="ap_date")
    target = c2.selectbox("Measure", targets, key="ap_target",
                          index=targets.index(setup["value_col"]) if setup["value_col"] in targets else 0)
    agg_label = c3.selectbox("Combine rows by", list(_AGG), key="ap_agg", disabled=target == _ROW_COUNT)
    inv = {v: k for k, v in _FREQ.items()}
    freq = _FREQ[c4.selectbox("Period", list(_FREQ), index=list(_FREQ).index(inv[setup["freq"]]), key="ap_freq")]
    value_col = None if target == _ROW_COUNT else target
    agg = "count" if target == _ROW_COUNT else _AGG[agg_label]
    if st.button("Run period check", type="primary", key="ap_run"):
        st.session_state["ap_params"] = (base_key(), date_col, value_col, freq, agg, sens)
    p = st.session_state.get("ap_params")
    if not p or p[0] != base_key():
        return
    status, res = _safe("anomaly_periods", p[1:], lambda: series_anomalies(df, *p[1:]))
    if status != "ok":
        empty_state("Check not possible", res) if status == "insufficient" else st.warning(res)
        return
    fig = charts.anomaly_series_chart(res.series, f"{FREQ_LABELS[res.freq].title()} values and flagged periods",
                                      "Rows" if p[4] == "count" else str(p[2]))
    if fig is not None:
        show_figure(fig)
    for w in res.warnings:
        st.info(w)
    if res.findings.empty:
        st.info("No period stands out at this sensitivity.")
        return
    st.markdown(f"**{len(res.findings)} unusual {FREQ_LABELS[res.freq]}(s)**")
    out = res.findings[["row", "direction", "value", "score", "explanation"]].rename(columns={"row": "Period start"})
    out["Period start"] = pd.to_datetime(out["Period start"]).dt.date
    st.dataframe(out, hide_index=True)

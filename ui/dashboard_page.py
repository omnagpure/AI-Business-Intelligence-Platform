"""Dashboard: KPIs, trends, breakdowns, distributions and correlations for the active dataset.

Everything shown is computed from the active (cleaned, if applied) dataset using the roles from the
Semantics page. Nothing is displayed for an empty dataset.
"""
from __future__ import annotations

from dataclasses import replace
from html import escape

import pandas as pd
import streamlit as st

from analytics.correlation import correlation
from analytics.descriptive import (FREQ_LABELS, aggregate_time, analysable_numeric_columns, categorical_summary,
                                  date_summary, datetime_series, grouped_summary, groupable_columns, numeric_summary)
from analytics.formatting import format_value
from analytics.kpis import OK, DataFilter, compute_kpis, primary_column, primary_date_column
from config.analytics_config import DEFAULT_ANALYTICS
from core.logging_config import get_logger
from semantic.column_lexicon import DOMAIN_LABELS
from ui import state
from ui.analytics_state import cached, show_figure
from ui.layout import empty_state, eyebrow, no_dataset_notice, page_title
from visualization import charts

log = get_logger("ui.dashboard")
MAX_FILTER_VALUES = 60
_AGG = {"Sum": "sum", "Average": "mean", "Median": "median", "Row count": "count"}
_FREQ = {"Month": "M", "Week": "W", "Day": "D"}


def _safe(section: str, fn) -> None:
    """A failure in one section must not take down the rest of the page."""
    try:
        fn()
    except Exception:
        log.exception("Dashboard section failed: %s", section)
        st.warning(f"The {section} section could not be built for this dataset.")


def _filters(df: pd.DataFrame, profile, model, date_col: str | None) -> DataFilter:
    with st.expander("Filters and settings", expanded=False):
        c1, c2 = st.columns(2)
        date_range = None
        if date_col is not None:
            d = cached("date_bounds", date_col, lambda: (lambda s: (s.min(), s.max()))(datetime_series(df[date_col]).dropna()))
            if pd.notna(d[0]):
                with c1:
                    date_range = st.date_input(f"Date range ({date_col})", value=(d[0].date(), d[1].date()),
                                               min_value=d[0].date(), max_value=d[1].date(), key="dash_dates")
        cats = groupable_columns(df, profile, model, DEFAULT_ANALYTICS)
        with c2:
            chosen = st.multiselect("Filter by columns (up to 2)", cats, max_selections=2, key="dash_filter_cols")
        selected = []
        for col in chosen:
            vc = df[col].astype("string").value_counts().head(MAX_FILTER_VALUES)
            vals = st.multiselect(f"{col} is one of", list(vc.index), key=f"dash_vals_{col}",
                                  help=f"Showing the {MAX_FILTER_VALUES} most frequent values." if df[col].nunique() > MAX_FILTER_VALUES else None)
            if vals:
                selected.append((col, tuple(vals)))
        start = end = None
        if date_range is not None and len(date_range) == 2 and date_col is not None:
            full = (d[0].date(), d[1].date())
            if tuple(date_range) != full:
                start, end = str(date_range[0]), str(date_range[1])
        return DataFilter(tuple(selected), date_col if start else None, start, end)


def _kpi_section(df, model, config, flt) -> None:
    key_params = (config, flt.key())
    report = cached("kpis", key_params, lambda: compute_kpis(df, model, config, flt))
    if not report.results:
        empty_state("Nothing to show", report.notes[0] if report.notes else "No data.")
        return
    for n in report.notes:
        st.info(n)
    shown = [r for r in report.ok()]
    eyebrow("Key figures")
    for i in range(0, len(shown), 4):
        cols = st.columns(4)
        for col, r in zip(cols, shown[i:i + 4]):
            col.metric(r.name, format_value(r.value, r.unit), help=f"{r.formula}" + (f"\n\n{r.period}" if r.period else ""))
            if r.period:
                col.caption(r.period)
    with st.expander("Formulas, columns and limitations"):
        rows = [{"KPI": r.name, "Value": format_value(r.value, r.unit), "Formula": r.formula,
                 "Columns": ", ".join(r.columns), "Filters": "; ".join(r.filters) or "none",
                 "Limitations": " ".join(r.limitations), "Calculated at (UTC)": r.calculated_at} for r in shown]
        st.dataframe(pd.DataFrame(rows), hide_index=True)
    unavailable = report.unavailable()
    if unavailable:
        with st.expander(f"{len(unavailable)} KPI(s) not shown, and why"):
            st.dataframe(pd.DataFrame([{"KPI": r.name, "Status": r.status.replace("_", " "), "Reason": r.reason}
                                       for r in unavailable]), hide_index=True)
            st.caption("A KPI appears only when the columns it needs have the right roles. "
                       "Change roles on the Semantics page and this list updates.")


def _trend_section(df, profile, model, date_col, flt) -> None:
    if date_col is None:
        st.caption("No usable date column, so there is no trend. Set a Date / time role on the Semantics page.")
        return
    nums = analysable_numeric_columns(df, profile, model)
    default = primary_column(df, model, "revenue")
    c1, c2, c3 = st.columns(3)
    agg_label = c1.selectbox("Measure", list(_AGG), key="dash_t_agg")
    value = c2.selectbox("Column", nums, index=nums.index(default) if default in nums else 0, key="dash_t_val",
                         disabled=agg_label == "Row count") if nums else None
    freq = _FREQ[c3.selectbox("Period", list(_FREQ), key="dash_t_freq")]
    agg = _AGG[agg_label]
    t = cached("trend", (date_col, value, agg, freq, flt.key()), lambda: aggregate_time(df, date_col, value, freq, agg))
    fig = charts.line_chart(t, "period", "value", f"{agg_label}{'' if agg == 'count' else ' of ' + str(value)} by {FREQ_LABELS[freq]}",
                            y_title="Rows" if agg == "count" else str(value), partial=~t["complete"])
    if fig is None:
        st.caption("Nothing to plot for this selection.")
        return
    show_figure(fig)
    if (~t["complete"]).any():
        st.caption(f"Periods marked × are only partly covered by the data (first or last {FREQ_LABELS[freq]}). "
                   "They are excluded from growth figures.")
    d = date_summary(df, date_col)
    if d.missing_periods:
        st.caption(f"The date column has {d.missing_periods} empty period(s) between {d.min.date()} and {d.max.date()} "
                   f"(first gap starts {d.gaps[0][0]}).")


def _breakdown_section(df, profile, model, config, flt) -> None:
    groups = groupable_columns(df, profile, model, config)
    nums = analysable_numeric_columns(df, profile, model)
    if not groups:
        st.caption("No categorical column with 2 to "
                   f"{config.max_group_cardinality} distinct values was found.")
        return
    cat_default = next((c for c in model.columns_with_role("category") if c in groups), groups[0])
    c1, c2, c3 = st.columns(3)
    by = c1.selectbox("Group by", groups, index=groups.index(cat_default), key="dash_b_by")
    agg_label = c2.selectbox("Measure", list(_AGG), key="dash_b_agg")
    rev = primary_column(df, model, "revenue")
    value = c3.selectbox("Column", nums, index=nums.index(rev) if rev in nums else 0, key="dash_b_val",
                         disabled=agg_label == "Row count") if nums else None
    agg = _AGG[agg_label]
    t = cached("grouped", (by, value, agg, config, flt.key()), lambda: grouped_summary(df, by, value, agg, config))
    left, right = st.columns([3, 2])
    with left:
        fig = (charts.share_chart if agg in ("sum", "count") and len(t) <= 6 else charts.bar_chart)(
            t, "group", "value", f"{agg_label}{'' if agg == 'count' else ' of ' + str(value)} by {by}")
        if fig is None:
            st.caption("Nothing to plot for this selection.")
        else:
            show_figure(fig)
    with right:
        st.dataframe(t.rename(columns={"group": by, "value": "Value", "rows": "Rows", "share_pct": "Share %"}),
                     hide_index=True)
        st.caption(f"Top {config.top_n} groups"
                   + (" with the rest combined as 'Other'." if agg in ("sum", "count") else "; averages and medians cannot be combined into 'Other'."))


def _distribution_section(df, profile, model, flt) -> None:
    nums = analysable_numeric_columns(df, profile, model)
    if not nums:
        st.caption("No numeric measure columns (identifiers, coordinates and constants are skipped).")
        return
    col = st.selectbox("Column", nums, key="dash_d_col")
    left, right = st.columns(2)
    with left:
        fig = charts.histogram(df[col], f"Distribution of {col}")
        show_figure(fig) if fig else st.caption("No numeric values to plot.")
    with right:
        fig = charts.box_plot(df, col, f"Spread of {col}")
        show_figure(fig) if fig else st.caption("No numeric values to plot.")
    summary = cached("numeric_summary", (tuple(nums), flt.key()), lambda: numeric_summary(df, nums))
    st.dataframe(summary.rename_axis("Column").reset_index().round(3), hide_index=True)
    cats = [c for c in groupable_columns(df, profile, model)]
    if cats:
        pick = st.selectbox("Most frequent values of", cats, key="dash_d_cat")
        s = categorical_summary(df, [pick])[pick]
        st.dataframe(s.top.rename(columns={"value": pick, "count": "Rows", "pct": "Share %"}).round(2), hide_index=True)
        if s.note:
            st.caption(s.note)


def _correlation_section(df, profile, model, config, flt) -> None:
    nums = analysable_numeric_columns(df, profile, model)
    res = cached("correlation", (tuple(nums), config, flt.key()), lambda: correlation(df, nums, "pearson", config))
    if res.matrix.empty:
        st.caption(res.note or "Not enough suitable numeric columns.")
        return
    if not res.pairs.empty:
        keep = list(dict.fromkeys(list(res.pairs["a"]) + list(res.pairs["b"])))[:charts.MAX_HEATMAP_COLUMNS]
    else:
        keep = list(res.matrix.columns)[:charts.MAX_HEATMAP_COLUMNS]
    fig = charts.heatmap(res.matrix.loc[keep, keep], "Pearson correlation")
    if fig is not None:
        show_figure(fig)
    st.dataframe(res.pairs.drop(columns=["abs_r"]).round(3).rename(columns={"a": "Column A", "b": "Column B", "r": "r", "n": "Rows", "note": "Note"}),
                 hide_index=True)
    st.caption(f"Pairs with |r| below {config.corr_min_abs} are hidden. Correlation shows association, not cause. "
               + (res.note or ""))
    if res.excluded:
        st.caption("Left out: " + "; ".join(f"{c} ({why})" for c, why in list(res.excluded.items())[:8]))
    if len(nums) >= 2:
        a, b = st.columns(2)
        x = a.selectbox("Scatter X", nums, key="dash_s_x")
        y = b.selectbox("Scatter Y", nums, index=1, key="dash_s_y")
        fig = charts.scatter(df, x, y, f"{y} vs {x}", config.scatter_max_points)
        show_figure(fig) if fig else st.caption("No complete pairs of values to plot.")


def render_dashboard_page() -> None:
    result = state.get_result()
    if result is None:
        page_title("Dashboard", "KPIs and charts computed from your data.")
        no_dataset_notice()
        return
    ds = result.dataset
    page_title("Dashboard", ds.filename + (" · synthetic sample data" if ds.is_sample else ""))
    if ds.is_sample:
        st.warning("This is synthetic sample data generated for demonstration. It is not real.")
    try:
        df, profile, model = state.get_analysis_df(), state.get_analysis_profile(), state.get_semantic_model()
    except Exception:
        log.exception("Dashboard inputs unavailable")
        st.error("The dataset could not be prepared for analysis. Check the Data quality and Semantics pages.")
        return
    if df is None or profile is None or model is None or len(df) == 0:
        empty_state("Nothing to analyse", "The active dataset has no rows.")
        return
    st.caption(f"Domain: {DOMAIN_LABELS[model.domain.domain]}"
               + (" (chosen by you)" if model.domain.source == "user" else f" · confidence {model.domain.confidence:.0%}")
               + f" · {len(model.overridden())} role override(s)"
               + (" · cleaned data" if state.get_cleaning() else " · original data"))

    config = DEFAULT_ANALYTICS
    date_col = primary_date_column(df, model)
    flt = _filters(df, profile, model, date_col)
    with st.expander("Comparison settings"):
        c1, c2 = st.columns(2)
        cmp_label = c1.selectbox("Growth compares the latest complete period with",
                                 ["Previous period", "Same period last year"], key="dash_cmp")
        freq_label = c2.selectbox("Period for growth", list(_FREQ), key="dash_growth_freq")
        config = replace(config, comparison="year_over_year" if cmp_label.startswith("Same") else "previous_period",
                         time_freq=_FREQ[freq_label])
    fdf = cached("filtered_df", flt.key(), lambda: flt.apply(df)) if flt.active else df
    _safe("KPI", lambda: _kpi_section(df, model, config, flt))
    if len(fdf) == 0:
        return
    st.write("")
    t1, t2, t3, t4 = st.tabs(["Trend", "Breakdown", "Distribution", "Relationships"])
    with t1:
        _safe("trend", lambda: _trend_section(fdf, profile, model, date_col, flt))
    with t2:
        _safe("breakdown", lambda: _breakdown_section(fdf, profile, model, config, flt))
    with t3:
        _safe("distribution", lambda: _distribution_section(fdf, profile, model, flt))
    with t4:
        _safe("correlation", lambda: _correlation_section(fdf, profile, model, config, flt))

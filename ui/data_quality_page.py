"""Data quality page: issues, missing values and duplicate records."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from config.settings import Settings
from ui import state
from ui.layout import fmt_int, no_dataset_notice, page_title

_STATUS = {"good": "Good", "review": "Needs review", "poor": "Problems found"}


def render_data_quality_page(settings: Settings) -> None:
    page_title("Data quality", "Issues found in the uploaded data. Nothing has been changed or removed.")
    result = state.get_result()
    if result is None:
        no_dataset_notice()
        return
    p, q, df = result.profile, result.quality, result.dataset.df
    c = st.columns(4)
    c[0].metric("Status", _STATUS[q.status])
    c[1].metric("Completeness", f"{q.completeness_pct:.1f}%")
    c[2].metric("Duplicate rows", fmt_int(p.duplicate_rows))
    c[3].metric("Issues", f"{len(q.by_severity('high'))} high · {len(q.by_severity('medium'))} medium "
                          f"· {len(q.by_severity('low'))} low")

    t_issues, t_missing, t_dups = st.tabs(["Issues", "Missing values", "Duplicates"])
    with t_issues:
        if not q.issues:
            st.success("No data quality issues were detected.")
        else:
            sev = st.multiselect("Severity", ["high", "medium", "low"], default=["high", "medium", "low"])
            recs = [i for i in q.to_records() if i["severity"] in sev]
            table = pd.DataFrame(recs).rename(columns={
                "severity": "Severity", "code": "Issue", "column": "Column", "message": "Details",
                "affected_rows": "Rows affected", "suggestion": "Suggested action"})
            st.dataframe(table, hide_index=True)
        st.caption("Cleaning actions for these issues are added in a later release.")

    with t_missing:
        miss = pd.DataFrame([{"Column": c.name, "Missing": c.missing, "Missing %": c.missing_pct}
                             for c in p.columns_profile if c.missing]).sort_values("Missing", ascending=False) \
            if any(c.missing for c in p.columns_profile) else pd.DataFrame()
        if miss.empty:
            st.success("No missing values.")
        else:
            st.dataframe(miss, hide_index=True, column_config={
                "Missing %": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100)})

    with t_dups:
        if not p.duplicate_rows:
            st.success("No exact duplicate rows.")
        else:
            st.caption(f"{fmt_int(p.duplicate_rows)} rows repeat an earlier row. "
                       "Duplicates may be legitimate, so review before removing any.")
            dup = df[df.duplicated(keep=False)]
            st.dataframe(dup.head(settings.preview_rows), hide_index=True)
            if len(dup) > settings.preview_rows:
                st.caption(f"Showing the first {settings.preview_rows} of {fmt_int(len(dup))} rows involved.")

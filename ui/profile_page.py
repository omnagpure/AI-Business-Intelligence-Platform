"""Dataset profile: columns, numeric, categorical and date summaries."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from ingestion.profiler import DatasetProfile
from ui import state
from ui.layout import fmt_int, no_dataset_notice, page_title

_KIND_LABEL = {"numeric": "Numeric", "categorical": "Categorical", "temporal": "Date/time",
               "boolean": "Boolean", "text": "Free text", "identifier": "Identifier", "empty": "Empty"}


def _columns_table(p: DatasetProfile) -> pd.DataFrame:
    rows = []
    for c in p.columns_profile:
        flags = [f for f, on in (("identifier", c.is_identifier), ("constant", c.is_constant),
                                 ("stored as text", c.stored_as_text)) if on]
        rows.append({"Column": c.name, "Type": _KIND_LABEL[c.kind], "Stored as": c.dtype,
                     "Missing": c.missing, "Missing %": c.missing_pct, "Unique": c.unique,
                     "Cardinality": c.cardinality, "Examples": ", ".join(c.examples[:3]),
                     "Flags": ", ".join(flags)})
    return pd.DataFrame(rows)


def render_profile_page() -> None:
    page_title("Profile", "Structure and statistics of the uploaded data.")
    result = state.get_result()
    if result is None:
        no_dataset_notice()
        return
    p = result.profile
    c = st.columns(5)
    c[0].metric("Rows", fmt_int(p.rows))
    c[1].metric("Columns", p.columns)
    c[2].metric("Duplicate rows", fmt_int(p.duplicate_rows))
    c[3].metric("Missing cells", f"{p.missing_cells_pct:.1f}%")
    c[4].metric("Memory", f"{p.memory_mb:.1f} MB")

    t_cols, t_num, t_cat, t_date = st.tabs(["Columns", "Numeric", "Categorical", "Dates"])
    with t_cols:
        st.dataframe(_columns_table(p), hide_index=True, column_config={
            "Missing %": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100)})
        if p.potential_primary_keys:
            st.caption("Possible primary keys: " + ", ".join(p.potential_primary_keys))
        if p.possible_targets:
            st.caption("Possible target columns (name-based heuristic): " + ", ".join(p.possible_targets))

    with t_num:
        rows = [{"Column": c.name, "Count": s["count"], "Mean": s["mean"], "Std": s["std"],
                 "Min": s["min"], "Q1": s["q1"], "Median": s["median"], "Q3": s["q3"],
                 "Max": s["max"], "Skew": s["skew"], "Zeros": s["zeros"], "Negatives": s["negatives"]}
                for c in p.columns_profile if (s := c.numeric)]
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True)
            st.caption("Identifier-like numeric columns are excluded from this table.")
        else:
            st.caption("No numeric columns found.")

    with t_cat:
        cats = [c for c in p.columns_profile if c.top_values]
        if not cats:
            st.caption("No categorical columns found.")
        else:
            name = st.selectbox("Column", [c.name for c in cats], key="cat_col")
            col = p.column(name)
            st.caption(f"{fmt_int(col.unique)} distinct values · showing the top {len(col.top_values)}")
            st.dataframe(pd.DataFrame(col.top_values).rename(
                columns={"value": "Value", "count": "Count", "pct": "Share"}), hide_index=True,
                column_config={"Share": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100)})

    with t_date:
        rows = [{"Column": c.name, "Earliest": t["min"], "Latest": t["max"], "Span (days)": t["span_days"],
                 "Unparsed": t["unparsed"], "Stored as text": c.stored_as_text}
                for c in p.columns_profile if (t := c.temporal)]
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True)
        else:
            st.caption("No date columns detected.")

"""Data explorer: searchable, sortable, paginated preview with column types and roles.

Only one page of rows (config.preview_rows, default 100) is ever sent to the browser.
"""
from __future__ import annotations

import math

import pandas as pd
import streamlit as st

from config.analytics_config import DEFAULT_ANALYTICS
from core.logging_config import get_logger
from semantic.column_lexicon import ROLE_LABELS
from ui import state
from ui.analytics_state import cached
from ui.layout import empty_state, eyebrow, no_dataset_notice, page_title

log = get_logger("ui.explorer")
MAX_PREVIEW_ROWS = DEFAULT_ANALYTICS.preview_rows
MAX_SEARCH_COLUMNS = 12
DEFAULT_SHOWN_COLUMNS = 12


def search_rows(df: pd.DataFrame, query: str, columns: list[str]) -> pd.Series:
    """Boolean mask of rows where any of `columns` contains `query` (case-insensitive, literal text)."""
    q = query.strip()
    if not q or not columns:
        return pd.Series(True, index=df.index)
    mask = pd.Series(False, index=df.index)
    for col in columns:
        mask |= df[col].astype("string").str.contains(q, case=False, regex=False, na=False)
    return mask


def page_slice(df: pd.DataFrame, page: int, page_size: int = MAX_PREVIEW_ROWS) -> pd.DataFrame:
    """Rows of one page (1-based). Never returns more than page_size rows."""
    page = max(1, int(page))
    return df.iloc[(page - 1) * page_size: page * page_size]


def column_table(df: pd.DataFrame, profile, model) -> pd.DataFrame:
    rows = []
    for c in profile.columns_profile:
        sem = model.columns.get(c.name) if model else None
        rows.append({"Column": c.name, "Type": c.kind, "Role": ROLE_LABELS.get(sem.role, sem.role) if sem else "n/a",
                     "Role source": sem.source if sem else "", "Missing %": round(c.missing_pct, 1), "Distinct": c.unique})
    return pd.DataFrame(rows)


def render_explorer_page() -> None:
    page_title("Data explorer", "Search, sort and page through the active dataset.")
    df, profile = state.get_analysis_df(), state.get_analysis_profile()
    if df is None or profile is None:
        no_dataset_notice()
        return
    if len(df) == 0:
        empty_state("No rows", "The active dataset is empty.")
        return
    try:
        model = state.get_semantic_model()
    except Exception:
        log.exception("Semantic model unavailable in explorer")
        model = None
    st.caption(f"{len(df):,} rows · {df.shape[1]} columns · "
               + ("cleaned data" if state.get_cleaning() else "original data")
               + " · roles can be changed on the Semantics page")

    eyebrow("Find rows")
    text_cols = [c.name for c in profile.columns_profile if c.kind in ("categorical", "text", "identifier", "boolean")]
    c1, c2 = st.columns([2, 3])
    query = c1.text_input("Search", placeholder="Text to find", key="ex_query")
    search_cols = c2.multiselect("Search in", list(df.columns), default=text_cols[:MAX_SEARCH_COLUMNS // 2], key="ex_search_cols",
                                 help=f"Up to {MAX_SEARCH_COLUMNS} columns. Matching is case-insensitive, plain text.")
    search_cols = search_cols[:MAX_SEARCH_COLUMNS]
    c3, c4, c5 = st.columns([3, 2, 3])
    sort_col = c3.selectbox("Sort by", ["(original order)"] + list(df.columns), key="ex_sort")
    descending = c4.toggle("Descending", key="ex_desc")
    shown_default = list(df.columns[:DEFAULT_SHOWN_COLUMNS])
    shown = c5.multiselect("Columns to show", list(df.columns), default=shown_default, key=f"ex_cols_{state.get_result().dataset.fingerprint}")
    shown = shown or shown_default

    def build() -> pd.DataFrame:
        found = df[search_rows(df, query, search_cols)] if query.strip() else df
        if sort_col != "(original order)":
            found = found.sort_values(sort_col, ascending=not descending, kind="mergesort", na_position="last")
        return found

    result = cached("explorer", (query, tuple(search_cols), sort_col, descending), build) if (query.strip() or sort_col != "(original order)") else df
    total = len(result)
    if total == 0:
        empty_state("No matching rows", "Try a different search text or search in more columns.")
    else:
        pages = max(1, math.ceil(total / MAX_PREVIEW_ROWS))
        page = st.number_input("Page", min_value=1, max_value=pages, value=1, step=1, key="ex_page") if pages > 1 else 1
        view = page_slice(result[shown], int(page))
        first = (int(page) - 1) * MAX_PREVIEW_ROWS + 1
        st.caption(f"Rows {first:,} to {first + len(view) - 1:,} of {total:,}" + (f" matching '{query.strip()}'" if query.strip() else "")
                   + f" · {MAX_PREVIEW_ROWS} rows per page")
        st.dataframe(view, hide_index=True)

    st.write("")
    eyebrow("Columns")
    st.dataframe(column_table(df, profile, model), hide_index=True)

"""Shared filter bar: always visible under the page header, one filter state for every page that shows it.

The filter lives in session state (ui.state), not in widgets, so it survives moving between pages. Widgets are only
the editor: they start from the stored filter, and what they hold is written back each run. Pages that show the bar
apply the filter and key their cached results by it (ui.analytics_state.base_key). Home and Reports ignore it.
"""
from __future__ import annotations

from dataclasses import replace

import pandas as pd
import streamlit as st

from analytics.descriptive import datetime_series, groupable_columns
from analytics.kpis import DataFilter, primary_date_column
from config.analytics_config import DEFAULT_ANALYTICS
from ui import state
from ui.analytics_state import cached
from ui.layout import empty_state

MAX_VALUES = 60       # most frequent values offered per column
MAX_CATEGORY_FILTERS = 4
_NONE = "Add a filter on a column"


def begin_run() -> None:
    """Called once per script run: pages opt in again by drawing the bar."""
    state.set_filter_scope(False)


def filtered(df: pd.DataFrame) -> pd.DataFrame:
    """The rows the shared filter keeps (the same frame when nothing is filtered)."""
    flt = state.get_filter()
    return cached("filtered_df", flt.key(), lambda: flt.apply(df)) if flt.active else df


def _chip_label(col: str, values: tuple[str, ...]) -> str:
    shown = ", ".join(values[:2]) + (f" +{len(values) - 2}" if len(values) > 2 else "")
    return f"{col}: {shown}"


def _remove_category(col: str, vals_key: str) -> None:
    flt = state.get_filter()
    state.set_filter(replace(flt, categories=tuple(c for c in flt.categories if c[0] != col)))
    st.session_state.pop(vals_key, None)


def _remove_dates() -> None:
    flt = state.get_filter()
    state.set_filter(replace(flt, date_col=None, date_start=None, date_end=None))
    st.session_state["_fb_dates_reset"] = True


def _reset_all(prefix: str) -> None:
    state.set_filter(DataFilter())
    for key in [k for k in st.session_state if str(k).startswith(prefix)]:
        st.session_state.pop(key, None)
    st.session_state["_fb_dates_reset"] = True


def render_filter_bar(df: pd.DataFrame, model) -> None:
    """Draw the bar and store the filter it describes. Call before using the data on the page."""
    state.set_filter_scope(True)
    result, profile = state.get_result(), state.get_analysis_profile()
    if result is None or profile is None:
        return
    fp = result.dataset.fingerprint
    prefix = f"fb_{fp}_"
    flt = state.get_filter()
    date_col = primary_date_column(df, model) if model is not None else None
    cats = groupable_columns(df, profile, model, DEFAULT_ANALYTICS)

    with st.container(border=True):
        c_date, c_col, c_val, c_reset = st.columns([2.4, 2.2, 3.4, 1], vertical_alignment="bottom")
        start, end, bounds = flt.date_start, flt.date_end, None
        if date_col is not None:
            d = cached("date_bounds", date_col, lambda: (lambda s: (s.min(), s.max()))(datetime_series(df[date_col]).dropna()))
            if pd.notna(d[0]):
                bounds = (d[0].date(), d[1].date())
        if bounds is not None:
            key = f"{prefix}dates_{bounds[0]}_{bounds[1]}"
            if st.session_state.pop("_fb_dates_reset", False):
                st.session_state.pop(key, None)
            if key not in st.session_state:
                lo = max(bounds[0], pd.Timestamp(flt.date_start).date()) if flt.date_start else bounds[0]
                hi = min(bounds[1], pd.Timestamp(flt.date_end).date()) if flt.date_end else bounds[1]
                st.session_state[key] = (lo, max(lo, hi))
            with c_date:
                picked = st.date_input(f"Date range ({date_col})", min_value=bounds[0], max_value=bounds[1], key=key)
            if isinstance(picked, (tuple, list)) and len(picked) == 2:
                start, end = (None, None) if tuple(picked) == bounds else (str(picked[0]), str(picked[1]))
        cat_filters = dict(flt.categories)
        with c_col:
            options = [_NONE] + [c for c in cats]
            col_pick = st.selectbox("Filter by", options, key=f"{prefix}col")
        with c_val:
            if col_pick != _NONE:
                vc = cached("filter_values", col_pick, lambda: df[col_pick].astype("string").value_counts().head(MAX_VALUES))
                vkey = f"{prefix}vals_{col_pick}"
                if vkey not in st.session_state:
                    st.session_state[vkey] = [v for v in cat_filters.get(col_pick, ()) if v in set(vc.index)]
                if col_pick not in cat_filters and len(cat_filters) >= MAX_CATEGORY_FILTERS:
                    st.caption(f"Up to {MAX_CATEGORY_FILTERS} column filters at a time. Remove one to add another.")
                else:
                    picked_vals = st.multiselect(f"{col_pick} is one of", list(vc.index), key=vkey,
                                                 help=f"The {MAX_VALUES} most frequent values." if df[col_pick].nunique() > MAX_VALUES else None)
                    if picked_vals:
                        cat_filters[col_pick] = tuple(picked_vals)
                    else:
                        cat_filters.pop(col_pick, None)
            else:
                st.caption("Pick a column to filter its values.")
        new = DataFilter(tuple(cat_filters.items()), date_col if (start or end) else None, start, end)
        state.set_filter(new)
        with c_reset:
            st.button("Reset", key=f"{prefix}reset", disabled=not new.active, on_click=_reset_all, args=(prefix,), width="stretch")

        chips: list[tuple[str, object, tuple]] = []
        if new.date_col and (new.date_start or new.date_end):
            chips.append((f"{new.date_col}: {new.date_start or 'start'} to {new.date_end or 'end'}", _remove_dates, ()))
        for col, vals in new.categories:
            chips.append((_chip_label(col, vals), _remove_category, (col, f"{prefix}vals_{col}")))
        if chips:
            cols = st.columns(min(len(chips), 5) + 1)
            for i, (label, cb, args) in enumerate(chips[:5]):
                cols[i].button(label, key=f"{prefix}chip_{i}", icon=":material/close:", on_click=cb, args=args, help="Remove this filter")
        kept = len(filtered(df)) if new.active else len(df)
        st.caption(f"Showing {kept:,} of {len(df):,} rows" if new.active else f"All {len(df):,} rows")


def no_rows_notice() -> None:
    empty_state("No rows match these filters", "Remove a filter or press Reset to see your data again.")

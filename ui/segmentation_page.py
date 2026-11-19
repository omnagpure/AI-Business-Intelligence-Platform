"""Segmentation page (Phase 6C): group suppliers, customers or products with KMeans. Runs only when asked."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from analytics.segmentation import NOTE, SCALERS, build_entity_features, entity_candidates, run_segmentation
from config.analytics_config import DEFAULT_ANALYTICS
from core.logging_config import get_logger
from ui import state
from ui.progress_bar import ProgressBar
from ui.analysis_runner import safe_cached
from ui.analytics_state import base_key, show_figure
from ui.filters import filtered, no_rows_notice, render_filter_bar
from ui.layout import empty_state, no_dataset_notice, page_title
from visualization import charts

log = get_logger("ui.segmentation")


def _cluster(table, p):
    with ProgressBar("Segmenting") as bar:
        return run_segmentation(table, list(p[2]), p[3], p[4], progress=bar)


def render_segmentation_page() -> None:
    result = state.get_result()
    if result is None:
        page_title("Segmentation", "Group similar suppliers, customers or products.")
        no_dataset_notice()
        return
    ds = result.dataset
    page_title("Segmentation", ds.filename + (" · synthetic sample data" if ds.is_sample else ""))
    try:
        df, model = state.get_analysis_df(), state.get_semantic_model()
    except Exception:
        log.exception("Segmentation inputs unavailable")
        st.error("The dataset could not be prepared for analysis. Check the Data quality and Column roles pages.")
        return
    if df is None or model is None or len(df) == 0:
        empty_state("Nothing to segment", "The active dataset has no rows.")
        return
    render_filter_bar(df, model)
    df = filtered(df)
    if len(df) == 0:
        no_rows_notice()
        return
    cands = entity_candidates(df, model)
    if not cands:
        empty_state("No entity column found",
                    f"Segmentation needs a Supplier, Customer, Product or Warehouse column with at least "
                    f"{DEFAULT_ANALYTICS.seg_min_entities} different values and {DEFAULT_ANALYTICS.seg_min_rows}+ rows each. "
                    "You can set column roles on the Column roles page.")
        return
    st.caption(NOTE + " It runs only when you press the button; nothing is changed in your data.")
    labels = {c["column"]: f"{c['column']} ({c['role']}, {c['entities']:,} values)" for c in cands}
    entity = st.selectbox("What to group", list(labels), format_func=labels.get, key="seg_entity")
    status, table = safe_cached("seg_features", (entity,), lambda: build_entity_features(df, model, entity),
                                "Entity features could not be built for this column.")
    if status != "ok":
        empty_state("Features not available", table) if status == "insufficient" else st.warning(table)
        return
    usable = list(table.frame.columns)
    if len(usable) < 2:
        empty_state("Too few features", "Fewer than two measures could be built for each entity from this data.")
        return
    for w in table.warnings:
        st.info(w)
    with st.expander(f"Features built per {entity} ({len(table.frame):,} entities)"):
        show = table.frame.rename(columns=table.labels).round(2)
        st.dataframe(show.head(DEFAULT_ANALYTICS.preview_rows))
        if table.excluded_features:
            st.caption("Left out: " + "; ".join(f"{k} ({v})" for k, v in table.excluded_features.items()))
    c1, c2, c3 = st.columns(3)
    feats = c1.multiselect("Features to use", usable, default=usable, format_func=table.labels.get, key="seg_feats")
    scaler = c2.selectbox("Scaling", list(SCALERS), format_func=SCALERS.get, key="seg_scaler")
    mode = c3.radio("Number of clusters", ["Automatic", "Choose"], horizontal=True, key="seg_mode")
    k = c3.slider("K", 2, DEFAULT_ANALYTICS.seg_k_max, 3, key="seg_k") if mode == "Choose" else None
    if st.button("Run segmentation", type="primary", key="seg_run"):
        st.session_state["seg_params"] = (base_key(), entity, tuple(feats), k, scaler)
    p = st.session_state.get("seg_params")
    if not p or p[0] != base_key():
        return
    status, res = safe_cached("seg_result", p[1:], lambda: _cluster(table, p),
                              "Segmentation could not be completed for this selection.")
    if status != "ok":
        empty_state("Segmentation not possible", res) if status == "insufficient" else st.warning(res)
        return
    st.markdown(f"**{res.k} clusters** ({res.k_mode} choice) · silhouette **{res.silhouette:.2f}**. {res.quality}")
    for w in res.warnings:
        st.warning(w)
    a, b = st.columns(2)
    sizes = res.profile.reset_index().rename(columns={"index": "cluster"})
    sizes["label"] = "Cluster " + sizes.iloc[:, 0].astype(str)
    fig = charts.bar_chart(sizes, "label", "entities", f"Number of {p[1]} values per cluster", "Entities")
    if fig is not None:
        with a:
            show_figure(fig)
    sc = res.scores
    fig = charts.line_chart(sc, "k", "silhouette", "Silhouette by number of clusters (higher is better)", "Silhouette")
    if fig is not None:
        with b:
            show_figure(fig)
    st.markdown("**What makes each cluster different**")
    summary = pd.DataFrame({"Cluster": list(res.traits), "Entities": [int(res.profile.loc[c, "entities"]) for c in res.traits],
                            "Share": [f"{res.profile.loc[c, 'share']:.0%}" for c in res.traits],
                            "In plain words": list(res.traits.values())})
    st.dataframe(summary, hide_index=True)
    st.markdown("**Average of each feature per cluster**")
    st.dataframe(res.profile.drop(columns=["share"]).rename(columns=res.labels).round(2))
    with st.expander("Elbow data (inertia) and silhouette for every K"):
        st.dataframe(sc.round(3), hide_index=True)
    if len(res.features) >= 2:
        x1, x2 = st.columns(2)
        fx = x1.selectbox("Chart x", list(res.features), format_func=res.labels.get, key="seg_x")
        fy = x2.selectbox("Chart y", list(res.features), index=1, format_func=res.labels.get, key="seg_y")
        fig = charts.segment_scatter(res.assignments, fx, fy, "cluster", f"{res.labels[fy]} vs {res.labels[fx]} by cluster",
                                     res.labels[fx], res.labels[fy])
        if fig is not None:
            show_figure(fig)
    with st.expander("Entities and their cluster"):
        st.dataframe(res.assignments.head(DEFAULT_ANALYTICS.preview_rows).rename(columns=res.labels).round(2), hide_index=True)
        st.caption(f"Showing up to {DEFAULT_ANALYTICS.preview_rows} rows.")

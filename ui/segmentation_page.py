"""Segmentation page: group suppliers, customers, products or warehouses with KMeans and say what sets each group apart.

Runs as soon as the page opens (first entity column, all features, automatic number of segments) and on every
setting change; results are cached per setting and filter and shared with the Home card. The page leads with how
clear the grouping is, names each segment by its strongest real difference, and shows which features separate the
segments and what share of the business each one holds. Nothing in the data is changed.
"""
from __future__ import annotations

from html import escape

import pandas as pd
import streamlit as st

from analytics.segmentation import NOTE, SCALERS, build_entity_features, entity_candidates, run_segmentation
from config.analytics_config import DEFAULT_ANALYTICS
from core.logging_config import get_logger
from semantic.column_lexicon import DOMAIN_LABELS
from ui import state
from ui.analysis_runner import safe_cached
from ui.analytics_state import base_key, show_figure
from ui.filters import filtered, no_rows_notice, render_filter_bar
from ui.layout import badge, callout, empty_state, lede, meta_line, no_dataset_notice, page_title, section
from ui.progress_bar import ProgressBar
from visualization import charts

log = get_logger("ui.segmentation")
_ROLE_WORD = {"supplier": "suppliers", "customer": "customers", "product": "products", "warehouse": "warehouses"}
_AUTO = "Automatic"


def _cluster(table, p):
    """p = (base_key, entity, features, k, scaler). Shared with the Home card (same cache entry)."""
    with ProgressBar("Segmenting") as bar:
        return run_segmentation(table, list(p[2]), p[3], p[4], progress=bar)


def _fmt(v: float, unit: str) -> str:
    if v != v:
        return "–"
    if unit == "percent":
        return f"{v:.1f}%"
    if unit == "days":
        return f"{v:,.1f} d"
    a = abs(v)
    for limit, div, suffix in ((1e9, 1e9, "B"), (1e6, 1e6, "M"), (1e4, 1e3, "k")):
        if a >= limit:
            return f"{v / div:,.1f}{suffix}"
    return f"{v:,.0f}" if a >= 100 else f"{v:,.2f}".rstrip("0").rstrip(".")


def render_segmentation_page() -> None:
    result = state.get_result()
    if result is None:
        page_title("Segmentation", "Group similar suppliers, customers or products.")
        no_dataset_notice()
        return
    ds = result.dataset
    page_title("Segmentation", "Groups of similar suppliers, customers or products, and what sets them apart")
    try:
        df_all, model = state.get_analysis_df(), state.get_semantic_model()
    except Exception:
        log.exception("Segmentation inputs unavailable")
        st.error("The dataset could not be prepared for analysis. Check the Data quality and Column roles pages.")
        return
    if df_all is None or model is None or len(df_all) == 0:
        empty_state("Nothing to segment", "The active dataset has no rows.")
        return
    df = filtered(df_all)
    dom = model.domain.domain
    meta_line([ds.filename + (" (synthetic sample)" if ds.is_sample else ""),
               f"{len(df):,} of {len(df_all):,} rows" if state.get_filter().active else f"{len(df):,} rows"],
              badge(DOMAIN_LABELS.get(dom, dom), "accent" if model.is_supply_chain else "neutral"))
    render_filter_bar(df_all, model)
    df = filtered(df_all)
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
    by_col = {c["column"]: c for c in cands}
    labels = {c["column"]: f"{c['column']} ({c['entities']:,} {_ROLE_WORD.get(c['role'], 'values')})" for c in cands}

    with st.container(border=True, key="seg_setup"):
        c1, c2, c3 = st.columns([3, 1.4, 1], vertical_alignment="bottom")
        entity = c1.selectbox("What to group", list(labels), format_func=labels.get, key="seg_entity",
                              help="Columns that group the rows the same way (such as an id and its name) are shown once.")
        k_pick = c2.selectbox("Number of segments", [_AUTO] + list(range(2, DEFAULT_ANALYTICS.seg_k_max + 1)), key="seg_k_pick",
                              help="Automatic tries 2 to 8 and keeps the clearest split that has no tiny segment.")
        status, table = safe_cached("seg_features", (entity,), lambda: build_entity_features(df, model, entity),
                                    "Entity features could not be built for this column.")
        usable = list(table.frame.columns) if status == "ok" else []
        with c3.popover("Options", icon=":material/tune:", width="stretch"):
            feats = st.multiselect("Measures to group on", usable, default=usable, key="seg_feats",
                                   format_func=(table.labels.get if status == "ok" else str))
            scaler = st.selectbox("Scaling", list(SCALERS), format_func=SCALERS.get, key="seg_scaler")
            st.caption(NOTE)
    if status != "ok":
        empty_state("Features not available", table) if status == "insufficient" else st.warning(table)
        return
    if len(usable) < 2:
        empty_state("Too few measures", "Fewer than two measures could be built for each entity from this data.")
        return
    if len(feats) < 2:
        st.caption("Choose at least two measures to group on.")
        return
    k = None if k_pick == _AUTO else int(k_pick)
    p = (base_key(), entity, tuple(feats), k, scaler)
    st.session_state["seg_params"] = p
    status, res = safe_cached("seg_result", p[1:], lambda: _cluster(table, p),
                              "Segmentation could not be completed for this selection.")
    if status != "ok":
        empty_state("Segmentation not possible", res) if status == "insufficient" else st.warning(res)
        return
    _show(df, res, table, entity, by_col[entity])


VOLUME = ("rows", "units", "total_value")
DELIVERY = ("on_time_rate", "avg_delay", "lead_time", "fill_rate", "defect_rate", "stockout_rate")


def _value_feature(res) -> str | None:
    """The total to split by segment: money first (a revenue total, then any other summed amount), then orders, then units."""
    money = [f for f in res.features if f.startswith("sum_") and res.units.get(f) == "amount"] + \
            [f for f in res.features if f.startswith("sum_") and "value" in f.lower()]
    for f in ("total_value", *money, "rows", "units"):
        if f in res.assignments.columns and res.assignments[f].notna().any() and res.assignments[f].sum() > 0:
            return f
    return None


def _and(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _is_volume(f: str) -> bool:
    return f in VOLUME or f.startswith("sum_")


def _show(df, res, table, entity, cand) -> None:
    word = _ROLE_WORD.get(cand["role"], f"{entity} values")
    n = len(res.assignments)
    prof = res.profile
    sep = res.separation[res.separation > 0]
    top = [res.labels[f] for f in sep.index[:3] if sep[f] >= 0.2]
    all_close = all(v == "Close to average" for v in res.names.values())

    # ---- the answer
    structure = "clear" if res.silhouette >= 0.5 else "moderate" if res.silhouette >= 0.25 else "weak"
    text = f"The {n:,} {word} fall into {res.k} segments. The split is {structure} (fit {res.silhouette:.2f} of 1)"
    if structure == "weak":
        text += ": the segments overlap, so treat them as a convenient grouping rather than distinct types"
    text += "."
    if all_close:
        text += (" Every segment is within a few percent of the overall average on every measure, so in practice these "
                 f"{word} are alike.")
    elif top:
        text += f" They differ most in {_and(top)}."
        lead_feats = [f for f in sep.index[:3] if sep[f] >= 0.2]
        delivery = [f for f in res.features if f in DELIVERY]
        if lead_feats and all(_is_volume(f) for f in lead_feats) and delivery and \
                max(float(res.separation.get(f, 0)) for f in delivery) < 0.1:
            text += (" In other words they split by size (how much they handle), not by delivery performance: "
                     f"{_and([res.labels[f].lower() if res.labels[f][1:2].islower() else res.labels[f] for f in delivery])} "
                     "hardly differ between segments.")
    lede(text)
    if cand.get("same_as"):
        st.caption(f"{entity} groups the rows exactly like {', '.join(cand['same_as'])}, so only one of them is offered.")

    # ---- one card per segment
    vf = _value_feature(res)
    a = res.assignments
    total = float(a[vf].sum()) if vf else 0.0
    vlabel = res.labels.get(vf, "") if vf else ""
    cards = []
    for c in prof.index:
        share = float(prof.loc[c, "share"])
        chip = ""
        if vf and total > 0:
            vshare = float(a.loc[a["cluster"] == c, vf].sum()) / total
            tone = "good" if vshare > share * 1.25 else "flat"         # a smaller share is not "bad" in itself
            chip = f'<span class="chip {tone}">{vshare:.0%} of {escape(vlabel.replace("Total ", "").lower() if vlabel.startswith("Total ") else vlabel.lower())}</span>'
        trait = res.traits.get(c, "")
        cards.append(
            f'<div class="kpi compact seg"><div class="kpi-label"><span class="seg-dot" style="background:'
            f'{charts.PALETTE[(c - 1) % len(charts.PALETTE)]}"></span>Segment {c} · {int(prof.loc[c, "entities"]):,} '
            f'{escape(word)} ({share:.0%})</div><div class="kpi-value">{escape(res.names.get(c, f"Segment {c}"))}</div>'
            f'<div class="kpi-delta">{chip}</div><div class="kpi-foot">{escape(trait[:1].upper() + trait[1:])}</div></div>')
    cols = min(len(cards), 4)
    st.html(f'<div class="kpi-grid" style="grid-template-columns:repeat({cols}, minmax(0, 1fr))">' + "".join(cards) + "</div>")

    notes = list(res.warnings)
    if notes:
        callout("Good to know", " ".join(notes), "info")

    # ---- what separates them / how much business each holds
    section("What sets the segments apart")
    left, right = st.columns(2, gap="medium")
    with left:
        with st.container(border=True, key="seg_sep"):
            t = pd.DataFrame({"label": [res.labels[f] for f in res.separation.index], "value": 100 * res.separation.to_numpy()})
            fig = charts.bar_chart(t, "label", "value", "How much each measure differs between segments", "Percent", unit="%")
            if fig is not None:
                show_figure(fig)
            st.markdown('<div class="view-insight">The share of each measure\'s spread that lies between the segments '
                        'rather than within them. Near 0%: the segments do not differ on it.</div>', unsafe_allow_html=True)
    with right:
        with st.container(border=True, key="seg_mix"):
            if vf and total > 0:
                mix = pd.DataFrame({"label": [f"{c}. {res.names.get(c, '')}" for c in prof.index],
                                    "entities": [100 * float(prof.loc[c, "share"]) for c in prof.index],
                                    "value": [100 * float(a.loc[a["cluster"] == c, vf].sum()) / total for c in prof.index]})
                v = vlabel.replace("Total ", "") if vlabel.startswith("Total ") else vlabel
                fig = charts.segment_mix(mix, f"Share of {word} and of {v}", v, word)
                if fig is not None:
                    show_figure(fig)
                big = mix.loc[(mix["value"] - mix["entities"]).idxmax()]
                st.markdown(f'<div class="view-insight">Segment {escape(big["label"])} holds {big["entities"]:.0f}% of the '
                            f'{escape(word)} but {big["value"]:.0f}% of {escape(v)}.</div>', unsafe_allow_html=True)
            else:
                st.caption("No total (value, orders or units) to split by segment.")

    # ---- profiles
    section("Segment profiles", "Average per entity in each segment, against everyone")
    rows = []
    for c in prof.index:
        row = {"Segment": f"{c}. {res.names.get(c, '')}", word.capitalize(): int(prof.loc[c, "entities"])}
        for f in res.features:
            row[res.labels[f]] = _fmt(float(prof.loc[c, f]), res.units.get(f, "number"))
        rows.append(row)
    row = {"Segment": "All", word.capitalize(): n}
    for f in res.features:
        row[res.labels[f]] = _fmt(float(res.overall[f]), res.units.get(f, "number"))
    rows.append(row)
    st.dataframe(pd.DataFrame(rows), hide_index=True,
                 column_config={word.capitalize(): st.column_config.NumberColumn(format="localized")})

    # ---- explore
    section("Explore", "Each dot is one entity; diamonds are segment averages")
    feats = list(sep.index) + [f for f in res.features if f not in sep.index]
    # default pair: the strongest separator, and the next strongest that is not a near-copy of it
    second = 1 if len(feats) > 1 else 0
    if len(feats) > 2:
        corr = res.assignments[feats].corr().abs()
        second = next((i for i, f in enumerate(feats[1:], 1) if corr.loc[feats[0], f] < 0.8), 1)
    with st.container(border=True, key="seg_scatter"):
        x1, x2 = st.columns(2)
        fx = x1.selectbox("Horizontal", feats, format_func=res.labels.get, key="seg_x")
        fy = x2.selectbox("Vertical", feats, index=second, format_func=res.labels.get, key="seg_y")
        fig = charts.segment_scatter(res.assignments, fx, fy, "cluster", f"{res.labels[fy]} vs {res.labels[fx]}",
                                     res.labels[fx], res.labels[fy], names=res.names, entity=entity)
        if fig is not None:
            show_figure(fig)

    t1, t2, t3 = st.tabs([f"{word.capitalize()} and their segment", "Choosing the number of segments", "Measures used"])
    with t1:
        out = res.assignments.rename(columns=res.labels).copy()
        out.insert(2, "Segment name", out["cluster"].map(res.names))
        out = out.rename(columns={"cluster": "Segment"})
        st.dataframe(out.head(DEFAULT_ANALYTICS.preview_rows).round(2), hide_index=True)
        st.download_button("Download all as CSV", out.to_csv(index=False).encode("utf-8"),
                           file_name=f"segments_{entity}.csv".replace(" ", "_"), mime="text/csv", key="seg_dl")
        st.caption(f"Showing up to {DEFAULT_ANALYTICS.preview_rows:,} of {len(out):,} rows; the download has all of them.")
    with t2:
        sc = res.scores
        fig = charts.line_chart(sc, "k", "silhouette", "Fit for each number of segments (higher is clearer)", "Fit (silhouette)",
                                "Number of segments")
        if fig is not None:
            fig.add_vline(x=res.k, line=dict(color=charts.MUTED, width=1, dash="dash"))
            show_figure(fig)
        show = sc.rename(columns={"k": "Segments", "inertia": "Within-segment spread", "silhouette": "Fit",
                                  "min_share": "Smallest segment"}).copy()
        show["Smallest segment"] = show["Smallest segment"] * 100
        st.dataframe(show, hide_index=True, column_config={
            "Fit": st.column_config.NumberColumn(format="%.3f"),
            "Within-segment spread": st.column_config.NumberColumn(format="%.0f"),
            "Smallest segment": st.column_config.NumberColumn(format="%.1f%%")})
        st.caption(f"{res.k} segments were {'chosen automatically' if res.k_mode == 'automatic' else 'chosen by you'}. "
                   f"Automatic keeps the best fit among splits whose smallest segment holds at least "
                   f"{DEFAULT_ANALYTICS.seg_min_cluster_share:.0%} of the {word}. {res.quality}")
    with t3:
        built = pd.DataFrame({"Measure": [table.labels[c] for c in table.frame.columns],
                              "Used": ["Yes" if c in res.features else "No" for c in table.frame.columns]})
        st.dataframe(built, hide_index=True)
        if table.excluded_features:
            st.caption("Left out: " + "; ".join(f"{k.split('_', 1)[-1] if k.startswith(('sum_', 'mean_')) else k} ({v})"
                                                for k, v in table.excluded_features.items()))
        st.caption(f"Scaling: {SCALERS[res.scaler]}. Missing values are filled with the median of the measure. " + NOTE)

"""Dashboard: KPIs, suggested views and an explorer (trend, breakdown, distribution, relationships).

Everything shown is computed from the active (cleaned, if applied) dataset using the roles from the Column roles page.
KPI cards carry the KPI recomputed on each complete period (sparkline and change), so a trend always means exactly
what the KPI means. Nothing is displayed for an empty dataset.
"""
from __future__ import annotations

from dataclasses import replace
from html import escape
from urllib.parse import quote

import pandas as pd
import streamlit as st

from analytics.correlation import correlation
from analytics.data_shifts import describe_shifts, detect_shifts
from analytics.descriptive import (
    FREQ_LABELS,
    aggregate_time,
    analysable_numeric_columns,
    categorical_summary,
    date_summary,
    datetime_series,
    groupable_columns,
    grouped_summary,
    numeric_summary,
)
from analytics.formatting import format_value
from analytics.kpi_relevance import prioritise
from analytics.kpi_trend import KpiHistory, kpi_history, lower_is_better
from analytics.kpis import DataFilter, KPIResult, _Ctx, _category, _order_id, compute_kpis, primary_column, primary_date_column
from analytics.primary_columns import PICKABLE, primary_candidates, with_primary
from analytics.supply_chain_views import SuggestedView, suggest_views
from config.analytics_config import DEFAULT_ANALYTICS
from core.logging_config import get_logger
from semantic.column_lexicon import DOMAIN_LABELS, token_set
from ui import state
from ui.analytics_state import cached, show_figure
from ui.filters import render_filter_bar
from ui.layout import badge, empty_state, no_dataset_notice, page_title, section
from ui.shared_results import kpi_source
from visualization import charts

log = get_logger("ui.dashboard")
MAX_FILTER_VALUES = 60
HEADLINE = 8                     # cards in the key-metrics grid (two rows of four)
_AGG = {"Sum": "sum", "Average": "mean", "Median": "median", "Row count": "count"}
_FREQ = {"Month": "M", "Week": "W", "Day": "D"}
# A change in these is neither good nor bad on its own (more stock is not better), so it is shown in grey.
NEUTRAL_KEYS = frozenset({"stock_units", "stock_value", "days_of_cover", "row_count", "inventory_turnover",
                          "supplier_top_share", "supplier_top3_share"})
# Columns to break supply-chain data down by, best first: how it ships, where it goes, where it comes from.
SC_GROUP_ORDER = ({"mode", "carrier", "method"}, {"market", "region"}, {"warehouse", "supplier", "vendor"},
                  {"shipping", "delivery", "country"})
SC_GROUP_SKIP = {"status", "risk", "flag", "id"}


def _safe(section: str, fn) -> None:
    """A failure in one section must not take down the rest of the page."""
    try:
        fn()
    except Exception:
        log.exception("Dashboard section failed: %s", section)
        st.warning(f"The {section} section could not be built for this dataset.")


_section = section


# ---- header ---------------------------------------------------------------------------------------
def _coverage(df: pd.DataFrame, date_col: str | None) -> str:
    if not date_col:
        return ""
    d = datetime_series(df[date_col]).dropna()
    return "" if d.empty else f"{d.min():%d %b %Y} – {d.max():%d %b %Y}"


def _header(ds, df, fdf, model, date_col, flt) -> None:
    page_title("Dashboard", "KPIs and charts computed from " + ds.filename)
    rows = f"{len(fdf):,} of {len(df):,} rows" if flt.active else f"{len(df):,} rows"
    parts = [escape(ds.filename), rows]
    cov = _coverage(fdf, date_col)
    if cov:
        parts.append(f"{escape(cov)} <span class='dash-faint'>({escape(date_col)})</span>")
    dom = model.domain.domain
    tags = badge(DOMAIN_LABELS.get(dom, dom), "accent" if model.is_supply_chain else "neutral")
    if flt.active:
        tags += " " + badge(f"{len(flt.describe())} filter(s) on", "warn")
    sep = '<span class="dash-sep">·</span>'
    st.markdown(f'<div class="dash-meta">{sep.join(f"<span>{p}</span>" for p in parts)} {tags}</div>', unsafe_allow_html=True)


def _settings(df, model, config):
    """Comparison period and primary-column choices, tucked into one popover."""
    with st.popover("Settings", icon=":material/tune:"):
        st.markdown("**Comparison**")
        freq_label = st.selectbox("Cards compare the latest complete", list(_FREQ), key="dash_growth_freq",
                                  help="Sparklines and changes on the KPI cards use complete periods of this length.")
        cmp_label = st.selectbox("Growth KPIs compare it with", ["Previous period", "Same period last year"], key="dash_cmp")
        config = replace(config, comparison="year_over_year" if cmp_label.startswith("Same") else "previous_period",
                         time_freq=_FREQ[freq_label])
        _safe("primary columns", lambda: _primary_picker(df, model))
    return config


def _primary_picker(df, model) -> None:
    """Let the user choose which column leads when several share a role (e.g. duplicate sales columns)."""
    cands = primary_candidates(df, model)
    if not cands:
        return
    st.markdown("**Primary columns**")
    st.caption("Several columns could play the same role. The first is used for KPIs and charts.")
    picks = st.session_state.setdefault("dash_primary_picks", {})
    for role, label in PICKABLE:
        if role not in cands:
            continue
        cur, options = cands[role]
        choice = st.selectbox(label, options, index=0, key=f"dash_primary_{role}",
                              help="Your choice is saved as a role override and changes the KPIs and charts.")
        if choice != cur:
            ov = state.get_semantic_overrides()
            state.set_semantic_overrides(with_primary(ov["roles"], picks.get(role), choice, role), ov["domain"])
            picks[role] = choice
            st.rerun()


def _shift_notice(fdf, model, date_col, freq) -> None:
    order = _order_id(_Ctx(fdf, model, DEFAULT_ANALYTICS, DataFilter()))
    text = cached("data_shifts", (date_col, order, freq), lambda: describe_shifts(detect_shifts(fdf, date_col, order, freq)))
    if text:
        st.markdown(f'<div class="dash-callout"><span class="dash-callout-t">Data check</span>{escape(text)}</div>',
                    unsafe_allow_html=True)


# ---- KPI cards ------------------------------------------------------------------------------------
def _spark(points: list[float]) -> str:
    """Sparkline as two CSS-masked layers (soft area, then line) so it takes the theme's accent colour.
    Streamlit's HTML sanitizer removes inline <svg>, but a data-URI mask in a style attribute is kept."""
    if len(points) < 2:
        return ""
    lo, hi = min(points), max(points)
    span = (hi - lo) or 1.0
    w, h, pad = 200.0, 40.0, 3.0
    xy = [(i * w / (len(points) - 1), pad + (h - 2 * pad) * (1 - (v - lo) / span)) for i, v in enumerate(points)]
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in xy)
    head = f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {w:.0f} {h:.0f}' preserveAspectRatio='none'>"
    area_svg = head + f"<polygon points='0,{h:.0f} {line} {w:.0f},{h:.0f}'/></svg>"
    line_svg = head + (f"<polyline points='{line}' fill='none' stroke='black' stroke-width='1.8' "
                       "vector-effect='non-scaling-stroke' stroke-linejoin='round' stroke-linecap='round'/></svg>")

    def layer(cls: str, svg: str) -> str:
        url = "data:image/svg+xml;utf8," + quote(svg)
        return f'<div class="{cls}" style="-webkit-mask-image:url(&quot;{url}&quot;);mask-image:url(&quot;{url}&quot;)"></div>'
    return f'<div class="kpi-spark" aria-hidden="true">{layer("kpi-area", area_svg)}{layer("kpi-line", line_svg)}</div>'


def _period_name(label: str) -> str:
    try:
        p = pd.Period(label)
        return f"{p.start_time:%b %Y}" if p.freqstr.startswith("M") else (
            f"week of {p.start_time:%d %b}" if p.freqstr.startswith("W") else f"{p.start_time:%d %b %Y}")
    except Exception:
        return label


def _delta_html(r: KPIResult, h: KpiHistory) -> str:
    last, prev = _period_name(h.periods[-1]), _period_name(h.periods[-2])
    ctx = f'<span class="kpi-vs">{escape(last)} vs {escape(prev)}</span>'
    if h.flat:
        return f'<div class="kpi-delta"><span class="chip flat">No change</span>{ctx}</div>'
    up = h.change > 0
    if r.key in NEUTRAL_KEYS:
        tone = "flat"
    else:
        good = (not up) if lower_is_better(r.key) else up
        tone = "good" if good else "bad"
    txt = f"{abs(h.change):,.1f} pts" if h.change_unit == "pts" else f"{abs(h.change):,.1f}%"
    return f'<div class="kpi-delta"><span class="chip {tone}">{"▲" if up else "▼"} {txt}</span>{ctx}</div>'


def _foot(r: KPIResult, h: KpiHistory | None) -> str:
    if h is not None:
        return f"{h.period_label.capitalize()}ly values, last {len(h.points)} complete {h.period_label}s".replace("Dayly", "Daily")
    if "(latest date)" in r.name:
        return "Latest date in the data"
    if "(all dates)" in r.name:
        return "Across every date"
    return "Whole selection"


def _card(r: KPIResult, h: KpiHistory | None) -> str:
    tip = r.formula + ("\n\n" + "\n".join(r.limitations[:3]) if r.limitations else "")
    name = r.name.replace(" (latest date)", "").replace(" (all dates)", "")
    body = _delta_html(r, h) + _spark(h.points) if h is not None else '<div class="kpi-delta"></div><div class="kpi-spark-empty"></div>'
    return (f'<div class="kpi" title="{escape(tip)}"><div class="kpi-label">{escape(name)}</div>'
            f'<div class="kpi-value">{escape(format_value(r.value, r.unit))}</div>{body}'
            f'<div class="kpi-foot">{escape(_foot(r, h))}</div></div>')


def _mini(r: KPIResult) -> str:
    tip = r.formula + ("\n\n" + "\n".join(r.limitations[:3]) if r.limitations else "")
    return (f'<div class="kpi-mini" title="{escape(tip)}"><div class="kpi-label">{escape(r.name)}</div>'
            f'<div class="kpi-mini-v">{escape(format_value(r.value, r.unit))}</div></div>')


def _kpi_section(df, fdf, model, config, flt, date_col) -> None:
    if flt.active:
        report = cached("kpis", (config, flt.key()), lambda: compute_kpis(df, model, config, flt))
    else:                                    # same cache entry as the Insights and Reports pages: computed once for all three
        status, report = kpi_source(df, model, config, "The KPIs could not be calculated.")()
        if status != "ok":
            st.warning(str(report))
            return
    if not report.results:
        empty_state("Nothing to show", report.notes[0] if report.notes else "No data.")
        return
    display = prioritise(report, HEADLINE)
    shown = list(display.headline)
    freq = config.time_freq if config.time_freq in ("M", "W", "D") else "M"
    hist = cached("kpi_history", (tuple(r.key for r in shown), date_col, freq, flt.key()),
                  lambda: kpi_history(fdf, model, config, date_col, freq, tuple(r.key for r in shown)))
    word = {"M": "month", "W": "week", "D": "day"}[freq]
    _section("Key metrics", f"Cards compare the latest complete {word} with the one before; hover a card for its formula")
    st.html('<div class="kpi-grid">' + "".join(_card(r, hist.get(r.key)) for r in shown) + "</div>")

    more = [r for r in display.more if r.key != "row_count"]
    if more:
        groups: dict[str, list[KPIResult]] = {}
        for r in more:
            groups.setdefault(r.group, []).append(r)
        html = "".join(f'<div class="kpi-group"><div class="kpi-group-t">{escape(g)}</div>'
                       f'<div class="kpi-mini-grid">{"".join(_mini(r) for r in rs)}</div></div>'
                       for g, rs in groups.items())
        _section("More metrics")
        st.html(f'<div class="kpi-groups">{html}</div>')

    everything = shown + more
    n_rel, n_other = len(display.unavailable_relevant), len(display.unavailable_other)
    with st.expander("How these numbers are calculated"
                     + (f" · {n_rel} relevant KPI(s) could not be calculated" if n_rel else "")):
        tabs = st.tabs(["Formulas and columns", f"Not calculated ({n_rel})", f"Not applicable ({n_other})", "Notes"])
        with tabs[0]:
            st.dataframe(pd.DataFrame([{"KPI": r.name, "Value": format_value(r.value, r.unit), "Formula": r.formula,
                                        "Columns": ", ".join(r.columns), "Filters": "; ".join(r.filters) or "none",
                                        "Limitations": " ".join(r.limitations)} for r in everything]), hide_index=True)
        for tab, group in ((tabs[1], display.unavailable_relevant), (tabs[2], display.unavailable_other)):
            with tab:
                if group:
                    st.dataframe(pd.DataFrame([{"KPI": r.name, "Status": r.status.replace("_", " "), "Reason": r.reason}
                                               for r in group]), hide_index=True)
                else:
                    st.caption("None.")
        with tabs[3]:
            notes = list(report.notes) + ["Changes on the cards compare complete periods only; a partly covered first "
                                          "or last period is never used. Rates change in percentage points (pts)."]
            for n in notes:
                st.caption(n)


# ---- suggested views ------------------------------------------------------------------------------
def _view_figure(v: SuggestedView):
    """Turn a SuggestedView into a Plotly figure (None when nothing is plottable)."""
    if v.kind == "histogram":
        d = v.data["days"]
        return charts.histogram(d, v.title, "Days", marks={"Median": float(d.median()), "90th percentile": float(d.quantile(0.9))})
    if v.kind == "bar":
        return charts.bar_chart(v.data, "label", v.y[0], v.title, "Percent" if v.unit == "%" else v.y[0],
                                unit=v.unit, ref=v.ref, ref_label="Overall")
    if v.kind == "line":
        return charts.view_line(v.data, v.x, v.y[0], v.title, "Percent of items" if v.unit == "%" else v.y[0],
                                ref=v.ref, percent=v.unit == "%")
    if v.kind == "lines":
        return charts.multi_line_chart(v.data, v.x, v.y, v.title)
    return None


def _view_card(v: SuggestedView) -> None:
    with st.container(border=True, key=f"view_{v.key}"):
        fig = _view_figure(v)
        if fig is None:
            st.caption(f"{v.title}: nothing to plot.")
        else:
            show_figure(fig)
        st.markdown(f'<div class="view-insight">{escape(v.caption)}</div>', unsafe_allow_html=True)


def _views_section(df, model, config, flt) -> None:
    views = cached("suggested_views_v2", (config, flt.key()),
                   lambda: suggest_views(df, model, config, flt, max_views=4, skip=("trend",)))
    if not views:
        return
    _section("Supply-chain views" if model.is_supply_chain else "Suggested views",
             "Chosen from the columns in your data")
    pairs = [views[i:i + 2] for i in range(0, len(views), 2)]
    for pair in pairs:
        if len(pair) == 1:
            _view_card(pair[0])
            continue
        cols = st.columns(2, gap="medium")
        for col, v in zip(cols, pair):
            with col:
                _view_card(v)


# ---- explore tabs ---------------------------------------------------------------------------------
def _trend_section(df, profile, model, date_col, flt) -> None:
    if date_col is None:
        st.caption("No usable date column, so there is no trend. Set a Date / time role on the Column roles page.")
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
        st.caption(f"Hollow markers are periods only partly covered by the data (first or last {FREQ_LABELS[freq]}). "
                   "They are excluded from growth figures.")
    d = date_summary(df, date_col)
    if d.missing_periods:
        st.caption(f"The date column has {d.missing_periods} empty period(s) between {d.min.date()} and {d.max.date()} "
                   f"(first gap starts {d.gaps[0][0]}).")


def _default_group(df, model, groups: list[str]) -> str:
    """Supply chain: how it ships or where it goes (shipping mode, market). Otherwise the main category column."""
    if model.is_supply_chain:
        for words in SC_GROUP_ORDER:
            hit = [g for g in groups if token_set(g) & words and not token_set(g) & SC_GROUP_SKIP
                   and df[g].nunique(dropna=True) > 2]
            if hit:
                return hit[0]
    cat, _ = _category(_Ctx(df, model, DEFAULT_ANALYTICS, DataFilter()))
    if cat in groups:
        return cat
    return next((c for c in model.columns_with_role("category") if c in groups), groups[0])


def _breakdown_section(df, profile, model, config, flt) -> None:
    groups = groupable_columns(df, profile, model, config)
    nums = analysable_numeric_columns(df, profile, model)
    if not groups:
        st.caption("No categorical column with 2 to "
                   f"{config.max_group_cardinality} distinct values was found.")
        return
    c1, c2, c3 = st.columns(3)
    by = c1.selectbox("Group by", groups, index=groups.index(_default_group(df, model, groups)), key="dash_b_by")
    agg_label = c2.selectbox("Measure", list(_AGG), key="dash_b_agg")
    rev = primary_column(df, model, "revenue")
    value = c3.selectbox("Column", nums, index=nums.index(rev) if rev in nums else 0, key="dash_b_val",
                         disabled=agg_label == "Row count") if nums else None
    agg = _AGG[agg_label]
    t = cached("grouped", (by, value, agg, config, flt.key()), lambda: grouped_summary(df, by, value, agg, config))
    left, right = st.columns([3, 2])
    with left:
        title = f"{agg_label}{'' if agg == 'count' else ' of ' + str(value)} by {by}"
        fig = (charts.share_chart(t, "group", "value", title) if agg in ("sum", "count") and len(t) <= 6
               else charts.bar_chart(t, "group", "value", title))
        if fig is None:
            st.caption("Nothing to plot for this selection.")
        else:
            show_figure(fig)
    with right:
        table = t.rename(columns={"group": by, "value": "Value", "rows": "Rows", "share_pct": "Share %"}).copy()
        table["Value"] = table["Value"].round(0 if table["Value"].abs().max() >= 1000 else 2)
        cfg = {"Value": st.column_config.NumberColumn(format="localized"),
               "Rows": st.column_config.NumberColumn(format="localized")}
        if "Share %" in table:
            cfg["Share %"] = st.column_config.NumberColumn(format="%.1f%%")
        st.dataframe(table, hide_index=True, column_config=cfg)
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
    st.dataframe(summary.rename_axis("Column").reset_index().round(2), hide_index=True)
    cats = [c for c in groupable_columns(df, profile, model)]
    if cats:
        pick = st.selectbox("Most frequent values of", cats, key="dash_d_cat")
        s = categorical_summary(df, [pick])[pick]
        st.dataframe(s.top.rename(columns={"value": pick, "count": "Rows", "pct": "Share %"}).round(1), hide_index=True,
                     column_config={"Share %": st.column_config.NumberColumn(format="%.1f%%"),
                                    "Rows": st.column_config.NumberColumn(format="localized")})
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


# ---- page -----------------------------------------------------------------------------------------
def render_dashboard_page() -> None:
    result = state.get_result()
    if result is None:
        page_title("Dashboard")
        no_dataset_notice()
        return
    ds = result.dataset
    try:
        df, profile, model = state.get_analysis_df(), state.get_analysis_profile(), state.get_semantic_model()
    except Exception:
        log.exception("Dashboard inputs unavailable")
        page_title("Dashboard")
        st.error("The dataset could not be prepared for analysis. Check the Data quality and Column roles pages.")
        return
    if df is None or profile is None or model is None or len(df) == 0:
        page_title("Dashboard")
        empty_state("Nothing to analyse", "The active dataset has no rows.")
        return

    date_col = primary_date_column(df, model)
    head, tools = st.columns([5, 1], vertical_alignment="bottom")
    with tools:
        config = _settings(df, model, DEFAULT_ANALYTICS)
    flt = state.get_filter()
    fdf = cached("filtered_df", flt.key(), lambda: flt.apply(df)) if flt.active else df
    with head:
        _header(ds, df, fdf, model, date_col, flt)
    render_filter_bar(df, model)
    flt = state.get_filter()                       # the bar may have changed it in this run
    fdf = cached("filtered_df", flt.key(), lambda: flt.apply(df)) if flt.active else df
    if len(fdf) == 0:
        empty_state("No rows match the filters", "Widen the date range or remove a filter.")
        return
    freq = config.time_freq if config.time_freq in ("M", "W", "D") else "M"
    _safe("data check", lambda: _shift_notice(fdf, model, date_col, freq))
    _safe("KPI", lambda: _kpi_section(df, fdf, model, config, flt, date_col))
    _safe("suggested views", lambda: _views_section(df, model, config, flt))
    _section("Explore", "Pick any column to chart")
    t1, t2, t3, t4 = st.tabs(["Trend", "Breakdown", "Distribution", "Relationships"])
    with t1:
        _safe("trend", lambda: _trend_section(fdf, profile, model, date_col, flt))
    with t2:
        _safe("breakdown", lambda: _breakdown_section(fdf, profile, model, config, flt))
    with t3:
        _safe("distribution", lambda: _distribution_section(fdf, profile, model, flt))
    with t4:
        _safe("correlation", lambda: _correlation_section(fdf, profile, model, config, flt))

"""Anomaly page: unusual records and unusual periods. Nothing is removed or changed.

Records: several methods screen the chosen columns; a record matters more when more methods agree, so the page leads
with agreement (exact counts over every flag, not only the strongest shown). Periods: each period against a robust
expected range; supply-chain KPIs (on-time rate, shipping time, cancellations) can be checked per period too.
Both checks run as soon as the page opens and are cached per setting and filter.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from analytics.anomaly_detection import (METHOD_LABELS, METHODS, NOTE, context_columns, default_columns, eligible_columns,
                                         feature_comparison, run_anomaly_detection, series_anomalies)
from analytics.data_shifts import detect_shifts
from analytics.derived_targets import derived_targets, frame_with
from analytics.descriptive import FREQ_LABELS, analysable_numeric_columns, numeric_series
from analytics.forecasting import suggest_setup
from analytics.formatting import format_value
from analytics.kpis import DataFilter, _Ctx, _order_id
from config.analytics_config import DEFAULT_ANALYTICS
from core.exceptions import BIError, InsufficientDataError
from core.logging_config import get_logger
from semantic.column_lexicon import DOMAIN_LABELS
from ui import state
from ui.analytics_state import base_key, cached, show_figure
from ui.filters import filtered, no_rows_notice, render_filter_bar
from ui.layout import badge, callout, empty_state, lede, meta_line, no_dataset_notice, page_title, section, stat_cards
from ui.progress_bar import ProgressBar
from visualization import charts

log = get_logger("ui.anomaly")
_SENS = {1: "Strict", 2: "Cautious", 3: "Balanced", 4: "Broad", 5: "Sensitive"}
_SENS_HELP = "Strict flags only extreme values; Sensitive also flags milder ones. Balanced suits most data."
_FREQ = {"Day": "D", "Week": "W", "Month": "M"}
_ROW_COUNT = "Rows (count per period)"
STRONG = 3                         # methods that must agree for "high agreement"
MAX_REVIEW = 200                   # records listed for review


def _safe(name, params, fn):
    """Cached; failures become ('insufficient'|'error', message) so one check never breaks the page."""
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
    page_title("Anomaly detection", "Records and periods that stand out, and how sure the methods are")
    try:
        df_all, profile, model = state.get_analysis_df(), state.get_analysis_profile(), state.get_semantic_model()
    except Exception:
        log.exception("Anomaly inputs unavailable")
        st.error("The dataset could not be prepared for analysis. Check the Data quality and Column roles pages.")
        return
    if df_all is None or profile is None or model is None or len(df_all) == 0:
        empty_state("Nothing to examine", "The active dataset has no rows.")
        return
    df = filtered(df_all)
    parts = [ds.filename + (" (synthetic sample)" if ds.is_sample else ""),
             f"{len(df):,} of {len(df_all):,} rows" if state.get_filter().active else f"{len(df):,} rows"]
    dom = model.domain.domain
    meta_line(parts, badge(DOMAIN_LABELS.get(dom, dom), "accent" if model.is_supply_chain else "neutral"))
    render_filter_bar(df_all, model)
    df = filtered(df_all)
    if len(df) == 0:
        no_rows_notice()
        return
    head, sens_col = st.columns([3, 2], vertical_alignment="center")
    head.caption(NOTE + " Flagged records are never deleted or changed.")
    sens = sens_col.select_slider("Sensitivity", options=list(_SENS), value=DEFAULT_ANALYTICS.anomaly_sensitivity,
                                  format_func=_SENS.get, key="an_sens", help=_SENS_HELP)
    t1, t2 = st.tabs(["Unusual records", "Unusual periods"])
    with t1:
        _records_tab(df, profile, model, sens)
    with t2:
        _periods_tab(df, profile, model, sens)


def screen(df, chosen, methods, sens):
    """Screen records the way the page does (same cache entry), for the page and the Home cards."""
    def run():
        with ProgressBar("Screening for unusual values") as bar:
            return run_anomaly_detection(df, list(chosen), tuple(methods), sens, progress=bar)
    return _safe("anomaly_values_v2", (tuple(chosen), tuple(methods), sens), run)


# ---- records ---------------------------------------------------------------------------------------
def _records_tab(df, profile, model, sens) -> None:
    cols, skipped = eligible_columns(df, profile, model)
    if not cols:
        empty_state("No suitable numeric columns", "Anomaly screening needs measure columns with enough distinct values.")
        return
    defaults = cached("an_default_cols", tuple(cols), lambda: default_columns(df, cols))
    with st.container(border=True, key="an_setup"):
        c1, c2 = st.columns([4, 1], vertical_alignment="bottom")
        chosen = c1.multiselect("Columns to screen", cols, default=defaults, key="an_cols",
                                help="Chosen by role (money, volumes, times first); copies of a column already chosen "
                                     "are left out so one record is not counted twice.")
        with c2.popover("Methods", icon=":material/tune:", width="stretch"):
            methods = st.multiselect("Methods", list(METHODS), default=list(METHODS), format_func=METHOD_LABELS.get,
                                     key="an_methods", label_visibility="collapsed")
            st.caption("Robust z-score and IQR fences judge each column on its own; z-score only runs on roughly "
                       "bell-shaped columns; Isolation Forest looks at the columns together.")
    if not chosen or not methods:
        st.caption("Choose at least one column and one method.")
        return
    st.session_state["an_params"] = (base_key(), tuple(chosen), tuple(methods), sens)
    status, rep = screen(df, chosen, methods, sens)
    if status != "ok":
        empty_state("Detection not possible", rep) if status == "insufficient" else st.warning(rep)
        return
    _records_result(df, model, rep, chosen, skipped)


def _records_result(df, model, rep, chosen, skipped) -> None:
    n = max(rep.examined_rows, 1)
    ran = max(rep.methods_run, 1)
    need = min(STRONG, ran)
    strong = rep.agreed(need)
    anyflag = len(rep.votes)
    cv = rep.column_votes.copy()
    cv["share"] = 100 * cv["agreed"] / n
    top = cv.sort_values("agreed", ascending=False).iloc[0] if len(cv) and cv["agreed"].max() > 0 else None

    if anyflag == 0:
        lede(f"Nothing stands out: none of the {n:,} records was flagged by any method at this sensitivity.")
    else:
        lede(f"{len(strong):,} of {n:,} records ({100 * len(strong) / n:.1f}%) are unusual according to "
             f"{'all ' + str(ran) if need == ran else f'at least {need} of the {ran}'} methods that ran; these are the ones "
             f"to review first. Another {anyflag - len(strong):,} were flagged by fewer methods."
             + (f" Most agreed flags fall in {top['feature']}." if top is not None else ""))
    stat_cards([
        {"label": "Records screened", "value": f"{n:,}", "sub": f"{len(chosen)} column(s), {ran} method(s)",
         "foot": "Rows with at least one value in the chosen columns"},
        {"label": "High agreement", "value": f"{len(strong):,}",
         "chip": (f"{100 * len(strong) / n:.1f}% of records", "bad" if len(strong) / n > 0.05 else "flat"),
         "foot": f"Flagged by {need}+ methods: review these first"},
        {"label": "Flagged by any method", "value": f"{anyflag:,}", "chip": (f"{100 * anyflag / n:.1f}%", "flat"),
         "foot": "Includes records only one method found unusual"},
        {"label": "Column with most agreed flags", "value": "None" if top is None else str(top["feature"]),
         "sub": "" if top is None else f"{int(top['agreed']):,} records ({top['share']:.1f}%)",
         "foot": "Records flagged in this column by 2+ methods"},
    ])
    # "showing the N strongest" notes no longer matter (agreement is counted over every flag); skew notes do
    skewed = list(dict.fromkeys(w.split(" (")[0] for w in rep.warnings if "probably skewed" in w))
    other = [w for w in rep.warnings if "showing the" not in w and "probably skewed" not in w]
    if skewed:
        other.insert(0, f"{', '.join(skewed)} {'has' if len(skewed) == 1 else 'have'} a long tail: one method flags more than "
                        "10% of values there. Those are usually ordinary large or small values, not errors; the "
                        "high-agreement count above is the better guide.")
    if other:
        callout("Good to know", " ".join(other), "info")
    if anyflag == 0:
        return

    left, right = st.columns([2, 3], gap="medium")
    with left:
        with st.container(border=True, key="an_cols_chart"):
            t = cv.rename(columns={"feature": "label", "share": "value"})[["label", "value"]].sort_values("value", ascending=False)
            fig = charts.bar_chart(t, "label", "value", "Share of records with agreed flags", "Percent", unit="%")
            if fig is not None:
                show_figure(fig)
            st.markdown('<div class="view-insight">A flag counts here when at least two of the single-column methods '
                        'agree on that column. A column with many flags is often just skewed (a long tail of large orders), '
                        'not wrong.</div>', unsafe_allow_html=True)
    with right:
        _review_table(df, model, strong if len(strong) else rep.votes, need if len(strong) else 1, ran)

    with st.expander("Results per method and column"):
        s = rep.summary.rename(columns={"method": "Method", "feature": "Column", "examined": "Values checked",
                                        "flagged": "Flagged", "share": "Share", "threshold": "Threshold",
                                        "status": "Status", "note": "Note"}).copy()
        s["Method"] = s["Method"].map(METHOD_LABELS)
        s["Share"] = s["Share"] * 100
        s["Status"] = s["Status"].str.capitalize()
        st.dataframe(s, hide_index=True, column_config={"Share": st.column_config.NumberColumn(format="%.2f%%"),
                                                        "Values checked": st.column_config.NumberColumn(format="localized"),
                                                        "Flagged": st.column_config.NumberColumn(format="localized")})
    if skipped:
        with st.expander(f"Numeric columns not screened ({len(skipped)})"):
            st.dataframe(pd.DataFrame({"Column": list(skipped), "Reason": list(skipped.values())}), hide_index=True)


def _review_table(df, model, votes, need, ran) -> None:
    section("Records to review", f"Most agreement first; up to {MAX_REVIEW} shown")
    v = votes.head(MAX_REVIEW).copy()
    feats = v["features"].str.split(", ")
    used = {c for fs in feats for c in fs if c in df.columns}
    stats = {}
    for c in used:
        x = numeric_series(df[c])
        med = float(x.median())
        stats[c] = (x, med, float((x - med).abs().median()))
    rows, short = [], []
    for r, fs in zip(v["row"], feats):
        best, best_z = None, -1.0                     # the value furthest from typical, in robust deviations
        for c in fs:
            if c not in stats:
                continue
            x, med, mad = stats[c]
            val = x.loc[r]
            val = val.iloc[0] if isinstance(val, pd.Series) else val
            if pd.isna(val):
                continue
            z = abs(0.6745 * (val - med) / mad) if mad else 0.0
            if z > best_z:
                best, best_z = (c, val, med), z
        rows.append(f"{best[0]}: {best[1]:,.2f} (typical {best[2]:,.2f})" if best else "Unusual mix of values")
        named = [c for c in fs if c in df.columns]
        short.append(", ".join(named[:2]) + (f" +{len(named) - 2} more" if len(named) > 2 else "") if named
                     else "several columns together")
    v["features"] = short
    table = pd.DataFrame({"Row": v["row"].to_numpy(), "Methods agreeing": [f"{m} of {ran}" for m in v["methods"]],
                          "Most unusual value": rows, "Unusual in": v["features"].to_numpy()})
    ctx = [c for c in context_columns(df, model) if c in df.columns]
    if ctx:
        extra = df.loc[v["row"].to_numpy(), ctx].reset_index(drop=True)
        table = pd.concat([table, extra], axis=1)
    st.dataframe(table, hide_index=True, height=360)
    pick = st.selectbox("Compare one record with typical values", list(v["row"].head(50)), key="an_row",
                        format_func=lambda r: f"Row {r}")
    fc = feature_comparison(df, pick, sorted(used, key=list(df.columns).index))
    if len(fc):
        fc = fc.rename(columns={"Feature": "Column", "Median": "Typical (median)", "Robust deviation": "How unusual"})
        st.dataframe(fc, hide_index=True, column_config={
            "Value": st.column_config.NumberColumn(format="%.2f"), "Typical (median)": st.column_config.NumberColumn(format="%.2f"),
            "How unusual": st.column_config.NumberColumn(format="%.1f", help="Robust deviations from the median; "
                                                                               "beyond about 3.5 is unusual")})


# ---- periods ---------------------------------------------------------------------------------------
def _periods_tab(df, profile, model, sens) -> None:
    dates = [c for c in df.columns if c in model.columns and model.role_of(c) == "datetime"]
    if not dates:
        empty_state("No date column", "This check needs a column with the Date / time role. Set one on the Column roles page.")
        return
    setup = suggest_setup(df, model)
    derived = cached("derived_targets", (), lambda: derived_targets(df, model))
    measures = list(derived) + [_ROW_COUNT] + analysable_numeric_columns(df, profile, model)
    default = setup["value_col"] if setup["value_col"] in measures else measures[0]
    with st.container(border=True, key="ap_setup"):
        c1, c2, c3 = st.columns([3, 1.2, 1], vertical_alignment="bottom")
        target = c1.selectbox("Measure", measures, index=measures.index(default), key="ap_target",
                              help="Items starting with KPI: are supply-chain measures worked out per period.")
        inv = {v: k for k, v in _FREQ.items()}
        freq = _FREQ[c2.selectbox("Period", list(_FREQ), index=list(_FREQ).index(inv[setup["freq"]]), key="ap_freq")]
        with c3.popover("Options", icon=":material/tune:", width="stretch"):
            date_col = st.selectbox("Date column", dates, key="ap_date",
                                    index=dates.index(setup["date_col"]) if setup["date_col"] in dates else 0)
            agg = "sum"
            if target not in derived and target != _ROW_COUNT:
                agg = {"Sum": "sum", "Average": "mean"}[st.selectbox("Combine rows in a period by", ["Sum", "Average"],
                                                                     key="ap_agg")]
    if target in derived:
        t = derived[target]
        frame, value_col, agg, unit = frame_with(df, t), target, t.agg, t.unit
    elif target == _ROW_COUNT:
        frame, value_col, agg, unit = df, None, "count", "count"
    else:
        frame, value_col, unit = df, target, "number"
    params = (date_col, target, freq, agg, sens)
    st.session_state["ap_params"] = (base_key(), *params)
    status, res = _safe("anomaly_periods_v2", params, lambda: series_anomalies(frame, date_col, value_col, freq, agg, sens))
    if status != "ok":
        empty_state("Check not possible", res) if status == "insufficient" else st.warning(res)
        return
    per = FREQ_LABELS[res.freq]
    name = target.replace("KPI: ", "").replace(" (%)", "").replace(" (days)", "")
    pct = unit == "percent"
    order = _order_id(_Ctx(df, model, DEFAULT_ANALYTICS, DataFilter()))
    shifts = cached("fc_shifts", (date_col, order, freq), lambda: detect_shifts(df, date_col, order, freq))
    s = res.series
    shift = next((x for x in sorted(shifts, key=lambda x: x.start) if s["period"].min() < x.start <= s["period"].max()), None)
    f = res.findings
    fmt = (lambda v: f"{v:,.1f}%") if pct else (lambda v: format_value(v, "days") if unit == "days" else f"{v:,.2f}"
                                               if abs(v) < 100 else f"{v:,.0f}")
    if f.empty:
        lede(f"No {per} stands out: all {len(s)} {per}s of {name} stay within their expected range.")
    else:
        hi, lo = int((f["direction"] == "high").sum()), int((f["direction"] == "low").sum())
        worst = f.iloc[0]
        wp = pd.Timestamp(worst["row"])
        when = f"{wp:%b %Y}" if res.freq == "M" else (f"the week of {wp:%d %b %Y}" if res.freq == "W" else f"{wp:%d %b %Y}")
        exp_s = s.set_index("period")["baseline"]
        gaps = [abs(float(v) - float(exp_s.get(pd.Timestamp(p), float("nan")))) for p, v in zip(f["row"], f["value"])]
        gaps = [g for g in gaps if g == g]
        size = ""
        if gaps:
            if pct:
                size = f" They differ from expected by {min(gaps):,.1f} to {max(gaps):,.1f} pts."
            else:
                rel = [abs(float(v) - float(exp_s.get(pd.Timestamp(p)))) / abs(float(exp_s.get(pd.Timestamp(p)))) * 100
                       for p, v in zip(f["row"], f["value"]) if float(exp_s.get(pd.Timestamp(p), 0) or 0)]
                if rel:
                    size = f" They differ from expected by {min(rel):,.0f}% to {max(rel):,.0f}%."
        lede(f"{len(f)} of {len(s)} {per}s fall outside the expected range of {name} ({hi} high, {lo} low). The most unusual "
             f"is {when}: {fmt(worst['value'])} against about {fmt(float(exp_s.get(wp)))} expected.{size}"
             + (" The series is very steady, so even small moves stand out; judge whether that size matters."
                if gaps and pct and max(gaps) < 3 else ""))
    if shift is not None:
        callout("Data check", f"The data changes shape from {shift.start:%b %Y} ({shift.change_text()}). A flag at that point may come from the change in the data "
                "rather than from a business event.")
    with st.container(border=True, key="ap_chart"):
        y_title = "Rows" if value_col is None else name
        fig = charts.anomaly_series_chart(s, f"{name} by {per}", y_title, percent=pct,
                                          shift_at=None if shift is None else shift.start)
        if fig is not None:
            show_figure(fig)
        st.markdown('<div class="view-insight">The shaded band is the expected range: the trend of nearby periods'
                    + (" plus the usual repeating pattern" if res.seasonal else "")
                    + f". A {per} outside it is flagged; widen or narrow the band with Sensitivity.</div>",
                    unsafe_allow_html=True)
    for w in res.warnings:
        callout("Good to know", w, "info")
    if f.empty:
        return
    section(f"Unusual {per}s", "Most unusual first")
    exp = s.set_index("period")["baseline"]
    rows = []
    for _, r in f.iterrows():
        p = pd.Timestamp(r["row"])
        e = float(exp.get(p, float("nan")))
        cause = ""
        if shift is not None and abs((p - shift.start).days) <= {"D": 1, "W": 7, "M": 31}[res.freq]:
            cause = "Same period as a change in the data"
        rows.append({"Period": p.date(), "Direction": "High" if r["direction"] == "high" else "Low",
                     "Actual": float(r["value"]), "Expected": e,
                     "Difference": (r["value"] - e) if pct else ((r["value"] - e) / abs(e) * 100 if e else None),
                     "How unusual": float(r["score"]), "Note": cause})
    out = pd.DataFrame(rows)
    big = not pct and unit != "days" and out["Actual"].abs().max() >= 100
    out[["Actual", "Expected"]] = out[["Actual", "Expected"]].round(0 if big else 2)
    num = "%.1f%%" if pct else ("localized" if big else "%.2f")
    st.dataframe(out, hide_index=True, column_config={
        "Actual": st.column_config.NumberColumn(format=num), "Expected": st.column_config.NumberColumn(format=num),
        "Difference": st.column_config.NumberColumn("Difference" + (" (pts)" if pct else " (%)"),
                                                    format="%+.1f" if pct else "%+.1f%%"),
        "How unusual": st.column_config.NumberColumn(format="%.1f", help="Robust deviations from the expected value")})

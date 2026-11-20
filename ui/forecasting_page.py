"""Forecasting page: pick a target and a period, validate models chronologically, show the forecast and what it means.

The forecast runs as soon as the page opens (with the suggested target) and whenever a setting changes; results are
cached per setting and filter. Supply-chain KPIs (orders, on-time rate, shipping time, cancellations) can be
forecast like columns: their values per period are the KPI engine's own definitions (analytics.derived_targets).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import streamlit as st

from analytics.data_shifts import detect_shifts
from analytics.derived_targets import derived_targets, frame_with
from analytics.descriptive import FREQ_LABELS, datetime_series
from analytics.forecasting import _MODELS, DEFAULT_HORIZON, forecast_series, suggest_setup, summarise_forecast, target_columns
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

log = get_logger("ui.forecasting")
_FREQ = {"Day": "D", "Week": "W", "Month": "M"}
_AGG = {"Sum": "sum", "Average": "mean", "Row count": "count"}
_ROW_COUNT = "Rows (count per period)"
_AMOUNT_ROLES = {"revenue", "cost", "profit", "balance", "price"}
_COUNT_ROLES = {"quantity"}


@dataclass(frozen=True)
class Target:
    label: str
    value_col: str | None          # column in the frame passed to the forecast (None = row count)
    agg: str
    unit: str                      # format_value unit
    bounds: tuple[float, float] | None = None
    lower_is_better: bool | None = None       # None: an increase is neither good nor bad
    derived: bool = False
    note: str = ""


def _targets(df, profile, model) -> dict[str, Target]:
    """Supply-chain KPIs first, then the row count, then numeric columns."""
    out: dict[str, Target] = {}
    for label, t in cached("derived_targets", (), lambda: derived_targets(df, model)).items():
        out[label] = Target(label, label, t.agg, t.unit, t.bounds, None if t.unit == "count" else t.lower_is_better,
                            True, t.note)
    out[_ROW_COUNT] = Target(_ROW_COUNT, None, "count", "count")
    for c in target_columns(df, profile, model):
        role = model.role_of(c) if c in model.columns else ""
        unit = "amount" if role in _AMOUNT_ROLES else "count" if role in _COUNT_ROLES else "number"
        out[c] = Target(c, c, "sum", unit)
    return out


def _fmt(v: float, unit: str) -> str:
    if unit == "count" and abs(v) < 100:
        return f"{v:,.1f}"            # a forecast of 4.6 orders a day is not 5
    return format_value(v, unit)


def _run(frame, date_col, t: Target, freq, horizon, model_key):
    """Cached; returns ('ok', result) or ('insufficient'|'error', message) so failures are cached too."""
    def compute():
        try:
            with ProgressBar("Forecasting") as bar:
                return "ok", forecast_series(frame, date_col, t.value_col, freq, t.agg, horizon, DEFAULT_ANALYTICS, model_key,
                                             progress=bar, bounds=t.bounds)
        except InsufficientDataError as exc:
            return "insufficient", exc.user_message
        except BIError as exc:
            return "error", exc.user_message
        except Exception:
            log.exception("Forecasting failed")
            return "error", "The forecast could not be built for this selection."
    return cached("forecast_v2", (date_col, t.label, t.agg, freq, horizon, model_key, DEFAULT_ANALYTICS), compute)


def forecast_for(df, profile, model, label: str, date_col: str, freq: str, horizon: int, model_key=None, agg=None):
    """Forecast one target the way the page does (same cache entry), for the page and the Home cards.
    `label` is a key of _targets (a column, the row count, or a KPI target). Returns (status, result or message)."""
    targets = _targets(df, profile, model)
    t = targets[label]
    if agg is not None and not t.derived and t.value_col is not None:
        t = Target(t.label, t.value_col, agg, t.unit)
    frame = frame_with(df, cached("derived_targets", (), lambda: derived_targets(df, model))[t.label]) if t.derived else df
    return _run(frame, date_col, t, freq, horizon, model_key), t


def _header(ds, df_all, df, model, date_col, flt_active) -> None:
    parts = [ds.filename + (" (synthetic sample)" if ds.is_sample else ""),
             f"{len(df):,} of {len(df_all):,} rows" if flt_active else f"{len(df):,} rows"]
    if date_col:
        d = datetime_series(df[date_col]).dropna()
        if len(d):
            parts.append(f"{d.min():%d %b %Y} – {d.max():%d %b %Y}")
    dom = model.domain.domain
    meta_line(parts, badge(DOMAIN_LABELS.get(dom, dom), "accent" if model.is_supply_chain else "neutral"))


def render_forecasting_page() -> None:
    result = state.get_result()
    if result is None:
        page_title("Forecasting", "Validated forecasts computed from your data.")
        no_dataset_notice()
        return
    ds = result.dataset
    page_title("Forecasting", "Validated against your own history before it is shown")
    try:
        df_all, profile, model = state.get_analysis_df(), state.get_analysis_profile(), state.get_semantic_model()
    except Exception:
        log.exception("Forecasting inputs unavailable")
        st.error("The dataset could not be prepared for analysis. Check the Data quality and Column roles pages.")
        return
    if df_all is None or profile is None or model is None or len(df_all) == 0:
        empty_state("Nothing to forecast", "The active dataset has no rows.")
        return
    setup = suggest_setup(df_all, model)
    _header(ds, df_all, filtered(df_all), model, setup["date_col"], state.get_filter().active)
    render_filter_bar(df_all, model)
    df = filtered(df_all)
    if len(df) == 0:
        no_rows_notice()
        return
    dates = [c for c in df.columns if c in model.columns and model.role_of(c) == "datetime"]
    if not dates:
        empty_state("No date column", "Forecasting needs a column with the Date / time role. Set one on the Column roles page.")
        return

    targets = _targets(df, profile, model)
    labels = list(targets)
    default = setup["value_col"] if setup["value_col"] in targets else labels[0]
    with st.container(border=True, key="fc_setup"):
        c1, c2, c3, c4 = st.columns([3, 1.2, 1.4, 1], vertical_alignment="bottom")
        label = c1.selectbox("What to forecast", labels, index=labels.index(default), key="fc_target",
                             help="Items starting with KPI: are supply-chain measures worked out per period, "
                                  "with the same definitions as the dashboard.")
        default_freq = {v: k for k, v in _FREQ.items()}[setup["freq"]]
        freq = _FREQ[c2.selectbox("Period", list(_FREQ), index=list(_FREQ).index(default_freq), key="fc_freq")]
        horizon = int(c3.number_input(f"{FREQ_LABELS[freq].capitalize()}s ahead", 1, DEFAULT_ANALYTICS.forecast_max_horizon,
                                      DEFAULT_HORIZON[freq], key=f"fc_h_{freq}"))
        with c4.popover("Options", icon=":material/tune:", width="stretch"):
            date_col = st.selectbox("Date column", dates, key="fc_date",
                                    index=dates.index(setup["date_col"]) if setup["date_col"] in dates else 0)
            agg = None
            if not targets[label].derived and targets[label].value_col is not None:
                agg = _AGG[st.selectbox("Combine rows in a period by", ["Sum", "Average"], key="fc_agg")]
            names = {"Automatic (best validated)": None, **{lab: key for key, lab, *_ in _MODELS}}
            model_key = names[st.selectbox("Model", list(names), key="fc_model",
                                           help="Automatic picks the simplest model that is not clearly worse in "
                                                "validation. A model you choose still shows its validation error.")]
    st.session_state["fc_params"] = (base_key(), date_col, label, freq, horizon, model_key)
    (status, r), t = forecast_for(df, profile, model, label, date_col, freq, horizon, model_key, agg)
    if status == "insufficient":
        empty_state("This data cannot support a forecast", r)
        return
    if status == "error":
        st.warning(r)
        return
    _show(r, t, df, model)


def _shift(df, model, r):
    order = _order_id(_Ctx(df, model, DEFAULT_ANALYTICS, DataFilter()))
    shifts = cached("fc_shifts", (r.date_col, order, r.freq), lambda: detect_shifts(df, r.date_col, order, r.freq))
    start = r.history["period"].iloc[0]
    inside = [s for s in shifts if s.start > start]
    return min(inside, key=lambda s: s.start) if inside else None


def _show(r, t: Target, df, model) -> None:
    sm = summarise_forecast(r)
    scores = {s.key: s for s in r.scores}
    chosen, base = scores[r.chosen], scores[r.baseline]
    per = FREQ_LABELS[r.freq]
    name = t.label.replace("KPI: ", "").replace(" (%)", "").replace(" (days)", "")
    pct = t.unit == "percent"
    unit_word = {"percent": "", "days": " days"}.get(t.unit, "")
    shift = _shift(df, model, r)

    # ---- the answer, in one sentence
    when = f"{sm.next_period:%d %b %Y}" if r.freq == "D" else (f"the week of {sm.next_period:%d %b %Y}" if r.freq == "W"
                                                               else f"{sm.next_period:%B %Y}")
    total = ""
    if sm.horizon_total is not None and r.horizon > 1:
        total = f", and {_fmt(sm.horizon_total, t.unit)} over the next {r.horizon} {per}s"
        if sm.change_pct is not None:
            total += f" ({sm.change_pct:+.1f}% vs the previous {r.horizon})"
    elif r.horizon > 1 and sm.change_pct is not None:
        total = f"; the next {r.horizon} {per}s average {_fmt(sm.horizon_mean, t.unit)}" + \
                (f" ({sm.horizon_mean - sm.previous_mean:+.1f} pts" if pct else f" ({sm.change_pct:+.1f}%") + \
                f" vs the previous {r.horizon})"
    lede(f"{name} is forecast at {_fmt(sm.next_value, t.unit)} for {when} (likely between "
         f"{_fmt(sm.next_lower, t.unit)} and {_fmt(sm.next_upper, t.unit)}){total}.")
    if shift is not None:
        after = int((r.history["period"] >= shift.start).sum())
        callout("Data check", f"The data changes shape from {shift.start:%b %Y} ({shift.change_text()}). The model learns from both sides of the change, and only "
                f"{after} {per}(s) come after it, so the forecast mainly carries the latest level forward. Check whether "
                "the source export changed before relying on it.")

    # ---- cards
    if sm.horizon_total is not None and r.horizon > 1:
        ahead_label, ahead_value = f"Next {r.horizon} {per}s, total", _fmt(sm.horizon_total, t.unit)
        prev_text = f"vs {_fmt(sm.previous_total, t.unit)} in the previous {r.horizon}"
    else:
        ahead_label, ahead_value = f"Next {r.horizon} {per}s, average", _fmt(sm.horizon_mean, t.unit)
        prev_text = f"vs {_fmt(sm.previous_mean, t.unit)} in the previous {r.horizon}"
    chg = None
    if sm.change_pct is not None:
        delta = sm.horizon_mean - sm.previous_mean if pct else sm.change_pct
        txt = f"{'▲' if delta > 0 else '▼'} {abs(delta):,.1f}{' pts' if pct else '%'}"
        flat = abs(delta) < (0.1 if pct else 0.5)
        if flat:
            chg = ("No change", "flat")
        elif t.lower_is_better is None:
            chg = (txt, "flat")
        else:
            chg = (txt, "good" if (delta < 0) == t.lower_is_better else "bad")
    err = sm.typical_error_pct
    good_fit = err is not None and err < 10
    beat = r.improvement_vs_baseline_pct
    stat_cards([
        {"label": f"Next {per}", "value": _fmt(sm.next_value, t.unit),
         "sub": when, "foot": f"{r.interval_level:.0%} range {_fmt(sm.next_lower, t.unit)} – {_fmt(sm.next_upper, t.unit)}",
         "tip": "The range comes from how far the model was off when tested on past periods it had not seen."},
        {"label": ahead_label, "value": ahead_value, "chip": chg, "sub": prev_text,
         "foot": "Compared with the same number of periods just before"},
        {"label": "Typical error in testing", "value": "n/a" if err is None else f"±{err:,.1f}%",
         "chip": None if err is None else (("Reliable", "good") if good_fit else ("Rough guide", "bad") if err > 25
                                           else ("Moderate", "flat")),
         "sub": f"{_fmt(chosen.mae, t.unit)}{unit_word if t.unit == 'days' else ''} per {per} on average",
         "foot": f"Tested on {len(r.backtest)} past {per}s the model had not seen",
         "tip": "Mean absolute error in validation, as a share of the average actual value."},
        {"label": "Model", "value": chosen.label.split(" (")[0],
         "chip": (f"{beat:,.0f}% better than baseline", "good") if sm.beats_baseline and beat is not None
         else ("Same as baseline", "flat"),
         "sub": "",
         "foot": f"Best of {sum(s.status == 'ok' for s in r.scores)} models; baseline: {base.label.split(' (')[0].lower()}",
         "tip": "The baseline repeats the last value (or the same period last cycle). A model is only chosen if it beats "
                "that in testing; among similar models the simplest wins."},
    ])

    section("Forecast", "Shaded: likely range · dotted: the model's predictions on past periods it had not seen")
    with st.container(border=True, key="fc_chart"):
        y_title = "Rows" if r.agg == "count" and t.value_col is None else name
        title = f"{name} by {per}" if t.derived or r.agg == "count" else f"{'Total' if r.agg == 'sum' else 'Average'} {name} by {per}"
        fig = charts.forecast_chart(r.history, r.forecast, title, y_title, r.backtest, f"{r.interval_level:.0%} range",
                                    shift_at=None if shift is None else shift.start, percent=pct)
        if fig is not None:
            show_figure(fig)

    notes = [w for w in r.warnings]
    if t.note:
        notes.insert(0, t.note)
    if notes:
        callout("Good to know", " ".join(notes), "info")

    tab1, tab2, tab3 = st.tabs(["Forecast values", "Model comparison", "How this forecast was made"])
    with tab1:
        out = r.forecast.rename(columns={"period": "Period", "forecast": "Forecast", "lower": "Low", "upper": "High"}).copy()
        out["Period"] = pd.to_datetime(out["Period"]).dt.date
        big = not pct and t.unit != "days" and abs(out["Forecast"]).max() >= 100
        out[["Forecast", "Low", "High"]] = out[["Forecast", "Low", "High"]].round(0 if big else 1)
        fmt = "%.1f%%" if pct else ("localized" if big else "%.1f")
        cfg = {c: st.column_config.NumberColumn(format=fmt) for c in ("Forecast", "Low", "High")}
        st.dataframe(out, hide_index=True, column_config=cfg)
        st.caption(f"Low and High bound the {r.interval_level:.0%} range for each {per} on its own; they do not add up "
                   "to a range for the total.")
    with tab2:
        sc = r.scores_frame()
        sc["Selected"] = sc["Selected"].map({True: "✓", False: ""})
        sc["Status"] = sc["Status"].str.capitalize()
        st.dataframe(sc, hide_index=True,
                     column_config={c: st.column_config.NumberColumn(format="%.2f") for c in ("MAE", "RMSE", "MASE", "MAPE %")})
        st.caption(f"Errors are measured on {r.folds} test window(s) of {r.validation_horizon} {per}(s), each forecast only "
                   "from the periods before it. MASE below 1 beats repeating the previous value. MAPE is shown only "
                   "when no actual value is zero or tiny.")
    with tab3:
        for line in r.limitations:
            st.markdown(f"- {line}")

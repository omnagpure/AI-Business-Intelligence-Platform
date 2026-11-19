"""Forecasting page (Phase 6A): pick a date and a target, validate models chronologically, show the forecast."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from analytics.descriptive import FREQ_LABELS
from analytics.forecasting import _MODELS, DEFAULT_HORIZON, forecast_series, suggest_setup, target_columns
from config.analytics_config import DEFAULT_ANALYTICS
from core.exceptions import BIError, InsufficientDataError
from core.logging_config import get_logger
from ui import state
from ui.progress_bar import ProgressBar
from ui.analytics_state import base_key, cached, show_figure
from ui.filters import filtered, no_rows_notice, render_filter_bar
from ui.layout import empty_state, eyebrow, no_dataset_notice, page_title
from visualization import charts

log = get_logger("ui.forecasting")
_FREQ = {"Day": "D", "Week": "W", "Month": "M"}
_AGG = {"Sum": "sum", "Average": "mean", "Row count": "count"}
_ROW_COUNT = "(row count)"


def _run(df, date_col, value_col, freq, agg, horizon, model=None):
    """Cached; returns ('ok', result) or ('insufficient'|'error', message) so failures are cached too."""
    def compute():
        try:
            with ProgressBar("Forecasting") as bar:
                return "ok", forecast_series(df, date_col, value_col, freq, agg, horizon, DEFAULT_ANALYTICS, model, progress=bar)
        except InsufficientDataError as exc:
            return "insufficient", exc.user_message
        except BIError as exc:
            return "error", exc.user_message
        except Exception:
            log.exception("Forecasting failed")
            return "error", "The forecast could not be built for this selection."
    return cached("forecast", (date_col, value_col, freq, agg, horizon, model, DEFAULT_ANALYTICS), compute)


def render_forecasting_page() -> None:
    result = state.get_result()
    if result is None:
        page_title("Forecasting", "Validated forecasts computed from your data.")
        no_dataset_notice()
        return
    ds = result.dataset
    page_title("Forecasting", ds.filename + (" · synthetic sample data" if ds.is_sample else ""))
    if ds.is_sample:
        st.warning("This is synthetic sample data generated for demonstration. It is not real.")
    try:
        df, profile, model = state.get_analysis_df(), state.get_analysis_profile(), state.get_semantic_model()
    except Exception:
        log.exception("Forecasting inputs unavailable")
        st.error("The dataset could not be prepared for analysis. Check the Data quality and Column roles pages.")
        return
    if df is None or profile is None or model is None or len(df) == 0:
        empty_state("Nothing to forecast", "The active dataset has no rows.")
        return
    render_filter_bar(df, model)
    df = filtered(df)
    if len(df) == 0:
        no_rows_notice()
        return

    dates = [c for c in df.columns if c in model.columns and model.role_of(c) == "datetime"]
    setup = suggest_setup(df, model)
    if not dates:
        empty_state("No date column", "Forecasting needs a column with the Date / time role. Set one on the Column roles page.")
        return
    targets = [_ROW_COUNT] + target_columns(df, profile, model)
    c1, c2, c3 = st.columns(3)
    date_col = c1.selectbox("Date column", dates, index=dates.index(setup["date_col"]) if setup["date_col"] in dates else 0,
                            key="fc_date")
    target = c2.selectbox("Forecast target", targets, key="fc_target",
                          index=targets.index(setup["value_col"]) if setup["value_col"] in targets else 0)
    agg_label = c3.selectbox("Combine rows by", list(_AGG), key="fc_agg", disabled=target == _ROW_COUNT)
    c4, c5, c6 = st.columns(3)
    default_freq = {v: k for k, v in _FREQ.items()}[setup["freq"]]
    freq_label = c4.selectbox("Period", list(_FREQ), index=list(_FREQ).index(default_freq), key="fc_freq")
    freq = _FREQ[freq_label]
    horizon = int(c5.number_input(f"Forecast horizon ({FREQ_LABELS[freq]}s ahead)", 1, DEFAULT_ANALYTICS.forecast_max_horizon,
                                  DEFAULT_HORIZON[freq], key=f"fc_h_{freq}"))
    labels = {"Automatic (best validated)": None, **{label: key for key, label, *_ in _MODELS}}
    model_key = labels[c6.selectbox("Model", list(labels), key="fc_model",
                                    help="Automatic picks the simplest model that is not clearly worse in validation. "
                                         "Choosing one yourself still shows its validation error.")]
    value_col = None if target == _ROW_COUNT else target
    agg = "count" if target == _ROW_COUNT else _AGG[agg_label]

    if st.button("Run forecast", type="primary", key="fc_run"):
        st.session_state["fc_params"] = (base_key(), date_col, value_col, freq, agg, horizon, model_key)
    params = st.session_state.get("fc_params")
    if not params or params[0] != base_key():
        st.caption("Choose a target and press Run forecast. Models are compared with rolling-origin validation "
                   "(no random splits), and the simplest model that is not clearly worse is used.")
        return
    status, payload = _run(df, *params[1:])
    if status == "insufficient":
        empty_state("This data cannot support a forecast", payload)
        return
    if status == "error":
        st.warning(payload)
        return
    _show(payload)


def _show(r) -> None:
    scores = {s.key: s for s in r.scores}
    chosen, base = scores[r.chosen], scores[r.baseline]
    target = r.value_col or "rows"
    eyebrow("Result")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Selected model", chosen.label.split(" (")[0])
    m2.metric("Validation MAE", f"{chosen.mae:,.2f}")
    m3.metric("Baseline MAE", f"{base.mae:,.2f}", help=f"Baseline: {base.label}")
    m4.metric("Improvement vs baseline", "n/a" if r.improvement_vs_baseline_pct is None else f"{r.improvement_vs_baseline_pct:.1f}%")
    if r.chosen == r.baseline:
        st.info("No more complex model was clearly better than the baseline, so the baseline is used.")
    agg_text = "Rows" if r.agg == "count" else f"{r.agg.title()} of {target}"
    fig = charts.forecast_chart(r.history, r.forecast, f"{agg_text} by {FREQ_LABELS[r.freq]}: history and forecast",
                                "Rows" if r.agg == "count" else target, r.backtest, f"{r.interval_level:.0%} interval")
    if fig is not None:
        show_figure(fig)
    for w in r.warnings:
        st.warning(w)
    tab1, tab2, tab3 = st.tabs(["Model comparison", "Forecast values", "Method and limitations"])
    with tab1:
        st.dataframe(r.scores_frame(), hide_index=True)
        st.caption(f"Scores are out-of-sample errors over {r.folds} fold(s) of {r.validation_horizon} {FREQ_LABELS[r.freq]}(s). "
                   "MASE below 1 beats repeating the previous value. MAPE is shown only when no actual value is zero or tiny.")
    with tab2:
        out = r.forecast.rename(columns={"period": "Period", "forecast": "Forecast", "lower": "Lower", "upper": "Upper"})
        out["Period"] = pd.to_datetime(out["Period"]).dt.date
        st.dataframe(out, hide_index=True)
    with tab3:
        for t in r.limitations:
            st.markdown(f"- {t}")

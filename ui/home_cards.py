"""Finding cards on Home: one plain-language line per analysis, each with an Open button.

Every card runs the analysis with the same defaults and cache key as its full page, so opening the page later
(or pressing Run there) reuses the stored result. A card for an analysis the data cannot support says why.
"""
from __future__ import annotations

from dataclasses import dataclass
from html import escape

import streamlit as st

from analytics.anomaly_detection import METHODS, default_columns, eligible_columns
from analytics.descriptive import FREQ_LABELS
from analytics.drivers import available_date_parts, candidate_features, driver_targets
from analytics.forecasting import DEFAULT_HORIZON, suggest_setup, target_columns
from analytics.segmentation import SCALERS, build_entity_features, entity_candidates
from config.analytics_config import DEFAULT_ANALYTICS
from core.logging_config import get_logger
from ui import nav
from ui.analysis_runner import safe_cached
from ui.analytics_state import base_key
from ui.anomaly_page import screen as _screen
from ui.drivers_page import _explain
from ui.forecasting_page import _ROW_COUNT, forecast_for
from ui.segmentation_page import _cluster

log = get_logger("ui.home_cards")


@dataclass(frozen=True)
class Card:
    key: str            # page key for the Open button
    name: str           # small label above the headline
    headline: str       # the one-line result, or the reason it was skipped
    detail: str = ""
    skipped: bool = False


# ---- plain-language summaries (pure, tested) ---------------------------------------------------
def summarise_forecast(r) -> Card:
    unit = {"D": "day", "W": "week", "M": "month"}[r.freq]
    what = "Row count" if r.agg == "count" else f"{r.agg.title()} of {r.value_col}"
    recent = r.history["value"].tail(r.horizon).mean()
    ahead = r.forecast["forecast"].mean()
    if recent and recent == recent and abs(recent) > 1e-12:
        pct = (ahead - recent) / abs(recent) * 100
        move = "stay about flat" if abs(pct) < 1 else f"{'rise' if pct > 0 else 'fall'} about {abs(pct):.0f}%"
        head = f"{what} is expected to {move} over the next {r.horizon} {unit}(s)"
    else:
        head = f"{what}: forecast for the next {r.horizon} {unit}(s) is ready"
    scores = {s.key: s for s in r.scores}
    model = scores[r.chosen].label.split(" (")[0]
    note = "No model clearly beat the simple baseline, so the baseline is used." if r.chosen == r.baseline else f"Model: {model}, validated on past periods."
    return Card("forecasting", "Forecast", head, note)


def summarise_anomalies(rep, n_cols: int) -> Card:
    votes = getattr(rep, "votes", None)
    if votes is not None and len(votes):
        ran = max(getattr(rep, "methods_run", 1), 1)
        need = min(3, ran)
        n = len(rep.agreed(need))
        return Card("anomalies", "Anomalies", f"{n:,} record(s) flagged by {need}+ methods",
                    f"{len(votes):,} flagged by at least one method across {n_cols} column(s). "
                    "Nothing is changed or removed.")
    n = int(rep.findings["row"].nunique()) if len(rep.findings) else 0
    if n == 0:
        return Card("anomalies", "Anomalies", "Nothing unusual found at balanced sensitivity",
                    f"Screened {n_cols} numeric column(s). Nothing is ever changed or removed.")
    return Card("anomalies", "Anomalies", f"{n:,} unusual record(s) flagged",
                f"Screened {n_cols} numeric column(s), strongest first. Nothing is changed or removed.")


def summarise_segments(res, entity: str) -> Card:
    names = getattr(res, "names", {}) or {}
    n = len(res.assignments) if hasattr(res, "assignments") else None
    head = f"{n:,} {entity} values fall into {res.k} segments" if n else f"Your {entity} values fall into {res.k} groups"
    if names and not all(v == "Close to average" for v in names.values()):
        biggest = int(res.profile["entities"].idxmax())
        detail = f"Largest: {names.get(biggest, '')} ({float(res.profile.loc[biggest, 'share']):.0%}). {res.quality}"
    else:
        detail = res.quality
    return Card("segmentation", "Segments", head, detail)


def summarise_drivers(res, target: str) -> Card:
    if not res.reliable or res.importance.empty:
        return Card("drivers", "Drivers", f"No column clearly explains {target}",
                    "The model did not predict held-out rows better than a simple guess.")
    top = str(res.importance.iloc[0]["feature"])
    return Card("drivers", "Drivers", f"{top} is the column most associated with {target}",
                "An association found on held-out rows, not proof of cause.")


# ---- one function per card: same defaults and cache keys as the page ---------------------------
def _forecast_card(df, profile, model) -> Card:
    dates = [c for c in df.columns if c in model.columns and model.role_of(c) == "datetime"]
    if not dates:
        return Card("forecasting", "Forecast", "Skipped: no date column", "Forecasting needs a column with the Date / time role.", True)
    setup = suggest_setup(df, model)
    targets = [_ROW_COUNT] + target_columns(df, profile, model)
    target = setup["value_col"] if setup["value_col"] in targets else _ROW_COUNT
    date_col = setup["date_col"] if setup["date_col"] in dates else dates[0]
    order = ["D", "W", "M"]
    status = payload = None
    for freq in order[order.index(setup["freq"]):]:      # sparse data: try a coarser period before giving up
        (status, payload), _ = forecast_for(df, profile, model, target, date_col, freq, DEFAULT_HORIZON[freq])
        if status != "insufficient":
            break
    if status != "ok":
        return Card("forecasting", "Forecast", "Skipped: not enough data for a validated forecast" if status == "insufficient"
                    else "Could not be completed", payload, True)
    return summarise_forecast(payload)


def _anomaly_card(df, profile, model) -> Card:
    cols, _ = eligible_columns(df, profile, model)
    if not cols:
        return Card("anomalies", "Anomalies", "Skipped: no suitable numeric columns",
                    "Screening needs measure columns with enough distinct values.", True)
    chosen, sens = tuple(default_columns(df, cols)), DEFAULT_ANALYTICS.anomaly_sensitivity
    status, rep = _screen(df, chosen, METHODS, sens)
    if status != "ok":
        return Card("anomalies", "Anomalies", "Skipped: detection not possible" if status == "insufficient" else "Could not be completed", rep, True)
    return summarise_anomalies(rep, len(chosen))


def _segment_card(df, model) -> Card:
    cands = entity_candidates(df, model)
    if not cands:
        return Card("segmentation", "Segments", "Skipped: no supplier, customer or product column",
                    "Grouping needs an entity column with enough different values.", True)
    entity = cands[0]["column"]
    status, table = safe_cached("seg_features", (entity,), lambda: build_entity_features(df, model, entity),
                                "Entity features could not be built for this column.")
    if status != "ok" or len(table.frame.columns) < 2:
        return Card("segmentation", "Segments", "Skipped: too few measures to group by",
                    table if status != "ok" else "Fewer than two measures could be built per entity.", True)
    feats = tuple(table.frame.columns)
    p = (base_key(), entity, feats, None, next(iter(SCALERS)))
    status, res = safe_cached("seg_result", p[1:], lambda: _cluster(table, p), "Segmentation could not be completed for this selection.")
    if status != "ok":
        return Card("segmentation", "Segments", "Skipped: grouping not possible" if status == "insufficient" else "Could not be completed", res, True)
    return summarise_segments(res, entity)


def _driver_card(df, profile, model) -> Card:
    targets = driver_targets(df, profile, model)
    if not targets:
        return Card("drivers", "Drivers", "Skipped: no suitable target to explain",
                    "A target needs a numeric measure with enough distinct values, or a Yes/No column.", True)
    target = list(targets)[0]
    num, cat, _ = candidate_features(df, profile, model, target)
    feats = tuple(num + cat + available_date_parts(df, model, target))
    p = (base_key(), target, feats)
    status, res = safe_cached("driver_result", p[1:], lambda: _explain(df, profile, model, p),
                              "Driver analysis could not be completed for this selection.")
    if status != "ok":
        return Card("drivers", "Drivers", "Skipped: analysis not possible" if status == "insufficient" else "Could not be completed", res, True)
    return summarise_drivers(res, target)


def _explorer_card(df) -> Card:
    return Card("explorer", "Explorer", f"Browse all {len(df):,} rows", "Search, sort and page through your data.")


# ---- rendering --------------------------------------------------------------------------------
def _draw(card: Card) -> None:
    tone = " card-skip" if card.skipped else ""
    st.markdown(f'<div class="card-k">{escape(card.name)}</div><div class="card-t{tone}">{escape(card.headline)}</div>'
                f'<div class="card-d">{escape(card.detail)}</div>', unsafe_allow_html=True)
    if st.button("Open", key=f"home_open_{card.key}", type="secondary"):
        nav.go(card.key)


def render_finding_cards(df, profile, model) -> None:
    """Five cards in a grid. Each is computed and drawn in turn, so the first load shows progress."""
    st.markdown('<div class="eyebrow">What your data says</div>', unsafe_allow_html=True)
    top = st.columns(3, gap="medium")
    bottom = st.columns(3, gap="medium")
    slots = [("Forecast", top[0], lambda: _forecast_card(df, profile, model)),
             ("Anomalies", top[1], lambda: _anomaly_card(df, profile, model)),
             ("Segments", top[2], lambda: _segment_card(df, model)),
             ("Drivers", bottom[0], lambda: _driver_card(df, profile, model)),
             ("Explorer", bottom[1], lambda: _explorer_card(df))]
    for label, col, build in slots:
        with col, st.container(border=True):
            try:
                with st.spinner(f"Checking {label.lower()}..."):
                    card = build()
            except Exception:
                log.exception("Home card %s failed", label)
                card = Card(label.lower(), label, "Could not be completed", "Open the page to see details.", True)
            _draw(card)

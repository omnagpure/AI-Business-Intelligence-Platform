"""Session cache for Phase 5 results.

Every result is stored under a key built from the dataset fingerprint, the applied cleaning actions, the
semantic model (roles and domain), the user's overrides and the analysis-specific parameters (filters,
settings). Changing any of them gives a different key, so stale results are never shown.
"""
from __future__ import annotations

import json
from typing import Callable, TypeVar

import streamlit as st

from analytics.kpis import cache_key
from ui import state

T = TypeVar("T")
_CACHE = "analytics_cache"
_MAX_ENTRIES = 40


def base_key() -> str | None:
    """Fingerprint of everything an analysis depends on, or None when there is nothing to analyse."""
    result, model = state.get_result(), state.get_semantic_model()
    if result is None or model is None:
        return None
    cs = state.get_cleaning()
    return cache_key(result.dataset.fingerprint, cs.result.actions_fingerprint if cs else "",
                     model.fingerprint(), json.dumps(state.get_semantic_overrides(), sort_keys=True, default=str))


def cached(name: str, params: object, compute: Callable[[], T]) -> T:
    """Return the stored result for (base_key, name, params) or compute and store it."""
    store: dict = st.session_state.setdefault(_CACHE, {})
    key = cache_key(base_key(), name, params)
    if key in store:
        return store[key]
    value = compute()
    store[key] = value
    while len(store) > _MAX_ENTRIES:             # drop the oldest entries
        store.pop(next(iter(store)))
    return value


def show_figure(fig) -> None:
    """Render a Plotly figure across Streamlit versions."""
    try:
        st.plotly_chart(fig, width="stretch", theme=None)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, theme=None)

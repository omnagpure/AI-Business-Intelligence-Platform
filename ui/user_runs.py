"""Results the user already produced on the analysis pages (with their own settings), for Insights and Reports to reuse.

Nothing is computed here: a result is returned only if it is in the session cache for the current dataset, cleaning,
roles and parameters. Anything not run on its page falls back to the default-settings run.
"""
from __future__ import annotations

import streamlit as st

from config.analytics_config import DEFAULT_ANALYTICS
from insights.default_runs import PAGE_NAMES  # noqa: F401  (re-exported for the pages)
from ui.analytics_state import base_key, peek



def user_runs() -> dict[str, object]:
    """{analysis name: result} for analyses the user has run on their own page for the current data."""
    bk, out = base_key(), {}
    specs = (("forecast", "fc_params", "forecast", lambda p: (*p[1:], DEFAULT_ANALYTICS)),
             ("anomalies", "an_params", "anomaly_values", lambda p: p[1:]),
             ("segmentation", "seg_params", "seg_result", lambda p: p[1:]),
             ("drivers", "dr_params", "driver_result", lambda p: p[1:]))
    for analysis, key, cache_name, params in specs:
        p = st.session_state.get(key)
        if not p or p[0] != bk:
            continue
        hit = peek(cache_name, params(p))
        if hit is not None and hit[0] == "ok":
            out[analysis] = hit[1]
    return out

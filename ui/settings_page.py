"""Configuration (read-only) and system status, including the optional semantic model."""
from __future__ import annotations

import platform

import pandas as pd
import streamlit as st

from config.settings import APP_VERSION, SUPPORTED_EXTENSIONS, Settings
from ui import state
from ui.layout import eyebrow, page_title


def _sklearn_version() -> str:
    try:
        import sklearn
        return sklearn.__version__
    except ImportError:
        return "not installed"


def render_settings_page(settings: Settings) -> None:
    page_title("Settings", "Current configuration and system status. Change values in the .env file and restart.")
    eyebrow("Configuration")
    st.dataframe(pd.DataFrame({
        "Setting": ["Maximum upload size", "Preview rows", "Log level", "Supported formats", "Version"],
        "Value": [f"{settings.max_upload_mb} MB", settings.preview_rows, settings.log_level,
                  ", ".join(SUPPORTED_EXTENSIONS), APP_VERSION]}), hide_index=True)

    st.write("")
    eyebrow("Semantic name model (optional)")
    status = state.get_name_model_status()
    use = state.get_use_model()
    st.dataframe(pd.DataFrame({
        "Item": ["Python", "scikit-learn", "Trained model", "Blended into suggestions"],
        "Status": [platform.python_version(), _sklearn_version(),
                   f"Loaded ({status.version})" if status.available else status.reason,
                   "Yes" if use and status.available else "No (rules only)"]}), hide_index=True)
    new = st.toggle("Use the trained name model", value=use and status.available, disabled=not status.available,
                    help="Off by default. The Semantics page explains what the model does and how it was evaluated.")
    if status.available and new != use:
        state.set_use_model(new)
        st.rerun()
    st.caption("Processed locally. Uploaded data is held in memory for the session and is not written to disk.")

"""Configuration (read-only) and system status, including the optional semantic model."""
from __future__ import annotations

import platform

import pandas as pd
import streamlit as st

from config.settings import APP_VERSION, SUPPORTED_EXTENSIONS, Settings
from semantic.semantic_model import MODEL_MODE_LABELS, MODEL_MODES
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
    mode = state.get_model_mode()
    st.dataframe(pd.DataFrame({
        "Item": ["Python", "scikit-learn", "Trained model", "How it is used"],
        "Status": [platform.python_version(), _sklearn_version(),
                   f"Loaded ({status.version})" if status.available else status.reason,
                   MODEL_MODE_LABELS[mode] if status.available else "Not used (rules only)"]}), hide_index=True)
    if st.session_state.get("settings_model_mode") != mode:     # follow the saved setting (also changeable on the other page)
        st.session_state["settings_model_mode"] = mode
    st.radio("Use the trained name model", MODEL_MODES, horizontal=True, format_func=MODEL_MODE_LABELS.get,
             key="settings_model_mode", disabled=not status.available,
             on_change=lambda: state.set_model_mode(st.session_state["settings_model_mode"]),
             help="Automatic: supply-chain data keeps the rules in charge (the model only fills columns they "
                  "cannot identify); other data gets the model's vote on every column.")
    st.caption("Processed locally. Files you upload and reports you build are saved on this server under your account only, "
               "and you can delete them from the Upload and Reports pages.")

"""Read-only view of the active configuration."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from config.settings import APP_VERSION, SUPPORTED_EXTENSIONS, Settings
from ui.layout import page_title


def render_settings_page(settings: Settings) -> None:
    page_title("Settings", "Current configuration. Change values in the .env file and restart.")
    st.dataframe(pd.DataFrame({
        "Setting": ["Maximum upload size", "Preview rows", "Log level", "Supported formats", "Version"],
        "Value": [f"{settings.max_upload_mb} MB", settings.preview_rows, settings.log_level,
                  ", ".join(SUPPORTED_EXTENSIONS), APP_VERSION]}), hide_index=True)

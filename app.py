"""Streamlit entry point. Run: streamlit run app.py"""
from __future__ import annotations

import streamlit as st

from config.settings import APP_NAME, load_settings
from core.logging_config import setup_logging
from ui.data_quality_page import render_data_quality_page
from ui.layout import configure_page
from ui.profile_page import render_profile_page
from ui.settings_page import render_settings_page
from ui.sidebar import render_sidebar_footer
from ui.upload_page import render_upload_page


def main() -> None:
    settings = load_settings()
    setup_logging(settings)
    configure_page()

    def upload() -> None:
        render_upload_page(settings)

    def profile() -> None:
        render_profile_page()

    def quality() -> None:
        render_data_quality_page(settings)

    def config() -> None:
        render_settings_page(settings)

    nav = st.navigation([
        st.Page(upload, title="Upload", url_path="upload", icon=":material/upload_file:", default=True),
        st.Page(profile, title="Profile", url_path="profile", icon=":material/table_chart:"),
        st.Page(quality, title="Data quality", url_path="quality", icon=":material/fact_check:"),
        st.Page(config, title="Settings", url_path="settings", icon=":material/settings:"),
    ])
    with st.sidebar:
        st.markdown(f"**{APP_NAME}**")
    render_sidebar_footer()
    nav.run()


main()

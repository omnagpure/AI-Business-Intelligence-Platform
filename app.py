"""Streamlit entry point.

Run `streamlit run server.py` (adds the session-cookie routes, so sign-in survives reloads and "Remember me" works),
or `streamlit run app.py` (sign-in lasts for the open tab only).

Visitors who are not signed in get only the public pages (homepage, sign in, registration): the analysis pages are not
registered for them at all, so no protected page code can run without an authenticated session.
"""
from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import streamlit as st

from config.settings import load_settings
from core.logging_config import setup_logging
from ui import auth_state
from ui.account_page import render_account_page
from ui.anomaly_page import render_anomaly_page
from ui.auth_pages import render_login_page, render_register_page
from ui.dashboard_page import render_dashboard_page
from ui.data_quality_page import render_data_quality_page
from ui.drivers_page import render_drivers_page
from ui.explorer_page import render_explorer_page
from ui.filters import begin_run
from ui.forecasting_page import render_forecasting_page
from ui.insights_page import render_insights_page
from ui.landing_page import render_landing_page
from ui.layout import configure_page, context_line
from ui.nav import register as register_pages
from ui.overview_page import render_overview_page
from ui.profile_page import render_profile_page
from ui.public_ui import apply_public_theme
from ui.reports_page import render_reports_page
from ui.segmentation_page import render_segmentation_page
from ui.semantic_page import render_semantic_page
from ui.settings_page import render_settings_page
from ui.sidebar import render_sidebar_footer
from ui.upload_page import render_upload_page

_PROTECTED_PATHS = {"overview", "upload", "profile", "quality", "semantics", "dashboard", "explorer", "forecasting", "anomalies",
                    "segmentation", "drivers", "insights", "reports", "settings", "account"}


def _requested_path() -> str:
    try:
        return urlparse(st.context.url or "").path.rstrip("/").rsplit("/", 1)[-1]
    except Exception:
        return ""


def public_app() -> None:
    """Homepage, sign in and registration. Nothing here reads user data."""
    pages = {
        "landing": st.Page(render_landing_page, title="AI Business Intelligence Platform", url_path="home", icon=":material/insights:", default=True),
        "login": st.Page(render_login_page, title="Sign in", url_path="login", icon=":material/login:"),
        "register": st.Page(render_register_page, title="Create account", url_path="register", icon=":material/person_add:"),
    }
    register_pages(pages)
    page = st.navigation(list(pages.values()), position="hidden")
    requested = _requested_path()
    if requested in _PROTECTED_PATHS and not st.session_state.get("_auth_asked"):   # a deep link to a protected page
        st.session_state["_auth_asked"] = True
        auth_state.set_next(requested)
        st.switch_page(pages["login"])
    page.run()


def main() -> None:
    settings = load_settings()
    setup_logging(settings)
    user = auth_state.resolve_user()
    configure_page(sidebar=user is not None)
    if auth_state.has_pending_redirect():            # browser is about to leave for the cookie route
        apply_public_theme()
        auth_state.render_pending_redirect()
        return
    if user is None:
        public_app()
        return
    begin_run()

    def overview() -> None:
        render_overview_page()

    def upload() -> None:
        render_upload_page(settings)

    def profile() -> None:
        render_profile_page()

    def quality() -> None:
        render_data_quality_page(settings)

    def semantics() -> None:
        render_semantic_page()

    def dashboard() -> None:
        render_dashboard_page()

    def explorer() -> None:
        render_explorer_page()

    def forecasting() -> None:
        render_forecasting_page()

    def anomalies() -> None:
        render_anomaly_page()

    def segmentation() -> None:
        render_segmentation_page()

    def drivers() -> None:
        render_drivers_page()

    def insights() -> None:
        render_insights_page()

    def reports() -> None:
        render_reports_page()

    def config() -> None:
        render_settings_page(settings)

    def account() -> None:
        render_account_page()

    pages = {
        "home": st.Page(overview, title="Home", url_path="overview", icon=":material/home:", default=True),
        "upload": st.Page(upload, title="Upload", url_path="upload", icon=":material/upload_file:"),
        "profile": st.Page(profile, title="Profile", url_path="profile", icon=":material/table_chart:"),
        "quality": st.Page(quality, title="Data quality", url_path="quality", icon=":material/fact_check:"),
        "roles": st.Page(semantics, title="Column roles", url_path="semantics", icon=":material/category:"),
        "dashboard": st.Page(dashboard, title="Dashboard", url_path="dashboard", icon=":material/monitoring:"),
        "explorer": st.Page(explorer, title="Data explorer", url_path="explorer", icon=":material/table_view:"),
        "forecasting": st.Page(forecasting, title="Forecasting", url_path="forecasting", icon=":material/trending_up:"),
        "anomalies": st.Page(anomalies, title="Anomalies", url_path="anomalies", icon=":material/crisis_alert:"),
        "segmentation": st.Page(segmentation, title="Segmentation", url_path="segmentation", icon=":material/groups:"),
        "drivers": st.Page(drivers, title="Drivers", url_path="drivers", icon=":material/insights:"),
        "insights": st.Page(insights, title="Insights", url_path="insights", icon=":material/lightbulb:"),
        "reports": st.Page(reports, title="Reports", url_path="reports", icon=":material/description:"),
        "settings": st.Page(config, title="Settings", url_path="settings", icon=":material/settings:"),
        "account": st.Page(account, title="Account", url_path="account", icon=":material/account_circle:"),
    }
    register_pages(pages)
    nav = st.navigation({
        "Start": [pages["home"], pages["upload"]],
        "Understand your data": [pages["profile"], pages["quality"], pages["roles"]],
        "Find insights": [pages[k] for k in ("dashboard", "explorer", "forecasting", "anomalies", "segmentation", "drivers", "insights")],
        "Deliver": [pages["reports"]],
        "System": [pages["settings"], pages["account"]],
    })
    try:
        st.logo(str(Path(__file__).parent / "assets" / "logo.svg"), size="large")
    except Exception:  # a missing or unsupported logo must never stop the app
        pass
    target = auth_state.pop_target()                 # just signed in within this tab: open the requested page
    if target and target in pages_by_path(pages):
        st.switch_page(pages_by_path(pages)[target])
    render_sidebar_footer()
    context_line()
    nav.run()


def pages_by_path(pages: dict) -> dict:
    return {p.url_path: p for p in pages.values()}


main()

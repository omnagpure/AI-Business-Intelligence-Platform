"""Page configuration, light CSS polish and small shared widgets."""
from __future__ import annotations

import streamlit as st

from config.settings import APP_NAME

_CSS = """
<style>
#MainMenu, footer, [data-testid="stDecoration"], [data-testid="stStatusWidget"] {display:none;}
header[data-testid="stHeader"] {background: transparent;}
.block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1280px;}
h1 {font-size: 1.75rem !important; font-weight: 600 !important; margin-bottom: .2rem;}
h2, h3 {font-weight: 600 !important;}
[data-testid="stMetricLabel"] p {font-size: .8rem; opacity: .7;}
[data-testid="stMetricValue"] {font-size: 1.55rem;}
section[data-testid="stSidebar"] {border-right: 1px solid #222a31;}
[data-testid="stFileUploaderDropzone"] {border-style: dashed;}
</style>
"""


def configure_page() -> None:
    st.set_page_config(page_title=APP_NAME, layout="wide", initial_sidebar_state="expanded")
    st.markdown(_CSS, unsafe_allow_html=True)


def page_title(title: str, caption: str = "") -> None:
    st.title(title)
    if caption:
        st.caption(caption)


def no_dataset_notice() -> None:
    with st.container(border=True):
        st.markdown("**No dataset loaded**")
        st.caption("Open Upload in the sidebar and add a CSV or Excel file, or load the sample data.")


def fmt_int(n: int) -> str:
    return f"{n:,}"

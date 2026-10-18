"""Upload page: validate, parse, profile and preview a dataset."""
from __future__ import annotations

import streamlit as st

from config.settings import EXCEL_EXTENSIONS, SUPPORTED_EXTENSIONS, Settings
from core.exceptions import BIError
from core.logging_config import get_logger
from ingestion.file_loader import fingerprint_bytes, list_excel_sheets
from ingestion.pipeline import ingest_bytes, ingest_sample
from ui import state
from ui.layout import fmt_int, page_title

log = get_logger("ui.upload")


def _show_loaded(settings: Settings) -> None:
    result = state.get_result()
    if result is None:
        return
    ds, prof, q = result.dataset, result.profile, result.quality
    if ds.is_sample:
        st.warning("This is synthetic sample data generated for demonstration. It is not real.")
    c = st.columns(4)
    c[0].metric("Rows", fmt_int(prof.rows))
    c[1].metric("Columns", prof.columns)
    c[2].metric("File size", f"{ds.size_bytes / 1024 / 1024:.1f} MB" if ds.size_bytes else "n/a")
    c[3].metric("Completeness", f"{q.completeness_pct:.1f}%")
    kind_label = "Excel" if ds.file_type == "excel" else "Delimited text"
    details = [f"Format: {kind_label}"]
    if ds.sheet:
        details.append(f"Sheet: {ds.sheet}")
    if ds.encoding:
        details.append(f"Encoding: {ds.encoding}")
    if ds.delimiter:
        details.append(f"Delimiter: {'tab' if ds.delimiter == chr(9) else ds.delimiter!r}")
    st.caption(" · ".join(details))
    for w in ds.warnings:
        st.warning(w)
    st.subheader("Preview")
    n = min(settings.preview_rows, prof.rows)
    st.dataframe(ds.df.head(n), hide_index=True, height=360)
    st.caption(f"Showing the first {fmt_int(n)} of {fmt_int(prof.rows)} rows. "
               "The uploaded data is not modified.")


def render_upload_page(settings: Settings) -> None:
    page_title("Upload", f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}. Any dataset works, up to {settings.max_upload_mb} MB.")
    left, right = st.columns([3, 1], vertical_alignment="bottom")
    uploaded = left.file_uploader("Data file", type=[e.lstrip(".") for e in SUPPORTED_EXTENSIONS],
                                  label_visibility="collapsed")
    if right.button("Load sample data", help="Synthetic sales data for trying the app"):
        state.set_result(ingest_sample())
        st.rerun()

    if uploaded is not None:
        try:
            data = uploaded.getvalue()
            sheet = None
            if uploaded.name.lower().endswith(EXCEL_EXTENSIONS):
                sheets = state.cached_sheets(uploaded.file_id)
                if sheets is None:
                    sheets = list_excel_sheets(data, uploaded.name)
                    state.cache_sheets(uploaded.file_id, sheets)
                if len(sheets) > 1:
                    sheet = st.selectbox("Worksheet", sheets)
                else:
                    sheet = sheets[0]
            current = state.get_result()
            if current is None or current.dataset.fingerprint != fingerprint_bytes(data, sheet):
                with st.spinner("Reading and profiling file..."):
                    state.set_result(ingest_bytes(data, uploaded.name, max_bytes=settings.max_upload_bytes,
                                                  sheet=sheet))
                st.rerun()
        except BIError as exc:
            log.info("Upload rejected: %s", type(exc).__name__)
            state.clear_result()
            st.error(exc.user_message)
        except Exception:
            log.exception("Unexpected error while loading upload")
            state.clear_result()
            st.error("Something went wrong while reading this file. Check that it is a valid CSV or "
                     "XLSX file. Details were written to the application log.")

    if state.get_result() is None:
        with st.container(border=True):
            st.markdown("**No dataset loaded**")
            st.caption("Drop a file above to see its structure, missing values and quality issues.")
    else:
        _show_loaded(settings)

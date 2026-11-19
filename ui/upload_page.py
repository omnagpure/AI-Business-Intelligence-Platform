"""Upload page: validate, parse, profile and preview a dataset."""
from __future__ import annotations

from datetime import datetime

import streamlit as st

from config.settings import EXCEL_EXTENSIONS, SUPPORTED_EXTENSIONS, Settings
from core.exceptions import BIError
from core.logging_config import get_logger
from ingestion.file_loader import fingerprint_bytes, list_excel_sheets
from ingestion.pipeline import ingest_bytes, ingest_sample
from storage import repository
from ui import auth_state, notify, state
from ui.progress_bar import ProgressBar
from ui.layout import empty_state, fmt_int, md_escape, page_title

log = get_logger("ui.upload")
LARGE_FILE_BYTES = 5 * 1024 * 1024
_UPLOADER_VER = "upload_widget_version"
_FLASH = "upload_flash"


def _reset_uploader() -> None:
    """A fresh file uploader, so a file left in it does not replace a dataset opened another way."""
    st.session_state[_UPLOADER_VER] = st.session_state.get(_UPLOADER_VER, 0) + 1


def _store(data: bytes, filename: str, sheet: str | None, loaded) -> str | None:
    """Save the upload to the signed-in user's datasets. Returns the record id, or None if it could not be saved."""
    user = auth_state.require_user()
    try:
        rec = repository.save_dataset(user.id, data, filename, sheet, loaded.dataset.fingerprint, loaded.profile.rows, loaded.profile.columns)
        return rec.id
    except Exception:
        log.exception("Could not store the uploaded dataset")
        st.session_state[_FLASH] = "The file was loaded, but it could not be saved to your datasets. It is available for this session only."
        return None


def _when(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%d %b %Y, %H:%M")


def open_saved(settings: Settings, dataset_id: str, quiet: bool = False) -> bool:
    """Make one of the signed-in user's saved datasets the active one. Returns True on success; the caller reruns.
    quiet: do not show errors (used when reopening the last dataset automatically)."""
    user = auth_state.require_user()
    try:
        rec, data = repository.read_dataset(user.id, dataset_id)          # ownership is checked here
        with st.spinner(f"Opening {rec.filename}..."):
            loaded = ingest_bytes(data, rec.filename, max_bytes=settings.max_upload_bytes, sheet=rec.sheet)
    except repository.NotFound:
        if not quiet:
            st.error("That dataset is no longer available.")
        return False
    except BIError as exc:
        if not quiet:
            st.error(exc.user_message)
        return False
    except Exception:
        log.exception("Could not open a saved dataset")
        if not quiet:
            st.error("The saved dataset could not be opened. Details were written to the application log.")
        return False
    _reset_uploader()
    state.set_result(loaded, dataset_id=rec.id)
    return True


def _open_saved(settings: Settings, dataset_id: str) -> None:
    if open_saved(settings, dataset_id):
        notify.toast(f"Opened {state.get_result().dataset.filename}")
        st.rerun()


def _saved_datasets(settings: Settings) -> None:
    """The signed-in user's own uploads, newest first. Other users' datasets are never listed or reachable."""
    user = auth_state.require_user()
    records = repository.list_datasets(user.id)
    st.subheader("Your datasets")
    if not records:
        st.caption("No saved datasets yet.")
        return
    active = state.get_dataset_id()
    for rec in records:
        with st.container(border=True):
            info, open_col, del_col = st.columns([6, 1.3, 1.2], vertical_alignment="center")
            meta = f"{fmt_int(rec.rows)} rows · {rec.columns} columns · {rec.size_bytes / 1024 / 1024:.1f} MB · uploaded {_when(rec.created_at)}"
            if rec.sheet:
                meta = f"Sheet {rec.sheet} · " + meta
            info.markdown(f"**{md_escape(rec.filename)}**" + (" :green-badge[Active]" if rec.id == active else ""))
            info.caption(meta)
            if open_col.button("Open", key=f"ds_open_{rec.id}", disabled=rec.id == active, width="stretch"):
                _open_saved(settings, rec.id)
            with del_col.popover("Delete", width="stretch"):
                st.caption("Delete this dataset and its stored file? Reports already built from it are kept.")
                if st.button("Delete dataset", key=f"ds_del_{rec.id}", type="primary"):
                    try:
                        repository.delete_dataset(user.id, rec.id)
                    except repository.NotFound:
                        pass
                    if rec.id == active:
                        state.clear_result()
                    notify.toast(f"Deleted {rec.filename}")
                    st.rerun()


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
                                  label_visibility="collapsed", key=f"uploader_{st.session_state.get(_UPLOADER_VER, 0)}")
    if right.button("Load sample data", help="Synthetic sales data for trying the app"):
        _reset_uploader()
        state.set_result(ingest_sample())
        st.rerun()
    if (flash := st.session_state.pop(_FLASH, None)) is not None:
        st.warning(flash)

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
                if len(data) >= LARGE_FILE_BYTES:            # a bar for large files; small ones load too fast to need it
                    with ProgressBar("Reading and profiling the file", numbered=True) as bar:
                        loaded = ingest_bytes(data, uploaded.name, max_bytes=settings.max_upload_bytes, sheet=sheet, progress=bar)
                else:
                    with st.spinner("Reading and profiling file..."):
                        loaded = ingest_bytes(data, uploaded.name, max_bytes=settings.max_upload_bytes, sheet=sheet)
                state.set_result(loaded, dataset_id=_store(data, uploaded.name, sheet, loaded))
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
        empty_state("No dataset loaded", "Drop a file above, or open one of your saved datasets below.")
    else:
        _show_loaded(settings)
    st.write("")
    _saved_datasets(settings)

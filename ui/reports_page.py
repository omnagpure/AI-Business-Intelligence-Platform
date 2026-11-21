"""Reports page: builds a PDF report (and an HTML report if ticked) from computed results only, plus CSV downloads."""
from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import PurePath

import pandas as pd
import streamlit as st

from analytics.kpis import cache_key
from core.estimates import describe_estimate
from core.logging_config import get_logger
from reporting.assembly import ReportContext, Sources, assemble_report
from reporting.report_builder import AUTO_TITLES, safe_filename
from reporting.theme import ACCENTS
from storage import repository
from ui import auth_state, notify, state
from ui.analytics_state import base_key
from ui.layout import empty_state, md_escape, no_dataset_notice, page_title
from ui.progress_bar import ProgressBar
from ui.shared_results import advanced_sources, findings_source, kpi_source
from ui.user_runs import user_runs

log = get_logger("ui.reports")
_PKG = "report_package"
_CSV = "report_csv_bytes"
CLASSIFICATIONS = ["Internal", "Confidential", "Public"]
MAX_LOGO_BYTES = 1_000_000


def _log_csv(cleaning) -> bytes | None:
    if cleaning is None:
        return None
    rows = [{"step": e.step, "operation": e.op, "column": e.column, "status": e.status, "rows_before": e.rows_before,
             "rows_after": e.rows_after, "cells_changed": e.cells_changed, "detail": e.detail} for e in cleaning.log]
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")


def _branding() -> dict:
    """Optional organisation name, logo, accent colour and classification for the cover and page headers."""
    with st.expander("Branding (optional)"):
        b1, b2, b3 = st.columns([2, 1, 1])
        org = b1.text_input("Prepared by (your organisation)", value="", max_chars=80, key="rep_org")
        accent_name = b2.selectbox("Accent colour", list(ACCENTS), key="rep_accent")
        cls = b3.selectbox("Classification", CLASSIFICATIONS, key="rep_cls")
        up = st.file_uploader("Logo (PNG or JPG, up to 1 MB)", type=["png", "jpg", "jpeg"], key="rep_logo")
        logo = None
        if up is not None:
            data = up.getvalue()
            if len(data) > MAX_LOGO_BYTES:
                st.warning("The logo is larger than 1 MB and was not used.")
            elif not (data[:8] == b"\x89PNG\r\n\x1a\n" or data[:3] == b"\xff\xd8\xff"):
                st.warning("The logo is not a PNG or JPG image and was not used.")
            else:
                logo = data
    return {"organisation": org.strip(), "accent": ACCENTS[accent_name], "classification": cls, "logo": logo}


def _build(result, df, profile, model, title: str, prepared: str, advanced: bool, with_html: bool, brand: dict | None = None) -> dict:
    cs = state.get_cleaning()
    quality = cs.quality if cs is not None else result.quality
    cleaning = cs.result if cs is not None else None
    brand = brand or {}
    ctx = ReportContext(result.dataset.filename, result.dataset.is_sample, result.dataset.fingerprint, profile, quality, len(df), model,
                        cleaning, title, prepared, df=df, organisation=brand.get("organisation", ""), logo=brand.get("logo"),
                        accent=brand.get("accent", ACCENTS["Navy"]), classification=brand.get("classification", "Internal"))
    sources = Sources(kpi_source(df, model), findings_source(df, model), advanced_sources(df, profile, model))
    with ProgressBar("Building the report", numbered=True) as bar:
        out = assemble_report(ctx, sources, user_results=user_runs() if advanced else {}, advanced=advanced, with_html=with_html, progress=bar)
        skipped = list(bar.skipped)
    files = dict(out.files)
    log_bytes = _log_csv(cleaning)
    if log_bytes is not None:
        files["log"] = log_bytes
    return {"data": out.data, "files": files, "errors": out.errors, "skipped": skipped, "html": with_html}


def _save_history(pkg: dict, dataset_name: str, title: str) -> None:
    """Keep the built files in the signed-in user's report history."""
    user = auth_state.require_user()
    for fmt in ("pdf", "html"):
        data = pkg["files"].get(fmt)
        if not data:
            continue
        try:
            repository.save_report(user.id, state.get_dataset_id(), dataset_name, title, fmt, data)
        except Exception:
            log.exception("Could not save a report to history")
            st.warning("The report was built, but it could not be saved to your report history.")
            return


def _history() -> None:
    """The signed-in user's previously built reports. Each download is read through the ownership check."""
    user = auth_state.require_user()
    records = repository.list_reports(user.id)
    st.write("")
    st.subheader("Report history")
    if not records:
        st.caption("No reports yet.")
        return
    for rec in records:
        with st.container(border=True):
            info, dl, rm = st.columns([6, 1.5, 1.2], vertical_alignment="center")
            info.markdown(f"**{md_escape(rec.title)}** · {rec.fmt.upper()}")
            info.caption(f"{rec.dataset_name} · {datetime.fromtimestamp(rec.created_at):%d %b %Y, %H:%M} · {rec.size_bytes / 1024:,.0f} KB")
            dl.download_button("Download", data=lambda rid=rec.id: repository.read_report(user.id, rid)[1],
                               file_name=f"{safe_filename(PurePath(rec.dataset_name).stem)}_report.{rec.fmt}",
                               mime="application/pdf" if rec.fmt == "pdf" else "text/html", key=f"hist_dl_{rec.id}",
                               on_click="ignore", width="stretch")
            with rm.popover("Delete", width="stretch"):
                st.caption("Delete this report from your history?")
                if st.button("Delete report", key=f"hist_del_{rec.id}", type="primary"):
                    try:
                        repository.delete_report(user.id, rec.id)
                    except repository.NotFound:
                        pass
                    notify.toast("Report deleted")
                    st.rerun()


def render_reports_page() -> None:
    _render_current()
    _history()


def _render_current() -> None:
    result = state.get_result()
    if result is None:
        page_title("Reports", "Download a finished report built from your computed results.")
        no_dataset_notice()
        return
    ds = result.dataset
    page_title("Reports", ds.filename + (" · synthetic sample data" if ds.is_sample else ""))
    try:
        df, profile, model = state.get_analysis_df(), state.get_analysis_profile(), state.get_semantic_model()
    except Exception:
        log.exception("Report inputs unavailable")
        st.error("The dataset could not be prepared for analysis. Check the Data quality and Column roles pages.")
        return
    if df is None or profile is None or model is None or len(df) == 0:
        empty_state("Nothing to report", "The active dataset has no rows.")
        return

    auto = AUTO_TITLES.get(model.domain.domain, AUTO_TITLES["general"])
    c1, c2 = st.columns(2)
    title = c1.text_input("Report title", value="", max_chars=90, key="rep_title", placeholder=f"Automatic: {auto}",
                          help="Leave empty to use a title chosen from the data's business area.")
    prepared = c2.text_input("Prepared for (optional)", value="", max_chars=80, key="rep_for")
    brand = _branding()
    advanced = st.checkbox("Include forecast, anomaly, segment and driver analyses (default settings; can take a while on large files)",
                           value=True, key="rep_adv")
    with_html = st.checkbox("Also build HTML", value=False, key="rep_html",
                            help="A self-contained web page version of the same report. It adds a little time, so it is off by default.")
    key = cache_key(base_key(), title, prepared, advanced, with_html, brand["organisation"], brand["accent"], brand["classification"],
                    hashlib.sha256(brand["logo"] or b"").hexdigest()[:12])
    hint = describe_estimate("report", len(df)) if advanced else None
    if hint:
        st.caption(hint)
    pkg = st.session_state.get(_PKG)
    if st.button("Build PDF and HTML reports" if with_html else "Build PDF report", type="primary", key="rep_build"):
        try:
            pkg = {"key": key, **_build(result, df, profile, model, title, prepared, advanced, with_html, brand)}
            _save_history(pkg, ds.filename, pkg["data"].title if pkg.get("data") is not None else (title or auto))
        except Exception:
            log.exception("Report build failed")
            st.error("The report could not be built for this dataset.")
            pkg = None
        st.session_state[_PKG] = pkg
        st.session_state.pop(_CSV, None)
    if pkg is None:
        _csv_download(df, PurePath(ds.filename).stem)
        return
    if pkg["key"] != key:
        st.info("The dataset, roles, cleaning or the options above changed since this report was built. Build it again to refresh.")
        return
    _downloads(pkg, safe_filename(PurePath(ds.filename).stem))
    _csv_download(df, safe_filename(PurePath(ds.filename).stem))


def _downloads(pkg: dict, stem: str) -> None:
    data, files = pkg["data"], pkg["files"]
    labels = {"pdf": "PDF", "html": "HTML"}
    for fmt in pkg["errors"]:
        st.warning(f"The {labels.get(fmt, fmt)} file could not be created. The other format is unaffected.")
    for line in pkg.get("skipped", []):
        st.caption(f"Skipped: {line}")
    if data is not None:
        st.markdown(f"**{data.title}** · report {data.report_id} · {len(data.included)} section(s) included, {len(data.skipped)} not included")
    specs = [("pdf", "PDF report", ".pdf", "application/pdf"), ("html", "HTML report", ".html", "text/html")]
    for fmt, label, ext, mime in specs:
        if fmt in files:
            st.download_button(f"{label} ({len(files[fmt]) / 1024:,.0f} KB)", files[fmt], file_name=f"{stem}_report{ext}", mime=mime,
                               key=f"dl_{fmt}", type="primary" if fmt == "pdf" else "secondary")
    if "log" in files:
        st.download_button("Transformation log (CSV)", files["log"], file_name=f"{stem}_transformation_log.csv", mime="text/csv", key="dl_log")
    if data is not None:
        _preview(data)


def _preview(data) -> None:
    """The report's executive summary as it appears on its first page, so the reader knows what they are downloading."""
    with st.container(border=True):
        st.markdown("**Executive summary preview**")
        if data.lede:
            st.write(data.lede)
        if data.tiles:
            cols = st.columns(len(data.tiles))
            for col, t in zip(cols, data.tiles):
                col.metric(t.label, t.value, t.note or None, delta_color={"good": "normal", "bad": "inverse", "warn": "inverse"}.get(t.status, "off"),
                           delta_arrow="off")
        if data.actions:
            st.markdown("**Priority actions**")
            st.markdown("\n".join(f"{k}. {a.action}" for k, a in enumerate(data.actions, 1)))
    with st.expander("Contents of the report"):
        rows = [{"Section": "Executive summary", "Part": "Summary"}]
        rows += [{"Section": f"{s.label}  {s.title}", "Part": "Body" if s.part == "main" else "Appendix"}
                 for s in data.sections if s.status == "ok"]
        rows.append({"Section": f"{data.method_label}  Methodology and limitations", "Part": "Appendix"})
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        if data.skipped:
            st.caption("Not included: " + "; ".join(f"{t} ({r.rstrip('.')})" for t, r in data.skipped) + ".")


def _csv_download(df: pd.DataFrame, stem: str) -> None:
    """The analysed data as CSV, built only when asked for (it can take seconds and tens of MB on large files)."""
    cs = state.get_cleaning()
    with st.expander("Analysed dataset (CSV)"):
        cached = st.session_state.get(_CSV)
        if cached is None or cached[0] != base_key():
            if st.button("Prepare CSV", key="rep_csv_prep"):
                data = cs.csv_bytes if cs is not None and cs.csv_bytes is not None else df.to_csv(index=False).encode("utf-8")
                st.session_state[_CSV] = cached = (base_key(), data)
        if cached is not None and cached[0] == base_key():
            st.download_button(f"Analysed dataset (CSV, {len(cached[1]) / 1024 / 1024:,.1f} MB)", cached[1], file_name=f"{stem}_data.csv",
                               mime="text/csv", key="dl_csv")

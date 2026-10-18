"""Data quality page: issues, missing values and duplicate records."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from cleaning.cleaner import apply_actions, select_with_dependencies
from config.settings import Settings
from core.exceptions import BIError
from core.logging_config import get_logger
from ingestion.profiler import profile_dataframe
from ingestion.quality_report import build_quality_report
from ui import state
from ui.layout import fmt_int, no_dataset_notice, page_title

log = get_logger("ui.quality")
_STATUS = {"good": "Good", "review": "Needs review", "poor": "Problems found"}


def render_data_quality_page(settings: Settings) -> None:
    page_title("Data quality", "Issues found in the uploaded data and the cleaning actions you can apply. "
                               "The uploaded data is never modified.")
    result = state.get_result()
    if result is None:
        no_dataset_notice()
        return
    p, q, df = result.profile, result.quality, result.dataset.df
    c = st.columns(4)
    c[0].metric("Status", _STATUS[q.status])
    c[1].metric("Completeness", f"{q.completeness_pct:.1f}%")
    c[2].metric("Duplicate rows", fmt_int(p.duplicate_rows))
    c[3].metric("Issues", f"{len(q.by_severity('high'))} high · {len(q.by_severity('medium'))} medium "
                          f"· {len(q.by_severity('low'))} low")

    t_issues, t_missing, t_dups, t_clean, t_hist = st.tabs(
        ["Issues", "Missing values", "Duplicates", "Cleaning", "History"])
    with t_issues:
        if not q.issues:
            st.success("No data quality issues were detected.")
        else:
            sev = st.multiselect("Severity", ["high", "medium", "low"], default=["high", "medium", "low"])
            recs = [i for i in q.to_records() if i["severity"] in sev]
            table = pd.DataFrame(recs).rename(columns={
                "severity": "Severity", "code": "Issue", "column": "Column", "message": "Details",
                "affected_rows": "Rows affected", "suggestion": "Suggested action"})
            st.dataframe(table, hide_index=True)
        st.caption("Use the Cleaning tab to review and apply fixes for these issues.")

    with t_missing:
        miss = pd.DataFrame([{"Column": c.name, "Missing": c.missing, "Missing %": c.missing_pct}
                             for c in p.columns_profile if c.missing]).sort_values("Missing", ascending=False) \
            if any(c.missing for c in p.columns_profile) else pd.DataFrame()
        if miss.empty:
            st.success("No missing values.")
        else:
            st.dataframe(miss, hide_index=True, column_config={
                "Missing %": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100)})

    with t_dups:
        if not p.duplicate_rows:
            st.success("No exact duplicate rows.")
        else:
            st.caption(f"{fmt_int(p.duplicate_rows)} rows repeat an earlier row. "
                       "Duplicates may be legitimate, so review before removing any.")
            dup = df[df.duplicated(keep=False)]
            st.dataframe(dup.head(settings.preview_rows), hide_index=True)
            if len(dup) > settings.preview_rows:
                st.caption(f"Showing the first {settings.preview_rows} of {fmt_int(len(dup))} rows involved.")

    with t_clean:
        _render_cleaning(result, settings)
    with t_hist:
        _render_history()


_RISK = {"safe": "Safe", "review": "Review", "destructive": "Removes data"}


def _render_cleaning(result, settings: Settings) -> None:
    proposals = state.get_proposals()
    if not proposals:
        st.success("No cleaning actions are suggested for this dataset.")
        return
    st.caption("Tick the actions to apply, then press Apply. Only safe format fixes are pre-ticked; "
               "anything that fills values, caps outliers or removes rows or columns is your decision.")
    table = pd.DataFrame([{
        "id": a.id, "Apply": a.selected_by_default, "Column": a.column or "(all columns)",
        "Found": a.condition, "Proposed action": a.proposed_action, "Rows affected": a.affected_rows,
        "Why": a.reason, "Expected consequence": a.consequence, "Risk": _RISK[a.risk]} for a in proposals]
    ).set_index("id")
    edited = st.data_editor(
        table, hide_index=True, key=f"clean_editor_{result.dataset.fingerprint}",
        disabled=[c for c in table.columns if c != "Apply"],
        column_config={"Apply": st.column_config.CheckboxColumn(width="small"),
                       "Rows affected": st.column_config.NumberColumn(format="%d")})
    chosen_ids = list(edited.index[edited["Apply"]])
    chosen = select_with_dependencies(proposals, chosen_ids)
    extra = len(chosen) - len(chosen_ids)
    if extra > 0:
        st.caption(f"{extra} prerequisite action(s) (such as a type conversion) will be added automatically.")
    risky = [a for a in chosen if a.risk == "destructive"]
    if risky:
        st.warning(f"{len(risky)} selected action(s) remove rows or columns. Check the 'Rows affected' column first.")
    left, right = st.columns([1, 1])
    if left.button("Apply selected actions", type="primary", disabled=not chosen):
        try:
            with st.spinner("Applying cleaning actions..."):
                res = apply_actions(result.dataset.df, chosen, source_fingerprint=result.dataset.fingerprint)
                prof = profile_dataframe(res.df)
                state.set_cleaning(state.CleaningState(res, prof, build_quality_report(res.df, prof)))
            st.rerun()
        except BIError as exc:
            st.error(exc.user_message)
        except Exception:
            log.exception("Cleaning failed")
            st.error("Cleaning failed unexpectedly. The original data is unchanged. Details are in the log.")
    cs = state.get_cleaning()
    if right.button("Discard cleaned data", disabled=cs is None):
        state.clear_cleaning()
        st.rerun()

    if cs is not None:
        st.subheader("Before and after")
        r, p0, q0 = cs.result, result.profile, result.quality
        rows = [("Rows", p0.rows, cs.profile.rows), ("Columns", p0.columns, cs.profile.columns),
                ("Completeness %", q0.completeness_pct, cs.quality.completeness_pct),
                ("Duplicate rows", p0.duplicate_rows, cs.profile.duplicate_rows),
                ("Quality issues", len(q0.issues), len(cs.quality.issues))]
        st.dataframe(pd.DataFrame(rows, columns=["Measure", "Original", "Cleaned"]), hide_index=True)
        if r.skipped_count:
            st.info(f"{r.skipped_count} action(s) were skipped. See the History tab for the reasons.")
        st.caption("Cleaned data is now the active dataset for later analysis. Open History to download it.")


def _render_history() -> None:
    cs = state.get_cleaning()
    if cs is None:
        st.caption("No cleaning has been applied yet. Original data is in use.")
        return
    r = cs.result
    st.caption(f"{r.applied_count} applied · {r.skipped_count} skipped · "
               f"{fmt_int(r.rows_before)} -> {fmt_int(r.rows_after)} rows · "
               f"{r.cols_before} -> {r.cols_after} columns · settings fingerprint {r.actions_fingerprint}")
    log_df = pd.DataFrame(r.log_records())
    st.dataframe(log_df, hide_index=True)
    if r.rename_map:
        st.caption("Original column names: " + ", ".join(f"{new} = {old}" for new, old in r.rename_map.items()))
    c1, c2 = st.columns(2)
    c1.download_button("Download transformation log (CSV)", log_df.to_csv(index=False).encode("utf-8"),
                       file_name="transformation_log.csv", mime="text/csv")
    if cs.csv_bytes is None:
        cs.csv_bytes = r.df.to_csv(index=False).encode("utf-8")
    c2.download_button("Download cleaned data (CSV)", cs.csv_bytes, file_name="cleaned_data.csv", mime="text/csv")

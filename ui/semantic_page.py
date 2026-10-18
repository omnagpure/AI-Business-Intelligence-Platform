"""Semantics page: what each column represents and what business area the data belongs to."""
from __future__ import annotations

from html import escape

import pandas as pd
import streamlit as st

from core.exceptions import BIError
from core.logging_config import get_logger
from semantic.column_lexicon import DOMAIN_LABELS, DOMAINS, ROLE_LABELS, ROLES
from semantic.semantic_model import SemanticModel
from ui import state
from ui.layout import badge, eyebrow, no_dataset_notice, note_list, page_title

log = get_logger("ui.semantic")
_KIND = {"numeric": "Numeric", "categorical": "Categorical", "temporal": "Date/time", "boolean": "Boolean",
         "text": "Free text", "identifier": "Identifier", "empty": "Empty"}
_SOURCE = {"rules": "Auto", "model": "Model", "user": "You"}
_LABEL_TO_ROLE = {v: k for k, v in ROLE_LABELS.items()}
_AUTO = "Auto-detect"


def _pct(x: float) -> str:
    return f"{round(x * 100)}%"


def _render_domain(model: SemanticModel) -> None:
    d, ov = model.domain, state.get_semantic_overrides()
    left, right = st.columns([3, 2], gap="large")
    with left:
        eyebrow("Business domain")
        tone = "accent" if d.source == "user" else ("neutral" if d.domain == "general" else "accent")
        tag = badge("Set by you" if d.source == "user" else f"Detected · {_pct(d.confidence)} confidence", tone)
        st.markdown(f'<div class="kv-title">{escape(DOMAIN_LABELS[d.domain])} {tag}</div>'
                    + note_list(list(d.evidence)), unsafe_allow_html=True)
        if d.alternatives:
            st.caption("Also considered: " + ", ".join(f"{DOMAIN_LABELS[k]} {_pct(v)}" for k, v in d.alternatives))
        if d.domain == "general" and d.source != "user":
            st.caption("Low domain confidence does not limit analysis. General-purpose statistics still apply.")
    with right:
        options = [_AUTO] + [DOMAIN_LABELS[x] for x in DOMAINS]
        current = _AUTO if ov["domain"] is None else DOMAIN_LABELS[ov["domain"]]
        choice = st.selectbox("Override domain", options, index=options.index(current), key="sem_domain_choice",
                              help="Choose a domain manually if the detected one is wrong.")
        if choice != current:
            new = None if choice == _AUTO else next(k for k, v in DOMAIN_LABELS.items() if v == choice)
            state.set_semantic_overrides(ov["roles"], new)
            st.rerun()


def _table(model: SemanticModel, flt: str) -> pd.DataFrame:
    rows = []
    for c in model.columns.values():
        if flt == "Needs review" and c.column not in model.needs_review():
            continue
        if flt == "Overridden" and c.source != "user":
            continue
        rows.append({
            "Column": c.column, "Kind": _KIND.get(c.kind, c.kind), "Role": ROLE_LABELS[c.role],
            "Confidence": round(c.confidence * 100), "Source": _SOURCE[c.source],
            "Evidence": "; ".join(c.evidence),
            "Also possible": ", ".join(f"{ROLE_LABELS[r]} {_pct(s)}" for r, s in c.alternatives)})
    return pd.DataFrame(rows, columns=["Column", "Kind", "Role", "Confidence", "Source", "Evidence",
                                       "Also possible"]).set_index("Column")


def _render_columns(model: SemanticModel) -> None:
    ov = state.get_semantic_overrides()
    n = len(model.columns)
    strong = sum(c.confidence >= 0.7 for c in model.columns.values())
    c = st.columns(4)
    c[0].metric("Columns analysed", n)
    c[1].metric("High confidence", strong)
    c[2].metric("Needs review", len(model.needs_review()))
    c[3].metric("Your overrides", len(model.overridden()))

    st.subheader("Column roles")
    flt = st.radio("Show", ["All", "Needs review", "Overridden"], horizontal=True, label_visibility="collapsed")
    table = _table(model, flt)
    if table.empty:
        st.caption("No columns match this filter.")
        return
    st.caption("Change a role in the Role column, then press Save. Later analysis uses your corrected meaning.")
    edited = st.data_editor(
        table, key=f"sem_editor_{model.fingerprint()}_{state.semantic_editor_version()}_{flt}",
        disabled=[x for x in table.columns if x != "Role"],
        column_config={
            "Role": st.column_config.SelectboxColumn(options=[ROLE_LABELS[r] for r in ROLES], required=True),
            "Confidence": st.column_config.ProgressColumn(format="%d%%", min_value=0, max_value=100),
            "Evidence": st.column_config.TextColumn(width="large")})
    changed = {col: lbl for col, lbl in edited["Role"].items() if lbl != table.at[col, "Role"]}

    left, right, _ = st.columns([1, 1, 4])
    if left.button("Save changes", type="primary", disabled=not changed):
        roles = dict(ov["roles"])
        for col, label in changed.items():
            cur = model.columns[col]
            auto = cur.alternatives[0][0] if cur.source == "user" and cur.alternatives else cur.role
            role = _LABEL_TO_ROLE[label]
            if role == auto:
                roles.pop(col, None)
            else:
                roles[col] = role
        state.set_semantic_overrides(roles, ov["domain"])
        st.rerun()
    if right.button("Reset all", disabled=not (ov["roles"] or ov["domain"])):
        state.set_semantic_overrides({}, None)
        st.rerun()


def render_semantic_page() -> None:
    page_title("Semantics", "What each column represents and what kind of business the data describes. "
                            "Corrections you make here apply to all later analysis.")
    if state.get_result() is None:
        no_dataset_notice()
        return
    try:
        model = state.get_semantic_model()
    except BIError as exc:
        st.error(exc.user_message)
        return
    except Exception:
        log.exception("Semantic analysis failed")
        st.error("Column analysis failed unexpectedly. Your data is unchanged. Details were written to the log.")
        return
    if model is None:
        no_dataset_notice()
        return
    if state.get_cleaning() is not None:
        st.caption("Based on the cleaned dataset.")
    _render_domain(model)
    st.divider()
    _render_columns(model)
    with st.expander("How suggestions are made"):
        st.markdown(
            "Each column is scored against every role using its name (vocabulary and phrases), its detected "
            "type, its value distribution (for example 0 to 1 for rates, whole numbers for quantities) and "
            "relationships between columns (for example revenue = quantity x price). Scores are heuristics, "
            "not probabilities. Columns with no strong signal stay Unknown rather than being guessed, and "
            "columns whose names suggest personal data are excluded. The domain is inferred from column "
            "vocabulary plus the roles found. Check anything marked for review.")

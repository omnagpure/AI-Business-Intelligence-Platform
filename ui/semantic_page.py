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
from ui.layout import badge, chips, empty_state, eyebrow, no_dataset_notice, note_list, page_title

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


def _render_summary(model: SemanticModel) -> None:
    n = len(model.columns)
    strong = sum(c.confidence >= 0.7 for c in model.columns.values())
    c = st.columns(4)
    c[0].metric("Columns analysed", n)
    c[1].metric("High confidence", strong)
    c[2].metric("Needs review", len(model.needs_review()))
    c[3].metric("Your overrides", len(model.overridden()))


def _render_editor(model: SemanticModel) -> None:
    ov = state.get_semantic_overrides()
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


def _render_review(model: SemanticModel) -> None:
    flagged = model.needs_review()
    if not flagged:
        empty_state("Nothing to review", "Every column was classified with reasonable confidence or set by you.")
        return
    st.caption("Low-confidence suggestions. Pick the right meaning with one click, or leave a column as it is.")
    ov = state.get_semantic_overrides()
    shown = flagged[:12]
    for col in shown:
        c = model.columns[col]
        options = [r for r, _ in c.alternatives if r != c.role][:2]
        if c.role != "unknown":
            options.append("unknown")
        with st.container(border=True):
            left, right = st.columns([3, 2], vertical_alignment="center")
            with left:
                tone = "warn" if c.confidence < 0.3 else "neutral"
                st.markdown(f'<div class="kv-title" style="font-size:.95rem">{escape(col)} '
                            f'{badge(ROLE_LABELS[c.role] + " · " + _pct(c.confidence), tone)}</div>'
                            + note_list(list(c.evidence)[:3]), unsafe_allow_html=True)
            with right:
                for role in options:
                    if st.button(f"Use {ROLE_LABELS[role]}", key=f"rev_{model.fingerprint()}_{col}_{role}",
                                 use_container_width=True):
                        roles = dict(ov["roles"])
                        roles[col] = role
                        state.set_semantic_overrides(roles, ov["domain"])
                        st.rerun()
    if len(flagged) > len(shown):
        st.caption(f"{len(flagged) - len(shown)} more in the Column roles tab (filter: Needs review).")


def _render_mix(model: SemanticModel) -> None:
    groups: dict[str, list[str]] = {}
    for c in model.columns.values():
        groups.setdefault(c.role, []).append(c.column)
    counts = pd.Series({ROLE_LABELS[r]: len(v) for r, v in sorted(groups.items(), key=lambda kv: -len(kv[1]))})
    left, right = st.columns([2, 3], gap="large")
    with left:
        eyebrow("Columns per role")
        st.bar_chart(counts, horizontal=True, height=max(180, 32 * len(counts)))
    with right:
        eyebrow("Where each role was found")
        for role, cols in sorted(groups.items(), key=lambda kv: (kv[0] == "unknown", -len(kv[1]))):
            st.markdown(f"**{ROLE_LABELS[role]}**")
            chips([(" ", col) for col in cols[:12]] + ([(" ", f"+{len(cols) - 12} more")] if len(cols) > 12 else []))


def _pct_str(x: float) -> str:
    return f"{x * 100:.0f}%"


def _render_model_tab() -> None:
    status, use = state.get_name_model_status(), state.get_use_model()
    st.markdown("An optional helper that reads **only the column name** and suggests a role. "
                "Its opinion is blended with the rules; your overrides always win. Column values never reach it.")
    if not status.available:
        st.info(status.reason)
        st.caption("To train it (optional), run these from the project folder, then press Check again:")
        st.code("python -m training.make_data\npython -m training.train_semantic_model\n"
                "python -m training.evaluate_semantic_model", language="powershell")
        if st.button("Check again"):
            state.reload_name_model()
            st.rerun()
        return
    new = st.toggle("Blend the trained name model into suggestions", value=use,
                    help="Off by default. Turning it on re-runs the column analysis.")
    if new != use:
        state.set_use_model(new)
        st.rerun()
    c = st.columns(3)
    c[0].metric("Model version", status.version)
    c[1].metric("Trained", status.trained_at or "n/a")
    c[2].metric("Roles known", status.n_roles)
    m = status.metrics
    if m:
        st.markdown("**Evaluation on names it never saw** (model vs the rules using names alone)")
        st.dataframe(pd.DataFrame([
            {"Test set": f"Held-out synthetic names (n={m['held_out_n']})",
             "Trained model": _pct_str(m["held_out_accuracy"]), "Rules, names only": _pct_str(m["baseline_accuracy"])},
            {"Test set": f"Hand-labelled real headers (n={m['real_headers_n']})",
             "Trained model": _pct_str(m["real_headers_accuracy"]), "Rules, names only": _pct_str(m["real_baseline_accuracy"])},
        ]), hide_index=True)
        st.caption("Synthetic and hand-labelled names only. This does not measure accuracy on your own data. "
                   "Where the model does not beat the rules, treat it as a second opinion, not an upgrade.")
    else:
        st.caption("No evaluation results match this model file. Run python -m training.evaluate_semantic_model.")
    if st.button("Check again"):
        state.reload_name_model()
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
    _render_summary(model)
    n_review = len(model.needs_review())
    t_cols, t_review, t_mix, t_model = st.tabs(
        ["Column roles", f"Review queue ({n_review})" if n_review else "Review queue", "Role mix", "Trained model"])
    with t_cols:
        _render_editor(model)
    with t_review:
        _render_review(model)
    with t_mix:
        _render_mix(model)
    with t_model:
        _render_model_tab()
    with st.expander("How suggestions are made"):
        st.markdown(
            "Each column is scored against every role using its name (vocabulary and phrases), its detected "
            "type, its value distribution (for example 0 to 1 for rates, whole numbers for quantities) and "
            "relationships between columns (for example revenue = quantity x price). Scores are heuristics, "
            "not probabilities. Columns with no strong signal stay Unknown rather than being guessed, and "
            "columns whose names suggest personal data are excluded. The domain is inferred from column "
            "vocabulary plus the roles found. If you switch on the optional trained name model (Trained model tab), its "
            "name-only opinion is blended in at a small weight. Check anything marked for review.")

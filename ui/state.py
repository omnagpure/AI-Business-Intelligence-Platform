"""Thin wrapper over st.session_state so pages never touch raw keys."""
from __future__ import annotations

import streamlit as st

import json
from dataclasses import dataclass

import pandas as pd

from cleaning.cleaner import CleaningResult, propose_actions
from cleaning.actions import CleaningAction
from ingestion.pipeline import IngestionResult
from ingestion.profiler import DatasetProfile
from ingestion.quality_report import QualityReport
from semantic.semantic_model import SemanticModel, build_semantic_model
from semantic.trained_name_model import ModelStatus, TrainedNameModel, load_name_model

_KEY = "ingestion_result"
_SHEETS = "sheet_cache"
_CLEAN = "cleaning_state"
_PROPS = "cleaning_proposals"
_SEM_OVR = "semantic_overrides"
_SEM = "semantic_model_cache"
_SEM_VER = "semantic_editor_version"
_USE_MODEL = "use_name_model"


@dataclass
class CleaningState:
    """Cleaned dataset plus its own profile/quality report (computed once, when applied)."""
    result: CleaningResult
    profile: DatasetProfile
    quality: QualityReport
    csv_bytes: bytes | None = None


def get_result() -> IngestionResult | None:
    return st.session_state.get(_KEY)


def set_result(result: IngestionResult) -> None:
    st.session_state[_KEY] = result
    st.session_state.pop(_CLEAN, None)     # a new dataset invalidates cleaning and proposals
    st.session_state.pop(_PROPS, None)
    for key in (_SEM_OVR, _SEM):
        st.session_state.pop(key, None)


def clear_result() -> None:
    for key in (_KEY, _SHEETS, _CLEAN, _PROPS, _SEM_OVR, _SEM):
        st.session_state.pop(key, None)


def cached_sheets(file_id: str) -> list[str] | None:
    entry = st.session_state.get(_SHEETS)
    return entry[1] if entry and entry[0] == file_id else None


def cache_sheets(file_id: str, sheets: list[str]) -> None:
    st.session_state[_SHEETS] = (file_id, sheets)


def get_proposals() -> list[CleaningAction]:
    """Cleaning proposals for the active dataset, computed once per dataset fingerprint."""
    result = get_result()
    if result is None:
        return []
    fp = result.dataset.fingerprint
    cached = st.session_state.get(_PROPS)
    if cached is None or cached[0] != fp:
        cached = (fp, propose_actions(result.dataset.df, result.profile))
        st.session_state[_PROPS] = cached
    return cached[1]


def get_cleaning() -> CleaningState | None:
    """Cleaning state, only if it was built from the dataset that is active now."""
    cs = st.session_state.get(_CLEAN)
    result = get_result()
    if cs is None or result is None or cs.result.source_fingerprint != result.dataset.fingerprint:
        return None
    return cs


def set_cleaning(cs: CleaningState) -> None:
    st.session_state[_CLEAN] = cs


def clear_cleaning() -> None:
    st.session_state.pop(_CLEAN, None)


def get_analysis_df() -> pd.DataFrame | None:
    """The dataset later analytics should use: cleaned data if cleaning was applied, else the original."""
    cs = get_cleaning()
    if cs is not None:
        return cs.result.df
    result = get_result()
    return result.dataset.df if result is not None else None


def get_analysis_profile() -> DatasetProfile | None:
    cs = get_cleaning()
    if cs is not None:
        return cs.profile
    result = get_result()
    return result.profile if result is not None else None


def get_semantic_overrides() -> dict:
    ov = st.session_state.get(_SEM_OVR) or {"roles": {}, "domain": None}
    return {"roles": dict(ov["roles"]), "domain": ov["domain"]}


def set_semantic_overrides(roles: dict[str, str], domain: str | None) -> None:
    st.session_state[_SEM_OVR] = {"roles": dict(roles), "domain": domain}
    st.session_state[_SEM_VER] = st.session_state.get(_SEM_VER, 0) + 1


def semantic_editor_version() -> int:
    return st.session_state.get(_SEM_VER, 0)


def get_semantic_model() -> SemanticModel | None:
    """Semantic model for the analysis dataset. Rebuilt only when the data, cleaning or overrides change."""
    result, df, profile = get_result(), get_analysis_df(), get_analysis_profile()
    if result is None or df is None or profile is None:
        return None
    cs, ov = get_cleaning(), get_semantic_overrides()
    nm = get_active_name_model()
    key = (result.dataset.fingerprint, cs.result.actions_fingerprint if cs else "",
           json.dumps(ov, sort_keys=True), nm.version if nm else "rules-only")
    cached = st.session_state.get(_SEM)
    if cached is not None and cached[0] == key:
        return cached[1]
    model = build_semantic_model(df, profile, ov["roles"], ov["domain"], get_active_name_model())
    st.session_state[_SEM] = (key, model)
    return model


# ---- optional trained name model -------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _load_name_model_cached() -> tuple[TrainedNameModel | None, ModelStatus]:
    return load_name_model()


def get_name_model_status() -> ModelStatus:
    return _load_name_model_cached()[1]


def reload_name_model() -> None:
    """Re-check the model folder (after training) and refresh dependent results."""
    _load_name_model_cached.clear()
    st.session_state.pop(_SEM, None)


def get_use_model() -> bool:
    """User preference. Off by default: the model is optional and only blended in when switched on."""
    return bool(st.session_state.get(_USE_MODEL, False))


def set_use_model(value: bool) -> None:
    st.session_state[_USE_MODEL] = bool(value)
    st.session_state[_SEM_VER] = st.session_state.get(_SEM_VER, 0) + 1


def get_active_name_model() -> TrainedNameModel | None:
    model, _ = _load_name_model_cached()
    return model if (model is not None and get_use_model()) else None

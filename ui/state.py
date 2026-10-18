"""Thin wrapper over st.session_state so pages never touch raw keys."""
from __future__ import annotations

import streamlit as st

from ingestion.pipeline import IngestionResult

_KEY = "ingestion_result"
_SHEETS = "sheet_cache"


def get_result() -> IngestionResult | None:
    return st.session_state.get(_KEY)


def set_result(result: IngestionResult) -> None:
    st.session_state[_KEY] = result


def clear_result() -> None:
    st.session_state.pop(_KEY, None)
    st.session_state.pop(_SHEETS, None)


def cached_sheets(file_id: str) -> list[str] | None:
    entry = st.session_state.get(_SHEETS)
    return entry[1] if entry and entry[0] == file_id else None


def cache_sheets(file_id: str, sheets: list[str]) -> None:
    st.session_state[_SHEETS] = (file_id, sheets)

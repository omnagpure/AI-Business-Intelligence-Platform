"""Data structures shared by all cleaning modules. No Streamlit, no I/O."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd
from pandas.api import types as pdt

RISK_SAFE = "safe"                # changes representation only (e.g. text -> number)
RISK_REVIEW = "review"            # changes values; check the effect
RISK_DESTRUCTIVE = "destructive"  # removes rows or columns


class SkipAction(Exception):
    """Raised by an operation that cannot run on the current data. The reason is logged."""


def is_text_series(s: pd.Series) -> bool:
    return pdt.is_object_dtype(s) or pdt.is_string_dtype(s)


def is_numeric_series(s: pd.Series) -> bool:
    return pdt.is_numeric_dtype(s) and not pdt.is_bool_dtype(s)


def count_changed(old: pd.Series, new: pd.Series) -> int:
    """Cells whose value differs between two aligned Series (missing == missing is unchanged)."""
    diff = old.astype(object) != new.astype(object)
    diff &= ~(old.isna() & new.isna())
    return int(diff.sum())


@dataclass(frozen=True)
class CleaningAction:
    """A proposed transformation. `params` hold everything needed to replay it deterministically."""
    id: str
    op: str
    column: str | None
    params: dict[str, Any]
    title: str
    condition: str          # what was found in the data
    proposed_action: str    # what will be done
    affected_rows: int
    reason: str
    consequence: str        # expected effect of applying
    risk: str = RISK_REVIEW
    selected_by_default: bool = False
    group: str | None = None            # only one action per group is applied
    requires: tuple[str, ...] = ()      # ids of actions that must run first

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LogEntry:
    step: int
    action_id: str
    op: str
    column: str | None
    status: str             # "applied" | "skipped"
    rows_before: int
    rows_after: int
    cells_changed: int
    detail: str
    params: dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""

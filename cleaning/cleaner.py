"""Cleaning pipeline: propose actions, apply the chosen ones, keep a reproducible log.

The input DataFrame is never modified. Applying the same actions to the same input always gives
the same output (all values an action needs, such as bounds and label maps, are stored in its params).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd

from cleaning import duplicate_handler, labels, missing_values, outlier_handler, type_inference
from cleaning.actions import CleaningAction, LogEntry, SkipAction
from config.analytics_config import DEFAULT_CLEANING, CleaningConfig
from core.exceptions import CleaningError
from ingestion.profiler import DatasetProfile

OPS = {**type_inference.OPS, **labels.OPS, **missing_values.OPS, **duplicate_handler.OPS, **outlier_handler.OPS}

# Execution order. Labels are standardised before trimming (their maps use raw values); types are
# converted before value-based steps; removals come late; renaming is last so every action can
# refer to the original column names.
_ORDER = ["standardize_labels", "strip_whitespace", "convert_numeric", "convert_datetime",
          "convert_boolean", "negative_to_missing", "fill_missing", "clip_outliers",
          "drop_outlier_rows", "drop_missing_rows", "drop_duplicates", "drop_column",
          "normalize_columns"]


@dataclass
class CleaningResult:
    df: pd.DataFrame
    log: list[LogEntry]
    rename_map: dict[str, str]          # cleaned column name -> original column name
    rows_before: int
    cols_before: int
    source_fingerprint: str = ""
    actions_fingerprint: str = ""
    applied_at: str = ""

    @property
    def rows_after(self) -> int:
        return len(self.df)

    @property
    def cols_after(self) -> int:
        return self.df.shape[1]

    @property
    def applied_count(self) -> int:
        return sum(e.status == "applied" for e in self.log)

    @property
    def skipped_count(self) -> int:
        return sum(e.status == "skipped" for e in self.log)

    def log_records(self) -> list[dict]:
        return [{"step": e.step, "action": e.action_id, "operation": e.op, "column": e.column,
                 "status": e.status, "rows_before": e.rows_before, "rows_after": e.rows_after,
                 "cells_changed": e.cells_changed, "detail": e.detail,
                 "parameters": json.dumps(e.params, default=str, sort_keys=True)} for e in self.log]


def propose_actions(df: pd.DataFrame, profile: DatasetProfile,
                    cfg: CleaningConfig = DEFAULT_CLEANING) -> list[CleaningAction]:
    """Inspect the data and propose cleaning actions. Reads only; nothing is applied."""
    actions: list[CleaningAction] = []
    actions += type_inference.propose_type_actions(df, profile)
    actions += labels.propose_label_actions(df, profile, cfg)
    actions += missing_values.propose_missing_actions(df, profile, cfg)
    actions += duplicate_handler.propose_duplicate_actions(df, profile.duplicate_rows, cfg)
    actions += outlier_handler.propose_outlier_actions(df, profile, cfg)
    for c in profile.columns_profile:
        if c.is_constant and c.kind != "empty":
            actions.append(CleaningAction(
                f"drop_column:{c.name}", "drop_column", c.name, {}, "Drop constant column",
                "Every value is identical.", "Remove the column.", 0,
                "A constant column carries no information for analysis.",
                "The column disappears from the cleaned data.", "destructive", False))
    actions += labels.propose_name_action(df)
    seen: set[str] = set()
    unique = []
    for a in actions:               # a column can be proposed for dropping by more than one rule
        if a.id not in seen:
            seen.add(a.id)
            unique.append(a)
    return unique


def select_with_dependencies(all_actions: list[CleaningAction], selected_ids: list[str]) -> list[CleaningAction]:
    """Return the selected actions plus anything they require, in proposal order."""
    by_id = {a.id: a for a in all_actions}
    wanted: set[str] = set()
    stack = [i for i in selected_ids if i in by_id]
    while stack:
        i = stack.pop()
        if i in wanted:
            continue
        wanted.add(i)
        stack += [r for r in by_id[i].requires if r in by_id]
    return [a for a in all_actions if a.id in wanted]


def actions_fingerprint(actions: list[CleaningAction]) -> str:
    """Stable hash of what will be done (independent of selection order). Use as a cache key."""
    payload = sorted(([a.id, a.op, a.column, a.params] for a in actions), key=lambda x: x[0])
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


def apply_actions(df: pd.DataFrame, actions: list[CleaningAction],
                  source_fingerprint: str = "") -> CleaningResult:
    """Apply actions to a copy of df. Actions that cannot run are skipped and logged, never silent."""
    for a in actions:
        if a.op not in OPS:
            raise CleaningError("A cleaning action is not recognised.", detail=f"unknown op {a.op!r}")
    work = df.copy()
    ordered = sorted(enumerate(actions), key=lambda t: (_ORDER.index(t[1].op), t[0]))
    log: list[LogEntry] = []
    used_groups: dict[str, str] = {}
    rename_map: dict[str, str] = {}

    def add(step, a, status, before, cells, detail):
        log.append(LogEntry(step, a.id, a.op, a.column, status, before, len(work), cells, detail,
                            dict(a.params), datetime.now(timezone.utc).isoformat(timespec="seconds")))

    for step, (_, a) in enumerate(ordered, 1):
        before = len(work)
        if a.group and a.group in used_groups:
            add(step, a, "skipped", before, 0, f"Conflicts with '{used_groups[a.group]}', which was already applied.")
            continue
        if a.column is not None and a.column not in work.columns:
            add(step, a, "skipped", before, 0, f"Column '{a.column}' is not present.")
            continue
        try:
            work, cells, detail = OPS[a.op](work, a.column, a.params)
        except SkipAction as exc:
            add(step, a, "skipped", before, 0, f"Skipped: {exc}.")
            continue
        if a.group:
            used_groups[a.group] = a.id
        if a.op == "normalize_columns":
            m = a.params["mapping"]
            rename_map = {new: old for old, new in m.items() if new in work.columns}
        add(step, a, "applied", before, cells, detail)

    return CleaningResult(work, log, rename_map, len(df), df.shape[1], source_fingerprint,
                          actions_fingerprint(actions), datetime.now(timezone.utc).isoformat(timespec="seconds"))

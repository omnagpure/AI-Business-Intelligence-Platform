"""Missing-value handling. Imputation is only ever proposed, never applied by default."""
from __future__ import annotations

import pandas as pd

from cleaning.actions import (RISK_DESTRUCTIVE, RISK_REVIEW, CleaningAction, SkipAction, is_numeric_series)
from config.analytics_config import CleaningConfig
from ingestion.profiler import DatasetProfile


def fill_missing(df: pd.DataFrame, column: str, params: dict) -> tuple[pd.DataFrame, int, str]:
    s = df[column]
    n = int(s.isna().sum())
    if n == 0:
        raise SkipAction("no missing values")
    strategy = params["strategy"]
    if strategy in ("median", "mean"):
        if not is_numeric_series(s):
            raise SkipAction("column is not numeric (convert its type first)")
        value = s.median() if strategy == "median" else s.mean()
        if pd.isna(value):
            raise SkipAction("no valid values to compute the fill value from")
    elif strategy == "mode":
        mode = s.mode(dropna=True)
        if mode.empty:
            raise SkipAction("no valid values to compute the fill value from")
        value = mode.iloc[0]
    elif strategy == "constant":
        value = params["value"]
        if is_numeric_series(s) and not isinstance(value, (int, float)):
            raise SkipAction("constant is not numeric")
    else:
        raise SkipAction(f"unknown strategy {strategy!r}")
    df[column] = s.fillna(value)
    return df, n, f"Filled {n} missing value(s) using {strategy} = {value!r}."


def drop_missing_rows(df: pd.DataFrame, column: str, params: dict) -> tuple[pd.DataFrame, int, str]:
    mask = df[column].isna()
    n = int(mask.sum())
    if n == 0:
        raise SkipAction("no missing values")
    return df.loc[~mask].copy(), n, f"Removed {n} row(s) where '{column}' is missing."


def drop_column(df: pd.DataFrame, column: str, params: dict) -> tuple[pd.DataFrame, int, str]:
    return df.drop(columns=[column]), len(df), f"Dropped column '{column}'."


OPS = {"fill_missing": fill_missing, "drop_missing_rows": drop_missing_rows, "drop_column": drop_column}


def propose_missing_actions(df: pd.DataFrame, profile: DatasetProfile, cfg: CleaningConfig) -> list[CleaningAction]:
    out: list[CleaningAction] = []
    for c in profile.columns_profile:
        if c.kind == "empty":
            out.append(CleaningAction(
                f"drop_column:{c.name}", "drop_column", c.name, {}, "Drop empty column",
                "Column is entirely empty.", "Remove the column.", 0, "It carries no information.",
                "The column disappears from the cleaned data.", RISK_REVIEW, True))
            continue
        if not c.missing or c.is_identifier:
            continue
        pct = c.missing_pct
        if pct >= cfg.drop_column_missing_pct:
            out.append(CleaningAction(
                f"drop_column:{c.name}", "drop_column", c.name, {}, "Drop mostly-empty column",
                f"{pct}% of values are missing.", "Remove the column.", 0,
                "Imputing most of a column would invent data.",
                "The column disappears from the cleaned data. Check collection first if it matters.",
                RISK_DESTRUCTIVE, False))
            continue
        needs = (f"convert_numeric:{c.name}",) if (c.kind == "numeric" and c.stored_as_text) else ()
        base = dict(column=c.name, affected_rows=c.missing, requires=needs, selected_by_default=False)
        cond = f"{c.missing} value(s) missing ({pct}%)."
        if c.kind == "numeric" and pct <= cfg.fill_max_missing_pct:
            out.append(CleaningAction(
                params={"strategy": "median"},
                title="Fill missing numbers with the median", condition=cond,
                proposed_action="Replace missing values with the column median.",
                reason="The median is robust to outliers; zero would distort totals and averages.",
                consequence="Averages stay similar; variance shrinks slightly. Imputed values are not real observations.",
                risk=RISK_REVIEW, group=f"missing:{c.name}", id=f"fill_missing:{c.name}", op="fill_missing", **base))
        elif c.kind == "categorical" and pct <= cfg.fill_max_missing_pct:
            out.append(CleaningAction(
                params={"strategy": "constant", "value": "Unknown"},
                title="Label missing categories 'Unknown'", condition=cond,
                proposed_action="Replace missing values with 'Unknown'.",
                reason="Keeps these rows in group-bys without guessing a category.",
                consequence="'Unknown' appears as its own category.",
                risk=RISK_REVIEW, group=f"missing:{c.name}", id=f"fill_missing:{c.name}", op="fill_missing", **base))
        elif c.kind == "temporal" and pct <= cfg.drop_rows_max_missing_pct:
            out.append(CleaningAction(
                params={},
                title="Remove rows with missing dates", condition=cond,
                proposed_action="Remove rows where this date is missing.",
                reason="Rows without a date cannot be placed on a timeline; dates should not be guessed.",
                consequence="Row count decreases; totals over time exclude these rows.",
                risk=RISK_DESTRUCTIVE, group=f"missing:{c.name}", id=f"drop_missing_rows:{c.name}",
                op="drop_missing_rows", **base))
    return out

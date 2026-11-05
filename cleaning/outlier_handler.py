"""Outlier and impossible-value handling. Nothing here is ever selected by default."""
from __future__ import annotations

import re

import pandas as pd

from cleaning.actions import (RISK_DESTRUCTIVE, RISK_REVIEW, CleaningAction, SkipAction, is_numeric_series)
from cleaning.type_inference import to_numeric_clean
from config.analytics_config import CleaningConfig
from ingestion.profiler import DatasetProfile

_NON_NEGATIVE = re.compile(r"(qty|quantity|units|count|age|price)", re.IGNORECASE)
_NO_OUTLIER_RULE = re.compile(r"(^|[^a-z])(lat|latitude|lon|lng|longitude|zip|zipcode|postal)($|[^a-z])", re.IGNORECASE)


def iqr_bounds(s: pd.Series, k: float) -> tuple[float, float] | None:
    v = s.dropna()
    if len(v) == 0:
        return None
    q1, q3 = float(v.quantile(0.25)), float(v.quantile(0.75))
    iqr = q3 - q1
    if iqr <= 0:
        return None
    return q1 - k * iqr, q3 + k * iqr


def _require_numeric(s: pd.Series) -> None:
    if not is_numeric_series(s):
        raise SkipAction("column is not numeric (convert its type first)")


def clip_outliers(df: pd.DataFrame, column: str, params: dict) -> tuple[pd.DataFrame, int, str]:
    s = df[column]
    _require_numeric(s)
    lo, hi = params["lower"], params["upper"]
    n = int(((s < lo) | (s > hi)).sum())
    if n == 0:
        raise SkipAction("no values outside the bounds")
    df[column] = s.clip(lo, hi)
    return df, n, f"Capped {n} value(s) to [{lo:.6g}, {hi:.6g}]."


def drop_outlier_rows(df: pd.DataFrame, column: str, params: dict) -> tuple[pd.DataFrame, int, str]:
    s = df[column]
    _require_numeric(s)
    lo, hi = params["lower"], params["upper"]
    mask = (s < lo) | (s > hi)
    n = int(mask.sum())
    if n == 0:
        raise SkipAction("no values outside the bounds")
    return df.loc[~mask].copy(), n, f"Removed {n} row(s) with '{column}' outside [{lo:.6g}, {hi:.6g}]."


def negative_to_missing(df: pd.DataFrame, column: str, params: dict) -> tuple[pd.DataFrame, int, str]:
    s = df[column]
    _require_numeric(s)
    mask = s < 0
    n = int(mask.sum())
    if n == 0:
        raise SkipAction("no negative values")
    df[column] = s.mask(mask)
    return df, n, f"Set {n} negative value(s) to missing."


OPS = {"clip_outliers": clip_outliers, "drop_outlier_rows": drop_outlier_rows,
       "negative_to_missing": negative_to_missing}


def propose_outlier_actions(df: pd.DataFrame, profile: DatasetProfile, cfg: CleaningConfig) -> list[CleaningAction]:
    out: list[CleaningAction] = []
    for c in profile.columns_profile:
        if c.kind != "numeric" or c.is_identifier or c.is_constant or _NO_OUTLIER_RULE.search(c.name):
            continue
        s = df[c.name]
        vals = to_numeric_clean(s) if c.stored_as_text else s
        needs = (f"convert_numeric:{c.name}",) if c.stored_as_text else ()
        if vals.notna().sum() < cfg.outlier_min_values:
            continue
        if _NON_NEGATIVE.search(c.name):
            neg = int((vals < 0).sum())
            if neg:
                out.append(CleaningAction(
                    f"negative_to_missing:{c.name}", "negative_to_missing", c.name, {},
                    "Treat negative values as missing", f"{neg} negative value(s) in '{c.name}'.",
                    "Set negative values to missing.", neg,
                    "Negatives in this kind of column can be returns/refunds or data-entry errors.",
                    "Only apply if they are errors; returns would be lost from totals.",
                    RISK_REVIEW, False, requires=needs))
        b = iqr_bounds(vals, cfg.outlier_iqr_k)
        if b is None:
            continue
        lo, hi = b
        n = int(((vals < lo) | (vals > hi)).sum())
        if not n:
            continue
        cond = (f"{n} value(s) lie outside [{lo:.6g}, {hi:.6g}] "
                f"(IQR rule, multiplier {cfg.outlier_iqr_k:g}). They may be genuine.")
        grp = f"outliers:{c.name}"
        out.append(CleaningAction(
            f"clip_outliers:{c.name}", "clip_outliers", c.name, {"lower": lo, "upper": hi},
            "Cap extreme values", cond, "Cap values at the bounds (winsorise).", n,
            "Limits the pull of extreme values on averages and models.",
            "Extreme but real values are altered; totals change.", RISK_REVIEW, False, group=grp, requires=needs))
        out.append(CleaningAction(
            f"drop_outlier_rows:{c.name}", "drop_outlier_rows", c.name, {"lower": lo, "upper": hi},
            "Remove rows with extreme values", cond, "Remove those rows.", n,
            "Use when the values are confirmed errors.",
            "Rows are lost from every analysis, not just this column.", RISK_DESTRUCTIVE, False,
            group=grp, requires=needs))
    return out

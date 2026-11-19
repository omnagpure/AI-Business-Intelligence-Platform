"""Type conversions: numbers/dates/booleans stored as text. Proposals + operations."""
from __future__ import annotations

import re

import pandas as pd

from cleaning.actions import (RISK_REVIEW, RISK_SAFE, CleaningAction, SkipAction, is_text_series)
from ingestion.profiler import DatasetProfile
from ingestion.schema_detector import parse_datetime_series

_STRIP = re.compile(r"[,\s$€£₹%]")
_PAREN_NEG = re.compile(r"^\((.*)\)$")
_TRUE = {"true", "yes", "y", "t", "1"}
_FALSE = {"false", "no", "n", "f", "0"}


def to_numeric_clean(s: pd.Series) -> pd.Series:
    """Parse text like '1,234.50', '$5', '12%' or '(3)' (= -3). Unparseable values become NaN."""
    mask = s.isna()
    t = s.astype(str).str.strip().str.replace(_PAREN_NEG, r"-\1", regex=True)
    t = t.str.replace(_STRIP, "", regex=True)
    return pd.to_numeric(t, errors="coerce").mask(mask)


def to_boolean(s: pd.Series) -> pd.Series:
    key = s.astype(str).str.strip().str.lower()
    out = pd.Series(pd.NA, index=s.index, dtype="boolean")
    out[key.isin(_TRUE) & s.notna()] = True
    out[key.isin(_FALSE) & s.notna()] = False
    return out


def _coerced(old: pd.Series, new: pd.Series) -> int:
    return int((new.isna() & old.notna()).sum())


def convert_numeric(df: pd.DataFrame, column: str, params: dict) -> tuple[pd.DataFrame, int, str]:
    s = df[column]
    if not is_text_series(s):
        raise SkipAction("column is not text")
    new = to_numeric_clean(s)
    lost = _coerced(s, new)
    df[column] = new
    msg = f"Converted {int(s.notna().sum()) - lost} value(s) to numbers."
    if lost:
        msg += f" {lost} value(s) could not be parsed and are now missing."
    return df, int(s.notna().sum()) - lost, msg


def convert_datetime(df: pd.DataFrame, column: str, params: dict) -> tuple[pd.DataFrame, int, str]:
    s = df[column]
    if not is_text_series(s):
        raise SkipAction("column is not text")
    new = parse_datetime_series(s)
    lost = _coerced(s, new)
    df[column] = new
    ok = int(s.notna().sum()) - lost
    msg = f"Parsed {ok} value(s) as dates."
    if lost:
        msg += f" {lost} value(s) could not be parsed and are now missing."
    return df, ok, msg


def convert_boolean(df: pd.DataFrame, column: str, params: dict) -> tuple[pd.DataFrame, int, str]:
    s = df[column]
    if not is_text_series(s):
        raise SkipAction("column is not text")
    new = to_boolean(s)
    lost = _coerced(s, new)
    df[column] = new
    ok = int(s.notna().sum()) - lost
    return df, ok, f"Converted {ok} value(s) to True/False." + (f" {lost} unrecognised value(s) are now missing." if lost else "")


OPS = {"convert_numeric": convert_numeric, "convert_datetime": convert_datetime,
       "convert_boolean": convert_boolean}


def propose_type_actions(df: pd.DataFrame, profile: DatasetProfile) -> list[CleaningAction]:
    out: list[CleaningAction] = []
    for c in profile.columns_profile:
        s = df[c.name]
        if not is_text_series(s):
            continue
        nn = int(s.notna().sum())
        if c.kind == "numeric" and c.stored_as_text:
            lost = _coerced(s, to_numeric_clean(s))
            out.append(CleaningAction(
                f"convert_numeric:{c.name}", "convert_numeric", c.name, {},
                "Convert text to numbers",
                f"Numbers are stored as text ({nn} values).",
                "Convert to a numeric type (currency symbols, thousands separators and % are removed).",
                nn, "Calculations and charts need real numbers.",
                f"{lost} value(s) cannot be parsed and become missing." if lost else "No values are lost.",
                RISK_SAFE, True))
        elif c.kind == "temporal" and c.stored_as_text:
            lost = _coerced(s, parse_datetime_series(s))
            out.append(CleaningAction(
                f"convert_datetime:{c.name}", "convert_datetime", c.name, {},
                "Parse text as dates",
                f"Dates are stored as text ({nn} values).",
                "Parse into a datetime column.",
                nn, "Time-based analysis requires true dates.",
                f"{lost} value(s) cannot be parsed and become missing." if lost else "No values are lost.",
                RISK_SAFE, True))
        elif c.kind == "boolean":
            out.append(CleaningAction(
                f"convert_boolean:{c.name}", "convert_boolean", c.name, {},
                "Convert yes/no text to True/False",
                f"Column holds yes/no style text ({nn} values).",
                "Convert to a True/False column.",
                nn, "Boolean columns are easier to count and filter.",
                "Values other than yes/no/true/false/y/n/t/f/1/0 become missing.",
                RISK_REVIEW, False))
    return out

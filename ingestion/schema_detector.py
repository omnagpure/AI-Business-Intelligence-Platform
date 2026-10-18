"""Infer the kind of each column. Never modifies the input Series."""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass

import pandas as pd
from pandas.api import types as pdt

from config.analytics_config import DEFAULT_THRESHOLDS, QualityThresholds

_ID_NAME = re.compile(r"(^|[^a-z])(id|uuid|guid|key|sku|code|number|no|num|ref)($|[^a-z])|id$", re.IGNORECASE)
_CURRENCY = re.compile(r"[,\s$€£₹%]")
KINDS = ("numeric", "boolean", "temporal", "categorical", "text", "identifier", "empty")


@dataclass(frozen=True)
class ColumnSchema:
    name: str
    pandas_dtype: str
    kind: str                    # one of KINDS
    stored_as_text: bool = False  # numeric/temporal values stored as strings
    is_identifier: bool = False
    is_constant: bool = False


def _sample(s: pd.Series, n: int) -> pd.Series:
    nn = s.dropna()
    return nn if len(nn) <= n else nn.sample(n, random_state=0)


def numeric_text_ratio(s: pd.Series, n: int = 2000) -> float:
    """Share of non-null string values that parse as numbers (after stripping currency/commas)."""
    sample = _sample(s, n).astype(str)
    if sample.empty:
        return 0.0
    cleaned = sample.str.replace(_CURRENCY, "", regex=True)
    return float(pd.to_numeric(cleaned, errors="coerce").notna().mean())


_DATE_FORMATS = (
    "%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S", "%Y/%m/%d",
    "%m/%d/%Y", "%m/%d/%Y %H:%M", "%m/%d/%Y %H:%M:%S", "%d/%m/%Y", "%d/%m/%Y %H:%M",
    "%d-%m-%Y", "%d-%b-%Y", "%b %d, %Y", "%d %b %Y",
)


def _try_format(sample: pd.Series, fmt: str) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(pd.to_datetime(sample, format=fmt, errors="coerce").notna().mean())


def parse_datetime_series(s: pd.Series) -> pd.Series:
    """Parse strings to datetimes (NaT where invalid). Infers one format from a sample for speed
    and consistency; falls back to flexible parsing only if no common format fits."""
    if pdt.is_datetime64_any_dtype(s):
        return s
    strs = s.astype("string")
    sample = _sample(strs, 300)
    best, best_ratio = None, 0.0
    for fmt in _DATE_FORMATS:
        r = _try_format(sample, fmt)
        if r > best_ratio:
            best, best_ratio = fmt, r
        if r >= 0.999:
            break
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if best and best_ratio >= 0.8:
            return pd.to_datetime(strs, format=best, errors="coerce")
        return pd.to_datetime(strs, errors="coerce", format="mixed")


def date_text_ratio(s: pd.Series, n: int = 500) -> float:
    sample = _sample(s, n).astype(str)
    if sample.empty:
        return 0.0
    looks = sample.str.contains(r"[-/:]|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", case=False, regex=True)
    if looks.mean() < 0.9:
        return 0.0
    return float(parse_datetime_series(sample).notna().mean())


def is_boolean_like(s: pd.Series) -> bool:
    if pdt.is_bool_dtype(s):
        return True
    vals = {str(v).strip().lower() for v in s.dropna().unique()[:10]}
    return 0 < len(vals) <= 2 and vals <= {"true", "false", "yes", "no", "y", "n", "t", "f"}


def detect_column(s: pd.Series, thresholds: QualityThresholds = DEFAULT_THRESHOLDS) -> ColumnSchema:
    name, dtype = str(s.name), str(s.dtype)
    nn = s.dropna()
    n = len(nn)
    if n == 0:
        return ColumnSchema(name, dtype, "empty", is_constant=False)
    nunique = int(nn.nunique())
    constant = nunique == 1 and len(s) > 1
    ratio = nunique / n

    if pdt.is_bool_dtype(s) or (not pdt.is_numeric_dtype(s) and is_boolean_like(s)):
        return ColumnSchema(name, dtype, "boolean", is_constant=constant)
    if pdt.is_datetime64_any_dtype(s):
        return ColumnSchema(name, dtype, "temporal", is_constant=constant)

    if pdt.is_numeric_dtype(s):
        is_int = pdt.is_integer_dtype(s) or bool((nn % 1 == 0).all())
        name_hint = bool(_ID_NAME.search(name))
        ident = (name_hint and is_int and ratio >= 0.5 and nunique > 2) or (
            is_int and ratio >= thresholds.id_unique_ratio and n >= 20 and name_hint
        )
        if ident:
            return ColumnSchema(name, dtype, "identifier", is_identifier=True, is_constant=constant)
        if nunique == 2 and set(nn.unique()) <= {0, 1}:
            return ColumnSchema(name, dtype, "boolean", is_constant=constant)
        return ColumnSchema(name, dtype, "numeric", is_constant=constant)

    # object / string columns
    if numeric_text_ratio(s) >= thresholds.numeric_text_ratio:
        txt = _sample(s, 200).astype(str)
        looks_code = txt.str.match(r"^0\d+$").mean() > 0.5  # zero-padded codes stay as text
        if not looks_code:
            return ColumnSchema(name, dtype, "numeric", stored_as_text=True, is_constant=constant)
    if date_text_ratio(s) >= thresholds.date_text_ratio:
        return ColumnSchema(name, dtype, "temporal", stored_as_text=True, is_constant=constant)

    strs = _sample(s, thresholds.sample_size).astype(str)
    avg_len = float(strs.str.len().mean())
    avg_words = float(strs.str.split().str.len().mean())
    name_hint = bool(_ID_NAME.search(name))
    if (name_hint and ratio >= 0.5) or (ratio >= thresholds.id_unique_ratio and n >= 20 and avg_words < 3):
        return ColumnSchema(name, dtype, "identifier", is_identifier=True, is_constant=constant)
    if avg_len > 50 or avg_words >= 5:
        return ColumnSchema(name, dtype, "text", is_constant=constant)
    return ColumnSchema(name, dtype, "categorical", is_constant=constant)

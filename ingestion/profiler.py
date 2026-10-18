"""Dataset profiling. Pure functions; the input DataFrame is never modified."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as pdt

from config.analytics_config import DEFAULT_THRESHOLDS, QualityThresholds
from core.exceptions import DataValidationError, ProfilingError
from core.logging_config import get_logger
from ingestion.schema_detector import ColumnSchema, detect_column, parse_datetime_series

log = get_logger("ingestion.profiler")
_TARGET_HINT = re.compile(
    r"(target|label|churn|fraud|default|outcome|converted|stockout|late|defect|status|"
    r"sales|revenue|profit|amount|price|cost|demand|quantity|units)", re.IGNORECASE)


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    kind: str
    missing: int
    missing_pct: float
    unique: int
    cardinality: str                 # low | medium | high
    examples: list[str]
    is_identifier: bool
    is_constant: bool
    stored_as_text: bool
    numeric: dict[str, float] | None = None
    top_values: list[dict[str, Any]] | None = None
    temporal: dict[str, Any] | None = None
    suspicious: list[str] = field(default_factory=list)


@dataclass
class DatasetProfile:
    rows: int
    columns: int
    memory_mb: float
    duplicate_rows: int
    duplicate_pct: float
    total_missing: int
    missing_cells_pct: float
    columns_profile: list[ColumnProfile]
    potential_primary_keys: list[str]
    possible_targets: list[str]
    kind_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def column(self, name: str) -> ColumnProfile:
        for c in self.columns_profile:
            if c.name == name:
                return c
        raise KeyError(name)


def _cardinality(unique: int, t: QualityThresholds) -> str:
    if unique <= t.low_cardinality_max:
        return "low"
    return "medium" if unique <= t.medium_cardinality_max else "high"


def _examples(nn: pd.Series, k: int = 5) -> list[str]:
    return [str(v)[:40] for v in nn.drop_duplicates().head(k)]


def _numeric_stats(s: pd.Series) -> dict[str, float] | None:
    x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if x.empty:
        return None
    q = x.quantile([0.25, 0.5, 0.75])
    return {
        "count": int(x.count()), "mean": float(x.mean()),
        "std": float(x.std()) if len(x) > 1 else 0.0,
        "min": float(x.min()), "q1": float(q[0.25]), "median": float(q[0.5]),
        "q3": float(q[0.75]), "max": float(x.max()),
        "skew": float(x.skew()) if len(x) > 2 else 0.0,
        "zeros": int((x == 0).sum()), "negatives": int((x < 0).sum()),
    }


def _temporal_stats(s: pd.Series) -> dict[str, Any] | None:
    if pdt.is_datetime64_any_dtype(s):
        d = s.dropna()
        unparsed = 0
    else:
        parsed = parse_datetime_series(s)
        unparsed = int(parsed.isna().sum() - s.isna().sum())
        d = parsed.dropna()
    if d.empty:
        return None
    return {"min": str(d.min()), "max": str(d.max()),
            "span_days": int((d.max() - d.min()).days), "unparsed": max(unparsed, 0)}


def _top_values(nn: pd.Series, total: int, k: int) -> list[dict[str, Any]]:
    vc = nn.astype(str).value_counts().head(k)
    return [{"value": str(v)[:60], "count": int(c), "pct": round(100 * c / total, 2)} for v, c in vc.items()]


def profile_column(s: pd.Series, t: QualityThresholds = DEFAULT_THRESHOLDS) -> ColumnProfile:
    schema: ColumnSchema = detect_column(s, t)
    total = len(s)
    nn = s.dropna()
    missing = total - len(nn)
    unique = int(nn.nunique())
    prof = ColumnProfile(
        name=str(s.name), dtype=str(s.dtype), kind=schema.kind, missing=missing,
        missing_pct=round(100 * missing / total, 2) if total else 0.0, unique=unique,
        cardinality=_cardinality(unique, t), examples=_examples(nn),
        is_identifier=schema.is_identifier, is_constant=schema.is_constant,
        stored_as_text=schema.stored_as_text,
    )
    if schema.kind == "numeric":
        if schema.stored_as_text:
            cleaned = s.astype(str).str.replace(r"[,\s$€£₹%]", "", regex=True).where(s.notna())
            prof.numeric = _numeric_stats(cleaned)
        else:
            prof.numeric = _numeric_stats(s)
        if prof.numeric and not schema.stored_as_text:
            inf = int(np.isinf(pd.to_numeric(s, errors="coerce")).sum())
            if inf:
                prof.suspicious.append(f"{inf} infinite value(s)")
    elif schema.kind == "temporal":
        prof.temporal = _temporal_stats(s)
    elif schema.kind in ("categorical", "boolean"):
        prof.top_values = _top_values(nn, max(len(nn), 1), t.top_n)
    elif schema.kind == "identifier" and prof.unique < len(nn):
        prof.suspicious.append(f"{len(nn) - prof.unique} repeated identifier value(s)")
    return prof


def _possible_targets(cols: list[ColumnProfile]) -> list[str]:
    out = []
    for c in cols:
        if c.is_identifier or c.is_constant or c.kind not in ("numeric", "boolean", "categorical"):
            continue
        if c.kind == "categorical" and c.unique > 10:
            continue
        if _TARGET_HINT.search(c.name):
            out.append(c.name)
    return out


def profile_dataframe(df: pd.DataFrame, thresholds: QualityThresholds = DEFAULT_THRESHOLDS) -> DatasetProfile:
    """Profile a DataFrame without modifying it."""
    if df is None or df.shape[0] == 0 or df.shape[1] == 0:
        raise DataValidationError("Cannot profile an empty dataset.")
    try:
        cols = [profile_column(df[c], thresholds) for c in df.columns]
        dup = int(df.duplicated().sum())
        cells = df.shape[0] * df.shape[1]
        missing = int(df.isna().sum().sum())
        n = len(df)
        pks = [c.name for c in cols if c.unique == n and c.missing == 0
               and (c.is_identifier or c.kind in ("categorical", "text"))]
        kinds: dict[str, int] = {}
        for c in cols:
            kinds[c.kind] = kinds.get(c.kind, 0) + 1
        return DatasetProfile(
            rows=n, columns=df.shape[1],
            memory_mb=round(float(df.memory_usage(deep=False).sum()) / 1024 / 1024, 2),
            duplicate_rows=dup, duplicate_pct=round(100 * dup / n, 2),
            total_missing=missing, missing_cells_pct=round(100 * missing / cells, 2),
            columns_profile=cols, potential_primary_keys=pks,
            possible_targets=_possible_targets(cols), kind_counts=kinds,
        )
    except DataValidationError:
        raise
    except Exception as exc:
        log.exception("Profiling failed")
        raise ProfilingError("Profiling failed unexpectedly. See the log for details.",
                             detail=type(exc).__name__) from exc

"""Correlation between suitable numeric columns only. Pure pandas."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from analytics.descriptive import numeric_series
from config.analytics_config import DEFAULT_ANALYTICS, AnalyticsConfig


@dataclass(frozen=True)
class CorrelationResult:
    matrix: pd.DataFrame = field(compare=False)
    pairs: pd.DataFrame = field(compare=False)       # a, b, r, abs_r, n, note
    excluded: dict = field(default_factory=dict, compare=False)   # column -> reason
    method: str = "pearson"
    note: str = ""


def correlation(df: pd.DataFrame, columns: list[str], method: str = "pearson",
                config: AnalyticsConfig = DEFAULT_ANALYTICS) -> CorrelationResult:
    """Correlation matrix and the strongest pairs.

    Constant columns (zero variance) and columns with too few values are excluded before the matrix is
    built and listed in `excluded`. If there are more than `corr_max_columns` candidates, the first ones
    in the given order are used and the result says so. Correlation is association, not causation.
    """
    if method not in {"pearson", "spearman"}:
        raise ValueError("method must be pearson or spearman")
    excluded: dict[str, str] = {}
    data = {}
    for col in columns:
        x = numeric_series(df[col])
        v = x.dropna()
        if len(v) < config.corr_min_periods:
            excluded[col] = f"fewer than {config.corr_min_periods} values"
        elif v.nunique() < 2 or float(v.std(ddof=0)) == 0.0:
            excluded[col] = "constant (no variation)"
        else:
            data[col] = x
    note = ""
    names = list(data)
    if len(names) > config.corr_max_columns:
        note = f"Only the first {config.corr_max_columns} of {len(names)} suitable columns were used."
        for c in names[config.corr_max_columns:]:
            excluded[c] = "not used (column cap)"
        names = names[:config.corr_max_columns]
    empty_pairs = pd.DataFrame(columns=["a", "b", "r", "abs_r", "n", "note"])
    if len(names) < 2:
        return CorrelationResult(pd.DataFrame(), empty_pairs, excluded, method,
                                 note or "Fewer than two suitable numeric columns.")
    frame = pd.DataFrame({c: data[c] for c in names})
    mat = frame.corr(method=method, min_periods=config.corr_min_periods)
    mask = frame.notna().astype(int)
    counts = mask.T.dot(mask)
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            r = mat.loc[a, b]
            if pd.isna(r) or abs(r) < config.corr_min_abs:
                continue
            rows.append({"a": a, "b": b, "r": float(r), "abs_r": float(abs(r)), "n": int(counts.loc[a, b]),
                         "note": "near-duplicate columns" if abs(r) >= 0.999 else ""})
    pairs = (pd.DataFrame(rows, columns=empty_pairs.columns).sort_values("abs_r", ascending=False)
             .head(config.corr_top_pairs).reset_index(drop=True)) if rows else empty_pairs
    return CorrelationResult(mat, pairs, excluded, method, note)

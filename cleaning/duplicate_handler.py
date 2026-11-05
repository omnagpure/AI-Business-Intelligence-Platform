"""Exact and near-duplicate rows. Removal is only proposed and keeps the first occurrence."""
from __future__ import annotations

import pandas as pd

from cleaning.actions import RISK_DESTRUCTIVE, CleaningAction, SkipAction, is_text_series
from config.analytics_config import CleaningConfig


def normalized_key_frame(df: pd.DataFrame, subset: list[str] | None = None) -> pd.DataFrame:
    """Frame used only to compare rows: text trimmed, lower-cased, spaces collapsed."""
    cols = subset or list(df.columns)
    out = {}
    for c in cols:
        s = df[c]
        out[c] = (s.astype("string").str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
                  if is_text_series(s) else s)
    return pd.DataFrame(out)


def drop_duplicates(df: pd.DataFrame, column: str | None, params: dict) -> tuple[pd.DataFrame, int, str]:
    subset = params.get("subset")
    key = normalized_key_frame(df, subset) if params.get("normalize") else (df[subset] if subset else df)
    dup = key.duplicated(keep="first")
    n = int(dup.sum())
    if n == 0:
        raise SkipAction("no duplicate rows")
    kind = "near-duplicate" if params.get("normalize") else "duplicate"
    return df.loc[~dup].copy(), n, f"Removed {n} {kind} row(s), keeping the first occurrence. Original row numbers are kept."


OPS = {"drop_duplicates": drop_duplicates}


def propose_duplicate_actions(df: pd.DataFrame, exact: int, cfg: CleaningConfig) -> list[CleaningAction]:
    out: list[CleaningAction] = []
    if exact:
        out.append(CleaningAction(
            "drop_duplicates:exact", "drop_duplicates", None, {"normalize": False},
            "Remove exact duplicate rows", f"{exact} row(s) are identical to an earlier row.",
            "Remove repeats, keeping the first occurrence.", exact,
            "Repeated rows can double-count totals, but may also be genuine repeat transactions.",
            "Row count decreases; sums and counts drop accordingly.", RISK_DESTRUCTIVE, False,
            group="duplicates"))
    if df.shape[0] * df.shape[1] <= cfg.near_duplicate_max_cells:
        near_total = int(normalized_key_frame(df).duplicated().sum())
        if near_total > exact:
            out.append(CleaningAction(
                "drop_duplicates:near", "drop_duplicates", None, {"normalize": True},
                "Remove near-duplicate rows",
                f"{near_total} row(s) repeat an earlier row once case and spacing are ignored "
                f"({near_total - exact} more than exact matches).",
                "Remove rows that match an earlier row ignoring case and extra spaces.", near_total,
                "Rows differing only by capitalisation or spacing are usually the same record.",
                "Row count decreases; review sample rows first.", RISK_DESTRUCTIVE, False, group="duplicates"))
    return out

"""Text hygiene: inconsistent category labels, stray whitespace, column-name normalisation."""
from __future__ import annotations

import re
from collections import defaultdict

import pandas as pd

from cleaning.actions import RISK_REVIEW, RISK_SAFE, CleaningAction, SkipAction, count_changed, is_text_series
from config.analytics_config import CleaningConfig
from ingestion.profiler import DatasetProfile


def _collapse(v: str) -> str:
    return re.sub(r"\s+", " ", v.strip())


def build_label_mapping(s: pd.Series, max_cardinality: int = 2000) -> tuple[dict[str, str], int]:
    """Map raw labels to a canonical spelling. Variants that differ only by case or spacing are
    merged into the most frequent form. Returns (mapping, rows_changed). Deterministic."""
    strs = s[s.map(lambda v: isinstance(v, str))]
    counts = strs.value_counts()
    if len(counts) > max_cardinality:
        return {}, 0
    groups: dict[str, list[str]] = defaultdict(list)
    for raw in counts.index:
        groups[_collapse(raw).lower()].append(raw)
    mapping: dict[str, str] = {}
    rows = 0
    for raws in groups.values():
        form_counts: dict[str, int] = defaultdict(int)
        for r in raws:
            form_counts[_collapse(r)] += int(counts[r])
        canonical = sorted(form_counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        for r in raws:
            if r != canonical:
                mapping[r] = canonical
                rows += int(counts[r])
    return mapping, rows


def standardize_labels(df: pd.DataFrame, column: str, params: dict) -> tuple[pd.DataFrame, int, str]:
    s = df[column]
    if not is_text_series(s):
        raise SkipAction("column is not text")
    mapping = params["mapping"]
    new = s.map(lambda v: mapping.get(v, v) if isinstance(v, str) else v)
    n = count_changed(s, new)
    if n == 0:
        raise SkipAction("no matching labels found")
    df[column] = new
    return df, n, f"Standardised {n} value(s) across {len(mapping)} label variant(s)."


def strip_whitespace(df: pd.DataFrame, column: str, params: dict) -> tuple[pd.DataFrame, int, str]:
    s = df[column]
    if not is_text_series(s):
        raise SkipAction("column is not text")
    new = s.map(lambda v: v.strip() if isinstance(v, str) else v)
    n = count_changed(s, new)
    if n == 0:
        raise SkipAction("no leading/trailing spaces found")
    df[column] = new
    return df, n, f"Trimmed spaces from {n} value(s)."


def normalize_name(name: str) -> str:
    n = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(name).strip())
    n = re.sub(r"[\W_]+", "_", n).strip("_").lower()
    return n or "column"


def build_name_mapping(columns: list[str]) -> dict[str, str]:
    """Original -> normalised names (only those that change). Collisions get _2, _3 ..."""
    used: set[str] = {c for c in columns if normalize_name(c) == c}  # already-canonical names keep theirs
    mapping: dict[str, str] = {}
    for c in columns:
        if c in used:
            continue
        base = normalize_name(c)
        new, i = base, 2
        while new in used:
            new, i = f"{base}_{i}", i + 1
        used.add(new)
        if new != c:
            mapping[c] = new
    return mapping


def normalize_columns(df: pd.DataFrame, column: str | None, params: dict) -> tuple[pd.DataFrame, int, str]:
    mapping = {k: v for k, v in params["mapping"].items() if k in df.columns}
    if not mapping:
        raise SkipAction("no matching columns")
    return df.rename(columns=mapping), 0, f"Renamed {len(mapping)} column(s). Original names are kept in the log."


OPS = {"standardize_labels": standardize_labels, "strip_whitespace": strip_whitespace,
       "normalize_columns": normalize_columns}


def propose_label_actions(df: pd.DataFrame, profile: DatasetProfile, cfg: CleaningConfig) -> list[CleaningAction]:
    out: list[CleaningAction] = []
    for c in profile.columns_profile:
        s = df[c.name]
        if not is_text_series(s) or c.kind not in ("categorical", "text", "identifier"):
            continue
        if c.kind == "categorical":
            mapping, rows = build_label_mapping(s, cfg.max_label_cardinality)
            if mapping:
                examples = ", ".join(f"'{k}' -> '{v}'" for k, v in list(mapping.items())[:3])
                out.append(CleaningAction(
                    f"standardize_labels:{c.name}", "standardize_labels", c.name, {"mapping": mapping},
                    "Standardise category labels",
                    f"{len(mapping)} label(s) differ only by case or spacing ({examples}).",
                    "Merge each variant into the most frequent spelling.",
                    rows, "Counts and group-bys split one category into several otherwise.",
                    "Category totals change; the number of distinct labels shrinks.",
                    RISK_REVIEW, True))
                continue
        strs = s[s.map(lambda v: isinstance(v, str))]
        n = int((strs != strs.map(str.strip)).sum()) if len(strs) else 0
        if n:
            out.append(CleaningAction(
                f"strip_whitespace:{c.name}", "strip_whitespace", c.name, {},
                "Trim leading/trailing spaces", f"{n} value(s) have extra spaces.",
                "Remove leading and trailing spaces.", n,
                "Invisible spaces make equal values look different.", "Only whitespace changes.",
                RISK_SAFE, True))
    return out


def propose_name_action(df: pd.DataFrame) -> list[CleaningAction]:
    mapping = build_name_mapping([str(c) for c in df.columns])
    if not mapping:
        return []
    ex = ", ".join(f"'{k}' -> '{v}'" for k, v in list(mapping.items())[:3])
    return [CleaningAction(
        "normalize_columns", "normalize_columns", None, {"mapping": mapping},
        "Normalise column names", f"{len(mapping)} column name(s) are not in lower_snake_case ({ex}).",
        "Rename to lower_snake_case.", 0, "Consistent names are easier to reference in later analysis.",
        "Names change in the cleaned data; the originals are recorded in the log.", RISK_SAFE, False)]

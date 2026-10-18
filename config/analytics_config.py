"""Tunable thresholds for profiling and data-quality rules."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QualityThresholds:
    missing_warn_pct: float = 20.0       # column missingness that triggers a warning
    missing_high_pct: float = 50.0       # ... and a high-severity issue
    duplicate_warn_pct: float = 1.0      # share of duplicate rows that triggers a warning
    min_rows_warn: int = 30              # fewer rows than this limits most analyses
    id_unique_ratio: float = 0.98        # unique ratio for identifier heuristic
    numeric_text_ratio: float = 0.95     # share of values that must parse as numbers
    date_text_ratio: float = 0.90        # share of values that must parse as dates
    low_cardinality_max: int = 20
    medium_cardinality_max: int = 200
    sample_size: int = 2000              # rows sampled for expensive per-column checks
    top_n: int = 10


DEFAULT_THRESHOLDS = QualityThresholds()

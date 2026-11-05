"""Tunable thresholds for profiling and data-quality rules."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QualityThresholds:
    missing_warn_pct: float = 20.0
    missing_high_pct: float = 50.0
    duplicate_warn_pct: float = 1.0
    min_rows_warn: int = 30
    id_unique_ratio: float = 0.98
    numeric_text_ratio: float = 0.95
    date_text_ratio: float = 0.90
    low_cardinality_max: int = 20
    medium_cardinality_max: int = 200
    sample_size: int = 2000
    top_n: int = 10


DEFAULT_THRESHOLDS = QualityThresholds()


@dataclass(frozen=True)
class CleaningConfig:
    """Thresholds that decide which cleaning actions are proposed. Nothing is applied automatically."""
    fill_max_missing_pct: float = 30.0
    drop_column_missing_pct: float = 60.0
    drop_rows_max_missing_pct: float = 5.0
    outlier_iqr_k: float = 3.0
    outlier_min_values: int = 20
    near_duplicate_max_cells: int = 3_000_000
    max_label_cardinality: int = 2000


DEFAULT_CLEANING = CleaningConfig()


@dataclass(frozen=True)
class AnalyticsConfig:
    """Settings for descriptive analytics, correlations and KPIs (Phase 5)."""
    top_n: int = 10
    max_group_cardinality: int = 200
    corr_min_abs: float = 0.30
    corr_top_pairs: int = 15
    corr_max_columns: int = 40
    corr_min_periods: int = 10
    min_rows: int = 5
    time_freq: str = "M"
    comparison: str = "previous_period"
    concentration_top_share: float = 0.10
    max_general_numeric: int = 4
    preview_rows: int = 100
    scatter_max_points: int = 5000


DEFAULT_ANALYTICS = AnalyticsConfig()
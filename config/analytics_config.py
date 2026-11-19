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


@dataclass(frozen=True)
class CleaningConfig:
    """Thresholds that decide which cleaning actions are proposed. Nothing is applied automatically."""
    fill_max_missing_pct: float = 30.0     # propose imputation only up to this missingness
    drop_column_missing_pct: float = 60.0  # propose dropping a column at or above this missingness
    drop_rows_max_missing_pct: float = 5.0  # propose dropping rows with missing dates up to this share
    outlier_iqr_k: float = 3.0             # IQR multiplier; 3.0 flags only extreme values
    outlier_min_values: int = 20           # do not look for outliers in tiny columns
    near_duplicate_max_cells: int = 3_000_000  # skip near-duplicate scan on very large tables
    max_label_cardinality: int = 2000      # skip label standardisation on very high-cardinality columns


DEFAULT_CLEANING = CleaningConfig()


@dataclass(frozen=True)
class AnalyticsConfig:
    """Settings for descriptive analytics, correlations and KPIs (Phase 5)."""
    top_n: int = 10                       # categories shown before the rest are grouped as "Other"
    max_group_cardinality: int = 200      # columns with more distinct values are not used for grouping
    corr_min_abs: float = 0.30            # weaker correlations are not reported as pairs
    corr_top_pairs: int = 15
    corr_max_columns: int = 40            # cap for the correlation matrix
    corr_min_periods: int = 10            # minimum rows with both values present
    min_rows: int = 5                     # below this most KPIs are reported as not calculable
    time_freq: str = "M"                  # D | W | M for trends and growth
    comparison: str = "previous_period"   # previous_period | year_over_year
    concentration_top_share: float = 0.10 # share of customers used for the concentration KPI
    max_general_numeric: int = 4          # numeric columns summarised when no role-based KPI applies
    growth_max_row_ratio: float = 10.0    # growth is skipped when one period has >10x the rows of the other
    preview_rows: int = 100               # rows per page in the data explorer
    scatter_max_points: int = 5000        # sample size for scatter plots
    # Forecasting (Phase 6A)
    forecast_folds: int = 3               # rolling-origin validation folds (reduced automatically for short series)
    forecast_interval: float = 0.80       # coverage of the empirical prediction interval
    forecast_simpler_margin: float = 0.05  # a complex model must beat a simpler one by more than this share of MAE
    forecast_max_gap_share: float = 0.30  # refuse to forecast when more periods than this are empty
    forecast_max_horizon: int = 90        # periods ahead
    forecast_max_history: int = 1500      # only the latest periods are modelled (keeps fits fast)
    # Anomaly detection (Phase 6B)
    anomaly_sensitivity: int = 3          # 1 (strict, few flags) .. 5 (sensitive, many flags)
    anomaly_min_obs: int = 30             # non-missing values a column needs
    anomaly_min_unique: int = 10          # columns with fewer distinct values (flags, codes) are skipped
    anomaly_max_findings: int = 500       # per method and column; counts in the summary stay exact
    anomaly_max_features: int = 12        # Isolation Forest feature cap
    anomaly_if_min_rows: int = 200        # complete rows needed for Isolation Forest
    anomaly_if_max_fit_rows: int = 200000  # larger data is subsampled for fitting (all rows are still scored)
    anomaly_random_seed: int = 42
    # Segmentation (Phase 6C)
    seg_min_entities: int = 12            # entities (suppliers, customers...) needed to cluster
    seg_min_rows: int = 3                 # rows an entity needs to be included
    seg_k_max: int = 8                    # largest K tried
    seg_max_features: int = 10            # features built per entity
    seg_max_missing: float = 0.30         # a feature missing for more entities than this is dropped
    seg_min_cluster_share: float = 0.05   # automatic K avoids clusters smaller than this share
    seg_dup_corr: float = 0.98            # features this correlated are duplicates; the later one is dropped
    seg_random_seed: int = 42
    # Driver analysis (Phase 6D)
    driver_min_rows: int = 100            # rows with a known target needed
    driver_min_unique: int = 10           # numeric targets need at least this many distinct values
    driver_max_fit_rows: int = 20000      # larger data is sampled (fixed seed)
    driver_trees: int = 100
    driver_max_depth: int = 12
    driver_perm_repeats: int = 5
    driver_perm_rows: int = 5000          # test rows used for permutation importance
    driver_test_share: float = 0.25
    driver_min_r2: float = 0.10           # below this test R2 the ranking is flagged as unreliable
    driver_min_auc: float = 0.60          # same for a yes/no target (ROC AUC)
    driver_max_missing: float = 0.50      # features missing more often than this are left out
    driver_leak_corr: float = 0.95        # features this correlated with the target are treated as near-copies
    driver_leak_r2: float = 0.95          # a categorical column explaining this share of the target's variance is treated as a copy of it
    driver_dominance: float = 0.90        # warn when one column holds this share of all importance
    driver_max_categories: int = 20       # categorical features with more values are left out
    driver_random_seed: int = 42


DEFAULT_ANALYTICS = AnalyticsConfig()


@dataclass(frozen=True)
class InsightConfig:
    """Thresholds for the rule-based insight engine (Phase 7). Every threshold is shown on the insight it triggers.

    These are this app's reference levels, not industry standards; change them to match your business."""
    # growth / decline of a KPI against its comparison period (absolute percent change)
    growth_medium_pct: float = 10.0
    growth_high_pct: float = 25.0
    # share of revenue held by the top category (percent)
    category_share_medium: float = 40.0
    category_share_high: float = 60.0
    # KPIs where a LOW value is a concern: (kpi key, medium below, high below) in percent
    low_is_bad: tuple[tuple[str, float, float], ...] = (
        ("on_time_rate", 90.0, 80.0), ("fill_rate", 95.0, 85.0), ("otif", 85.0, 70.0))
    # KPIs where a HIGH value is a concern: (kpi key, medium at or above, high at or above) in percent
    high_is_bad: tuple[tuple[str, float, float], ...] = (
        ("stockout_rate", 5.0, 15.0), ("defect_rate", 2.0, 5.0), ("below_reorder_share", 10.0, 30.0),
        ("supplier_top_share", 40.0, 60.0), ("concentration", 60.0, 80.0))
    low_margin_pct: float = 5.0           # profit margin below this is flagged (negative margin is always high)
    # data quality
    missing_medium_pct: float = 20.0
    missing_high_pct: float = 50.0
    duplicate_medium_pct: float = 1.0
    min_rows_high: int = 30               # fewer rows than this is a high-severity data-size warning
    # anomalies: share of examined values flagged by one method (fraction)
    anomaly_share_medium: float = 0.03
    anomaly_share_high: float = 0.10
    # forecast: change of the forecast mean vs the same number of latest observed periods (percent)
    forecast_change_medium_pct: float = 10.0
    forecast_change_high_pct: float = 25.0
    forecast_good_improvement_pct: float = 10.0   # improvement over the baseline needed for "high" confidence
    # segments
    seg_small_share: float = 0.30         # a "small" segment holds at most this share of entities
    seg_value_ratio: float = 2.0          # its mean must be at least this many times the overall mean
    seg_value_share_high: float = 0.50    # ... and severity is high when it holds this share of the total
    seg_min_silhouette: float = 0.25      # below this the clusters are too weak to report on
    # drivers
    driver_top_share: float = 0.25        # the top factor must hold at least this share of importance
    # confidence from the number of rows analysed
    confidence_rows_high: int = 500
    confidence_rows_medium: int = 50
    max_items_listed: int = 5
    # data-pattern findings (read straight from the data)
    group_gap_medium_pct: float = 25.0    # (best group mean - worst group mean) as % of the overall mean
    group_gap_high_pct: float = 60.0
    group_min_rows: int = 30              # a group needs this many rows to be compared
    group_max_groups: int = 60
    flag_lift_medium_pct: float = 10.0    # mean with the flag vs without it
    flag_lift_high_pct: float = 25.0
    flag_min_share: float = 0.05          # both flag outcomes must cover at least this share of rows
    weekday_spread_medium_pct: float = 25.0   # (best weekday - worst weekday) as % of the average daily total
    weekday_spread_high_pct: float = 50.0
    weekday_min_days: int = 56
    pareto_top_share_medium: float = 40.0     # share held by the top 10% of entities (even spread would be about 10%)
    pareto_top_share_high: float = 60.0
    pareto_min_entities: int = 10
    cover_low_medium_days: float = 14.0   # days of cover below these levels
    cover_low_high_days: float = 7.0
    cover_excess_days: float = 90.0       # days of cover above this suggests excess stock
    fc_wape_medium_pct: float = 20.0      # error of a forecast column in the data against actual units
    fc_wape_high_pct: float = 35.0
    fc_bias_pct: float = 10.0             # systematic over- or under-forecast


DEFAULT_INSIGHTS = InsightConfig()

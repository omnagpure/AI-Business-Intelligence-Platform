"""Data types for insights. Every insight carries the evidence that triggered it (no evidence, no insight)."""
from __future__ import annotations

from dataclasses import dataclass, field

SEVERITIES = ("high", "medium", "low", "info")
CONFIDENCES = ("high", "medium", "low")
GROUPS = ("Performance", "Supply chain", "Data quality", "Forecast", "Anomalies", "Segments", "Drivers")


@dataclass(frozen=True)
class Evidence:
    """One computed fact behind an insight. `value` and `detail` are text produced from computed results only."""
    source: str                       # kpi | profile | forecast | anomaly | segmentation | drivers
    metric: str                       # e.g. revenue_growth, missing_pct, silhouette
    value: str
    detail: str = ""                  # formula or calculation, in words
    columns: tuple[str, ...] = ()
    period: str = ""


@dataclass(frozen=True)
class Insight:
    id: str                           # deterministic: rule + subject
    rule: str
    group: str
    title: str
    description: str
    evidence: tuple[Evidence, ...]
    severity: str                     # high | medium | low | info
    confidence: str                   # high | medium | low
    subject: str = ""                 # short label used by recommendation templates
    direction: str = ""               # e.g. down, up, low, high; used by recommendation templates
    period: str = ""
    thresholds: str = ""              # plain statement of the threshold that was crossed
    limitations: tuple[str, ...] = field(default_factory=tuple)
    priority: int = 0                 # set by prioritization.rank
    next_step: str = ""               # set by the engine from analytics.recommendations

    def __post_init__(self) -> None:
        if not self.evidence:
            raise ValueError("An insight needs at least one piece of evidence.")
        if self.severity not in SEVERITIES or self.confidence not in CONFIDENCES:
            raise ValueError("Unknown severity or confidence.")

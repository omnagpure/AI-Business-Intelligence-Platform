"""Runs every insight rule over the computed results. Deterministic: same inputs and config, same insights."""
from __future__ import annotations

from dataclasses import dataclass, field, replace

from analytics.recommendations import Recommendation, build_recommendations
from config.analytics_config import DEFAULT_INSIGHTS, InsightConfig
from core.logging_config import get_logger
from insights.evidence import Insight
from insights.insight_rules import RULES, InsightInputs
from insights.prioritization import rank

log = get_logger("insights.engine")

_INPUT_LABELS = {"kpis": "KPI", "profile": "data profile", "forecast": "forecast", "anomalies": "anomaly",
                 "segmentation": "segmentation", "drivers": "driver analysis", "findings": "data pattern"}


@dataclass(frozen=True)
class InsightReport:
    insights: tuple[Insight, ...]
    recommendations: tuple[Recommendation, ...]
    skipped: tuple[tuple[str, str], ...]        # (rule, reason) for rules that did not run
    inputs_used: tuple[str, ...]
    notes: tuple[str, ...] = field(default_factory=tuple)


def generate_insights(inputs: InsightInputs, config: InsightConfig = DEFAULT_INSIGHTS) -> InsightReport:
    """Each rule fails independently; a failing rule is listed as skipped and never stops the others."""
    reasons = dict(inputs.unavailable)
    found: list[Insight] = []
    skipped: list[tuple[str, str]] = []
    for rule in RULES:
        if getattr(inputs, rule.needs) is None:
            why = reasons.get(rule.needs) or f"The {_INPUT_LABELS[rule.needs]} results were not available."
            skipped.append((rule.name, why))
            continue
        try:
            found.extend(rule.fn(inputs, config))
        except Exception:                      # one broken rule must not hide the others
            log.exception("Insight rule %s failed", rule.name)
            skipped.append((rule.name, "The rule could not be evaluated on this data."))
    ranked = rank(found)
    recs = build_recommendations(ranked)
    by_id = {r.insight_id: r.text for r in recs}
    final = tuple(replace(i, next_step=by_id.get(i.id, "")) for i in ranked)
    used = tuple(f for f in _INPUT_LABELS if getattr(inputs, f) is not None)
    notes = []
    if not final:
        notes.append("No finding crossed a threshold. That is a result, not an error: the thresholds are shown on every insight "
                     "and can be changed.")
    return InsightReport(final, tuple(recs), tuple(skipped), used, tuple(notes))

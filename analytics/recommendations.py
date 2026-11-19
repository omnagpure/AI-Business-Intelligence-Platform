"""Template-based recommendations. Deterministic: the same insight always gives the same text, and every
recommendation points at the insight (and so the evidence) that triggered it. No numbers are invented here."""
from __future__ import annotations

from dataclasses import dataclass

from insights.evidence import Insight


@dataclass(frozen=True)
class Recommendation:
    insight_id: str
    text: str
    priority: int


_TEMPLATES: dict[tuple[str, str], str] = {
    ("growth", "down"): "Break the {subject} decline down by category, region or product on the Dashboard to find where it comes from, "
                        "and check whether the comparison period was unusual.",
    ("growth", "up"): "Find out which categories or periods drive the {subject} increase and whether it is repeatable.",
    ("category_share", "*"): "Review dependence on '{subject}': check its trend and whether other categories could be developed.",
    ("threshold", "low"): "Investigate why {subject} is low: split it by supplier, warehouse or period to find where the gap is concentrated.",
    ("threshold", "high"): "Investigate why {subject} is high: split it by supplier, warehouse, product or period to find the main contributors.",
    ("margin", "*"): "Compare cost and price by product or category to see where margin is lost, and check the cost column for errors.",
    ("kpi_unavailable", "*"): "Set the missing column roles on the Semantics page or add the missing columns to the data so these KPIs can be calculated.",
    ("missing", "*"): "Review '{subject}' and the other incomplete columns on the Data quality page before relying on analyses that use them.",
    ("duplicates", "*"): "Review the duplicate rows on the Data quality page and remove them only if they are errors.",
    ("small_dataset", "*"): "Add more history or rows before drawing conclusions; treat all results as indicative only.",
    ("forecast_direction", "up"): "Check that capacity, stock and budgets can support the forecast increase in {subject}.",
    ("forecast_direction", "down"): "Look into the forecast decline in {subject} and plan for it; confirm it against known events before acting.",
    ("forecast_reliability", "*"): "Use the forecast only as a rough guide; more history or additional drivers may be needed for a better model.",
    ("anomalies", "*"): "Open the Anomalies page and review the flagged '{subject}' values; correct genuine errors and keep genuine extremes.",
    ("segments", "*"): "Look at the entities in {subject} on the Segmentation page and decide whether they need a dedicated approach.",
    ("group_gap", "*"): "Look at why {subject} differs so much: review the best and worst groups and whether the gap is justified or fixable.",
    ("flag_effect", "up"): "Check whether the effect of {subject} is repeatable and worth the cost, ideally with a controlled comparison.",
    ("flag_effect", "down"): "Understand why rows with {subject} perform worse and whether they can be prevented.",
    ("weekday", "*"): "Plan staffing, stock and promotions around the weekday pattern in {subject}.",
    ("trend", "down"): "Check what changed in the latest month for {subject} (calendar, supply or demand) before it becomes a trend.",
    ("trend", "up"): "Check whether the latest increase in {subject} is sustainable and whether supply can keep up.",
    ("concentration", "*"): "Protect the top items by {subject} and review the long tail for possible simplification.",
    ("forecast_accuracy", "error"): "Review how '{subject}' is produced; improving it reduces stockouts and excess stock.",
    ("forecast_accuracy", "over"): "Reduce the systematic over-forecast in '{subject}' to avoid excess stock.",
    ("forecast_accuracy", "under"): "Correct the systematic under-forecast in '{subject}' to avoid stockouts.",
    ("cover", "low"): "Review reorder points and replenishment lead times to avoid stockouts.",
    ("cover", "high"): "Review order quantities and slow-moving items to release stock and cash.",
    ("drivers", "*"): "Treat the leading factor for {subject} as a lead to test, not a cause; confirm it with a controlled comparison before acting.",
}


def recommend(insight: Insight) -> str | None:
    t = _TEMPLATES.get((insight.rule, insight.direction)) or _TEMPLATES.get((insight.rule, "*"))
    return t.format(subject=insight.subject or "this item") if t else None


def build_recommendations(insights: list[Insight]) -> list[Recommendation]:
    out = []
    for i in insights:
        text = recommend(i)
        if text:
            out.append(Recommendation(i.id, text, i.priority))
    return out

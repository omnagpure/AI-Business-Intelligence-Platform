"""Template-based recommendations. Deterministic: the same insight always gives the same text, and every
recommendation points at the insight (and so the evidence) that triggered it. No numbers are invented here.

`recommend` gives the one-sentence next step used by reports; `plan` gives the fuller version the Insights page shows:
a kind (risk, opportunity, pattern, data), why it matters, a short action, a few concrete steps and the page to open.
"""
from __future__ import annotations

from dataclasses import dataclass

from insights.evidence import Insight


@dataclass(frozen=True)
class Recommendation:
    insight_id: str
    text: str
    priority: int


@dataclass(frozen=True)
class Plan:
    kind: str                       # risk | opportunity | pattern | data
    why: str                        # why it matters, in business terms
    action: str                     # short imperative headline
    steps: tuple[str, ...]          # concrete things to do, in order
    page: str = ""                  # nav key of the page that helps (ui.nav)


KINDS = {"risk": "Needs attention", "opportunity": "Opportunity", "pattern": "Pattern", "data": "Data note"}
PAGE_LABELS = {"dashboard": "Dashboard", "explorer": "Data explorer", "forecasting": "Forecasting", "anomalies": "Anomalies",
               "segmentation": "Segmentation", "drivers": "Drivers", "quality": "Data quality", "roles": "Column roles"}


_TEMPLATES: dict[tuple[str, str], str] = {
    ("growth", "down"): "Break the {subject} decline down by category, region or product on the Dashboard to find where it comes from, "
                        "and check whether the comparison period was unusual.",
    ("growth", "up"): "Find out which categories or periods drive the {subject} increase and whether it is repeatable.",
    ("category_share", "*"): "Review dependence on '{subject}': check its trend and whether other categories could be developed.",
    ("threshold", "low"): "Investigate why {subject} is low: split it by supplier, warehouse or period to find where the gap is concentrated.",
    ("threshold", "high"): "Investigate why {subject} is high: split it by supplier, warehouse, product or period to find the main contributors.",
    ("margin", "*"): "Compare cost and price by product or category to see where margin is lost, and check the cost column for errors.",
    ("kpi_unavailable", "*"): "Set the missing column roles on the Semantics page or add the missing columns to the data so these KPIs can be calculated.",
    ("missing", "empty"): "Drop the empty columns from the export if they are not needed; nothing else is required.",
    ("missing", "*"): "Review '{subject}' and the other incomplete columns on the Data quality page before relying on analyses that use them.",
    ("duplicates", "*"): "Review the duplicate rows on the Data quality page and remove them only if they are errors.",
    ("small_dataset", "*"): "Add more history or rows before drawing conclusions; treat all results as indicative only.",
    ("forecast_direction", "up"): "Check that capacity, stock and budgets can support the forecast increase in {subject}.",
    ("forecast_direction", "down"): "Look into the forecast decline in {subject} and plan for it; confirm it against known events before acting.",
    ("forecast_reliability", "*"): "Use the forecast only as a rough guide; more history or additional drivers may be needed for a better model.",
    ("anomalies", "*"): "Open the Anomalies page and review the flagged '{subject}' values; correct genuine errors and keep genuine extremes.",
    ("segments", "*"): "Look at the entities in {subject} on the Segmentation page and decide whether they need a dedicated approach.",
    ("group_gap", "by_design"): "Compare actual against promised times for each option of {subject} to see which one misses its promise.",
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

# (rule, direction) -> (kind, why it matters, action, steps, page). "{subject}" is filled in; "*" matches any direction.
_PLANS: dict[tuple[str, str], tuple[str, str, str, tuple[str, ...], str]] = {
    ("growth", "down"): ("risk", "A falling top line squeezes every margin and plan built on last period's level.",
                         "Find where the {subject} drop comes from",
                         ("Split {subject} by category, region and product on the Dashboard.",
                          "Check whether the comparison period was unusually strong (a promotion, a one-off order).",
                          "Confirm the latest period is complete and the export did not change."), "dashboard"),
    ("growth", "up"): ("opportunity", "Growth you understand is growth you can repeat.",
                       "Find out what is driving the {subject} increase",
                       ("Split {subject} by category, region and product on the Dashboard.",
                        "Check whether the increase comes from a few large orders or broad demand."), "dashboard"),
    ("category_share", "*"): ("risk", "When one category carries most of the revenue, a problem there hits the whole business.",
                              "Reduce dependence on '{subject}'",
                              ("Track the trend of '{subject}' on the Dashboard.",
                               "Identify the next categories that could grow."), "dashboard"),
    ("threshold", "low"): ("risk", "{subject} below its reference level means customers or operations feel the gap every day.",
                           "Find where {subject} falls short",
                           ("Split {subject} by supplier, warehouse, shipping mode or period on the Dashboard.",
                            "Open Drivers to see which columns go with the shortfall.",
                            "Agree a target and review it each period."), "drivers"),
    ("threshold", "high"): ("risk", "{subject} above its reference level adds cost, delay or lost sales.",
                            "Find the main contributors to {subject}",
                            ("Split {subject} by supplier, warehouse, product or period on the Dashboard.",
                             "Open Drivers to see which columns go with it.",
                             "Agree a target and review it each period."), "drivers"),
    ("margin", "*"): ("risk", "A thin or negative margin means growth can lose money.",
                      "Find where margin is lost",
                      ("Compare price and cost by product and category on the Dashboard.",
                       "Check the cost column for errors or missing values."), "dashboard"),
    ("kpi_unavailable", "*"): ("data", "Missing KPIs leave blind spots: these numbers cannot be tracked yet.",
                               "Make the missing KPIs calculable",
                               ("Check the column roles on the Column roles page; a role may just need confirming.",
                                "If the columns do not exist, add them to the next export."), "roles"),
    ("missing", "empty"): ("data", "Empty columns do no harm; they only add clutter.",
                           "Drop the empty columns if they are not needed", ("Remove them from the export next time.",), "quality"),
    ("missing", "*"): ("data", "Results that use incomplete columns rest on fewer rows and may be biased.",
                       "Fill or confirm the gaps in '{subject}'",
                       ("Review the incomplete columns on the Data quality page.",
                        "Fix them at the source if the values should exist."), "quality"),
    ("duplicates", "*"): ("data", "Duplicate rows can overstate totals and counts.",
                          "Check the duplicate rows",
                          ("Review them on the Data quality page.", "Remove them only if they are errors."), "quality"),
    ("small_dataset", "*"): ("data", "With so few rows every statistic and model is fragile.",
                             "Add more rows or history", ("Treat all results as indicative until more data is loaded.",), ""),
    ("forecast_direction", "up"): ("opportunity", "Higher expected {subject} needs stock, capacity and cash in place ahead of time.",
                                   "Prepare for higher {subject}",
                                   ("Check stock and capacity against the forecast on the Forecasting page.",
                                    "Confirm the forecast with sales or known events before committing spend."), "forecasting"),
    ("forecast_direction", "down"): ("risk", "Lower expected {subject} can leave excess stock and idle capacity.",
                                     "Plan for lower {subject}",
                                     ("Review the forecast and its range on the Forecasting page.",
                                      "Confirm it against known events before cutting orders."), "forecasting"),
    ("forecast_reliability", "*"): ("pattern", "The forecast is no better than repeating recent values, so use it with care.",
                                    "Use the forecast as a rough guide only",
                                    ("Add more history, or try a different period or measure on the Forecasting page.",), "forecasting"),
    ("anomalies", "*"): ("risk", "Extreme values can be errors that distort totals, or real events worth knowing about.",
                         "Review the unusual '{subject}' values",
                         ("Open the flagged records on the Anomalies page.",
                          "Correct genuine errors at the source; keep genuine extremes."), "anomalies"),
    ("segments", "*"): ("opportunity", "A small group carrying an outsized share deserves its own attention.",
                        "Give {subject} a dedicated approach",
                        ("Look at the entities in {subject} on the Segmentation page.",
                         "Decide whether they need their own service level, pricing or review cadence."), "segmentation"),
    ("group_gap", "by_design"): ("pattern", "Different service levels are meant to perform differently; the real question is "
                                            "whether each one keeps its promise.",
                                 "Check each option of {subject} against its promise",
                                 ("Open Drivers to compare actual against promised times per option.",), "drivers"),
    ("group_gap", "*"): ("pattern", "Large gaps between groups point to where performance can be improved or where mix matters.",
                         "Understand the gap in {subject}",
                         ("Compare the best and worst groups on the Dashboard.",
                          "Decide whether the gap is justified (different products, prices) or fixable."), "dashboard"),
    ("flag_effect", "up"): ("opportunity", "If the effect is real, more of {subject} could lift results.",
                            "Test whether {subject} is worth scaling",
                            ("Compare rows with and without {subject} on the Dashboard.",
                             "Confirm the effect with a controlled comparison and weigh it against the cost."), "dashboard"),
    ("flag_effect", "down"): ("risk", "Rows with {subject} perform worse, so preventing them pays off.",
                              "Find out why {subject} goes with worse results",
                              ("Compare rows with and without {subject} on the Dashboard.",
                               "Look for a preventable cause."), "dashboard"),
    ("weekday", "*"): ("pattern", "A steady weekday pattern is free planning information.",
                       "Plan around the weekday pattern",
                       ("Align staffing, stock and promotions with the busy and quiet days.",), "dashboard"),
    ("trend", "down"): ("risk", "One bad month can be noise; two is the start of a trend.",
                        "Check what changed for {subject} last month",
                        ("Look at the monthly trend on the Dashboard.",
                         "Rule out calendar effects, supply problems and data changes."), "dashboard"),
    ("trend", "up"): ("opportunity", "A rising month is good news if supply can keep up.",
                      "Check that the increase in {subject} can be sustained",
                      ("Look at the monthly trend on the Dashboard.", "Check stock and capacity for the coming months."), "dashboard"),
    ("concentration", "*"): ("risk", "A few items carry most of the value: losing one of them hurts.",
                             "Protect the top {subject} values",
                             ("Make sure the top items have reliable supply and stock.",
                              "Review the long tail for items that could be simplified."), "dashboard"),
    ("forecast_accuracy", "error"): ("risk", "Forecast error turns directly into stockouts or excess stock.",
                                     "Improve how '{subject}' is produced",
                                     ("Compare it with actuals by product and period.",
                                      "Try the validated model on the Forecasting page as a benchmark."), "forecasting"),
    ("forecast_accuracy", "over"): ("risk", "Systematic over-forecasting builds excess stock and ties up cash.",
                                    "Reduce the over-forecast in '{subject}'",
                                    ("Correct the bias at the source and compare again next period.",), "forecasting"),
    ("forecast_accuracy", "under"): ("risk", "Systematic under-forecasting causes stockouts and lost sales.",
                                     "Correct the under-forecast in '{subject}'",
                                     ("Correct the bias at the source and compare again next period.",), "forecasting"),
    ("cover", "low"): ("risk", "Low cover means stock can run out before it is replenished.",
                       "Raise cover where stock is short",
                       ("Review reorder points and replenishment lead times.",), "dashboard"),
    ("cover", "high"): ("risk", "Excess cover ties up cash and space and risks obsolete stock.",
                        "Release excess stock",
                        ("Review order quantities and slow-moving items.",), "dashboard"),
    ("drivers", "*"): ("opportunity", "Knowing what goes with {subject} shows where to look first, and where a change is most "
                                      "likely to pay off.",
                       "Use the top driver of {subject} as a lead",
                       ("Open Drivers to see how {subject} changes across its groups.",
                        "Test a change on a small scale before rolling it out: the link is an association, not proof of cause."),
                       "drivers"),
}


def _lookup(table: dict, insight: Insight):
    return table.get((insight.rule, insight.direction)) or table.get((insight.rule, "*"))


def recommend(insight: Insight) -> str | None:
    t = _lookup(_TEMPLATES, insight)
    return t.format(subject=insight.subject or "this item") if t else None


def plan(insight: Insight) -> Plan:
    """The page's fuller recommendation. Unknown rules still get a plan built from the one-line next step."""
    subject = insight.subject or "this item"
    t = _lookup(_PLANS, insight)
    if t is None:
        kind = "data" if insight.group == "Data quality" else "pattern"
        return Plan(kind, "", recommend(insight) or "Review this finding", ())
    kind, why, action, steps, page = t
    fill = lambda x: x.replace("{subject}", subject)                      # noqa: E731  (subject may contain braces)
    why = fill(why)
    return Plan(kind, why[:1].upper() + why[1:], fill(action), tuple(fill(x) for x in steps), page)


def build_recommendations(insights: list[Insight]) -> list[Recommendation]:
    out = []
    for i in insights:
        text = recommend(i)
        if text:
            out.append(Recommendation(i.id, text, i.priority))
    return out

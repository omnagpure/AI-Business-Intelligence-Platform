"""Transparent ranking: priority = 10 x severity weight + confidence weight. No hidden scoring."""
from __future__ import annotations

from dataclasses import replace

from insights.evidence import GROUPS, Insight

SEVERITY_WEIGHT = {"high": 3, "medium": 2, "low": 1, "info": 0}
CONFIDENCE_WEIGHT = {"high": 3, "medium": 2, "low": 1}


def priority_score(i: Insight) -> int:
    return 10 * SEVERITY_WEIGHT[i.severity] + CONFIDENCE_WEIGHT[i.confidence]


def rank(insights: list[Insight]) -> list[Insight]:
    """Drop duplicate ids (first wins), score, and sort by priority, then group order, then id (stable and deterministic)."""
    seen: dict[str, Insight] = {}
    for i in insights:
        seen.setdefault(i.id, i)
    scored = [replace(i, priority=priority_score(i)) for i in seen.values()]
    return sorted(scored, key=lambda i: (-i.priority, GROUPS.index(i.group) if i.group in GROUPS else 99, i.id))

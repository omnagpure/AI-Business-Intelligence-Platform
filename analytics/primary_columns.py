"""Choosing the 'primary' column for a role (revenue, category, date) when several columns share it.

The KPI engine already ranks a user-set role first, so picking a primary column simply records a role override
for it. This module only lists the candidates and edits the override dict; it never touches the data.
"""
from __future__ import annotations

import pandas as pd

from analytics.kpis import DataFilter, _Ctx, _category, _revenue, primary_date_column
from config.analytics_config import DEFAULT_ANALYTICS
from semantic.semantic_model import SemanticModel

PICKABLE = (("revenue", "Revenue / amount column"), ("category", "Category column"), ("datetime", "Date column"))


def primary_candidates(df: pd.DataFrame, model: SemanticModel) -> dict[str, tuple[str, list[str]]]:
    """{role: (current primary, all candidates with the primary first)} for roles with 2+ candidates."""
    ctx = _Ctx(df, model, DEFAULT_ANALYTICS, DataFilter())
    out = {}
    for role, _ in PICKABLE:
        numeric = role == "revenue"
        cands = ctx.rank(role, numeric=numeric)
        if role == "category":
            cands = [c for c in cands if 2 <= df[c].nunique(dropna=True) <= DEFAULT_ANALYTICS.max_group_cardinality]
        cur = {"revenue": lambda: _revenue(ctx)[0], "category": lambda: _category(ctx)[0],
               "datetime": lambda: primary_date_column(df, model)}[role]()
        if cur is not None and len(cands) >= 2:
            out[role] = (cur, [cur] + [c for c in cands if c != cur])
    return out


def with_primary(roles: dict[str, str], previous: str | None, column: str, role: str) -> dict[str, str]:
    """New override dict with `column` set to `role`. A column the picker set earlier (`previous`) loses its
    override so only one column is user-ranked first; overrides the user made elsewhere are left alone."""
    new = dict(roles)
    if previous and previous != column and new.get(previous) == role:
        del new[previous]
    new[column] = role
    return new


def explain_choice(model: SemanticModel, candidates: list[str]) -> list[dict]:
    """One row per candidate (chosen column first): why it was used or not. Reasons come only from what decides the ranking:
    a role you set, then name preference for the measure, then confidence, then column order."""
    chosen, top = candidates[0], max(model.columns[c].confidence for c in candidates)
    rows = []
    for c in candidates:
        conf, user = model.columns[c].confidence, model.columns[c].source == "user"
        if c == chosen:
            if user:
                why = "Set by you."
            elif conf >= top and all(conf > model.columns[o].confidence for o in candidates[1:]):
                why = f"Highest confidence ({round(conf * 100)}%)."
            elif conf >= top:
                why = f"Tied on confidence ({round(conf * 100)}%); its name matches the words preferred for this measure, or it comes first."
            else:
                why = "Its name matches the words preferred for this measure, which is ranked ahead of confidence."
        elif conf < model.columns[chosen].confidence:
            why = f"Lower confidence ({round(conf * 100)}%) than the chosen column."
        elif conf == model.columns[chosen].confidence:
            why = f"Same confidence ({round(conf * 100)}%); the chosen column ranks ahead on name or position."
        else:
            why = f"Higher confidence ({round(conf * 100)}%), but the chosen column's name matches the preferred words for this measure."
        rows.append({"Column": c, "Used": "Chosen" if c == chosen else "Not used", "Confidence": round(conf * 100), "Why": why})
    return rows

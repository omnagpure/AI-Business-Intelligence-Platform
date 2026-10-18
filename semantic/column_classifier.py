"""Rule-based semantic role classifier.

Combines column-name vocabulary, inferred kind, value distributions and relationships between
columns (e.g. revenue ~ quantity x price). Scores are transparent heuristics, not probabilities.
An optional trained name model (see `NameModel`) can be blended in; it is never required.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Protocol

import numpy as np
import pandas as pd
from pandas.api import types as pdt

from cleaning.type_inference import to_numeric_clean
from ingestion.profiler import ColumnProfile, DatasetProfile
from semantic.column_lexicon import (ID_TOKENS, LEXICON, LOCATION_NUMERIC_TOKENS, MONEY_ROLES, NUMERIC_ROLES,
                                     RATIO_TOKENS, ROLE_LABELS, SENSITIVE_TOKENS, tokenize, token_set)

MIN_SCORE = 0.30          # below this a column is reported as Unknown
SAMPLE_ROWS = 5000
MODEL_WEIGHT = 0.3
_ROLES = [r for r in ROLE_LABELS if r != "unknown"]
# Name tokens that describe an *attribute* (so "Customer City" is a location, not a customer).
_ATTR_TOKENS = frozenset().union(*(LEXICON[r].strong for r in (
    "revenue", "cost", "profit", "price", "quantity", "category", "location", "percentage", "balance", "duration")))


class NameModel(Protocol):
    """Optional trained model: returns {role: probability} for a column name."""
    def predict_proba(self, name: str) -> dict[str, float]: ...


@dataclass(frozen=True)
class ColumnSemantic:
    column: str
    role: str
    confidence: float
    evidence: tuple[str, ...]
    alternatives: tuple[tuple[str, float], ...]
    source: str            # "rules" | "model" | "user"
    kind: str


def _numeric_values(s: pd.Series, stored_as_text: bool) -> pd.Series | None:
    nn = s.dropna()
    if len(nn) > SAMPLE_ROWS:
        nn = nn.sample(SAMPLE_ROWS, random_state=0)
    if stored_as_text:
        v = to_numeric_clean(nn)
    elif pdt.is_numeric_dtype(nn) and not pdt.is_bool_dtype(nn):
        v = nn
    else:
        return None
    v = v.replace([np.inf, -np.inf], np.nan).dropna()
    return v if len(v) else None


@dataclass
class _Feat:
    name: str
    tokens: set[str]
    joined: str
    kind: str
    unique: int
    unique_ratio: float
    n: int
    vals: pd.Series | None
    is_int: bool = False
    vmin: float = 0.0
    vmax: float = 0.0


def _features(df: pd.DataFrame, c: ColumnProfile) -> _Feat:
    vals = _numeric_values(df[c.name], c.stored_as_text) if c.kind in ("numeric", "boolean", "identifier") else None
    n = max(len(df) - c.missing, 1)
    f = _Feat(c.name, token_set(c.name), " ".join(tokenize(c.name)), c.kind, c.unique, c.unique / n, n, vals)
    if vals is not None:
        f.is_int = bool((vals % 1 == 0).all())
        f.vmin, f.vmax = float(vals.min()), float(vals.max())
    return f


def _name_match(role: str, f: _Feat) -> tuple[float, str]:
    lex = LEXICON.get(role)
    if lex is None:
        return 0.0, ""
    phrase = next((p for p in lex.phrases if p in f.joined), None)
    strong = sorted(f.tokens & lex.strong)
    if phrase:
        return 0.7, f"Name contains the phrase '{phrase}'"
    if strong:
        return min(0.8, 0.6 + 0.1 * (len(strong) - 1)), f"Name contains '{strong[0]}'"
    weak = sorted(f.tokens & lex.weak)
    if weak:
        return 0.3, f"Name loosely suggests this ('{weak[0]}')"
    return 0.0, ""


def _score_role(role: str, f: _Feat) -> tuple[float, list[str]]:
    ev: list[str] = []
    name_pts, why = _name_match(role, f)
    pts = name_pts
    if why:
        ev.append(why)
    mult = 1.0
    numeric = f.kind == "numeric"
    if role in NUMERIC_ROLES:
        if numeric:
            pts += 0.15
            ev.append("Numeric column")
        else:
            mult *= 0.15
            ev.append(f"Column is {f.kind}, but this role needs numbers")
        if role in MONEY_ROLES and f.tokens & RATIO_TOKENS and role != "balance":
            mult *= 0.4
            ev.append("Name indicates a ratio, not an amount")
        if numeric and f.vals is not None:
            if role == "quantity" and f.is_int and f.vmin >= 0:
                pts += 0.2 if name_pts else 0.05
                ev.append("Whole, non-negative numbers")
            if role in ("revenue", "cost") and f.vmin < 0:
                pts -= 0.1
                ev.append("Contains negative values")
            if role == "profit" and f.vmin < 0 and name_pts:
                pts += 0.1
                ev.append("Negative values are plausible (losses)")
            if role in MONEY_ROLES and not f.is_int and name_pts:
                pts += 0.1
                ev.append("Decimal amounts")
            if role == "percentage":
                if -1.0 <= f.vmin and f.vmax <= 1.0 and not f.is_int:
                    pts += 0.3
                    ev.append("Values lie between -1 and 1")
                elif f.vmin >= 0 and f.vmax <= 100 and name_pts:
                    pts += 0.2
                    ev.append("Values lie between 0 and 100")
                elif f.vmax > 100:
                    mult *= 0.3
                    ev.append("Values exceed 100")
    elif role == "datetime":
        if f.kind == "temporal":
            pts += 0.6
            ev.append("Values parse as dates")
        else:
            mult *= 0.4
            ev.append("Column does not parse as dates")
    elif role == "identifier":
        idlike = bool(f.tokens & ID_TOKENS)
        if f.kind == "identifier":
            pts += 0.4
            ev.append("Mostly unique values")
        elif idlike and numeric and f.is_int:
            pts += 0.4
            ev.append("Whole-number code with an id-style name")
        elif idlike and f.kind == "categorical":
            pts += 0.25
            ev.append("Repeating code with an id-style name")
        elif f.unique_ratio >= 0.98 and f.n >= 20 and (f.kind != "numeric" or f.is_int):
            pts += 0.2
            ev.append("Almost every value is unique")
        if idlike and f.tokens & (LEXICON["customer"].strong | LEXICON["product"].strong):
            pts -= 0.15
            ev.append("A more specific entity role may apply")
    elif role == "text":
        if f.kind == "text":
            pts += 0.7
            ev.append("Long free-form values")
    elif role == "category":
        if f.kind == "categorical":
            pts += 0.4 + (0.1 if f.unique <= 20 else 0)
            ev.append(f"{f.unique} distinct values")
        if f.tokens & ID_TOKENS:
            pts -= 0.25
            ev.append("Name ends in an id-style token")
        elif f.kind == "boolean":
            pts += 0.2
            ev.append("Two-valued column")
    elif role in ("customer", "product"):
        if f.kind in ("identifier", "categorical", "text") or (numeric and f.is_int and f.tokens & ID_TOKENS):
            pts += 0.2
            if name_pts and f.tokens & ID_TOKENS:
                pts += 0.2
                ev.append("Entity name combined with an id/key token")
        elif numeric:
            mult *= 0.3
            ev.append("Plain numeric column, unlikely to name an entity")
        if name_pts and f.tokens & _ATTR_TOKENS:
            pts -= 0.3
            ev.append("Name describes an attribute of the entity (see other roles)")
    elif role == "location":
        if f.kind in ("categorical", "text"):
            pts += 0.15
        elif f.kind == "temporal" or (numeric and not f.tokens & LOCATION_NUMERIC_TOKENS):
            mult *= 0.2
    elif role == "target":
        if f.kind == "boolean":
            pts += 0.3
            ev.append("Two-valued column, typical for labels")
        elif f.kind in ("text", "temporal"):
            mult *= 0.3
    return max(0.0, min(1.0, pts)) * mult, ev


def _relationship_boosts(df: pd.DataFrame, feats: dict[str, _Feat],
                         scores: dict[str, dict[str, tuple[float, list[str]]]]) -> None:
    """Detect a ~ b x c (revenue = quantity x price) and a ~ b - c (profit = revenue - cost)."""
    cand = [n for n, f in feats.items() if f.kind == "numeric" and f.vals is not None and f.vals.std() > 0]
    cand.sort(key=lambda n: -max(scores[n][r][0] for r in NUMERIC_ROLES))
    cand = cand[:16]
    if len(cand) < 3:
        return
    idx = df.sample(min(len(df), 2000), random_state=0).index if len(df) > 2000 else df.index
    cols = {}
    for n in cand:
        s = df.loc[idx, n]
        cols[n] = to_numeric_clean(s) if not pdt.is_numeric_dtype(s) else s.astype(float)
    num = pd.DataFrame(cols)

    def boost(col: str, role: str, amount: float, text: str) -> None:
        sc, ev = scores[col][role]
        scores[col][role] = (min(1.0, sc + amount), ev + [text])

    def close(a: pd.Series, p: pd.Series) -> float | None:
        ok = a.notna() & p.notna()
        if ok.sum() < 20:
            return None
        diff = (a[ok] - p[ok]).abs() <= 0.01 * a[ok].abs() + 1e-9
        return float(diff.mean())

    done: set[str] = set()
    for b, c in combinations(cand, 2):
        prod = num[b] * num[c]
        for a in cand:
            if a in (b, c) or a in done:
                continue
            r = close(num[a], prod)
            if r is not None and r >= 0.97:
                qty, pr = (b, c) if (feats[b].is_int, -num[b].mean()) >= (feats[c].is_int, -num[c].mean()) else (c, b)
                txt = f"Relationship: '{a}' = '{b}' x '{c}' in {r:.0%} of sampled rows"
                boost(a, "revenue", 0.3, txt)
                boost(qty, "quantity", 0.25, txt)
                boost(pr, "price", 0.25, txt)
                done.add(a)
    for b, c in combinations(cand, 2):
        for first, second in ((b, c), (c, b)):
            if num[first].mean() <= num[second].mean() or num[second].mean() <= 0:
                continue
            diff = num[first] - num[second]
            plausible = scores[first]["revenue"][0] >= 0.3 and scores[second]["cost"][0] >= 0.3
            for a in cand:
                if a in (first, second) or a in done:
                    continue
                if not (plausible or scores[a]["profit"][0] >= 0.3):   # a-b-c alone is ambiguous
                    continue
                r = close(num[a], diff)
                if r is not None and r >= 0.97:
                    txt = f"Relationship: '{a}' = '{first}' - '{second}' in {r:.0%} of sampled rows"
                    boost(a, "profit", 0.3, txt)
                    boost(first, "revenue", 0.2, txt)
                    boost(second, "cost", 0.2, txt)
                    done.add(a)


def classify_columns(df: pd.DataFrame, profile: DatasetProfile,
                     name_model: NameModel | None = None) -> dict[str, ColumnSemantic]:
    """Suggest a semantic role for every column. Reads only; never changes `df`."""
    feats = {c.name: _features(df, c) for c in profile.columns_profile}
    scores: dict[str, dict[str, tuple[float, list[str]]]] = {
        n: {r: _score_role(r, f) for r in _ROLES} for n, f in feats.items()}
    _relationship_boosts(df, feats, scores)

    out: dict[str, ColumnSemantic] = {}
    for c in profile.columns_profile:
        f, sc = feats[c.name], scores[c.name]
        if c.kind == "empty":
            out[c.name] = ColumnSemantic(c.name, "unknown", 0.0, ("Column is empty",), (), "rules", c.kind)
            continue
        if f.tokens & SENSITIVE_TOKENS:
            out[c.name] = ColumnSemantic(c.name, "unknown", 0.0,
                                         ("Name suggests personal or sensitive data; excluded from analytics",),
                                         (), "rules", c.kind)
            continue
        source = "rules"
        rules_best, rules_top = min(((r, v[0]) for r, v in sc.items()), key=lambda t: (-t[1], t[0]))
        if name_model is not None:
            probs = name_model.predict_proba(c.name)
            if probs:
                sc = dict(sc)           # blend on a copy: the rules-only scores stay untouched
                for r, p in probs.items():
                    if r in sc and p > 0:
                        s0, e0 = sc[r]
                        sc[r] = ((1 - MODEL_WEIGHT) * s0 + MODEL_WEIGHT * p, e0 + [f"Trained name model: p={p:.2f}"])
        ranked = sorted(((r, v[0]) for r, v in sc.items()), key=lambda t: (-t[1], t[0]))
        (best, s1), s2 = ranked[0], ranked[1][1]
        if s1 < MIN_SCORE:
            out[c.name] = ColumnSemantic(c.name, "unknown", 0.0, ("No strong signal from name or values",),
                                         tuple((r, round(s, 2)) for r, s in ranked[:2] if s > 0.15), "rules", c.kind)
            continue
        if name_model is not None and (best != rules_best or rules_top < MIN_SCORE):
            source = "model"            # label a column "model" only when the model changed the outcome
        conf = min(0.97, s1) * (0.7 + 0.3 * min(1.0, (s1 - s2) / 0.4))
        alts = tuple((r, round(s, 2)) for r, s in ranked[1:4] if s >= 0.25)
        out[c.name] = ColumnSemantic(c.name, best, round(conf, 2), tuple(sc[best][1]), alts, source, c.kind)
    return out

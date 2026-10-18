"""Business-domain detection from column vocabulary and detected semantic roles."""
from __future__ import annotations

from dataclasses import dataclass

from semantic.column_classifier import ColumnSemantic
from semantic.column_lexicon import (DOMAIN_LABELS, DOMAIN_LEXICON, DOMAIN_ROLE_SUPPORT, token_set)

MIN_RAW_SCORE = 2.0       # below this the dataset is treated as general tabular data


@dataclass(frozen=True)
class DomainResult:
    domain: str
    confidence: float
    evidence: tuple[str, ...]
    alternatives: tuple[tuple[str, float], ...]     # (domain, share of total evidence)
    source: str = "rules"                           # "rules" | "user"


def detect_domain(columns: dict[str, ColumnSemantic]) -> DomainResult:
    raw: dict[str, float] = {}
    hits: dict[str, dict[str, str]] = {}
    present_roles = {c.role for c in columns.values() if c.role != "unknown"}
    for d, terms in DOMAIN_LEXICON.items():
        matched: dict[str, str] = {}
        for name in columns:
            for t in sorted(token_set(name) & terms):
                matched.setdefault(t, name)
        support = DOMAIN_ROLE_SUPPORT.get(d, frozenset()) & present_roles
        raw[d] = len(matched) + 0.5 * len(support)
        hits[d] = matched
        if support:
            hits[d]["__roles__"] = ", ".join(sorted(support))
    total = sum(raw.values())
    ranked = sorted(raw.items(), key=lambda kv: (-kv[1], kv[0]))
    top, top_raw = ranked[0]
    def conf_of(v: float) -> float:       # same scale for the winner and the alternatives
        return round((v / total) * min(1.0, v / 5.0), 2)

    alts = tuple((d, conf_of(v)) for d, v in ranked[1:4] if v > 0) if total else ()
    if top_raw < MIN_RAW_SCORE or total == 0:
        return DomainResult("general", 0.3, ("Too few domain-specific column names to identify a business area",),
                            tuple((d, conf_of(v)) for d, v in ranked[:3] if v > 0) if total else ())
    conf = conf_of(top_raw)
    ev = [f"Column '{col}' contains the term '{t}'" for t, col in hits[top].items() if t != "__roles__"][:6]
    if "__roles__" in hits[top]:
        ev.append(f"Detected roles typical for {DOMAIN_LABELS[top]}: {hits[top]['__roles__']}")
    return DomainResult(top, conf, tuple(ev), alts)

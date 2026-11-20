"""Business-domain detection from column vocabulary and detected semantic roles."""
from __future__ import annotations

from dataclasses import dataclass

from semantic.column_classifier import ColumnSemantic
from semantic.column_lexicon import (DOMAIN_LABELS, DOMAIN_LEXICON, DOMAIN_ROLE_SUPPORT, MIN_SUPPLY_CHAIN_TERMS,
                                     SUPPLY_CHAIN_ROLES, SUPPLY_CHAIN_TERMS, semantic_tokens)

MIN_RAW_SCORE = 2.0       # below this the dataset is treated as general tabular data
# Domains a supply-chain dataset is commonly mistaken for: order lines read as retail, shipment logs as
# operations, stock tables as inventory. Enough supply-chain-only terms move these to supply chain.
_SUPPLY_CHAIN_LOOKALIKES = frozenset({"retail", "operations", "inventory", "general"})


@dataclass(frozen=True)
class DomainResult:
    domain: str
    confidence: float
    evidence: tuple[str, ...]
    alternatives: tuple[tuple[str, float], ...]     # (domain, share of total evidence)
    source: str = "rules"                           # "rules" | "user"


def supply_chain_terms(columns: dict[str, ColumnSemantic]) -> dict[str, str]:
    """Supply-chain-only terms found in the column names: {term: first column containing it}."""
    found: dict[str, str] = {}
    for name in columns:
        for t in sorted(semantic_tokens(name) & SUPPLY_CHAIN_TERMS):
            found.setdefault(t, name)
    return found


def _supply_chain_confidence(n_terms: int, n_roles: int) -> float:
    return round(min(0.9, 0.35 + 0.08 * n_terms + 0.05 * n_roles), 2)


def detect_domain(columns: dict[str, ColumnSemantic]) -> DomainResult:
    raw: dict[str, float] = {}
    hits: dict[str, dict[str, str]] = {}
    present_roles = {c.role for c in columns.values() if c.role != "unknown"}
    for d, terms in DOMAIN_LEXICON.items():
        matched: dict[str, str] = {}
        for name in columns:
            for t in sorted(semantic_tokens(name) & terms):
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
        return round((v / total) * min(1.0, v / 5.0), 2) if total else 0.0

    sc_terms = supply_chain_terms(columns)
    sc_roles = SUPPLY_CHAIN_ROLES & present_roles
    if len(sc_terms) >= MIN_SUPPLY_CHAIN_TERMS and (top == "supply_chain" or top in _SUPPLY_CHAIN_LOOKALIKES):
        ev = [f"Column '{col}' contains the supply-chain term '{t}'" for t, col in sc_terms.items()][:6]
        if top != "supply_chain":
            ev.insert(0, f"{len(sc_terms)} supply-chain terms in the column names, so the "
                         f"{DOMAIN_LABELS[top].lower()} vocabulary is read as part of a supply chain")
        if sc_roles:
            ev.append(f"Detected roles typical for supply chain: {', '.join(sorted(sc_roles))}")
        conf = max(conf_of(raw["supply_chain"]), _supply_chain_confidence(len(sc_terms), len(sc_roles)))
        alts = tuple((d, conf_of(v)) for d, v in ranked if d != "supply_chain" and v > 0)[:3]
        return DomainResult("supply_chain", conf, tuple(ev), alts)

    alts = tuple((d, conf_of(v)) for d, v in ranked[1:4] if v > 0) if total else ()
    if top_raw < MIN_RAW_SCORE or total == 0:
        return DomainResult("general", 0.3, ("Too few domain-specific column names to identify a business area",),
                            tuple((d, conf_of(v)) for d, v in ranked[:3] if v > 0) if total else ())
    conf = conf_of(top_raw)
    ev = [f"Column '{col}' contains the term '{t}'" for t, col in hits[top].items() if t != "__roles__"][:6]
    if "__roles__" in hits[top]:
        ev.append(f"Detected roles typical for {DOMAIN_LABELS[top]}: {hits[top]['__roles__']}")
    return DomainResult(top, conf, tuple(ev), alts)


@dataclass(frozen=True)
class SupplyChainCheck:
    is_supply_chain: bool
    reason: str


def assess_supply_chain(columns: dict[str, ColumnSemantic], domain: DomainResult) -> SupplyChainCheck:
    """The one definition of "this is a supply-chain dataset", shared by the name-model switch and the KPI pages.

    A domain the user chose wins. Otherwise: the detected domain is supply chain, or a supplier or warehouse
    column is present."""
    if domain.source == "user":
        yes = domain.domain == "supply_chain"
        return SupplyChainCheck(yes, f"You set the business domain to {DOMAIN_LABELS[domain.domain]}")
    if domain.domain == "supply_chain":
        return SupplyChainCheck(True, "Detected business domain is supply chain")
    roles = sorted(SUPPLY_CHAIN_ROLES & {c.role for c in columns.values()})
    if roles:
        cols = [n for n, c in columns.items() if c.role in SUPPLY_CHAIN_ROLES][:2]
        return SupplyChainCheck(True, f"Has {' and '.join(roles)} column(s): {', '.join(cols)}")
    return SupplyChainCheck(False, f"Detected business domain is {DOMAIN_LABELS[domain.domain]}, with no supplier or "
                                   "warehouse columns")

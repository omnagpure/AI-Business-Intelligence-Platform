"""Semantic model: roles per column + business domain, with user overrides applied on top.

Downstream analytics should ask this object (columns_with_role) instead of guessing from names,
so a user's correction is honoured everywhere. Build it with `build_semantic_model`.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import pandas as pd

from core.exceptions import DataValidationError
from ingestion.profiler import DatasetProfile
from semantic.business_domain import DomainResult, SupplyChainCheck, assess_supply_chain, detect_domain
from semantic.column_classifier import ColumnSemantic, NameModel, decide_roles, score_columns
from semantic.column_lexicon import DOMAINS, ROLES

REVIEW_BELOW = 0.5      # confidence under which a suggestion is flagged for review

# How the optional trained name model is used.
#   auto: rules first. Supply-chain data (where the rules are tuned and the model is weaker) keeps the rules in
#         charge and the model may only fill columns the rules left Unknown; any other data gets the model's vote
#         on every column.
#   on:   the model votes on every column, whatever the data.
#   off:  rules only.
MODEL_MODES = ("auto", "on", "off")
MODEL_MODE_LABELS = {"auto": "Automatic", "on": "Always on", "off": "Off"}


@dataclass(frozen=True)
class NameModelUse:
    mode: str           # one of MODEL_MODES
    applied: str        # "all" | "unknown" (only columns the rules left Unknown) | "none"
    reason: str

    @property
    def summary(self) -> str:
        what = {"all": "used on every column", "unknown": "used only for columns the rules could not identify",
                "none": "not used"}[self.applied]
        return f"Name model {what}. {self.reason}."


@dataclass(frozen=True)
class SemanticModel:
    columns: dict[str, ColumnSemantic]
    domain: DomainResult
    supply_chain: SupplyChainCheck | None = None      # computed from columns and domain when not given
    name_model_use: NameModelUse = field(default_factory=lambda: NameModelUse("off", "none", "Rules only"))

    def __post_init__(self) -> None:
        if self.supply_chain is None:
            object.__setattr__(self, "supply_chain", assess_supply_chain(self.columns, self.domain))

    @property
    def is_supply_chain(self) -> bool:
        return self.supply_chain.is_supply_chain

    def role_of(self, column: str) -> str:
        return self.columns[column].role

    def columns_with_role(self, role: str) -> list[str]:
        return [n for n, c in self.columns.items() if c.role == role]

    def needs_review(self) -> list[str]:
        return [n for n, c in self.columns.items() if c.source != "user" and c.confidence < REVIEW_BELOW]

    def overridden(self) -> list[str]:
        return [n for n, c in self.columns.items() if c.source == "user"]

    def fingerprint(self) -> str:
        """Stable hash of roles and domain. Use as part of cache keys for dependent results."""
        payload = json.dumps([sorted((n, c.role) for n, c in self.columns.items()), self.domain.domain])
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def to_records(self) -> list[dict]:
        return [{"column": c.column, "kind": c.kind, "role": c.role, "confidence": c.confidence,
                 "source": c.source, "evidence": list(c.evidence), "alternatives": list(c.alternatives)}
                for c in self.columns.values()]


def validate_overrides(role_overrides: dict[str, str] | None, domain_override: str | None) -> None:
    for col, role in (role_overrides or {}).items():
        if role not in ROLES:
            raise DataValidationError(f"'{role}' is not a valid role for column '{col}'.")
    if domain_override is not None and domain_override not in DOMAINS:
        raise DataValidationError(f"'{domain_override}' is not a valid business domain.")


def _assemble(cols: dict[str, ColumnSemantic], role_overrides: dict[str, str] | None,
              domain_override: str | None) -> tuple[dict[str, ColumnSemantic], DomainResult]:
    """Apply user role overrides, then detect the domain from the final roles (so a correction can change the
    domain), then apply a manual domain choice if any."""
    cols = dict(cols)
    for name, role in (role_overrides or {}).items():
        if name not in cols:            # stale override (column renamed or dropped by cleaning)
            continue
        auto = cols[name]
        cols[name] = ColumnSemantic(name, role, 1.0, ("Set by user",),
                                    ((auto.role, auto.confidence),) if auto.role != role else (), "user", auto.kind)
    domain = detect_domain(cols)
    if domain_override is not None:
        domain = DomainResult(domain_override, 1.0, ("Chosen by user",),
                              ((domain.domain, domain.confidence),) if domain.domain != domain_override else (), "user")
    return cols, domain


def build_semantic_model(df: pd.DataFrame, profile: DatasetProfile,
                         role_overrides: dict[str, str] | None = None,
                         domain_override: str | None = None,
                         name_model: NameModel | None = None,
                         model_mode: str = "auto") -> SemanticModel:
    """Classify columns with the rules, decide whether the data is supply chain, then (if a name model is given)
    blend it in as `model_mode` says. The supply-chain decision is taken from the rules-only result, so the
    model can never talk its way into being used."""
    if model_mode not in MODEL_MODES:
        raise DataValidationError(f"'{model_mode}' is not a valid name-model mode.")
    validate_overrides(role_overrides, domain_override)
    feats, scores = score_columns(df, profile)
    cols, domain = _assemble(decide_roles(profile, feats, scores), role_overrides, domain_override)
    sc = assess_supply_chain(cols, domain)

    if name_model is None:
        use = NameModelUse(model_mode, "none", "No trained model is loaded" if model_mode != "off" else "Switched off")
        return SemanticModel(cols, domain, sc, use)
    if model_mode == "off":
        return SemanticModel(cols, domain, sc, NameModelUse("off", "none", "Switched off"))
    if model_mode == "on":
        use = NameModelUse("on", "all", "Set to always on")
    elif sc.is_supply_chain:
        use = NameModelUse("auto", "unknown", f"Supply-chain data, where the rules are more reliable ({_lc(sc.reason)})")
    else:
        use = NameModelUse("auto", "all", f"Not supply-chain data ({_lc(sc.reason)})")

    cols, domain = _assemble(decide_roles(profile, feats, scores, name_model, use.applied), role_overrides,
                             domain_override)
    return SemanticModel(cols, domain, assess_supply_chain(cols, domain), use)


def _lc(text: str) -> str:
    return text[:1].lower() + text[1:]

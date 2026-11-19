"""Semantic model: roles per column + business domain, with user overrides applied on top.

Downstream analytics should ask this object (columns_with_role) instead of guessing from names,
so a user's correction is honoured everywhere. Build it with `build_semantic_model`.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import pandas as pd

from core.exceptions import DataValidationError
from ingestion.profiler import DatasetProfile
from semantic.business_domain import DomainResult, detect_domain
from semantic.column_classifier import ColumnSemantic, NameModel, classify_columns
from semantic.column_lexicon import DOMAINS, ROLES

REVIEW_BELOW = 0.5      # confidence under which a suggestion is flagged for review


@dataclass(frozen=True)
class SemanticModel:
    columns: dict[str, ColumnSemantic]
    domain: DomainResult

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


def build_semantic_model(df: pd.DataFrame, profile: DatasetProfile,
                         role_overrides: dict[str, str] | None = None,
                         domain_override: str | None = None,
                         name_model: NameModel | None = None) -> SemanticModel:
    """Classify columns, apply user role overrides, then detect the domain from the final roles
    (so a correction can change the domain), then apply a manual domain choice if any."""
    validate_overrides(role_overrides, domain_override)
    cols = classify_columns(df, profile, name_model)
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
    return SemanticModel(cols, domain)

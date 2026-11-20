"""Roles must not change because of performance work: faster relationship search == original, and real files keep their roles."""
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import semantic.column_classifier as cc
from ingestion.pipeline import ingest_bytes
from semantic.semantic_model import build_semantic_model
from tests.reference_relationship_boosts import reference_relationship_boosts

GOLDEN = json.loads((Path(__file__).parent / "golden" / "roles.json").read_text(encoding="utf-8"))
REAL_DIRS = [Path(os.environ["BI_REAL_FILES_DIR"])] if os.environ.get("BI_REAL_FILES_DIR") else []
REAL_DIRS.append(Path(__file__).parent / "real_files")
# the four real files named in the handoff, plus the on-disk names they may have been saved under
REAL_FILES = ("DataCoSupplyChainDataset.csv", "supply_chain_dataset1.csv", "supply-chain-manufacturing-sample.csv",
              "inventory-supply-chain-v1_0_0-analysis.xlsx")


def _key(name: str) -> str:
    """'inventory-supply-chain-v1.0.0-analysis' and 'inventory-supply-chain-v1_0_0-analysis' are the same file."""
    return re.sub(r"[^0-9a-z]", "", Path(name).stem.lower())


def _find(name: str) -> Path | None:
    """The file in a real-files folder or any folder below it (downloads often unpack into sub-folders)."""
    for d in REAL_DIRS:
        if not d.is_dir():
            continue
        for cand in sorted(d.rglob(f"*{Path(name).suffix}")):
            if cand.is_file() and _key(name) in _key(cand.name):
                return cand
    return None


def _snapshot(df, profile):
    m = build_semantic_model(df, profile, {}, None, None)
    return m, {n: (c.role, c.confidence, c.evidence, c.alternatives) for n, c in m.columns.items()}


def _with_reference(df, profile):
    """Same classification, but with the original pandas relationship search swapped in."""
    keep = cc._relationship_boosts
    cc._relationship_boosts = reference_relationship_boosts
    try:
        return _snapshot(df, profile)
    finally:
        cc._relationship_boosts = keep


def _synthetic(n=600, seed=3):
    rng = np.random.default_rng(seed)
    qty = rng.integers(1, 40, n).astype(float)
    price = np.round(rng.uniform(2, 90, n), 2)
    cost = np.round(price * qty * rng.uniform(0.4, 0.8, n), 2)
    rev = np.round(qty * price, 2)
    df = pd.DataFrame({"x1": qty, "x2": price, "x3": rev, "x4": cost, "x5": rev - cost, "x6": rng.normal(size=n),
                       "x7": [f"{v:,.2f}" for v in rev], "x8": rng.integers(0, 2, n)})
    df.loc[rng.choice(n, 40, replace=False), "x1"] = np.nan
    return df


def test_numpy_relationship_search_matches_original_on_synthetic_data():
    df = _synthetic()
    r = ingest_bytes(df.to_csv(index=False).encode(), "s.csv", max_bytes=10**8)
    new_model, new = _snapshot(r.dataset.df, r.profile)
    _old_model, old = _with_reference(r.dataset.df, r.profile)
    assert new == old
    assert any(e.startswith("Relationship:") for v in new.values() for e in v[2]), "the test data must exercise the relationship search"


def test_numpy_relationship_search_matches_original_on_a_large_sample_path():
    df = pd.concat([_synthetic(900, s) for s in (1, 2, 3)], ignore_index=True)       # > 2000 rows: the sampled branch
    r = ingest_bytes(df.to_csv(index=False).encode(), "big.csv", max_bytes=10**8)
    assert _snapshot(r.dataset.df, r.profile)[1] == _with_reference(r.dataset.df, r.profile)[1]


def test_fewer_than_three_numeric_candidates_returns_immediately():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0] * 10, "b": [2.0, 4.0, 7.0] * 10, "name": ["x", "y", "z"] * 10})
    r = ingest_bytes(df.to_csv(index=False).encode(), "t.csv", max_bytes=10**6)
    assert _snapshot(r.dataset.df, r.profile)[1] == _with_reference(r.dataset.df, r.profile)[1]


def test_no_numeric_columns_is_safe():
    df = pd.DataFrame({"city": ["a", "b", "c", "d"] * 10, "note": ["u", "v", "w", "x"] * 10})
    r = ingest_bytes(df.to_csv(index=False).encode(), "t.csv", max_bytes=10**6)
    assert set(build_semantic_model(r.dataset.df, r.profile, {}, None, None).columns) == {"city", "note"}


@pytest.mark.parametrize("name", REAL_FILES)
def test_real_file_roles_match_original_code_and_frozen_golden(name):
    path = _find(name)
    if path is None:
        pytest.skip(f"{name} not found (set BI_REAL_FILES_DIR or use tests/real_files/)")
    r = ingest_bytes(path.read_bytes(), path.name, max_bytes=10**10)
    m, new = _snapshot(r.dataset.df, r.profile)
    assert new == _with_reference(r.dataset.df, r.profile)[1]                              # identical to the original algorithm
    key = next((k for k in GOLDEN if _key(k) in _key(path.name) or _key(path.name) in _key(k)), None)
    if key:                                                                                # and to the roles frozen before Batch 2
        g = GOLDEN[key]
        assert {n: c.role for n, c in m.columns.items()} == g["roles"]
        assert {n: c.confidence for n, c in m.columns.items()} == g["confidence"]
        assert (m.domain.domain, m.domain.confidence) == (g["domain"], g["domain_confidence"])
        assert m.is_supply_chain                                                         # all four are supply-chain files


def test_frozen_golden_covers_the_known_domains():
    # DataCo read as retail until the supply-chain term rule (docs/STATUS.md); all four real files are supply chain.
    assert {k: v["domain"] for k, v in GOLDEN.items()} == {k: "supply_chain" for k in GOLDEN}
    assert set(GOLDEN) >= {"DataCoSupplyChainDataset.csv", "supply_chain_dataset1.csv"}

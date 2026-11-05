"""Supply-chain vocabulary: new roles (supplier, warehouse, flag), domain, and no mis-summing."""
from __future__ import annotations

import numpy as np
import pandas as pd

from analytics.descriptive import analysable_numeric_columns
from ingestion.profiler import profile_dataframe
from semantic.semantic_model import build_semantic_model


def _frame(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "run_id": [f"R{i}" for i in range(n)],
        "Supplier_ID": rng.choice(["S1", "S2", "S3"], n),
        "Warehouse_ID": rng.choice(["W1", "W2"], n),
        "supplier_name": rng.choice(["Acme", "Bolt", "Cog"], n),
        "lead_time_days": rng.integers(2, 30, n),
        "on_time_delivery": rng.choice([0, 1], n),
        "stockout": rng.choice([0, 1], n),
        "is_delayed": rng.choice(["Yes", "No"], n),
        "on_time_pct": rng.uniform(80, 100, n).round(2),
        "ordered_qty": rng.integers(10, 100, n),
        "received_qty": rng.integers(10, 100, n),
        "delivery_date": pd.date_range("2025-01-01", periods=n).astype(str),
    })


def _model(df: pd.DataFrame):
    return build_semantic_model(df, profile_dataframe(df))


def test_supplier_and_warehouse_ids_get_entity_roles_not_identifier():
    m = _model(_frame())
    assert m.role_of("Supplier_ID") == "supplier"
    assert m.role_of("supplier_name") == "supplier"
    assert m.role_of("Warehouse_ID") == "warehouse"
    assert m.role_of("run_id") == "identifier"


def test_yes_no_status_columns_are_flags_not_identifiers():
    m = _model(_frame())
    for col in ("on_time_delivery", "stockout", "is_delayed"):
        assert m.role_of(col) == "flag", col


def test_continuous_on_time_percentage_stays_a_percentage():
    assert _model(_frame()).role_of("on_time_pct") == "percentage"


def test_lead_time_and_quantities_keep_their_roles():
    m = _model(_frame())
    assert m.role_of("lead_time_days") == "duration"
    assert m.role_of("ordered_qty") == m.role_of("received_qty") == "quantity"


def test_domain_is_supply_chain_for_supply_chain_columns():
    assert _model(_frame()).domain.domain == "supply_chain"


def test_entity_and_flag_columns_are_never_summed_or_averaged():
    df = _frame()
    m = _model(df)
    numeric = analysable_numeric_columns(df, profile_dataframe(df), m)
    assert not {"on_time_delivery", "stockout", "Supplier_ID", "Warehouse_ID"} & set(numeric)
    assert "lead_time_days" in numeric


def test_binary_zero_one_column_named_rate_is_not_a_percentage():
    df = pd.DataFrame({"approval_rate": np.tile([0, 1], 50)})
    assert _model(df).role_of("approval_rate") != "percentage"


def test_carrier_is_not_treated_as_supplier_by_name_alone():
    df = pd.DataFrame({"carrier": np.tile(["X", "Y"], 50), "qty": np.arange(100)})
    assert _model(df).role_of("carrier") != "supplier"

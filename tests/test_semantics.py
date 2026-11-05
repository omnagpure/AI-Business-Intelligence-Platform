import numpy as np
import pandas as pd
import pytest

from core.exceptions import DataValidationError
from ingestion.pipeline import ingest_sample
from ingestion.profiler import profile_dataframe
from semantic.column_lexicon import tokenize, token_set
from semantic.semantic_model import build_semantic_model


def model_for(df, **kw):
    return build_semantic_model(df, profile_dataframe(df), **kw)


def sales_df(n=120):
    rng = np.random.default_rng(1)
    qty = rng.integers(1, 9, n)
    price = rng.choice([9.99, 24.5, 49.0, 120.0], n)
    return pd.DataFrame({
        "order_id": [f"O{i:05d}" for i in range(n)],
        "order_date": (pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 300, n), unit="D")).strftime("%Y-%m-%d"),
        "customer_id": rng.integers(1000, 1040, n),
        "product_category": rng.choice(["Toys", "Home", "Books"], n),
        "quantity": qty, "unit_price": price, "revenue": (qty * price).round(2),
        "city": rng.choice(["Pune", "Delhi", "Mumbai"], n),
    })


def test_tokenize_handles_camel_snake_and_spaces():
    assert tokenize("OrderItemTotal") == ["order", "item", "total"]
    assert tokenize("order_item total (USD)") == ["order", "item", "total", "usd"]
    assert {"unit", "units"} <= token_set("Units_Sold")


def test_common_roles_are_identified_with_evidence():
    m = model_for(sales_df())
    expected = {"order_id": "identifier", "order_date": "datetime", "customer_id": "customer",
                "product_category": "category", "quantity": "quantity", "unit_price": "price",
                "revenue": "revenue", "city": "location"}
    assert {c: m.role_of(c) for c in expected} == expected
    for col in m.columns.values():
        assert col.evidence and 0.0 <= col.confidence <= 1.0


def test_unnamed_numeric_column_is_not_assumed_to_be_money():
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"x7": rng.normal(50, 9, 80), "k": rng.normal(5, 2, 80)})
    m = model_for(df)
    assert m.role_of("x7") == "unknown" and m.role_of("k") == "unknown"
    assert m.columns["x7"].confidence == 0.0


def test_relationship_detection_finds_revenue_quantity_price_without_names():
    rng = np.random.default_rng(3)
    a = rng.integers(1, 10, 100)
    b = rng.choice([2.5, 4.0, 10.0, 33.3], 100)
    m = model_for(pd.DataFrame({"f1": a, "f2": b, "f3": (a * b).round(2)}))
    assert (m.role_of("f1"), m.role_of("f2"), m.role_of("f3")) == ("quantity", "price", "revenue")
    assert any("Relationship" in e for e in m.columns["f3"].evidence)


def test_ratio_names_are_percentages_not_profit():
    rng = np.random.default_rng(4)
    df = pd.DataFrame({"profit_ratio": rng.uniform(-0.2, 0.4, 80), "profit": rng.normal(100, 30, 80)})
    m = model_for(df)
    assert m.role_of("profit_ratio") == "percentage" and m.role_of("profit") == "profit"


def test_sensitive_and_empty_columns_are_excluded():
    df = pd.DataFrame({"customer_email": ["a@x.com"] * 30, "empty": [None] * 30, "n": range(30)})
    m = model_for(df)
    assert m.role_of("customer_email") == "unknown" and "sensitive" in m.columns["customer_email"].evidence[0]
    assert m.role_of("empty") == "unknown"


def test_entity_attribute_names_prefer_attribute_role():
    df = pd.DataFrame({"customer_city": ["Pune", "Delhi"] * 20, "customer_name": [f"N{i}" for i in range(40)]})
    m = model_for(df)
    assert m.role_of("customer_city") == "location"


def test_user_override_changes_role_confidence_source_and_queries():
    df = sales_df()
    auto = model_for(df)
    m = model_for(df, role_overrides={"quantity": "balance"})
    c = m.columns["quantity"]
    assert (c.role, c.confidence, c.source) == ("balance", 1.0, "user")
    assert c.alternatives[0][0] == "quantity"
    assert m.columns_with_role("balance") == ["quantity"] and "quantity" not in m.columns_with_role("quantity")
    assert m.overridden() == ["quantity"]
    assert m.fingerprint() != auto.fingerprint()


def test_stale_override_for_missing_column_is_ignored_and_invalid_role_rejected():
    df = sales_df()
    assert model_for(df, role_overrides={"gone": "revenue"}).fingerprint() == model_for(df).fingerprint()
    with pytest.raises(DataValidationError):
        model_for(df, role_overrides={"revenue": "not_a_role"})
    with pytest.raises(DataValidationError):
        model_for(df, domain_override="astrology")


def test_domain_detection_and_manual_override():
    assert model_for(sales_df()).domain.domain == "retail"
    hr = pd.DataFrame({"employee_id": range(40), "salary": np.linspace(30000, 90000, 40),
                       "department": ["A", "B"] * 20, "hire_date": ["2020-01-01"] * 40, "tenure_years": range(40)})
    d = model_for(hr).domain
    assert d.domain == "hr" and d.confidence > 0.3 and d.evidence
    m = model_for(hr, domain_override="finance")
    assert m.domain.domain == "finance" and m.domain.source == "user" and m.domain.confidence == 1.0
    assert m.domain.alternatives[0][0] == "hr"


def test_unrecognisable_data_falls_back_to_general_without_blocking():
    df = pd.DataFrame({"alpha": range(50), "beta": [x * 0.37 for x in range(50)], "gamma": ["u", "v"] * 25})
    d = model_for(df).domain
    assert d.domain == "general" and d.confidence <= 0.3


def test_role_override_can_change_domain():
    df = pd.DataFrame({"ledger": range(40), "a2": np.linspace(0, 1, 40), "a3": [1.5] * 20 + [2.5] * 20})
    base = model_for(df).domain.domain
    shifted = model_for(df, role_overrides={"ledger": "balance", "a2": "cost", "a3": "profit"}).domain.domain
    assert base == "general" and shifted == "finance"


class FakeModel:
    def predict_proba(self, name):
        return {"revenue": 0.9} if name == "mystery" else {}


def test_optional_name_model_blends_in_and_is_not_required():
    rng = np.random.default_rng(5)
    df = pd.DataFrame({"mystery": rng.normal(100, 20, 60), "other": rng.normal(1, 1, 60)})
    assert model_for(df).role_of("mystery") == "unknown"
    m = model_for(df, name_model=FakeModel())
    assert m.role_of("mystery") == "revenue" and m.columns["mystery"].source == "model"
    assert any("Trained name model" in e for e in m.columns["mystery"].evidence)


def test_classification_does_not_modify_the_dataframe():
    df = sales_df()
    before = df.copy(deep=True)
    model_for(df)
    pd.testing.assert_frame_equal(df, before)


def test_sample_dataset_end_to_end():
    r = ingest_sample()
    m = build_semantic_model(r.dataset.df, r.profile)
    assert m.role_of("revenue") == "revenue" and m.domain.domain == "retail"
    assert m.to_records()[0]["column"] == "order_id"

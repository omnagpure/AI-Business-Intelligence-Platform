"""Supply-chain detection, the automatic name-model switch, and the supply-chain rule fixes."""
import numpy as np
import pandas as pd
import pytest

from core.exceptions import DataValidationError
from ingestion.profiler import profile_dataframe
from semantic.business_domain import assess_supply_chain
from semantic.semantic_model import SemanticModel, build_semantic_model
from tests.test_role_stability import REAL_FILES, _find

N = 200


def model_for(df, **kw):
    return build_semantic_model(df, profile_dataframe(df), **kw)


class ConstantModel:
    """Says `role` with high probability for every column name."""
    def __init__(self, role="revenue", p=0.95):
        self.role, self.p = role, p

    def predict_proba(self, name):
        return {self.role: self.p}


def dataco_like(n=N):
    """Order lines with shipping and delivery tracking (the DataCo shape): commerce words plus logistics words."""
    rng = np.random.default_rng(2)
    qty = rng.integers(1, 5, n)
    price = rng.choice([20.0, 59.99, 129.0], n)
    return pd.DataFrame({
        "Order Id": np.arange(n), "Order Date": pd.date_range("2024-01-01", periods=n, freq="D").strftime("%Y-%m-%d"),
        "Customer Id": rng.integers(1, 60, n), "Customer Segment": rng.choice(["Consumer", "Corporate"], n),
        "Product Name": rng.choice(["Shoe", "Ball", "Bag"], n), "Category Name": rng.choice(["Fitness", "Outdoor"], n),
        "Order Item Quantity": qty, "Product Price": price, "Sales": qty * price, "Order Item Discount Rate": rng.uniform(0, .2, n),
        "Days for shipping (real)": rng.integers(1, 7, n), "Days for shipment (scheduled)": rng.integers(1, 5, n),
        "Delivery Status": rng.choice(["Late delivery", "Shipping on time", "Advance shipping"], n),
        "Late_delivery_risk": rng.choice([0, 1], n), "Shipping Mode": rng.choice(["Standard Class", "First Class"], n)})


def superstore_like(n=N):
    """Retail order lines with a ship date and ship mode, but no logistics tracking: must stay retail."""
    rng = np.random.default_rng(3)
    qty = rng.integers(1, 9, n)
    price = rng.choice([9.99, 24.5, 120.0], n)
    return pd.DataFrame({
        "Order ID": [f"CA-{i}" for i in range(n)], "Order Date": pd.date_range("2024-01-01", periods=n, freq="D").strftime("%Y-%m-%d"),
        "Ship Date": pd.date_range("2024-01-03", periods=n, freq="D").strftime("%Y-%m-%d"),
        "Ship Mode": rng.choice(["First", "Standard"], n), "Customer ID": rng.choice([f"C{i}" for i in range(40)], n),
        "Segment": rng.choice(["Consumer", "Corporate"], n), "City": rng.choice(["Austin", "Boston"], n),
        "Product ID": rng.choice([f"P{i}" for i in range(30)], n), "Category": rng.choice(["Tech", "Office"], n),
        "Sales": qty * price, "Quantity": qty, "Discount": rng.choice([0, .1, .2], n), "Profit": rng.normal(20, 30, n).round(2)})


def stock_table(n=N):
    """Inventory with supplier and warehouse columns, plus one column the rules cannot place."""
    rng = np.random.default_rng(4)
    return pd.DataFrame({
        "Supplier_ID": rng.choice([f"SUP_{i}" for i in range(8)], n), "Warehouse_ID": rng.choice([f"WH_{i}" for i in range(4)], n),
        "Units_Sold": rng.integers(0, 60, n), "Inventory_Level": rng.integers(100, 900, n),
        "Lead_Time_Days": rng.integers(2, 20, n), "Unit_Cost": rng.uniform(2, 50, n).round(2),
        "mystery": rng.normal(100, 20, n)})


def hr_table(n=N):
    rng = np.random.default_rng(5)
    return pd.DataFrame({
        "EmpID": np.arange(n), "Department": rng.choice(["IT", "HR", "Sales"], n), "Age": rng.integers(20, 60, n),
        "JobSatisfaction": rng.integers(1, 5, n), "mystery": rng.normal(100, 20, n),
        "HireDate": pd.date_range("2015-01-01", periods=n, freq="W").strftime("%Y-%m-%d")})


# ---- supply-chain detection -------------------------------------------------------------------

def test_order_data_with_shipping_and_delivery_tracking_is_supply_chain():
    m = model_for(dataco_like())
    assert m.domain.domain == "supply_chain" and m.is_supply_chain
    assert m.domain.confidence >= 0.6
    assert any("supply-chain term" in e for e in m.domain.evidence)


def test_retail_orders_with_only_a_ship_date_stay_retail():
    m = model_for(superstore_like())
    assert m.domain.domain == "retail" and not m.is_supply_chain


def test_supplier_or_warehouse_column_makes_it_supply_chain_whatever_the_domain_label():
    m = model_for(stock_table())
    assert m.is_supply_chain


def test_a_domain_chosen_by_the_user_decides_supply_chain():
    assert model_for(hr_table(), domain_override="supply_chain").is_supply_chain
    assert not model_for(stock_table(), domain_override="retail").is_supply_chain


def test_semantic_model_built_directly_computes_its_supply_chain_check():
    m = model_for(dataco_like())
    direct = SemanticModel(m.columns, m.domain)
    assert direct.is_supply_chain and direct.supply_chain == assess_supply_chain(m.columns, m.domain)


# ---- the automatic switch ---------------------------------------------------------------------

def test_auto_on_supply_chain_keeps_rules_and_only_fills_unknown_columns():
    df = stock_table()
    rules = model_for(df, model_mode="off")
    auto = model_for(df, name_model=ConstantModel("revenue"), model_mode="auto")
    assert auto.name_model_use.mode == "auto" and auto.name_model_use.applied == "unknown"
    assert rules.role_of("mystery") == "unknown"
    assert auto.role_of("mystery") == "revenue" and auto.columns["mystery"].source == "model"
    for col in df.columns.drop("mystery"):                     # every column the rules identified is untouched
        assert auto.columns[col] == rules.columns[col]


def test_auto_on_other_data_lets_the_model_vote_on_every_column():
    auto = model_for(hr_table(), name_model=ConstantModel("quantity", 0.95), model_mode="auto")
    assert auto.name_model_use.applied == "all" and not auto.is_supply_chain
    assert any("Trained name model" in e for c in auto.columns.values() for e in c.evidence if c.role != "unknown")
    assert "Not supply-chain data" in auto.name_model_use.summary


def test_always_on_uses_the_model_everywhere_even_for_supply_chain():
    on = model_for(stock_table(), name_model=ConstantModel(), model_mode="on")
    assert on.name_model_use.applied == "all"


def test_off_and_no_model_mean_rules_only():
    df = stock_table()
    rules = model_for(df)
    off = model_for(df, name_model=ConstantModel(), model_mode="off")
    assert off.name_model_use.applied == "none" and off.columns == rules.columns
    assert rules.name_model_use.applied == "none" and "No trained model" in rules.name_model_use.reason


def test_the_switch_follows_a_domain_override():
    assert model_for(hr_table(), domain_override="supply_chain", name_model=ConstantModel()).name_model_use.applied == "unknown"
    assert model_for(stock_table(), domain_override="retail", name_model=ConstantModel()).name_model_use.applied == "all"


def test_unknown_mode_is_rejected():
    with pytest.raises(DataValidationError):
        model_for(hr_table(), model_mode="sometimes")


def test_model_cannot_give_a_role_the_values_rule_out():
    df = pd.DataFrame({"notes_blob": [f"free text number {i} with several words in it" for i in range(80)],
                       "mystery": np.random.default_rng(1).normal(5, 1, 80)})
    m = model_for(df, name_model=ConstantModel("revenue", 0.99), model_mode="on")
    assert m.role_of("notes_blob") != "revenue"
    assert m.role_of("mystery") == "revenue"


# ---- supply-chain rule fixes ------------------------------------------------------------------

def test_order_value_on_a_purchase_order_is_spend():
    rng = np.random.default_rng(6)
    df = pd.DataFrame({"purchase_order_id": [f"PO{i:05d}" for i in range(N)], "supplier_id": rng.choice(["S1", "S2", "S3"], N),
                       "warehouse_id": rng.choice(["W1", "W2"], N), "quantity_ordered": rng.integers(10, 900, N),
                       "order_value": rng.uniform(1000, 90000, N).round(2)})
    assert model_for(df).role_of("order_value") == "cost"


def test_order_value_on_a_sales_order_stays_revenue():
    rng = np.random.default_rng(7)
    df = pd.DataFrame({"order_id": np.arange(N), "customer_id": rng.choice(["C1", "C2", "C3"], N),
                       "product": rng.choice(["A", "B"], N), "order_value": rng.uniform(10, 900, N).round(2)})
    assert model_for(df).role_of("order_value") == "revenue"


def test_product_sku_is_a_product_and_a_two_valued_status_is_a_flag():
    rng = np.random.default_rng(8)
    df = pd.DataFrame({"product_sku": [f"product_sku_{i}" for i in range(N)], "Product Status": rng.choice([0, 1], N),
                       "units_produced": rng.integers(100, 900, N)})
    m = model_for(df)
    assert m.role_of("product_sku") == "product"
    assert m.role_of("Product Status") == "flag"


# ---- the real files, when available -----------------------------------------------------------

@pytest.mark.parametrize("name", REAL_FILES)
def test_real_supply_chain_files_keep_the_rules_in_charge(name):
    from ingestion.pipeline import ingest_bytes
    path = _find(name)
    if path is None:
        pytest.skip(f"{name} not found (set BI_REAL_FILES_DIR or use tests/real_files/)")
    r = ingest_bytes(path.read_bytes(), path.name, max_bytes=10**10)
    m = build_semantic_model(r.dataset.df, r.profile, name_model=ConstantModel("revenue", 0.99), model_mode="auto")
    rules = build_semantic_model(r.dataset.df, r.profile, model_mode="off")
    assert m.is_supply_chain and m.name_model_use.applied == "unknown"
    assert all(m.columns[c] == rules.columns[c] for c in rules.columns if rules.columns[c].role != "unknown")

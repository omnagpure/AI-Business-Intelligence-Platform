"""Semantic layer on supply-chain headers: aliases, glued names, misspellings, generic headers and context."""
import numpy as np
import pandas as pd

from ingestion.profiler import profile_dataframe
from semantic.column_lexicon import semantic_tokens, token_set
from semantic.semantic_model import build_semantic_model
from training.make_data import SC_HEADERS, SC_HOLDOUT, SC_TRAIN_SEEDS, _bases, sc_holdout_frame
from training.sc_frame import make_frame

FRESH = [("ship_dt", "datetime"), ("inv_on_hand", "quantity"), ("vendor_lead_days", "duration"), ("po_line_qty", "quantity"),
         ("carrier_charges", "cost"), ("supplier_otd_pct", "percentage"), ("grn_date", "datetime"), ("asn_number", "identifier"),
         ("fill_rate", "percentage"), ("stock_out_days", "duration"), ("unit_landed_cost", "cost"), ("wh_code", "warehouse"),
         ("dest_city", "location"), ("freight_cost_usd", "cost"), ("shipped_qty", "quantity"), ("dlvry_date", "datetime"),
         ("supplier_nm", "supplier"), ("vendr_name", "supplier"), ("delivery_lead_time", "duration"), ("backordered_units", "quantity"),
         ("defect_pct", "percentage"), ("scrap_cost", "cost"), ("stock_value", "balance"), ("WarehouseCode", "warehouse"),
         ("SupplierName", "supplier"), ("TotalFreightCost", "cost"), ("ReceivedQty", "quantity"), ("OnTimeFlag", "flag")]


def _roles(pairs, seed=0):
    df = make_frame(pairs, seed=seed)
    return build_semantic_model(df, profile_dataframe(df)), df


def test_short_forms_glued_names_and_misspellings_map_to_canonical_words():
    assert "supplier" in semantic_tokens("supp_no")
    assert {"vendor", "id"} <= semantic_tokens("vendorid")
    assert "supplier" in semantic_tokens("suppllier_id")
    assert "delivery" in semantic_tokens("delivary_date")
    assert "warehouse" in semantic_tokens("warehose_id")
    assert "freight" in semantic_tokens("transportation_cost")
    assert "supplier" not in token_set("supp_no")            # the plain tokeniser is unchanged; only the semantic layer expands


def test_unrelated_words_are_not_repaired_into_vocabulary():
    for name in ("carbon_footprint", "product_name", "region", "salary"):
        assert not semantic_tokens(name) & {"supplier", "warehouse", "vendor"}, name


def test_all_hand_labelled_supply_chain_headers_get_the_expected_role():
    seen, pairs = set(), []
    for n, r in SC_HEADERS:
        if n.lower() not in seen:
            seen.add(n.lower())
            pairs.append((n, r))
    model, _ = _roles(pairs)
    wrong = [(n, r, model.role_of(n)) for n, r in pairs if model.role_of(n) != r]
    assert not wrong, wrong


def test_headers_written_after_the_lexicon_also_classify_correctly():
    model, _ = _roles(FRESH, seed=3)
    wrong = [(n, r, model.role_of(n)) for n, r in FRESH if model.role_of(n) != r]
    assert len(wrong) <= 1, wrong


def test_inventory_turnover_is_a_ratio_not_revenue():
    model, _ = _roles([("Inventory_Turnover", "percentage"), ("Total_Turnover", "revenue")])
    assert model.role_of("Inventory_Turnover") != "revenue"
    assert model.role_of("Total_Turnover") == "revenue"


def test_weight_distance_and_capacity_are_not_units_sold():
    df = pd.DataFrame({"Shipment_Weight": np.arange(60) * 1.5 + 1, "Distance_KM": np.arange(60) + 10.0, "Storage_Capacity": np.arange(60) + 100})
    m = build_semantic_model(df, profile_dataframe(df))
    assert not any(m.role_of(c) == "quantity" for c in df.columns)


def test_generic_amount_with_purchasing_neighbours_is_cost():
    df = make_frame([("PO_Number", "identifier"), ("Supplier_ID", "supplier"), ("Amount", "money")])
    m = build_semantic_model(df, profile_dataframe(df))
    assert m.role_of("Amount") == "cost"
    assert any("Context" in e for e in m.columns["Amount"].evidence)


def test_generic_amount_with_freight_neighbour_is_spend_not_revenue():
    df = make_frame([("Shipment_ID", "identifier"), ("Freight_Cost", "money"), ("Amount", "money")])
    assert build_semantic_model(df, profile_dataframe(df)).role_of("Amount") == "cost"


def test_generic_amount_with_sales_neighbours_is_revenue():
    df = make_frame([("Order_ID", "identifier"), ("Customer_ID", "customer"), ("Amount", "money")])
    assert build_semantic_model(df, profile_dataframe(df)).role_of("Amount") == "revenue"


def test_generic_header_without_context_is_low_confidence_and_flagged_for_review():
    df = make_frame([("Amount", "money"), ("Notes_Count", "int")])
    m = build_semantic_model(df, profile_dataframe(df))
    assert m.columns["Amount"].confidence <= 0.45 and "Amount" in m.needs_review()
    assert any("Generic header" in e for e in m.columns["Amount"].evidence)


def test_image_link_columns_are_not_entities():
    df = pd.DataFrame({"Product_Image": [f"http://x/{i % 5}.jpg" for i in range(100)], "Product_Name": [f"P{i % 5}" for i in range(100)]})
    m = build_semantic_model(df, profile_dataframe(df))
    assert m.role_of("Product_Image") == "text" and m.role_of("Product_Name") == "product"


def test_holdout_headers_are_never_in_training_seeds():
    train_names = {n.lower().replace("_", " ") for n, _ in SC_TRAIN_SEEDS}
    assert not any(n.lower().replace("_", " ") in train_names for n, _ in SC_HOLDOUT)
    train_keys = {k for _, _, k in _bases()}
    hold = sc_holdout_frame()
    assert len(hold) >= 30
    assert not any(tuple(sorted(set(n.lower().replace("_", " ").split()))) in train_keys for n in hold["name"])


def test_person_name_columns_are_excluded_not_treated_as_customers():
    df = pd.DataFrame({"Customer_Fname": [f"N{i % 40}" for i in range(120)], "Customer_Id": [i % 30 for i in range(120)]})
    m = build_semantic_model(df, profile_dataframe(df))
    assert m.role_of("Customer_Fname") == "unknown" and m.role_of("Customer_Id") == "customer"

"""Build labelled column-name data for the optional semantic name model.

Design rules (so the evaluation means something):
* Seed names are written by hand, independently of semantic/column_lexicon.py.
* Seed names are split into train/test BEFORE any variants are generated, so no base name (and no
  case/prefix/suffix variant of it) is ever in both sets. The test set therefore measures
  generalisation to unseen names, not memorisation.
* A second, separate evaluation set is made of real column headers from public supply-chain
  datasets, labelled by hand (REAL_HEADERS). Those are never used for training.

All of this is synthetic or hand-labelled and covers names only. It is NOT a benchmark of real-world
accuracy. Run:  python -m training.make_data
"""
from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

import pandas as pd

from semantic.column_lexicon import ROLES, tokenize

SEED = 42
DATA_DIR = Path(__file__).resolve().parent / "data"
TEST_FRACTION = 0.3
VARIANTS_PER_BASE = 14

SEED_NAMES: dict[str, str] = {
    "revenue": "total sales, net revenue, gross sales, sales amount, order total, invoice amount, turnover, gmv, "
               "income, billing amount, revenue usd, sales value, amt net, txn amount, subscription revenue, mrr, "
               "arr, ad revenue, fees collected, receipts, gross merchandise value, booking value, sales inr, "
               "net sales, line total, order value, payment received, takings, proceeds, total billed, "
               "monthly recurring revenue, annual contract value, deal amount, cart value, sales per customer",
    "cost": "unit cost, cogs, total cost, expense, operating expense, opex, cost of goods sold, purchase cost, "
            "shipping cost, freight cost, marketing spend, ad spend, labor cost, material cost, overhead, "
            "expenditure, cost price, procurement cost, supplier cost, manufacturing cost, rent expense, "
            "salary expense, acquisition cost, spend usd, outgoing payments, cost per unit, handling fee, tax paid",
    "profit": "profit, gross profit, net profit, net income, ebitda, operating income, earnings, profit per order, "
              "benefit per order, contribution margin, profit amount, margin amount, operating profit, pbt, pat, "
              "net earnings, profit usd, gross margin value, order profit, trading profit, surplus, "
              "retained earnings, income after tax, profit loss, bottom line, earnings before tax, "
              "profit before tax, net gain",
    "price": "unit price, price, list price, mrp, msrp, selling price, product price, item price, price per unit, "
             "sale price, retail price, base price, price usd, discounted price, offer price, wholesale price, "
             "price per kg, ticket price, subscription price, plan price, average selling price, asp, "
             "monthly price, standard price, price inr, listing price, shelf price, avg unit price",
    "quantity": "quantity, qty, units sold, units, order quantity, items sold, number of items, stock level, "
                "inventory level, on hand, units produced, pieces, volume sold, qty ordered, quantity received, "
                "count of items, seats, headcount, number of units, shipped qty, defects, reorder point, "
                "demand forecast, units returned, qty shipped, no of orders, order count, line items, num units",
    "datetime": "date, order date, created at, updated at, timestamp, transaction date, invoice date, ship date, "
                "shipped on, delivery date, signup date, dt, event time, purchase date, due date, expected at, "
                "received at, posting date, hire date, start date, end date, last login, booking date, "
                "record date, month, period, week start, txn dt, created on, modified time, closing date",
    "identifier": "id, order id, transaction id, invoice number, uuid, record id, row id, sku id, warehouse id, "
                  "supplier id, machine id, run id, ticket id, case id, session id, order item id, "
                  "purchase order id, shipment id, tracking number, reference no, serial number, batch id, "
                  "employee id, lead id, account id, line id, txn ref, guid, index key, primary key, "
                  "voucher no, receipt no, department id, category id",
    "category": "category, segment, type, department name, status, channel, brand, tier, payment method, "
                "shipping mode, order status, delivery status, gender, plan type, subscription plan, "
                "customer segment, product category, department, class, grade, source, lead source, "
                "campaign type, payment type, marital status, job title, industry, device type, priority, "
                "size band, product status, mode of transport, carrier",
    "customer": "customer, customer id, client id, buyer, subscriber id, member id, user id, customer number, "
                "cust id, client name, account holder, customer ref, buyer id, shopper id, patron id, "
                "order customer id, cust no, user key, consumer id, client code, subscriber, member, user, "
                "account number, customer key, client ref, customer code, end user id, customer name, tenant id",
    "product": "product, product name, item name, sku, product id, article, item, product code, product title, "
               "merchandise, product sku, item id, goods, sku code, model name, product line, product number, "
               "stock keeping unit, item code, product key, catalog item, part number, part name, "
               "product card id, material name, article number, model number, product family, product item",
    "location": "city, state, country, region, zip code, postal code, address, street, latitude, longitude, lat, "
                "lng, location, territory, district, market, warehouse region, supplier country, store location, "
                "shipping city, destination, origin country, province, county, area, branch city, pincode, "
                "ship to state, billing country, order region, geo, site, zone, locality",
    "percentage": "percent, discount rate, profit ratio, conversion rate, churn rate, growth rate, margin pct, "
                  "tax rate, interest rate, share, utilization, ctr, click through rate, bounce rate, "
                  "attrition rate, discount percent, fill rate, defect rate, success ratio, occupancy, "
                  "yield pct, completion rate, open rate, return rate, win rate, retention rate, pct change, "
                  "proportion, cagr, uptime percentage",
    "balance": "balance, account balance, closing balance, opening balance, outstanding amount, current balance, "
               "ledger balance, equity, deposit, cash balance, available balance, loan balance, credit balance, "
               "debit balance, net worth, wallet balance, outstanding, due balance, principal outstanding, "
               "savings, bank balance, remaining balance, overdraft, liabilities, assets, receivable balance, "
               "payable balance, balance usd, carrying value",
    "duration": "duration, days for shipping, lead time days, tenure, age, hours worked, delivery days, "
                "processing time, downtime minutes, time to resolve, response time, session duration, "
                "days to ship, days since signup, wait time, cycle time, shipping delay, length of stay, "
                "experience years, minutes spent, handle time, turnaround time, days overdue, "
                "resolution hours, supplier lead time, tenure months, age in days, elapsed seconds, "
                "days scheduled",
    "target": "churn, churned, is fraud, fraud flag, default, late delivery risk, converted, outcome, label, "
              "target, response, will buy, attrition, is churned, stockout flag, defaulted, readmitted, "
              "survived, clicked, purchased, class label, risk flag, late, is late, cancelled, returned flag, "
              "y, positive outcome, subscribed, has left, loan default, delayed",
    "text": "description, comments, review, notes, feedback, remarks, message, summary, product description, "
            "customer review, ticket text, complaint text, email body, comment text, reviewer comments, "
            "details, narrative, bio, abstract, free text, additional info, review text, support notes, "
            "address notes, job description, memo, explanation, observation, caption, tweet text",
    "unknown": "x, x1, x2, var1, var 2, col3, field a, field b, unnamed 0, unnamed 4, misc, other, tmp, temp, "
               "aux, z, f1, f2, attr7, column 12, data, value1, val, metric a, param, q, k, extra, spare, "
               "zz, abc, flag3, v10, unnamed 12",
}

# Neutral affixes only: they carry no role information, so they teach the model to ignore noise.
_PREFIXES = ["", "", "", "", "src", "raw", "final", "tbl", "x"]
_SUFFIXES = ["", "", "", "", "v2", "1", "old", "new", "clean", "copy"]
_STYLES = ["snake", "camel", "pascal", "upper", "kebab", "title", "lower_space", "concat"]

# Hand-labelled headers from the public supply-chain files used while building this project.
# Ambiguous columns (for example "Order Item Discount", "SKU_ID") and personal-data columns are left out
# on purpose. These labels are one person's judgement, not ground truth. Never used for training.
REAL_HEADERS: dict[str, str] = {
    # DataCo supply chain
    "Type": "category", "Days for shipping (real)": "duration", "Days for shipment (scheduled)": "duration",
    "Benefit per order": "profit", "Sales per customer": "revenue", "Delivery Status": "category",
    "Late_delivery_risk": "target", "Category Id": "identifier", "Category Name": "category",
    "Customer City": "location", "Customer Country": "location", "Customer Id": "customer",
    "Customer Segment": "category", "Customer State": "location", "Customer Zipcode": "location",
    "Department Id": "identifier", "Department Name": "category", "Latitude": "location",
    "Longitude": "location", "Market": "location", "Order City": "location", "Order Country": "location",
    "Order Customer Id": "customer", "order date (DateOrders)": "datetime", "Order Id": "identifier",
    "Order Item Cardprod Id": "identifier", "Order Item Discount Rate": "percentage",
    "Order Item Id": "identifier", "Order Item Product Price": "price", "Order Item Profit Ratio": "percentage",
    "Order Item Quantity": "quantity", "Sales": "revenue", "Order Item Total": "revenue",
    "Order Profit Per Order": "profit", "Order Region": "location", "Order State": "location",
    "Order Status": "category", "Product Card Id": "identifier", "Product Category Id": "identifier",
    "Product Description": "text", "Product Name": "product", "Product Price": "price",
    "Product Status": "category", "shipping date (DateOrders)": "datetime", "Shipping Mode": "category",
    # supply_chain_dataset1
    "Date": "datetime", "Warehouse_ID": "identifier", "Supplier_ID": "identifier", "Region": "location",
    "Units_Sold": "quantity", "Inventory_Level": "quantity", "Supplier_Lead_Time_Days": "duration",
    "Reorder_Point": "quantity", "Order_Quantity": "quantity", "Unit_Cost": "cost", "Unit_Price": "price",
    "Demand_Forecast": "quantity",
    # manufacturing sample
    "run_id": "identifier", "product_sku": "product", "machine_id": "identifier",
    "units_produced": "quantity", "defects": "quantity", "downtime_minutes": "duration",
    # inventory workbook
    "purchase_order_id": "identifier", "ordered_at": "datetime", "expected_at": "datetime",
    "received_at": "datetime", "status": "category", "supplier_country": "location",
    "warehouse_region": "location", "line_count": "quantity", "quantity_ordered": "quantity",
    "quantity_received": "quantity",
}


def normalise(name: str) -> str:
    return " ".join(tokenize(name))


def _render(tokens: list[str], style: str) -> str:
    if style == "snake":
        return "_".join(tokens)
    if style == "upper":
        return "_".join(tokens).upper()
    if style == "kebab":
        return "-".join(tokens)
    if style == "title":
        return " ".join(t.capitalize() for t in tokens)
    if style == "lower_space":
        return " ".join(tokens)
    if style == "camel":
        return tokens[0] + "".join(t.capitalize() for t in tokens[1:])
    if style == "pascal":
        return "".join(t.capitalize() for t in tokens)
    return "".join(tokens)       # concat


def _variants(tokens: list[str], rng: random.Random, n: int) -> list[str]:
    out: list[str] = []
    for i in range(n):
        toks = list(tokens)
        pre, suf = rng.choice(_PREFIXES), rng.choice(_SUFFIXES)
        if i >= 2:               # keep a couple of clean variants per base name
            toks = ([pre] if pre else []) + toks + ([suf] if suf else [])
        out.append(_render(toks, _STYLES[i % len(_STYLES)] if i < len(_STYLES) else rng.choice(_STYLES)))
    return list(dict.fromkeys(out))


def _bases() -> list[tuple[str, str, tuple[str, ...]]]:
    """(role, base name, token key) with ambiguous duplicates removed."""
    seen: dict[tuple[str, ...], set[str]] = {}
    items = []
    for role, text in SEED_NAMES.items():
        for raw in (s.strip() for s in text.split(",")):
            if not raw:
                continue
            key = tuple(sorted(set(tokenize(raw))))
            seen.setdefault(key, set()).add(role)
            items.append((role, raw, key))
    return [(r, b, k) for r, b, k in items if len(seen[k]) == 1 and r in ROLES]


def build_datasets(seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train, test) frames with columns name, role, base. Disjoint by base name."""
    rng = random.Random(seed)
    train_rows, test_rows = [], []
    by_role: dict[str, list[tuple[str, tuple[str, ...]]]] = {}
    for role, base, key in _bases():
        by_role.setdefault(role, []).append((base, key))
    for role in sorted(by_role):
        bases = sorted(set(by_role[role]))
        rng.shuffle(bases)
        n_test = max(1, round(len(bases) * TEST_FRACTION))
        for i, (base, _) in enumerate(bases):
            target = test_rows if i < n_test else train_rows
            for v in _variants(tokenize(base), rng, VARIANTS_PER_BASE):
                target.append({"name": v, "role": role, "base": base})
    train, test = pd.DataFrame(train_rows), pd.DataFrame(test_rows)
    leak = set(train["base"]) & set(test["base"])
    if leak:                                           # cannot happen by construction; guard anyway
        raise RuntimeError(f"Train/test leakage on base names: {sorted(leak)[:5]}")
    return train, test


def real_headers_frame() -> pd.DataFrame:
    return pd.DataFrame({"name": list(REAL_HEADERS), "role": list(REAL_HEADERS.values())})


def label_distribution(df: pd.DataFrame) -> dict[str, int]:
    return dict(sorted(Counter(df["role"]).items()))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--out", type=Path, default=DATA_DIR)
    a = p.parse_args()
    train, test = build_datasets(a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    train.to_csv(a.out / "train.csv", index=False)
    test.to_csv(a.out / "test.csv", index=False)
    real_headers_frame().to_csv(a.out / "real_headers.csv", index=False)
    print(f"train: {len(train)} rows from {train['base'].nunique()} base names")
    print(f"test : {len(test)} rows from {test['base'].nunique()} base names (disjoint from train)")
    print(f"real : {len(REAL_HEADERS)} hand-labelled headers (evaluation only)")
    print("train label counts:", label_distribution(train))
    print(f"written to {a.out}")


if __name__ == "__main__":
    main()

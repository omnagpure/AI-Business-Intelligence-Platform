"""Vocabulary used by the rule-based semantic layer: roles, name tokens, domain terms.

Pure data plus tokenisation. Extend the sets here to teach the classifier new terminology.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

ROLE_LABELS = {
    "revenue": "Revenue", "cost": "Cost", "profit": "Profit", "price": "Unit price",
    "quantity": "Quantity", "datetime": "Date / time", "identifier": "Identifier",
    "category": "Category", "customer": "Customer", "product": "Product", "location": "Location",
    "percentage": "Percentage / rate", "balance": "Balance", "duration": "Duration",
    "target": "Target / label", "text": "Free text", "unknown": "Unknown",
}
ROLES = tuple(ROLE_LABELS)
NUMERIC_ROLES = frozenset({"revenue", "cost", "profit", "price", "quantity", "percentage", "balance", "duration"})
MONEY_ROLES = frozenset({"revenue", "cost", "profit", "price", "balance"})


@dataclass(frozen=True)
class Lex:
    strong: frozenset[str]
    weak: frozenset[str] = frozenset()
    phrases: tuple[str, ...] = ()


def _l(strong: str, weak: str = "", phrases: tuple[str, ...] = ()) -> Lex:
    return Lex(frozenset(strong.split()), frozenset(weak.split()), phrases)


# Tokens are singular, lower-case (see token_set, which also adds singular forms of plurals).
LEXICON: dict[str, Lex] = {
    "revenue": _l("revenue sale turnover gmv", "amount total gross income value", ("net sales", "order total")),
    "cost": _l("cost cogs expense expenditure spend", "fee charge outflow"),
    "profit": _l("profit earning ebitda", "benefit margin", ("net income", "gross profit")),
    "price": _l("price mrp msrp", "tariff", ("unit price", "list price")),
    "quantity": _l("quantity qty unit piece volume", "number num count no inventory stock level demand forecast defect reorder"),
    "datetime": _l("date datetime timestamp time created updated ordered shipped", "dt month year period"),
    "identifier": _l("id uuid guid key sku", "code ref reference number no num"),
    "category": _l("category type class segment group department brand tier channel mode gender", "status kind"),
    "customer": _l("customer client buyer member subscriber user", "consumer account"),
    "product": _l("product sku article merchandise goods", "item"),
    "location": _l("city state country region province zip zipcode postal address street location "
                   "latitude longitude lat lon lng territory district market"),
    "percentage": _l("percent percentage pct ratio rate share proportion", "growth conversion ctr margin"),
    "balance": _l("balance outstanding equity deposit", "closing opening"),
    "duration": _l("duration day hour minute tenure age lead", "delay"),
    "target": _l("target label churn churned fraud default outcome converted response flag risk late",
                 "status delay"),
    "text": _l("description comment note review remark message summary feedback"),
}
LOCATION_NUMERIC_TOKENS = frozenset({"lat", "latitude", "lon", "lng", "longitude", "zip", "zipcode", "postal"})
RATIO_TOKENS = frozenset({"ratio", "rate", "pct", "percent", "percentage", "share", "proportion", "margin"})
SENSITIVE_TOKENS = frozenset({"password", "passwd", "pwd", "secret", "email", "phone", "mobile", "ssn", "passport", "iban"})
ID_TOKENS = frozenset({"id", "key", "code", "uuid", "guid"})

# ---- business domains ------------------------------------------------------------------------
DOMAIN_LABELS = {
    "retail": "Retail / e-commerce", "sales_crm": "Sales / CRM", "finance": "Finance",
    "inventory": "Inventory", "marketing": "Marketing", "saas": "SaaS / product analytics",
    "hr": "Human resources", "operations": "Operations / logistics", "general": "General tabular data",
}
DOMAINS = tuple(DOMAIN_LABELS)


def _d(s: str) -> frozenset[str]:
    return frozenset(s.split())


DOMAIN_LEXICON: dict[str, frozenset[str]] = {
    "retail": _d("order product category customer discount sale revenue cart basket store sku item brand price"),
    "sales_crm": _d("lead opportunity deal pipeline account stage quota rep prospect contact win lost"),
    "finance": _d("ledger debit credit balance expense income budget invoice transaction interest payment tax asset liability"),
    "inventory": _d("stock inventory warehouse reorder sku supplier lead onhand shelf"),
    "marketing": _d("campaign impression click ctr cpc conversion channel spend ad audience"),
    "saas": _d("subscription mrr arr churn plan user session signup active trial seat"),
    "hr": _d("employee salary department hire attrition tenure manager payroll job performance leave headcount"),
    "operations": _d("shipping shipment delivery carrier route late delay risk defect throughput production "
                     "supply schedule dispatch freight"),
}
DOMAIN_ROLE_SUPPORT: dict[str, frozenset[str]] = {
    "retail": frozenset({"product", "customer", "revenue", "quantity", "price"}),
    "sales_crm": frozenset({"customer", "revenue", "target"}),
    "finance": frozenset({"balance", "cost", "profit", "revenue"}),
    "inventory": frozenset({"product", "quantity", "cost"}),
    "marketing": frozenset({"percentage", "cost", "category"}),
    "saas": frozenset({"customer", "datetime", "target"}),
    "hr": frozenset({"duration"}),
    "operations": frozenset({"duration", "location", "target"}),
}

_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def tokenize(name: str) -> list[str]:
    """'Order Item Total' / 'orderItemTotal' / 'order_item_total' -> ['order', 'item', 'total']."""
    spaced = _CAMEL.sub(" ", str(name))
    return [t for t in re.split(r"[^0-9a-zA-Z]+", spaced.lower()) if t]


def token_set(name: str) -> set[str]:
    toks = tokenize(name)
    out = set(toks)
    out |= {t[:-1] for t in toks if len(t) > 3 and t.endswith("s") and not t.endswith("ss")}
    return out

"""Vocabulary used by the rule-based semantic layer: roles, name tokens, domain terms.

Pure data plus tokenisation. Extend the sets here to teach the classifier new terminology.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

ROLE_LABELS = {
    "revenue": "Revenue", "cost": "Cost", "profit": "Profit", "price": "Unit price",
    "quantity": "Quantity", "datetime": "Date / time", "identifier": "Identifier",
    "category": "Category", "customer": "Customer", "product": "Product", "location": "Location",
    "percentage": "Percentage / rate", "balance": "Balance", "duration": "Duration",
    "target": "Target / label", "text": "Free text", "unknown": "Unknown",
    "supplier": "Supplier / vendor", "warehouse": "Warehouse / facility", "flag": "Status flag (yes/no)",
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
    "cost": _l("cost cogs expense expenditure spend freight procurement", "fee charge outflow",
               ("freight amount", "shipping charge", "delivery fee", "shipping fee", "handling fee", "transport charge", "carrying cost")),
    "profit": _l("profit earning ebitda", "benefit margin", ("net income", "gross profit")),
    "price": _l("price mrp msrp", "tariff", ("unit price", "list price")),
    "quantity": _l("quantity qty unit piece volume moq",
                   "number num count no inventory stock level demand forecast defect reorder production backorder "
                   "reserved damaged scrap rejected accepted shortage cancelled returned variance",
                   ("on hand", "safety stock", "reorder level", "reorder quantity", "available stock", "closing inventory",
                    "opening inventory", "planned production", "actual production", "actual demand", "cancelled orders")),
    "datetime": _l("date datetime timestamp time created updated ordered shipped eta etd expiry expiration",
                   "dt month year period dispatch receiving delivered received"),
    "identifier": _l("id uuid guid key sku po awb", "code ref reference number no num tracking batch dock"),
    "category": _l("category type class segment group department brand tier channel mode gender carrier", "status kind reason terms result"),
    "customer": _l("customer client buyer member subscriber user", "consumer account"),
    "product": _l("product sku article merchandise goods", "item"),
    "location": _l("city state country region province zip zipcode postal address street location "
                   "latitude longitude lat lon lng territory district market origin destination zone"),
    "percentage": _l("percent percentage pct ratio rate share proportion utilization utilisation yield accuracy score rating",
                     "growth conversion ctr margin index fulfillment fulfilment",
                     ("inventory turnover", "turnover ratio", "stock turnover")),
    "balance": _l("balance outstanding equity deposit", "closing opening", ("inventory value", "stock value")),
    "duration": _l("duration day hour minute tenure age lead transit downtime dwell turnaround", "delay",
                   ("picking time", "packing time", "putaway time", "cycle time", "lead time", "transit time", "processing time",
                    "handling time", "wait time", "dock time", "days of inventory", "days of stock")),
    "target": _l("target label churn churned fraud default outcome converted response flag risk late",
                 "status delay"),
    "text": _l("description comment note review remark message summary feedback url link image photo thumbnail picture"),
    # supply chain
    "supplier": _l("supplier vendor manufacturer", "provider distributor"),
    "warehouse": _l("warehouse depot plant facility", "dc hub site"),
    "flag": _l("stockout backorder otif delayed ontime promotion promo",
               "flag", ("on time", "in full", "stock out", "is late", "is delayed", "was late")),
}

# Short forms and spelling variants -> the canonical token the lexicon knows. Conservative on purpose: a short
# token is only listed when it is not also a common word.
ALIASES: dict[str, str] = {
    "supp": "supplier", "suppl": "supplier", "sup": "supplier", "vend": "vendor", "vndr": "vendor", "vendr": "vendor",
    "wh": "warehouse", "whse": "warehouse", "wrhs": "warehouse", "whs": "warehouse",
    "shpmt": "shipment", "shpt": "shipment", "dlvry": "delivery", "delv": "delivery", "dlv": "delivery",
    "prod": "product", "inv": "inventory", "invt": "inventory", "stk": "stock", "ord": "order",
    "purch": "purchase", "proc": "procurement", "cst": "cost", "exp": "expected", "est": "estimated",
    "trans": "transit", "transport": "freight", "transportation": "freight", "logistic": "freight", "logistics": "freight",
    "amt": "amount", "tot": "total", "ttl": "total", "pct": "percent", "util": "utilization", "wt": "weight", "dist": "distance",
    "dur": "duration", "cnt": "count", "nbr": "number", "sts": "status", "stat": "status", "cust": "customer",
    "cstmr": "customer", "rcvd": "received", "shpd": "shipped", "qty": "quantity", "qnty": "quantity", "unts": "unit",
}
# Words whose misspellings we want to recognise (the lexicon vocabulary plus common supply-chain words).
_EXTRA_VOCAB = frozenset("delivery shipment inventory purchase procurement logistics dispatch receiving inspection production "
                         "warehouse supplier vendor quantity freight shipping transit downtime utilization accuracy".split())
_GLUE_SUFFIXES = frozenset("id no num number code key name ref reference date cost qty status type".split())
# Headers that say little on their own. Their role comes mostly from the values and the neighbouring columns,
# and their confidence is capped so they are never presented as certain.
IMAGE_TOKENS = frozenset("url link image photo thumbnail picture".split())      # a link to a file is not a product or segment
GENERIC_HEADERS = frozenset("amount value total code rate cost location reference ref time".split())
PROCUREMENT_CONTEXT = frozenset("po purchase supplier vendor procurement freight shipment shipping carrier transit warehouse".split())
SALES_CONTEXT = frozenset("customer sales sale invoice revenue product retail".split())
LOCATION_NUMERIC_TOKENS = frozenset({"lat", "latitude", "lon", "lng", "longitude", "zip", "zipcode", "postal"})
RATIO_TOKENS = frozenset({"ratio", "rate", "pct", "percent", "percentage", "share", "proportion", "margin"})
SENSITIVE_TOKENS = frozenset({"fname", "lname", "firstname", "lastname", "surname", "password", "passwd", "pwd", "secret", "email", "phone", "mobile", "ssn", "passport", "iban"})
ID_TOKENS = frozenset({"id", "key", "code", "uuid", "guid"})
ENTITY_ROLES = ("customer", "product", "supplier", "warehouse")

# ---- business domains ------------------------------------------------------------------------
DOMAIN_LABELS = {
    "retail": "Retail / e-commerce", "sales_crm": "Sales / CRM", "finance": "Finance",
    "inventory": "Inventory", "marketing": "Marketing", "saas": "SaaS / product analytics",
    "hr": "Human resources", "operations": "Operations / logistics", "supply_chain": "Supply chain", "general": "General tabular data",
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
    "supply_chain": _d("supplier vendor warehouse lead reorder stockout backorder fill otif delivery delivered carrier "
                       "shipment shipping inventory stock freight transit order received ordered defect "
                       "procurement purchase moq putaway pallet dock tracking eta"),
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
    "supply_chain": frozenset({"supplier", "warehouse", "flag", "duration", "quantity"}),
}

_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def tokenize(name: str) -> list[str]:
    """'Order Item Total' / 'orderItemTotal' / 'order_item_total' -> ['order', 'item', 'total']."""
    spaced = _CAMEL.sub(" ", str(name))
    return [t for t in re.split(r"[^0-9a-zA-Z]+", spaced.lower()) if t]


def _vocab() -> frozenset[str]:
    words = set(_EXTRA_VOCAB)
    for lex in LEXICON.values():
        words |= lex.strong | lex.weak
    return frozenset(w for w in words if len(w) >= 5)


_VOCAB = _vocab()


def semantic_tokens(name: str) -> set[str]:
    """token_set plus: short forms mapped to canonical words (supp_no -> supplier), glued names split
    (vendorid -> vendor, id) and one-letter-slip misspellings repaired (suppllier -> supplier).
    The original tokens are always kept, so this only ever adds evidence. Used by the semantic layer only."""
    base = token_set(name)
    out = set(base)
    for t in tokenize(name):
        if t in ALIASES:
            out.add(ALIASES[t])
        if len(t) >= 6 and t not in _VOCAB:
            for w in _VOCAB:                                     # glued: vendorid, supplierkey, transportcost
                if t.startswith(w) and t[len(w):] in _GLUE_SUFFIXES:
                    out |= {w, ALIASES.get(t[len(w):], t[len(w):])}
            near = difflib.get_close_matches(t, _VOCAB, n=1, cutoff=0.86)      # misspellings: suppllier, delivary
            if near and near[0][0] == t[0] and abs(len(near[0]) - len(t)) <= 2:
                out.add(near[0])
    return out


def token_set(name: str) -> set[str]:
    toks = tokenize(name)
    out = set(toks)
    out |= {t[:-1] for t in toks if len(t) > 3 and t.endswith("s") and not t.endswith("ss")}
    return out

"""Which KPIs deserve the headline cards. Pure, deterministic, no Streamlit.

The KPI engine computes every KPI whose roles exist. This module only orders and groups them for display:
supply-chain data leads with delivery, stock, quality and supplier KPIs; other data keeps the engine order.
Hidden KPIs stay available (nothing is deleted or invented).
"""
from __future__ import annotations

from dataclasses import dataclass

from analytics.kpis import OK, KPIReport, KPIResult

SUPPLY_GROUPS = frozenset({"Supply chain", "Procurement"})
SUPPLY_ORDER = ("on_time_rate", "otif", "fill_rate", "avg_lead_time", "avg_late_delay", "cancel_rate", "avg_delay", "stockout_rate",
                "below_reorder_share", "days_of_cover", "inventory_turnover", "defect_rate",
                "supplier_top_share", "supplier_top3_share", "total_spend", "spend_growth", "avg_po_value",
                "stock_units", "stock_value", "reorder_candidates", "order_count", "units",
                "total_revenue", "revenue_growth")
NOT_SUPPLY_GROUPS = frozenset({"Sales", "Profit and cost", "Customers"})
SUPPLY_KEYS = frozenset(SUPPLY_ORDER)


@dataclass(frozen=True)
class KPIDisplay:
    headline: tuple[KPIResult, ...]            # cards shown first
    more: tuple[KPIResult, ...]                # other calculated KPIs (expander)
    unavailable_relevant: tuple[KPIResult, ...]  # not calculated, but relevant to this data
    unavailable_other: tuple[KPIResult, ...]     # not calculated and not relevant here (kept for transparency)
    supply_chain_mode: bool


def is_supply_chain(report: KPIReport) -> bool:
    return report.domain == "supply_chain" or sum(1 for r in report.ok() if r.group in SUPPLY_GROUPS) >= 2


def prioritise(report: KPIReport, top: int = 6) -> KPIDisplay:
    """Pick up to `top` headline KPIs. Supply-chain mode ranks by SUPPLY_ORDER; otherwise engine order."""
    mode = is_supply_chain(report)
    ok = [r for r in report.ok() if r.key != "row_count"] or report.ok()
    if mode:
        rank = {k: i for i, k in enumerate(SUPPLY_ORDER)}
        ok = sorted(ok, key=lambda r: rank.get(r.key, len(rank)))        # stable: ties keep engine order
        headline_pool = [r for r in ok if r.key in SUPPLY_KEYS] or ok
    else:
        headline_pool = ok
    headline = tuple(headline_pool[:top])
    more = tuple(r for r in report.ok() if r not in headline)
    rel, other = [], []
    for r in report.unavailable():
        hide = mode and r.group in NOT_SUPPLY_GROUPS and r.key not in SUPPLY_KEYS
        (other if hide else rel).append(r)
    return KPIDisplay(headline, more, tuple(rel), tuple(other), mode)

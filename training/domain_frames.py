"""Small labelled tables from domains other than supply chain, for an end-to-end check of the semantic layer.

Each table has realistic headers and values with a plausible shape (ids unique, dates parse, flags two-valued ...),
and a hand label per column. They test the full classifier (names + values + relationships) the way the app uses
it, with the name model off and in automatic mode. Many headers here also appear among the training names (the
evaluation reports the subset the model never saw separately). Synthetic values: a sanity check, not a benchmark.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

N = 300


def _frames(seed: int = 0) -> list[tuple[str, pd.DataFrame, dict[str, str]]]:
    rng = np.random.default_rng(seed)
    days = pd.date_range("2024-01-01", periods=N, freq="D")
    out: list[tuple[str, pd.DataFrame, dict[str, str]]] = []

    hr = pd.DataFrame({
        "EmployeeNumber": np.arange(1, N + 1), "Age": rng.integers(20, 60, N), "Department": rng.choice(["IT", "HR", "Sales"], N),
        "JobRole": rng.choice(["Analyst", "Manager", "Engineer", "Rep"], N), "MonthlySalary": rng.integers(2000, 20000, N),
        "YearsAtCompany": rng.integers(0, 30, N), "TotalWorkingYears": rng.integers(0, 40, N),
        "PercentSalaryHike": rng.integers(10, 25, N), "Attrition": rng.choice(["Yes", "No"], N),
        "BusinessTravel": rng.choice(["Rarely", "Frequently", "Non-Travel"], N), "NumCompaniesWorked": rng.integers(0, 9, N),
        "HireDate": days.strftime("%Y-%m-%d")})
    out.append(("HR", hr, {"EmployeeNumber": "identifier", "Age": "duration", "Department": "category", "JobRole": "category",
                           "MonthlySalary": "cost", "YearsAtCompany": "duration", "TotalWorkingYears": "duration",
                           "PercentSalaryHike": "percentage", "Attrition": "target", "BusinessTravel": "category",
                           "NumCompaniesWorked": "quantity", "HireDate": "datetime"}))

    imp = rng.integers(1000, 100000, N)
    clicks = (imp * rng.uniform(0.005, 0.05, N)).astype(int)
    spend = (clicks * rng.uniform(0.2, 2.0, N)).round(2)
    mk = pd.DataFrame({
        "campaign_id": [f"CMP{i:04d}" for i in range(N)], "campaign_name": rng.choice(["Spring", "Summer", "Brand", "Retarget"], N),
        "day": days.strftime("%Y-%m-%d"), "channel": rng.choice(["search", "social", "display"], N), "impressions": imp,
        "clicks": clicks, "ad_spend": spend, "cpc": (spend / np.maximum(clicks, 1)).round(2),
        "conversions": rng.integers(0, 40, N), "roas": rng.uniform(0.5, 8, N).round(2), "ctr": (clicks / imp).round(4)})
    out.append(("Marketing", mk, {"campaign_id": "identifier", "campaign_name": "category", "day": "datetime", "channel": "category",
                                  "impressions": "quantity", "clicks": "quantity", "ad_spend": "cost", "cpc": "cost",
                                  "conversions": "quantity", "roas": "percentage", "ctr": "percentage"}))

    saas = pd.DataFrame({
        "account_id": [f"ACC{i:05d}" for i in range(N)], "signup_date": days.strftime("%Y-%m-%d"),
        "plan_tier": rng.choice(["free", "pro", "team"], N), "mrr": rng.choice([0, 29, 99, 299], N).astype(float),
        "seats": rng.integers(1, 50, N), "monthly_active_users": rng.integers(0, 500, N), "logins_30d": rng.integers(0, 300, N),
        "nps": rng.integers(-100, 101, N), "churned": rng.choice([0, 1], N), "billing_cycle": rng.choice(["monthly", "annual"], N),
        "auto_renew": rng.choice([True, False], N)})
    out.append(("SaaS", saas, {"account_id": "identifier", "signup_date": "datetime", "plan_tier": "category", "mrr": "revenue",
                               "seats": "quantity", "monthly_active_users": "quantity", "logins_30d": "quantity",
                               "nps": "percentage", "churned": "target", "billing_cycle": "category", "auto_renew": "flag"}))

    fin = pd.DataFrame({
        "journal_entry_id": [f"JE{i:06d}" for i in range(N)], "posting_date": days.strftime("%Y-%m-%d"),
        "gl_account": rng.choice(["1000 Cash", "1200 AR", "2000 AP", "4000 Sales"], N),
        "cost_center": rng.choice(["CC10", "CC20", "CC30"], N), "transaction_type": rng.choice(["invoice", "payment", "accrual"], N),
        "operating_expense": rng.uniform(100, 9000, N).round(2), "net_revenue": rng.uniform(1000, 50000, N).round(2),
        "closing_balance": rng.uniform(-5000, 80000, N).round(2), "interest_rate": rng.uniform(0.01, 0.12, N).round(4),
        "currency": rng.choice(["USD", "EUR"], N)})
    out.append(("Finance", fin, {"journal_entry_id": "identifier", "posting_date": "datetime", "gl_account": "category",
                                 "cost_center": "category", "transaction_type": "category", "operating_expense": "cost",
                                 "net_revenue": "revenue", "closing_balance": "balance", "interest_rate": "percentage",
                                 "currency": "category"}))
    return out


def domain_frames(seed: int = 0) -> list[tuple[str, pd.DataFrame, dict[str, str]]]:
    """[(title, dataframe, {column: hand label})] for HR, marketing, SaaS and finance."""
    return _frames(seed)

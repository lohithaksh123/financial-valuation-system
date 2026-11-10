"""Rule-based analyst commentary: turns ratios into plain-language findings."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from fivs.analysis.ratios import cagr
from fivs.models import Company

GOOD, WATCH, RISK = "good", "watch", "risk"


@dataclass
class Insight:
    severity: str
    title: str
    detail: str


def _known(*values: float) -> bool:
    return not any(pd.isna(v) for v in values)


def generate_insights(company: Company, ratios: pd.DataFrame) -> List[Insight]:
    fin, out = company.financials, []
    latest = ratios.iloc[-1]
    previous = ratios.iloc[-2] if len(ratios) > 1 else latest * np.nan
    add = lambda severity, title, detail: out.append(Insight(severity, title, detail))  # noqa: E731

    years = min(3, len(fin) - 1)
    growth = cagr(fin["revenue"], years) if years else None
    if growth is not None:
        if growth >= 0.10:
            add(GOOD, "Strong revenue growth", f"Revenue compounded at {growth:.1%} a year over {years} years.")
        elif growth < 0:
            add(RISK, "Shrinking revenue", f"Revenue declined {abs(growth):.1%} a year over {years} years.")

    if len(fin) > 1 and _known(fin["receivables"].iloc[-1], fin["receivables"].iloc[-2], latest.revenue_growth):
        if fin["receivables"].iloc[-2] > 0:
            receivable_growth = fin["receivables"].iloc[-1] / fin["receivables"].iloc[-2] - 1
            if receivable_growth > latest.revenue_growth + 0.15:
                add(WATCH, "Receivables outpacing sales",
                    f"Receivables grew {receivable_growth:.0%} against revenue growth of {latest.revenue_growth:.0%}. "
                    "This can signal looser credit terms or aggressive revenue recognition.")

    if not company.is_balance_sheet_financial:
        if _known(latest.ebit_margin, previous.ebit_margin):
            change = latest.ebit_margin - previous.ebit_margin
            if change >= 0.02:
                add(GOOD, "Expanding margins", f"Operating margin rose {change * 100:.1f} points to {latest.ebit_margin:.1%}.")
            elif change <= -0.02:
                add(WATCH, "Margin pressure", f"Operating margin fell {abs(change) * 100:.1f} points to {latest.ebit_margin:.1%}.")

        if _known(latest.roic):
            if latest.roic >= 0.15:
                add(GOOD, "High return on capital", f"ROIC of {latest.roic:.1%} points to a durable competitive position.")
            elif latest.roic < 0.06:
                add(WATCH, "Low return on capital", f"ROIC of {latest.roic:.1%} is likely below the cost of capital.")

        if _known(latest.cash_conversion) and fin["net_income"].iloc[-1] > 0:
            if latest.cash_conversion < 0.8:
                add(RISK, "Weak earnings quality",
                    f"Operating cash flow is only {latest.cash_conversion:.0%} of net income, so profit is not turning into cash.")
            elif latest.cash_conversion >= 1.1:
                add(GOOD, "Cash-backed earnings", f"Operating cash flow is {latest.cash_conversion:.0%} of net income.")

        if _known(latest.free_cash_flow) and latest.free_cash_flow < 0:
            add(RISK, "Negative free cash flow", "Capital spending exceeds cash generated from operations.")

        capex = fin["capex"] / fin["revenue"]
        if len(capex.dropna()) >= 3 and capex.iloc[-1] > capex.iloc[-3] * 1.5 and capex.iloc[-1] > 0.10:
            add(WATCH, "Capital spending surge",
                f"Capex rose to {capex.iloc[-1]:.0%} of revenue, from {capex.iloc[-3]:.0%} two years earlier. "
                "Near-term free cash flow understates earning power only if this spending earns its return.")

        if _known(latest.net_debt_to_ebitda):
            if latest.net_debt_to_ebitda > 4:
                add(RISK, "High leverage", f"Net debt is {latest.net_debt_to_ebitda:.1f}x EBITDA.")
            elif latest.net_debt_to_ebitda < 0:
                add(GOOD, "Net cash balance sheet", "Cash and short-term investments exceed total debt.")

        if _known(latest.interest_coverage) and latest.interest_coverage < 3:
            add(RISK, "Thin interest coverage", f"Operating income covers interest only {latest.interest_coverage:.1f}x.")

        if _known(latest.current_ratio) and latest.current_ratio < 0.9:
            add(WATCH, "Tight liquidity", f"Current ratio of {latest.current_ratio:.2f} means short-term liabilities exceed current assets.")
    else:
        if _known(latest.roe):
            if latest.roe >= 0.14:
                add(GOOD, "Strong return on equity", f"ROE of {latest.roe:.1%} is high for a balance-sheet lender.")
            elif latest.roe < 0.08:
                add(WATCH, "Low return on equity", f"ROE of {latest.roe:.1%} is likely below the cost of equity.")
        equity_to_assets = fin["equity"].iloc[-1] / fin["total_assets"].iloc[-1]
        add(GOOD if equity_to_assets >= 0.08 else WATCH, "Equity cushion",
            f"Book equity is {equity_to_assets:.1%} of total assets.")

    return out

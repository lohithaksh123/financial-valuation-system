"""Derives default valuation inputs from a company's own history."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from fivs.analysis.ratios import compute_ratios
from fivs.config import MARKETS, MarketDefaults
from fivs.models import Company

HISTORY_YEARS = 3
WACC_MIN_SPREAD_OVER_GROWTH = 0.03


@dataclass(frozen=True)
class CostOfCapital:
    risk_free: float
    equity_risk_premium: float
    raw_beta: Optional[float]
    beta: float
    cost_of_equity: float
    pre_tax_cost_of_debt: float
    tax_rate: float
    equity_weight: float
    wacc: float


@dataclass(frozen=True)
class DCFAssumptions:
    revenue_base: float
    growth_initial: float
    growth_terminal: float
    margin_initial: float
    margin_target: float
    tax_rate: float
    net_capex_pct: float  # capex above depreciation, as a share of revenue
    nwc_pct: float  # working capital needed per unit of new revenue
    wacc: float
    roic_spread: float = 0.03  # return on new capital in perpetuity, above WACC
    years: int = 10


@dataclass(frozen=True)
class ResidualIncomeAssumptions:
    book_value: float
    roe_initial: float
    cost_of_equity: float
    payout_ratio: float
    roe_terminal_spread: float = 0.01
    persistence: float = 0.60  # share of excess returns surviving each year after the horizon
    years: int = 10


def _recent_median(series: pd.Series, default: float) -> float:
    clean = series.replace([np.inf, -np.inf], np.nan).dropna().tail(HISTORY_YEARS)
    return float(clean.median()) if len(clean) else default


def effective_tax_rate(company: Company, market: MarketDefaults) -> float:
    fin = company.financials
    pretax = fin["pretax_income"].where(fin["pretax_income"] > 0)
    return float(np.clip(_recent_median(fin["income_tax"] / pretax, market.statutory_tax), 0.10, 0.35))


def adjusted_beta(raw_beta: Optional[float]) -> float:
    """Blume-adjusted beta, bounded so a thinly correlated quote cannot set the discount rate."""
    if raw_beta is None:
        return 1.0
    return float(np.clip(0.67 * raw_beta + 0.33, 0.70, 2.0))


def cost_of_capital(company: Company, risk_free: float, market: Optional[MarketDefaults] = None) -> CostOfCapital:
    market = market or MARKETS[company.market]
    tax = effective_tax_rate(company, market)
    beta = adjusted_beta(company.beta)
    cost_of_equity = risk_free + beta * market.equity_risk_premium

    fin = company.financials
    debt = fin["total_debt"].iloc[-1]
    implied_rate = _recent_median(fin["interest_expense"] / fin["total_debt"].where(fin["total_debt"] > 0), risk_free + 0.015)
    cost_of_debt = float(np.clip(implied_rate, risk_free + 0.005, risk_free + 0.06))

    debt_value = 0.0 if pd.isna(debt) else float(debt)
    equity_weight = company.market_cap / (company.market_cap + debt_value)
    wacc = equity_weight * cost_of_equity + (1 - equity_weight) * cost_of_debt * (1 - tax)
    wacc = max(wacc, market.terminal_growth + WACC_MIN_SPREAD_OVER_GROWTH)
    return CostOfCapital(
        risk_free, market.equity_risk_premium, company.beta, beta, cost_of_equity, cost_of_debt, tax, equity_weight, wacc
    )


def build_dcf_assumptions(company: Company, capital: CostOfCapital) -> DCFAssumptions:
    fin, market = company.financials, MARKETS[company.market]
    ratios = compute_ratios(fin)

    growth = np.clip(_recent_median(ratios["revenue_growth"], market.terminal_growth), -0.05, 0.25)
    margin_latest = ratios["ebit_margin"].dropna()
    net_capex = (fin["capex"] - fin["depreciation_amortization"]) / fin["revenue"]
    working_capital = (
        fin["receivables"].fillna(0) + fin["inventory"].fillna(0) - fin["accounts_payable"].fillna(0)
    ) / fin["revenue"]

    return DCFAssumptions(
        revenue_base=float(fin["revenue"].iloc[-1]),
        growth_initial=float(growth),
        growth_terminal=market.terminal_growth,
        margin_initial=float(margin_latest.iloc[-1]) if len(margin_latest) else 0.0,
        margin_target=_recent_median(ratios["ebit_margin"], 0.0),
        tax_rate=capital.tax_rate,
        # Both floored at zero: free cash from shrinking assets or stretching
        # suppliers is not something to capitalize for ten years.
        net_capex_pct=float(np.clip(_recent_median(net_capex, 0.02), 0.0, 0.25)),
        nwc_pct=float(np.clip(_recent_median(working_capital, 0.05), 0.0, 0.40)),
        wacc=capital.wacc,
    )


def build_residual_income_assumptions(company: Company, capital: CostOfCapital) -> ResidualIncomeAssumptions:
    fin = company.financials
    average_equity = fin["equity"].rolling(2, min_periods=1).mean()
    earning_years = fin["net_income"].where(fin["net_income"] > 0)
    return ResidualIncomeAssumptions(
        book_value=float(fin["equity"].iloc[-1]),
        roe_initial=float(np.clip(_recent_median(fin["net_income"] / average_equity, capital.cost_of_equity), 0.0, 0.35)),
        cost_of_equity=capital.cost_of_equity,
        payout_ratio=float(np.clip(_recent_median(fin["dividends_paid"] / earning_years, 0.35), 0.0, 0.90)),
    )

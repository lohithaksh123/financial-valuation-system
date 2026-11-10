"""Residual income (excess return) model, for banks and insurers.

Value = book equity + present value of future earnings above the cost of equity.
It avoids free cash flow and debt, neither of which means much for a lender.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np
import pandas as pd

from fivs.valuation.assumptions import ResidualIncomeAssumptions


@dataclass
class ResidualIncomeResult:
    assumptions: ResidualIncomeAssumptions
    equity_value: float
    value_per_share: float
    pv_excess_returns: float
    pv_terminal: float
    forecast: pd.DataFrame


def run_residual_income(a: ResidualIncomeAssumptions, shares: float) -> ResidualIncomeResult:
    if a.persistence >= 1 + a.cost_of_equity or a.persistence < 0:
        raise ValueError("Persistence must be between 0 and 1.")

    t = np.arange(1, a.years + 1, dtype=float)
    roe = a.roe_initial + (a.cost_of_equity + a.roe_terminal_spread - a.roe_initial) * t / a.years

    opening_book = np.empty(a.years)
    book = a.book_value
    for i, year_roe in enumerate(roe):
        opening_book[i] = book
        book *= 1 + year_roe * (1 - a.payout_ratio)

    residual = (roe - a.cost_of_equity) * opening_book
    discount = (1 + a.cost_of_equity) ** -t
    pv_residual = residual * discount

    terminal = residual[-1] * a.persistence / (1 + a.cost_of_equity - a.persistence)
    pv_terminal = float(terminal * discount[-1])
    equity_value = a.book_value + float(pv_residual.sum()) + pv_terminal

    forecast = pd.DataFrame(
        {"opening_book": opening_book, "roe": roe, "residual_income": residual, "pv_residual": pv_residual},
        index=pd.RangeIndex(1, a.years + 1, name="year"),
    )
    return ResidualIncomeResult(a, equity_value, equity_value / shares, float(pv_residual.sum()), pv_terminal, forecast)


def residual_income_sensitivity(
    a: ResidualIncomeAssumptions, shares: float,
    ke_steps: Sequence[float] = (-0.01, -0.005, 0.0, 0.005, 0.01),
    roe_steps: Sequence[float] = (-0.02, -0.01, 0.0, 0.01, 0.02),
) -> pd.DataFrame:
    """Value per share across cost of equity (rows) and starting ROE (columns)."""
    kes = [a.cost_of_equity + step for step in ke_steps]
    roes = [a.roe_initial + step for step in roe_steps]
    values = [
        [run_residual_income(replace(a, cost_of_equity=ke, roe_initial=roe), shares).value_per_share for roe in roes]
        for ke in kes
    ]
    return pd.DataFrame(values, index=kes, columns=roes)

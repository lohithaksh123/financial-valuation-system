"""Discounted cash flow model.

`project` is one vectorized implementation shared by the base case, the sensitivity
grid and the Monte Carlo simulation, so all three always agree.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from fivs.valuation.assumptions import DCFAssumptions


@dataclass
class Projection:
    revenue: np.ndarray
    margin: np.ndarray
    nopat: np.ndarray
    reinvestment: np.ndarray
    fcff: np.ndarray
    pv_fcff: np.ndarray
    pv_terminal: np.ndarray
    enterprise_value: np.ndarray


def project(
    revenue_base, growth_initial, growth_terminal, margin_initial, margin_target,
    tax_rate, net_capex_pct, nwc_pct, wacc, roic_spread, years: int,
) -> Projection:
    """Project free cash flow to the firm and discount it.

    Inputs may be scalars or 1-D arrays of equal length; each array element is one
    scenario. Outputs carry one row per scenario.
    """
    def column(value) -> np.ndarray:
        return np.atleast_1d(np.asarray(value, dtype=float))[:, np.newaxis]

    g0, gt, m0, mt, tax = map(column, (growth_initial, growth_terminal, margin_initial, margin_target, tax_rate))
    capex0, nwc, r, spread, base = map(column, (net_capex_pct, nwc_pct, wacc, roic_spread, revenue_base))

    if np.any(r <= gt):
        raise ValueError("WACC must exceed terminal growth, or the terminal value is undefined.")

    t = np.arange(1, years + 1, dtype=float)
    fade = t / years

    growth = g0 + (gt - g0) * (t - 1) / years
    revenue = base * np.cumprod(1 + growth, axis=1)
    opening_revenue = np.broadcast_to(base, (revenue.shape[0], 1))
    prior_revenue = np.concatenate([opening_revenue, revenue[:, :-1]], axis=1)
    margin = m0 + (mt - m0) * fade
    nopat = revenue * margin * (1 - tax)

    # Value-driver formula: growing at g with return on new capital R requires
    # reinvesting g/R of NOPAT. Today's capex fades to that sustainable level, so
    # a temporary investment cycle is not extrapolated forever.
    new_capital_return = r + spread
    sustainable_rate = gt / new_capital_return
    sustainable_capex = sustainable_rate * mt * (1 - tax)
    capex_pct = capex0 + (sustainable_capex - capex0) * fade
    reinvestment = capex_pct * revenue + nwc * (revenue - prior_revenue)
    fcff = nopat - reinvestment

    discount = (1 + r) ** -t
    pv_fcff = fcff * discount

    terminal_nopat = revenue[:, -1:] * (1 + gt) * mt * (1 - tax)
    terminal_value = terminal_nopat * (1 - sustainable_rate) / (r - gt)
    pv_terminal = terminal_value * discount[:, -1:]

    enterprise_value = pv_fcff.sum(axis=1) + pv_terminal[:, 0]
    return Projection(revenue, margin, nopat, reinvestment, fcff, pv_fcff, pv_terminal[:, 0], enterprise_value)


def _project_assumptions(a: DCFAssumptions, **replacements) -> Projection:
    values = dict(
        revenue_base=a.revenue_base, growth_initial=a.growth_initial, growth_terminal=a.growth_terminal,
        margin_initial=a.margin_initial, margin_target=a.margin_target, tax_rate=a.tax_rate,
        net_capex_pct=a.net_capex_pct, nwc_pct=a.nwc_pct, wacc=a.wacc, roic_spread=a.roic_spread, years=a.years,
    )
    values.update(replacements)
    return project(**values)


@dataclass
class DCFResult:
    assumptions: DCFAssumptions
    enterprise_value: float
    equity_value: float
    value_per_share: float
    terminal_share: float
    forecast: pd.DataFrame


def run_dcf(a: DCFAssumptions, net_debt: float, minority_interest: float, shares: float) -> DCFResult:
    p = _project_assumptions(a)
    enterprise_value = float(p.enterprise_value[0])
    equity_value = enterprise_value - net_debt - minority_interest
    forecast = pd.DataFrame(
        {
            "revenue": p.revenue[0], "ebit_margin": p.margin[0], "nopat": p.nopat[0],
            "reinvestment": p.reinvestment[0], "fcff": p.fcff[0], "pv_fcff": p.pv_fcff[0],
        },
        index=pd.RangeIndex(1, a.years + 1, name="year"),
    )
    return DCFResult(a, enterprise_value, equity_value, equity_value / shares, float(p.pv_terminal[0]) / enterprise_value, forecast)


def sensitivity_table(
    a: DCFAssumptions, net_debt: float, minority_interest: float, shares: float,
    wacc_steps: Sequence[float] = (-0.01, -0.005, 0.0, 0.005, 0.01),
    growth_steps: Sequence[float] = (-0.01, -0.005, 0.0, 0.005, 0.01),
) -> pd.DataFrame:
    """Value per share across WACC (rows) and terminal growth (columns)."""
    waccs = [a.wacc + step for step in wacc_steps]
    growths = [a.growth_terminal + step for step in growth_steps]
    grid_w, grid_g = np.meshgrid(waccs, growths, indexing="ij")
    grid_w_flat, grid_g_flat = grid_w.ravel(), grid_g.ravel()
    valid = grid_w_flat > grid_g_flat + 0.005

    values = np.full(grid_w_flat.shape, np.nan)
    if valid.any():
        ev = _project_assumptions(a, wacc=grid_w_flat[valid], growth_terminal=grid_g_flat[valid]).enterprise_value
        values[valid] = (ev - net_debt - minority_interest) / shares
    return pd.DataFrame(values.reshape(grid_w.shape), index=waccs, columns=growths)


def market_implied_growth(
    a: DCFAssumptions, net_debt: float, minority_interest: float, shares: float, price: float,
) -> Optional[float]:
    """Reverse DCF: the starting growth rate that justifies today's price (bisection)."""
    def gap(growth: float) -> float:
        ev = float(_project_assumptions(a, growth_initial=growth).enterprise_value[0])
        return (ev - net_debt - minority_interest) / shares - price

    low, high = -0.30, 0.80
    if gap(low) * gap(high) > 0:
        return None
    for _ in range(80):
        mid = (low + high) / 2
        if gap(low) * gap(mid) <= 0:
            high = mid
        else:
            low = mid
    return (low + high) / 2

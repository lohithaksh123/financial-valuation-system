"""Chooses the right models for a company and blends them into one view."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from fivs.models import Company, PeerMultiples
from fivs.valuation.assumptions import (
    CostOfCapital,
    build_dcf_assumptions,
    build_residual_income_assumptions,
    cost_of_capital,
)
from fivs.valuation.comps import CompsResult, value_by_comps
from fivs.valuation.dcf import DCFResult, market_implied_growth, run_dcf, sensitivity_table
from fivs.valuation.monte_carlo import MonteCarloResult, simulate
from fivs.valuation.residual_income import ResidualIncomeResult, residual_income_sensitivity, run_residual_income

INTRINSIC_WEIGHT, COMPS_WEIGHT = 0.60, 0.40
FAIR_VALUE_BAND = 0.15
HIGH_TERMINAL_SHARE = 0.85


@dataclass
class RangeRow:
    label: str
    low: float
    mid: float
    high: float


@dataclass
class ValuationResult:
    capital: CostOfCapital
    dcf: Optional[DCFResult] = None
    monte_carlo: Optional[MonteCarloResult] = None
    implied_growth: Optional[float] = None
    residual_income: Optional[ResidualIncomeResult] = None
    sensitivity: Optional[pd.DataFrame] = None
    comps: Optional[CompsResult] = None
    fair_value: Optional[float] = None
    upside: Optional[float] = None
    verdict: str = "Not enough data"
    method: str = ""
    ranges: List[RangeRow] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


def _apply(assumptions, overrides: Optional[Dict[str, Any]]):
    if not overrides:
        return assumptions
    fields = assumptions.__dataclass_fields__
    return replace(assumptions, **{k: v for k, v in overrides.items() if k in fields})


def value_company(
    company: Company,
    peers: Sequence[PeerMultiples],
    risk_free: float,
    overrides: Optional[Dict[str, Any]] = None,
    runs: int = 10_000,
    seed: int = 42,
) -> ValuationResult:
    capital = cost_of_capital(company, risk_free)
    result = ValuationResult(capital=capital)
    shares = company.shares_outstanding
    intrinsic: Optional[float] = None

    if company.is_balance_sheet_financial:
        assumptions = _apply(build_residual_income_assumptions(company, capital), overrides)
        result.residual_income = run_residual_income(assumptions, shares)
        result.sensitivity = residual_income_sensitivity(assumptions, shares)
        intrinsic = result.residual_income.value_per_share
        low, high = result.sensitivity.iloc[3, 1], result.sensitivity.iloc[1, 3]
        result.ranges.append(RangeRow("Residual income", float(low), intrinsic, float(high)))
        result.method = "Residual income + peer multiples"
        result.notes.append("Banks and insurers are valued on book equity and excess returns; a DCF does not apply.")
    else:
        assumptions = _apply(build_dcf_assumptions(company, capital), overrides)
        net_debt, minority = company.net_debt, company.minority_interest
        result.dcf = run_dcf(assumptions, net_debt, minority, shares)
        result.sensitivity = sensitivity_table(assumptions, net_debt, minority, shares)
        result.monte_carlo = simulate(assumptions, net_debt, minority, shares, company.price, runs=runs, seed=seed)
        if company.price is not None:
            result.implied_growth = market_implied_growth(assumptions, net_debt, minority, shares, company.price)
        result.method = "DCF + peer multiples"
        if assumptions.margin_target <= 0:
            result.notes.append("Operating margin is not positive, so the DCF is excluded from the blended value.")
        else:
            intrinsic = result.dcf.value_per_share
            result.ranges.append(RangeRow("DCF (Monte Carlo P10-P90)", result.monte_carlo.p10, intrinsic, result.monte_carlo.p90))
            if result.dcf.terminal_share > HIGH_TERMINAL_SHARE:
                result.notes.append(
                    f"{result.dcf.terminal_share:.0%} of DCF value comes from the terminal value, so it is highly assumption-driven."
                )

    if peers:
        result.comps = value_by_comps(company, peers)
        for row in (r for r in result.comps.rows if r.primary):
            result.ranges.append(RangeRow(f"{row.multiple} peers", row.low_value, row.median_value, row.high_value))

    if company.low_52w and company.high_52w:
        midpoint = (company.low_52w + company.high_52w) / 2
        result.ranges.append(RangeRow("52-week trading range", company.low_52w, midpoint, company.high_52w))

    comps_value = result.comps.value_per_share if result.comps else None
    if intrinsic is not None and comps_value is not None:
        result.fair_value = INTRINSIC_WEIGHT * intrinsic + COMPS_WEIGHT * comps_value
    elif intrinsic is not None:
        result.fair_value = intrinsic
        result.notes.append("No usable peer multiples, so fair value uses the intrinsic model alone.")
    elif comps_value is not None:
        result.fair_value = comps_value
        result.method = "Peer multiples only"

    if result.fair_value is not None and company.price is None:
        result.verdict = "Model value only"
    elif result.fair_value is not None:
        result.upside = result.fair_value / company.price - 1
        if result.upside > FAIR_VALUE_BAND:
            result.verdict = "Price below model"
        elif result.upside < -FAIR_VALUE_BAND:
            result.verdict = "Price above model"
        else:
            result.verdict = "Near model value"
    return result

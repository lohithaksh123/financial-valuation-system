"""Monte Carlo simulation over the DCF's most uncertain inputs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from fivs.valuation.assumptions import DCFAssumptions
from fivs.valuation.dcf import _project_assumptions

MIN_WACC_OVER_GROWTH = 0.015


@dataclass(frozen=True)
class Uncertainty:
    growth_sd: float = 0.03
    margin_sd: float = 0.02
    wacc_range: float = 0.01
    terminal_growth_range: float = 0.005


@dataclass
class MonteCarloResult:
    values: np.ndarray
    mean: float
    p10: float
    p50: float
    p90: float
    prob_above_price: Optional[float]


def _triangular(rng: np.random.Generator, centre: float, half_width: float, runs: int) -> np.ndarray:
    if half_width == 0:
        return np.full(runs, centre)
    return rng.triangular(centre - half_width, centre, centre + half_width, runs)


def simulate(
    a: DCFAssumptions, net_debt: float, minority_interest: float, shares: float, price: Optional[float],
    uncertainty: Uncertainty = Uncertainty(), runs: int = 10_000, seed: int = 42,
) -> MonteCarloResult:
    rng = np.random.default_rng(seed)
    growth = rng.normal(a.growth_initial, uncertainty.growth_sd, runs)
    margin = rng.normal(a.margin_target, uncertainty.margin_sd, runs)
    wacc = _triangular(rng, a.wacc, uncertainty.wacc_range, runs)
    terminal = _triangular(rng, a.growth_terminal, uncertainty.terminal_growth_range, runs)
    terminal = np.minimum(terminal, wacc - MIN_WACC_OVER_GROWTH)

    ev = _project_assumptions(
        a, growth_initial=growth, margin_target=margin, wacc=wacc, growth_terminal=terminal
    ).enterprise_value
    values = np.maximum((ev - net_debt - minority_interest) / shares, 0.0)  # limited liability
    return MonteCarloResult(
        values=values,
        mean=float(values.mean()),
        p10=float(np.percentile(values, 10)),
        p50=float(np.percentile(values, 50)),
        p90=float(np.percentile(values, 90)),
        prob_above_price=None if price is None else float((values > price).mean()),
    )

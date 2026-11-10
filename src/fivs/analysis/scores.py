"""Academic health scores: Piotroski F-Score and Altman Z''-Score."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class Signal:
    name: str
    passed: Optional[bool]  # None when the data needed is missing


@dataclass
class PiotroskiResult:
    signals: List[Signal] = field(default_factory=list)

    @property
    def score(self) -> int:
        return sum(1 for s in self.signals if s.passed)

    @property
    def tested(self) -> int:
        return sum(1 for s in self.signals if s.passed is not None)

    @property
    def label(self) -> str:
        if self.tested < 6:
            return "Insufficient data"
        share = self.score / self.tested
        return "Strong" if share >= 7 / 9 else "Weak" if share <= 3 / 9 else "Average"


@dataclass
class AltmanResult:
    z: float
    zone: str
    working_capital_to_assets: float
    retained_earnings_to_assets: float
    ebit_to_assets: float
    book_equity_to_liabilities: float


def _test(condition_a: float, condition_b: float, compare) -> Optional[bool]:
    if np.isnan(condition_a) or np.isnan(condition_b):
        return None
    return bool(compare(condition_a, condition_b))


def piotroski(fin: pd.DataFrame) -> Optional[PiotroskiResult]:
    """Nine pass/fail signals on profitability, leverage and efficiency (Piotroski, 2000)."""
    if len(fin) < 2:
        return None
    cur, prev = fin.iloc[-1], fin.iloc[-2]
    gt = lambda a, b: a > b  # noqa: E731
    lt = lambda a, b: a < b  # noqa: E731
    le = lambda a, b: a <= b  # noqa: E731

    roa_cur = cur.net_income / cur.total_assets
    roa_prev = prev.net_income / prev.total_assets
    # Debt absent from the filing is treated as zero, unless the whole line is missing.
    ltd_cur = np.nan_to_num(cur.long_term_debt) if not np.isnan(cur.total_debt) else np.nan
    ltd_prev = np.nan_to_num(prev.long_term_debt) if not np.isnan(prev.total_debt) else np.nan

    return PiotroskiResult([
        Signal("Net income is positive", _test(roa_cur, 0, gt)),
        Signal("Operating cash flow is positive", _test(cur.operating_cash_flow, 0, gt)),
        Signal("Return on assets improved", _test(roa_cur, roa_prev, gt)),
        Signal("Cash flow exceeds net income", _test(cur.operating_cash_flow, cur.net_income, gt)),
        Signal("Long-term debt ratio fell", _test(ltd_cur / cur.total_assets, ltd_prev / prev.total_assets, lt)),
        Signal("Current ratio improved",
               _test(cur.current_assets / cur.current_liabilities, prev.current_assets / prev.current_liabilities, gt)),
        # 1% tolerance: rounding and dilution from employee awards are not an equity raise.
        Signal("No meaningful share issuance", _test(cur.diluted_shares, prev.diluted_shares * 1.01, le)),
        Signal("Gross margin improved", _test(cur.gross_profit / cur.revenue, prev.gross_profit / prev.revenue, gt)),
        Signal("Asset turnover improved", _test(cur.revenue / cur.total_assets, prev.revenue / prev.total_assets, gt)),
    ])


def altman_z_double_prime(fin: pd.DataFrame) -> Optional[AltmanResult]:
    """Altman Z'' (1995): book-value model valid for non-manufacturers and emerging markets.

    The original Z-Score targets US manufacturers and needs market cap. Z'' is used
    so that one consistent model applies across both US and Indian companies.
    """
    row = fin.iloc[-1]
    x1 = (row.current_assets - row.current_liabilities) / row.total_assets
    x2 = row.retained_earnings / row.total_assets
    x3 = row.operating_income / row.total_assets
    x4 = row.equity / row.total_liabilities if row.total_liabilities else np.nan
    if any(np.isnan(x) for x in (x1, x2, x3, x4)):
        return None
    z = 6.56 * x1 + 3.26 * x2 + 6.72 * x3 + 1.05 * x4
    zone = "Safe" if z > 2.6 else "Distress" if z < 1.1 else "Grey"
    return AltmanResult(float(z), zone, float(x1), float(x2), float(x3), float(x4))

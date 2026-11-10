"""Year-by-year financial ratios computed from the canonical table."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

DEFAULT_TAX_FOR_ROIC = 0.25


def _div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def _average(series: pd.Series) -> pd.Series:
    """Two-point average balance; the first year falls back to its closing balance."""
    return series.rolling(2, min_periods=1).mean()


def compute_ratios(fin: pd.DataFrame) -> pd.DataFrame:
    ratios = pd.DataFrame(index=fin.index)
    ebit = fin["operating_income"]
    ebitda = ebit + fin["depreciation_amortization"]
    fcf = fin["operating_cash_flow"] - fin["capex"]
    tax_rate = _div(fin["income_tax"], fin["pretax_income"]).clip(0, 0.40).fillna(DEFAULT_TAX_FOR_ROIC)
    invested_capital = (
        fin["total_debt"].fillna(0) + fin["equity"] - fin["cash"].fillna(0) - fin["short_term_investments"].fillna(0)
    )

    ratios["revenue_growth"] = fin["revenue"].pct_change(fill_method=None)
    ratios["gross_margin"] = _div(fin["gross_profit"], fin["revenue"])
    ratios["ebit_margin"] = _div(ebit, fin["revenue"])
    ratios["ebitda_margin"] = _div(ebitda, fin["revenue"])
    ratios["net_margin"] = _div(fin["net_income"], fin["revenue"])
    ratios["fcf_margin"] = _div(fcf, fin["revenue"])
    ratios["roe"] = _div(fin["net_income"], _average(fin["equity"]))
    ratios["roa"] = _div(fin["net_income"], _average(fin["total_assets"]))
    ratios["roic"] = _div(ebit * (1 - tax_rate), _average(invested_capital).where(lambda s: s > 0))
    ratios["asset_turnover"] = _div(fin["revenue"], _average(fin["total_assets"]))
    ratios["current_ratio"] = _div(fin["current_assets"], fin["current_liabilities"])
    ratios["debt_to_equity"] = _div(fin["total_debt"], fin["equity"])
    ratios["net_debt_to_ebitda"] = _div(
        fin["total_debt"] - fin["cash"].fillna(0) - fin["short_term_investments"].fillna(0), ebitda
    )
    ratios["interest_coverage"] = _div(ebit, fin["interest_expense"])
    ratios["cash_conversion"] = _div(fin["operating_cash_flow"], fin["net_income"])
    ratios["free_cash_flow"] = fcf
    ratios["ebitda"] = ebitda
    return ratios


def cagr(series: pd.Series, years: int) -> Optional[float]:
    """Compound annual growth over the last `years` periods; None if undefined."""
    clean = series.dropna()
    if len(clean) <= years:
        return None
    start, end = clean.iloc[-years - 1], clean.iloc[-1]
    if start <= 0 or end <= 0:
        return None
    return float((end / start) ** (1 / years) - 1)

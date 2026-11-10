"""End-to-end analysis of one company, shared by the dashboard, CLI and screener."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from fivs.analysis.insights import Insight, generate_insights
from fivs.analysis.ratios import compute_ratios
from fivs.analysis.scores import AltmanResult, PiotroskiResult, altman_z_double_prime, piotroski
from fivs.data.quality import QualityReport, check_quality
from fivs.data.repository import CompanyRepository
from fivs.models import Company
from fivs.peers import default_peers
from fivs.valuation.engine import ValuationResult, value_company


@dataclass
class Analysis:
    company: Company
    quality: QualityReport
    ratios: pd.DataFrame
    piotroski: Optional[PiotroskiResult]
    altman: Optional[AltmanResult]
    insights: List[Insight]
    valuation: ValuationResult


def analyze_company(
    company: Company,
    repository: CompanyRepository,
    peer_tickers: Optional[Sequence[str]] = None,
    risk_free: Optional[float] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Analysis:
    ratios = compute_ratios(company.financials)
    financial = company.is_balance_sheet_financial
    tickers = default_peers(company) if peer_tickers is None else list(peer_tickers)
    rate = repository.risk_free(company.market) if risk_free is None else risk_free
    return Analysis(
        company=company,
        quality=check_quality(company),
        ratios=ratios,
        # Both scores were designed around industrial balance sheets.
        piotroski=None if financial else piotroski(company.financials),
        altman=None if financial else altman_z_double_prime(company.financials),
        insights=generate_insights(company, ratios),
        valuation=value_company(company, repository.peer_multiples(tickers), rate, overrides),
    )


def screen_row(analysis: Analysis) -> Dict[str, Any]:
    """One flat summary row for ranking companies against each other."""
    c, v, latest = analysis.company, analysis.valuation, analysis.ratios.iloc[-1]
    return {
        "Ticker": c.ticker,
        "Company": c.name,
        "Sector": c.sector,
        "Price": c.price,
        "Currency": c.currency,
        "Fair value": v.fair_value,
        "Upside": v.upside,
        "Verdict": v.verdict,
        "F-Score": f"{analysis.piotroski.score}/{analysis.piotroski.tested}" if analysis.piotroski else "n/a",
        "Altman Z''": analysis.altman.z if analysis.altman else None,
        "ROE": latest["roe"],
        "Op. margin": latest["ebit_margin"],
        "Data quality": analysis.quality.score,
    }

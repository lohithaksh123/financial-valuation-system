"""Comparable-company valuation using peer trading multiples."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from fivs.models import Company, PeerMultiples

MIN_PEERS = 3
# Multiples above these levels reflect depressed earnings, not valuation.
OUTLIER_CAPS = {"pe": 100.0, "ev_ebitda": 60.0, "price_sales": 40.0, "price_book": 30.0}


@dataclass
class CompRow:
    multiple: str
    peers_used: int
    low_multiple: float
    median_multiple: float
    high_multiple: float
    low_value: float
    median_value: float
    high_value: float
    primary: bool  # counted toward the headline comps value


@dataclass
class CompsResult:
    rows: List[CompRow]
    peer_table: pd.DataFrame
    value_per_share: Optional[float]


def _clean(peers: Sequence[PeerMultiples], attribute: str) -> np.ndarray:
    values = [getattr(peer, attribute) for peer in peers]
    cap = OUTLIER_CAPS[attribute]
    return np.array([v for v in values if v is not None and 0 < v <= cap], dtype=float)


def value_by_comps(company: Company, peers: Sequence[PeerMultiples]) -> CompsResult:
    latest, ebitda = company.latest, company.latest["operating_income"] + company.latest["depreciation_amortization"]
    shares = company.shares_outstanding
    bridge = company.net_debt + company.minority_interest
    financial = company.is_balance_sheet_financial

    # (label, peer attribute, target metric, is enterprise value multiple, primary)
    if financial:
        specs = [
            ("P/E", "pe", latest["net_income"], False, True),
            ("P/B", "price_book", latest["equity"], False, True),
        ]
    else:
        specs = [
            ("P/E", "pe", latest["net_income"], False, True),
            ("EV/EBITDA", "ev_ebitda", ebitda, True, True),
            ("P/S", "price_sales", latest["revenue"], False, False),
        ]

    rows: List[CompRow] = []
    for label, attribute, metric, is_ev, primary in specs:
        multiples = _clean(peers, attribute)
        if len(multiples) < MIN_PEERS or np.isnan(metric) or metric <= 0:
            continue
        q1, median, q3 = np.percentile(multiples, [25, 50, 75])
        implied = [(m * metric - (bridge if is_ev else 0.0)) / shares for m in (q1, median, q3)]
        rows.append(CompRow(label, len(multiples), float(q1), float(median), float(q3), *map(float, implied), primary))

    primary_values = [row.median_value for row in rows if row.primary and row.median_value > 0]
    peer_table = pd.DataFrame(
        [
            {"Ticker": p.ticker, "Company": p.name, "P/E": p.pe, "EV/EBITDA": p.ev_ebitda, "P/S": p.price_sales, "P/B": p.price_book}
            for p in peers
        ]
    )
    return CompsResult(rows, peer_table, float(np.mean(primary_values)) if primary_values else None)

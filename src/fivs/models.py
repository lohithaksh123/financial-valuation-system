"""The canonical company record every data source is normalized into."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# One row per fiscal year. Every provider produces exactly these columns, so the
# analysis and valuation code never needs to know where the data came from.
CANONICAL_FIELDS = [
    "revenue", "cogs", "gross_profit", "operating_income", "pretax_income", "income_tax",
    "net_income", "interest_expense", "depreciation_amortization",
    "total_assets", "current_assets", "total_liabilities", "current_liabilities", "equity",
    "minority_interest", "cash", "short_term_investments", "total_debt", "long_term_debt",
    "retained_earnings", "receivables", "inventory", "accounts_payable",
    "operating_cash_flow", "capex", "dividends_paid", "diluted_shares",
]

# Banks and insurers fund themselves with deposits and float, so free cash flow
# and enterprise value are not meaningful. Payment networks and exchanges are
# ordinary fee businesses and stay in the DCF path.
_NON_BALANCE_SHEET_FINANCIALS = {"credit services", "financial data & stock exchanges"}


class DataError(RuntimeError):
    """A company's data could not be retrieved or cannot be used."""


class SymbolNotFound(DataError):
    """No listed company exists under this symbol."""


def normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Force canonical columns, a sorted datetime index and float values."""
    frame = frame.reindex(columns=CANONICAL_FIELDS).astype(float)
    frame.index = pd.to_datetime(frame.index)
    frame.index.name = "fiscal_year_end"
    return frame.sort_index()


@dataclass
class Company:
    ticker: str
    name: str
    market: str
    currency: str
    sector: str
    industry: str
    price: Optional[float]  # None for unlisted companies entered by the user
    shares_outstanding: float
    market_cap: float
    financials: pd.DataFrame
    source: str
    beta: Optional[float] = None
    high_52w: Optional[float] = None
    low_52w: Optional[float] = None
    warnings: List[str] = field(default_factory=list)

    @property
    def latest(self) -> pd.Series:
        return self.financials.iloc[-1]

    @property
    def is_balance_sheet_financial(self) -> bool:
        return (
            self.sector.strip().lower() == "financial services"
            and self.industry.strip().lower() not in _NON_BALANCE_SHEET_FINANCIALS
        )

    @property
    def net_debt(self) -> float:
        """Debt less cash and short-term investments. Negative means net cash."""
        row = self.latest
        return float(
            np.nan_to_num(row["total_debt"])
            - np.nan_to_num(row["cash"])
            - np.nan_to_num(row["short_term_investments"])
        )

    @property
    def minority_interest(self) -> float:
        return float(np.nan_to_num(self.latest["minority_interest"]))

    def to_dict(self) -> Dict[str, Any]:
        table = self.financials.copy()
        table.index = table.index.strftime("%Y-%m-%d")
        rows = {
            year: {k: (None if pd.isna(v) else float(v)) for k, v in row.items()}
            for year, row in table.to_dict("index").items()
        }
        data = {k: v for k, v in self.__dict__.items() if k != "financials"}
        data["financials"] = rows
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Company":
        data = dict(data)
        frame = pd.DataFrame.from_dict(data.pop("financials"), orient="index")
        return cls(financials=normalize_frame(frame), **data)


@dataclass
class PeerMultiples:
    ticker: str
    name: str
    pe: Optional[float] = None
    ev_ebitda: Optional[float] = None
    price_sales: Optional[float] = None
    price_book: Optional[float] = None

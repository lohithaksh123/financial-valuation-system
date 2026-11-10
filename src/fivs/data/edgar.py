"""US fundamentals from SEC EDGAR's XBRL "company facts" API (audited 10-K data).

Raw company facts are messy in three ways this module deals with:

* The same number appears in several filings. For each period the most recently
  filed value wins, so restatements replace the original figures.
* A "fiscal year" fact tag can still carry quarterly or multi-year durations, so
  flow items are kept only when the period is roughly one year long.
* Companies change tags over time, so each field has a priority list of tags and
  the series are merged period by period.
"""
from __future__ import annotations

from datetime import date
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from fivs.config import SEC_USER_AGENT
from fivs.models import DataError, normalize_frame

TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

ANNUAL_FORMS = {"10-K", "10-K/A", "10-KT"}
MIN_YEAR_DAYS, MAX_YEAR_DAYS = 340, 380  # allows 52/53-week fiscal years

# Canonical field -> us-gaap tags in priority order.
DURATION_TAGS = {
    "revenue": ["RevenuesNetOfInterestExpense", "Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet"],
    "cogs": ["CostOfGoodsAndServicesSold", "CostOfRevenue", "CostOfGoodsSold", "CostOfServices"],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "pretax_income": ["IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
                      "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments"],
    "income_tax": ["IncomeTaxExpenseBenefit"],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "interest_expense": ["InterestExpense", "InterestExpenseDebt", "InterestExpenseNonoperating"],
    "depreciation_amortization": ["DepreciationDepletionAndAmortization", "DepreciationAmortizationAndAccretionNet",
                                  "DepreciationAndAmortization", "DepreciationAmortizationAndOther", "Depreciation"],
    "operating_cash_flow": ["NetCashProvidedByUsedInOperatingActivities",
                            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"],
    "dividends_paid": ["PaymentsOfDividends", "PaymentsOfDividendsCommonStock"],
    "diluted_shares": ["WeightedAverageNumberOfDilutedSharesOutstanding"],
}
INSTANT_TAGS = {
    "total_assets": ["Assets"],
    "current_assets": ["AssetsCurrent"],
    "total_liabilities": ["Liabilities"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "minority_interest": ["MinorityInterest"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
    "short_term_investments": ["MarketableSecuritiesCurrent", "ShortTermInvestments", "AvailableForSaleSecuritiesCurrent"],
    "retained_earnings": ["RetainedEarningsAccumulatedDeficit"],
    "receivables": ["AccountsReceivableNetCurrent", "ReceivablesNetCurrent"],
    "inventory": ["InventoryNet"],
    "accounts_payable": ["AccountsPayableCurrent", "AccountsPayableAndAccruedLiabilitiesCurrent"],
    # Debt components, combined by _total_debt and then dropped.
    "_debt_current_all": ["DebtCurrent"],
    "_ltd_current": ["LongTermDebtCurrent", "LongTermDebtAndCapitalLeaseObligationsCurrent"],
    "_short_term": ["ShortTermBorrowings", "CommercialPaper", "OtherShortTermBorrowings"],
    "long_term_debt": ["LongTermDebtNoncurrent", "LongTermDebtAndCapitalLeaseObligations"],
    "_ltd_all": ["LongTermDebt"],
}


def _days(fact: dict) -> Optional[int]:
    if "start" not in fact:
        return None
    return (date.fromisoformat(fact["end"]) - date.fromisoformat(fact["start"])).days


def _annual_facts(facts: Sequence[dict], is_duration: bool) -> Dict[str, float]:
    """Period end -> value for annual filings, keeping the latest-filed version."""
    best: Dict[str, Tuple[str, float]] = {}
    for fact in facts:
        if fact.get("form") not in ANNUAL_FORMS:
            continue
        if is_duration:
            days = _days(fact)
            if days is None or not MIN_YEAR_DAYS <= days <= MAX_YEAR_DAYS:
                continue
        elif "start" in fact:
            continue
        end, filed = fact["end"], fact.get("filed", "")
        if end not in best or filed >= best[end][0]:
            best[end] = (filed, float(fact["val"]))
    return {end: value for end, (_, value) in best.items()}


def _merge_tags(gaap: dict, tags: List[str], is_duration: bool) -> Dict[str, float]:
    merged: Dict[str, float] = {}
    for tag in tags:
        units = gaap.get(tag, {}).get("units", {})
        facts = units.get("USD") or units.get("shares") or []
        for end, value in _annual_facts(facts, is_duration).items():
            merged.setdefault(end, value)
    return merged


def _total_debt(frame: pd.DataFrame) -> pd.Series:
    """Build total debt without double-counting overlapping XBRL tags."""
    current = frame["_debt_current_all"].where(
        frame["_debt_current_all"].notna(),
        frame["_ltd_current"].fillna(0) + frame["_short_term"].fillna(0),
    )
    noncurrent = frame["long_term_debt"].where(
        frame["long_term_debt"].notna(),
        frame["_ltd_all"] - frame["_ltd_current"].fillna(0),
    )
    total = current.fillna(0) + noncurrent.fillna(0)
    known = frame[["_debt_current_all", "_ltd_current", "_short_term", "long_term_debt", "_ltd_all"]].notna().any(axis=1)
    return total.where(known)


def parse_company_facts(payload: dict, max_years: int = 10) -> pd.DataFrame:
    """Turn a raw company-facts document into the canonical annual table."""
    gaap = payload.get("facts", {}).get("us-gaap")
    if not gaap:
        raise DataError("No US-GAAP facts in this filing (foreign filers report under IFRS on Form 20-F).")

    columns = {name: _merge_tags(gaap, tags, True) for name, tags in DURATION_TAGS.items()}
    columns.update({name: _merge_tags(gaap, tags, False) for name, tags in INSTANT_TAGS.items()})

    # Duration net income defines the true fiscal year ends; instant facts on any
    # other date (for example opening balances) fall away in this alignment.
    fiscal_ends = sorted(columns["net_income"])[-max_years:]
    if not fiscal_ends:
        raise DataError("No annual net income found in SEC filings.")
    frame = pd.DataFrame({name: pd.Series(values, dtype=float) for name, values in columns.items()}).reindex(fiscal_ends)

    frame["total_debt"] = _total_debt(frame)
    frame["gross_profit"] = frame["gross_profit"].fillna(frame["revenue"] - frame["cogs"])
    # Some filers (Walmart, for one) never report a total-liabilities line.
    frame["total_liabilities"] = frame["total_liabilities"].fillna(
        frame["total_assets"] - frame["equity"] - frame["minority_interest"].fillna(0)
    )
    # Pretax + interest approximates EBIT, but not for banks, where interest is an
    # operating cost. Banks have no classified balance sheet, so that identifies them.
    has_classified_balance_sheet = frame["current_assets"].notna()
    frame["operating_income"] = frame["operating_income"].where(
        frame["operating_income"].notna() | ~has_classified_balance_sheet,
        frame["pretax_income"] + frame["interest_expense"].fillna(0),
    )
    return normalize_frame(frame)


class EdgarClient:
    def __init__(self, session: Optional[requests.Session] = None):
        self.session = session or requests.Session()
        self.session.headers["User-Agent"] = SEC_USER_AGENT
        retry = Retry(total=3, backoff_factor=0.6, status_forcelist=[429, 500, 502, 503, 504])
        self.session.mount("https://", HTTPAdapter(max_retries=retry))
        self._ciks: Dict[str, int] = {}

    def _get_json(self, url: str) -> dict:
        response = self.session.get(url, timeout=30)
        if response.status_code == 404:
            raise DataError(f"SEC has no data at {url}")
        response.raise_for_status()
        return response.json()

    def cik_for(self, ticker: str) -> int:
        if not self._ciks:
            rows = self._get_json(TICKER_MAP_URL).values()
            self._ciks = {row["ticker"].upper(): int(row["cik_str"]) for row in rows}
        try:
            return self._ciks[ticker.upper()]
        except KeyError:
            raise DataError(f"{ticker} is not in the SEC ticker list.") from None

    def annual_financials(self, ticker: str) -> pd.DataFrame:
        return parse_company_facts(self._get_json(FACTS_URL.format(cik=self.cik_for(ticker))))

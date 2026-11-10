"""Single entry point for company data: picks the source, normalizes and caches."""
from __future__ import annotations

from dataclasses import asdict, replace
from typing import Iterable, List, Optional

import pandas as pd

from fivs.config import MARKETS, market_for_ticker
from fivs.data.cache import Cache
from fivs.data.directory import COLUMNS as DIRECTORY_COLUMNS
from fivs.data.directory import CompanyDirectory, DirectoryClient
from fivs.data.edgar import EdgarClient
from fivs.data.yahoo import YahooClient
from fivs.models import Company, DataError, PeerMultiples, SymbolNotFound


def candidate_symbols(query: str, market: Optional[str] = None) -> List[str]:
    """Symbols to try for what the user typed, most likely first.

    The selected market only sets the order, so a list can mix US and Indian
    companies. A query with spaces is a company name and has no direct symbol.
    """
    symbol = query.strip().upper()
    if not symbol:
        raise DataError("Enter a company name or ticker.")
    if " " in symbol:
        return []
    if symbol.endswith((".NS", ".BO")):
        return [symbol]
    if "." in symbol:
        return [symbol.replace(".", "-")]  # share classes: BF.B is BF-B on Yahoo
    return [f"{symbol}.NS", symbol] if market == "IN" else [symbol, f"{symbol}.NS"]


def convert_currency(frame: pd.DataFrame, rate: float) -> pd.DataFrame:
    """Convert every money column at one rate, which leaves ratios and growth unchanged."""
    converted = frame.copy()
    money_columns = [column for column in converted.columns if column != "diluted_shares"]
    converted[money_columns] = converted[money_columns] * rate
    return converted


def convert_company(company: Company, rate: float, target_currency: str) -> Company:
    """Restate a company in another currency at one rate.

    Ratios, growth and multiples are unaffected, and every per-share value
    scales by exactly the rate because the valuation models are linear in currency.
    """
    def scaled(value: Optional[float]) -> Optional[float]:
        return None if value is None else value * rate

    return replace(
        company,
        currency=target_currency,
        price=scaled(company.price),
        market_cap=company.market_cap * rate,
        high_52w=scaled(company.high_52w),
        low_52w=scaled(company.low_52w),
        financials=convert_currency(company.financials, rate),
    )


class CompanyRepository:
    def __init__(self, cache: Optional[Cache] = None, edgar=None, yahoo=None, directory_client=None):
        self.cache = cache or Cache()
        self.edgar = edgar or EdgarClient()
        self.yahoo = yahoo or YahooClient()
        self.directory_client = directory_client or DirectoryClient()

    def directory(self) -> CompanyDirectory:
        """All listed companies. If one market's list fails, the other is still returned."""
        frames, problems = [], []
        for name, fetch in (("US", self.directory_client.us), ("India", self.directory_client.india)):
            key = f"directory:{name}"
            stored = self.cache.get(key)
            if stored:
                frames.append(pd.DataFrame(stored, columns=DIRECTORY_COLUMNS))
                continue
            try:
                frame = fetch()
            except DataError as error:
                problems.append(f"{name} list unavailable. {error}")
                continue
            self.cache.set(key, frame[DIRECTORY_COLUMNS].values.tolist())
            frames.append(frame)
        table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=DIRECTORY_COLUMNS)
        return CompanyDirectory(table.sort_values("Company", key=lambda names: names.str.lower(), ignore_index=True), problems)

    def find(self, query: str, market: Optional[str] = None) -> Company:
        """Load a company from a ticker or a company name, in either market."""
        for symbol in candidate_symbols(query, market):
            try:
                return self.get(symbol)
            except SymbolNotFound:
                continue
        try:
            matches = self.yahoo.search(query.split(".")[0].strip(), market)
        except Exception:
            matches = []
        for symbol in matches[:3]:
            try:
                return self.get(symbol)
            except SymbolNotFound:
                continue
        raise SymbolNotFound(
            f'Could not find "{query.strip()}". Try the full company name, or check for a recent name change. '
            "Only companies listed in the US or India are supported."
        )

    def get(self, ticker: str, refresh: bool = False) -> Company:
        key = f"company:{ticker}"
        if not refresh:
            stored = self.cache.get(key)
            if stored:
                return Company.from_dict(stored)
        company = self._fetch(ticker)
        self.cache.set(key, company.to_dict())
        return company

    def _fetch(self, ticker: str) -> Company:
        market = market_for_ticker(ticker)
        snapshot = self.yahoo.market_snapshot(ticker)
        statement_currency = snapshot.pop("financial_currency")

        warnings: List[str] = []
        source = "Yahoo Finance"
        if market == "US":
            try:
                frame = self.edgar.annual_financials(ticker)
                source = "SEC EDGAR (audited 10-K) + Yahoo Finance prices"
                statement_currency = "USD"
            except DataError as error:
                warnings.append(f"SEC data unavailable ({error}) Using Yahoo Finance fundamentals.")
                frame = self.yahoo.annual_financials(ticker)
        else:
            frame = self.yahoo.annual_financials(ticker)

        trading_currency = snapshot["currency"]
        if statement_currency != trading_currency:
            rate = self.yahoo.fx_rate(statement_currency, trading_currency)
            if rate is None:
                raise DataError(f"No {statement_currency}/{trading_currency} exchange rate available for {ticker}.")
            frame = convert_currency(frame, rate)
            warnings.append(
                f"Statements are reported in {statement_currency} and were converted to {trading_currency} "
                f"at today's rate of {rate:,.2f}."
            )
        return Company(ticker=ticker, market=market, financials=frame, source=source, warnings=warnings, **snapshot)

    def peer_multiples(self, tickers: Iterable[str]) -> List[PeerMultiples]:
        """Fetch peers one by one; a single bad symbol must not sink the comparison."""
        peers = []
        for ticker in tickers:
            key = f"peer:{ticker}"
            stored = self.cache.get(key)
            if stored:
                peers.append(PeerMultiples(**stored))
                continue
            try:
                peer = self.yahoo.peer_multiples(ticker)
            except Exception:
                continue
            self.cache.set(key, asdict(peer))
            peers.append(peer)
        return peers

    def fx_rate(self, source: str, target: str) -> Optional[float]:
        """Today's exchange rate, cached; None if it cannot be retrieved."""
        if source == target:
            return 1.0
        key = f"fx:{source}{target}"
        stored = self.cache.get(key)
        if stored is not None:
            return stored
        try:
            rate = self.yahoo.fx_rate(source, target)
        except Exception:
            return None
        if rate:
            self.cache.set(key, rate)
        return rate

    def risk_free(self, market: str) -> float:
        """Live US 10-year yield when available; otherwise the market default."""
        default = MARKETS[market].risk_free
        if market != "US":
            return default
        stored = self.cache.get("risk_free:US")
        if stored is not None:
            return stored
        try:
            rate = self.yahoo.us_risk_free()
        except Exception:
            rate = None
        if rate is None:
            return default
        self.cache.set("risk_free:US", rate)
        return rate

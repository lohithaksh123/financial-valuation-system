"""Market data for all companies, and fundamentals for Indian listings, from Yahoo Finance."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
import yfinance as yf

from fivs.models import DataError, PeerMultiples, SymbolNotFound, normalize_frame

US_EXCHANGES = {"NMS", "NGM", "NCM", "NYQ", "ASE"}
INDIAN_EXCHANGES = {"NSI", "BSE"}

# Canonical field -> Yahoo line items in priority order.
INCOME_ITEMS = {
    "revenue": ["Total Revenue", "Operating Revenue"],
    "cogs": ["Cost Of Revenue", "Reconciled Cost Of Revenue"],
    "gross_profit": ["Gross Profit"],
    "operating_income": ["Operating Income", "Total Operating Income As Reported"],
    "pretax_income": ["Pretax Income"],
    "income_tax": ["Tax Provision"],
    "net_income": ["Net Income Common Stockholders", "Net Income"],
    "interest_expense": ["Interest Expense", "Interest Expense Non Operating"],
    "depreciation_amortization": ["Reconciled Depreciation"],
    "diluted_shares": ["Diluted Average Shares"],
}
BALANCE_ITEMS = {
    "total_assets": ["Total Assets"],
    "current_assets": ["Current Assets"],
    "total_liabilities": ["Total Liabilities Net Minority Interest"],
    "current_liabilities": ["Current Liabilities"],
    "equity": ["Stockholders Equity", "Common Stock Equity"],
    "minority_interest": ["Minority Interest"],
    "cash": ["Cash And Cash Equivalents"],
    "short_term_investments": ["Other Short Term Investments"],
    "total_debt": ["Total Debt"],
    "long_term_debt": ["Long Term Debt"],
    "retained_earnings": ["Retained Earnings"],
    "receivables": ["Accounts Receivable"],
    "inventory": ["Inventory"],
    "accounts_payable": ["Accounts Payable"],
}
CASH_FLOW_ITEMS = {
    "operating_cash_flow": ["Operating Cash Flow"],
    "capex": ["Capital Expenditure"],
    "dividends_paid": ["Cash Dividends Paid"],
    "depreciation_amortization": ["Depreciation And Amortization"],
}


def _extract(statement: Optional[pd.DataFrame], items: Dict[str, list]) -> pd.DataFrame:
    if statement is None or statement.empty:
        return pd.DataFrame()
    columns = {}
    for name, labels in items.items():
        series = pd.Series(index=statement.columns, dtype=float)
        for label in labels:
            if label in statement.index:
                series = series.fillna(pd.to_numeric(statement.loc[label], errors="coerce"))
        columns[name] = series
    return pd.DataFrame(columns)


def statements_to_frame(income: pd.DataFrame, balance: pd.DataFrame, cash_flow: pd.DataFrame) -> pd.DataFrame:
    """Combine Yahoo's three statements into the canonical annual table."""
    income_rows = _extract(income, INCOME_ITEMS)
    if income_rows.empty:
        raise DataError("Yahoo Finance returned no income statement.")
    balance_rows = _extract(balance, BALANCE_ITEMS)
    cash_rows = _extract(cash_flow, CASH_FLOW_ITEMS)

    frame = income_rows.join(balance_rows, how="left").join(
        cash_rows.drop(columns="depreciation_amortization", errors="ignore"), how="left"
    )
    if "depreciation_amortization" in cash_rows:
        frame["depreciation_amortization"] = frame["depreciation_amortization"].fillna(
            cash_rows["depreciation_amortization"]
        )
    # Yahoo reports outflows as negatives; the canonical table stores magnitudes.
    for outflow in ("capex", "dividends_paid"):
        if outflow in frame:
            frame[outflow] = frame[outflow].abs()
    # Yahoo often pads the oldest column with an almost empty year.
    frame = frame[frame[["revenue", "net_income", "total_assets"]].notna().all(axis=1)]
    if frame.empty:
        raise DataError("Yahoo Finance returned no complete fiscal year.")
    return normalize_frame(frame)


def _positive(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def rank_search_results(quotes: List[Dict[str, Any]], market: Optional[str] = None) -> List[str]:
    """Keep ordinary US and Indian shares, listing the selected market first."""
    shares = [
        quote for quote in quotes
        if quote.get("quoteType") == "EQUITY"
        and quote.get("exchange") in US_EXCHANGES | INDIAN_EXCHANGES
        and "-BL." not in quote.get("symbol", "")  # Indian block-deal windows
    ]
    preferred = {"IN": INDIAN_EXCHANGES, "US": US_EXCHANGES}.get(market, set())
    shares.sort(key=lambda quote: quote["exchange"] not in preferred)  # stable: keeps Yahoo's order
    return [quote["symbol"] for quote in shares]


class YahooClient:
    def market_snapshot(self, ticker: str) -> Dict[str, Any]:
        try:
            info = yf.Ticker(ticker).info or {}
        except Exception as error:
            raise DataError(f"Could not reach Yahoo Finance for {ticker}: {error}") from error
        price = _positive(info.get("currentPrice")) or _positive(info.get("regularMarketPrice"))
        if price is None:
            raise SymbolNotFound(f"No market price found for {ticker}. Check the symbol.")
        market_cap = _positive(info.get("marketCap"))
        # Market cap / price counts every share class. Yahoo's sharesOutstanding
        # covers only the quoted class, which understates dual-class companies.
        shares = market_cap / price if market_cap else _positive(info.get("sharesOutstanding"))
        if shares is None:
            raise DataError(f"No share count found for {ticker}.")
        return {
            "name": info.get("longName") or info.get("shortName") or ticker,
            "sector": info.get("sector") or "Unknown",
            "industry": info.get("industry") or "Unknown",
            "currency": info.get("currency") or "USD",
            "financial_currency": info.get("financialCurrency") or info.get("currency") or "USD",
            "price": price,
            "shares_outstanding": shares,
            "market_cap": market_cap or price * shares,
            "beta": _positive(info.get("beta")),
            "high_52w": _positive(info.get("fiftyTwoWeekHigh")),
            "low_52w": _positive(info.get("fiftyTwoWeekLow")),
        }

    def annual_financials(self, ticker: str) -> pd.DataFrame:
        stock = yf.Ticker(ticker)
        try:
            return statements_to_frame(stock.income_stmt, stock.balance_sheet, stock.cashflow)
        except DataError:
            raise
        except Exception as error:
            raise DataError(f"Could not read Yahoo statements for {ticker}: {error}") from error

    def peer_multiples(self, ticker: str) -> PeerMultiples:
        info = yf.Ticker(ticker).info or {}
        return PeerMultiples(
            ticker=ticker,
            name=info.get("shortName") or ticker,
            pe=_positive(info.get("trailingPE")),
            ev_ebitda=_positive(info.get("enterpriseToEbitda")),
            price_sales=_positive(info.get("priceToSalesTrailing12Months")),
            price_book=_positive(info.get("priceToBook")),
        )

    def search(self, query: str, market: Optional[str] = None) -> List[str]:
        """Symbols of US and Indian listed shares matching a company name, best first."""
        quotes = yf.Search(query, max_results=10, news_count=0).quotes
        return rank_search_results(quotes, market)

    def fx_rate(self, source: str, target: str) -> Optional[float]:
        """Latest price of one unit of `source` in `target`."""
        history = yf.Ticker(f"{source}{target}=X").history(period="5d")
        return float(history["Close"].dropna().iloc[-1]) if not history.empty else None

    def us_risk_free(self) -> Optional[float]:
        """Latest 10-year US Treasury yield (^TNX quotes the yield in percent)."""
        history = yf.Ticker("^TNX").history(period="5d")
        return float(history["Close"].dropna().iloc[-1]) / 100 if not history.empty else None

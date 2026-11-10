"""Lists of every listed company, from the SEC and the National Stock Exchange of India."""
from __future__ import annotations

import io
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from fivs.config import SEC_USER_AGENT
from fivs.models import DataError

SEC_LIST_URL = "https://www.sec.gov/files/company_tickers_exchange.json"
NSE_LIST_URL = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"

US_EXCHANGES = {"Nasdaq", "NYSE", "CBOE", "TXSE"}  # leaves out over-the-counter names
COLUMNS = ["Company", "Ticker", "Exchange", "Market"]


def parse_sec_list(payload: dict) -> pd.DataFrame:
    rows = pd.DataFrame(payload["data"], columns=payload["fields"])
    rows = rows[rows["exchange"].isin(US_EXCHANGES)].drop_duplicates("ticker")
    return pd.DataFrame({
        "Company": rows["name"].str.strip(), "Ticker": rows["ticker"].str.upper(), "Exchange": rows["exchange"], "Market": "US",
    }).reset_index(drop=True)


def parse_nse_list(content: bytes) -> pd.DataFrame:
    rows = pd.read_csv(io.BytesIO(content), dtype=str)
    rows.columns = [column.strip() for column in rows.columns]
    return pd.DataFrame({
        "Company": rows["NAME OF COMPANY"].str.strip(), "Ticker": rows["SYMBOL"].str.strip() + ".NS", "Exchange": "NSE", "Market": "India",
    }).reset_index(drop=True)


class DirectoryClient:
    def __init__(self, session: Optional[requests.Session] = None):
        self.session = session or requests.Session()
        self.session.headers["User-Agent"] = SEC_USER_AGENT
        self.session.mount("https://", HTTPAdapter(max_retries=Retry(total=2, backoff_factor=0.6, status_forcelist=[429, 500, 502, 503, 504])))

    def _get(self, url: str) -> requests.Response:
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
        except requests.RequestException as error:
            raise DataError(f"Could not download the company list: {error}") from error
        return response

    def us(self) -> pd.DataFrame:
        return parse_sec_list(self._get(SEC_LIST_URL).json())

    def india(self) -> pd.DataFrame:
        return parse_nse_list(self._get(NSE_LIST_URL).content)


@dataclass
class CompanyDirectory:
    table: pd.DataFrame
    problems: List[str]

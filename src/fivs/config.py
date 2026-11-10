"""Runtime settings and per-market valuation defaults."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

CACHE_PATH = Path(os.environ.get("FIVS_CACHE_DIR", Path.home() / ".cache" / "fivs")) / "cache.sqlite3"
EDITS_PATH = CACHE_PATH.parent / "edits.sqlite3"
CACHE_TTL_SECONDS =int(os.environ.get("FIVS_CACHE_TTL", 6 * 3600))

# The SEC asks automated clients to identify themselves. Set SEC_USER_AGENT to
# "Your Name your.email@example.com" when running this for real.
SEC_USER_AGENT = os.environ.get("SEC_USER_AGENT", "FIVS-valuation-tool research@example.com")


@dataclass(frozen=True)
class MarketDefaults:
    """Starting points for valuation. All are editable in the dashboard."""

    code: str
    currency_symbol: str
    risk_free: float  # 10-year government bond yield, in local currency
    equity_risk_premium: float
    terminal_growth: float  # nominal, local currency; must stay below risk_free
    statutory_tax: float


MARKETS = {
    "US": MarketDefaults("US", "$", 0.043, 0.045, 0.025, 0.21),
    "IN": MarketDefaults("IN", "₹", 0.068, 0.065, 0.050, 0.2517),
}


def market_for_ticker(ticker: str) -> str:
    """NSE (.NS) and BSE (.BO) symbols are Indian; everything else is US."""
    return "IN" if ticker.upper().endswith((".NS", ".BO")) else "US"

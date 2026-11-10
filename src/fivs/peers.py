"""Default peer groups. Users can replace these in the dashboard."""
from __future__ import annotations

from typing import List

from fivs.models import Company

# Checked in order: industry-specific groups first, then sector-wide groups.
_INDUSTRY_PEERS = {
    ("US", "credit services"): ["V", "MA", "AXP", "COF", "SYF"],
    ("US", "banks - diversified"): ["JPM", "BAC", "WFC", "C"],
    ("US", "banks - regional"): ["USB", "PNC", "TFC", "FITB", "KEY"],
    ("US", "software - infrastructure"): ["MSFT", "ORCL", "ADBE", "PANW", "SNPS"],
    ("US", "software - application"): ["CRM", "NOW", "INTU", "WDAY", "ADSK"],
    ("US", "semiconductors"): ["NVDA", "AMD", "AVGO", "TXN", "QCOM"],
    ("US", "consumer electronics"): ["AAPL", "SONY", "HPQ", "DELL", "MSFT"],
    ("US", "discount stores"): ["WMT", "COST", "TGT", "DG", "DLTR"],
    ("US", "internet content & information"): ["GOOGL", "META", "PINS", "SNAP", "RDDT"],
    ("IN", "banks - regional"): ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "AXISBANK.NS", "KOTAKBANK.NS"],
    ("IN", "banks - diversified"): ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "AXISBANK.NS", "KOTAKBANK.NS"],
    ("IN", "information technology services"): ["TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS"],
    ("IN", "auto manufacturers"): ["MARUTI.NS", "TMPV.NS", "M&M.NS", "BAJAJ-AUTO.NS", "EICHERMOT.NS"],
    ("IN", "drug manufacturers - specialty & generic"): ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "LUPIN.NS"],
}
_SECTOR_PEERS = {
    ("US", "technology"): ["AAPL", "MSFT", "ORCL", "CSCO", "IBM"],
    ("US", "communication services"): ["GOOGL", "META", "NFLX", "DIS", "VZ"],
    ("US", "consumer cyclical"): ["AMZN", "HD", "MCD", "NKE", "LOW"],
    ("US", "consumer defensive"): ["PG", "KO", "PEP", "WMT", "COST"],
    ("US", "healthcare"): ["JNJ", "PFE", "MRK", "ABBV", "UNH"],
    ("US", "financial services"): ["JPM", "BAC", "GS", "MS", "BLK"],
    ("US", "energy"): ["XOM", "CVX", "COP", "EOG", "SLB"],
    ("US", "industrials"): ["CAT", "HON", "UNP", "GE", "DE"],
    ("US", "basic materials"): ["LIN", "SHW", "APD", "ECL", "NUE"],
    ("US", "utilities"): ["NEE", "DUK", "SO", "AEP", "D"],
    ("US", "real estate"): ["PLD", "AMT", "EQIX", "SPG", "O"],
    ("IN", "technology"): ["TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS"],
    ("IN", "financial services"): ["BAJFINANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "SHRIRAMFIN.NS"],
    ("IN", "energy"): ["RELIANCE.NS", "ONGC.NS", "BPCL.NS", "IOC.NS", "COALINDIA.NS"],
    ("IN", "consumer defensive"): ["HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS", "DABUR.NS"],
    ("IN", "consumer cyclical"): ["MARUTI.NS", "M&M.NS", "TITAN.NS", "BAJAJ-AUTO.NS", "TRENT.NS"],
    ("IN", "healthcare"): ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "APOLLOHOSP.NS"],
    ("IN", "industrials"): ["LT.NS", "SIEMENS.NS", "ABB.NS", "BEL.NS", "ADANIPORTS.NS"],
    ("IN", "basic materials"): ["ULTRACEMCO.NS", "TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS", "ASIANPAINT.NS"],
    ("IN", "communication services"): ["BHARTIARTL.NS", "INDUSTOWER.NS", "ZEEL.NS", "SUNTV.NS", "TATACOMM.NS"],
    ("IN", "utilities"): ["NTPC.NS", "POWERGRID.NS", "TATAPOWER.NS", "ADANIPOWER.NS", "JSWENERGY.NS"],
}


def default_peers(company: Company) -> List[str]:
    market, industry, sector = company.market, company.industry.lower(), company.sector.lower()
    group = _INDUSTRY_PEERS.get((market, industry)) or _SECTOR_PEERS.get((market, sector)) or []
    return [ticker for ticker in group if ticker != company.ticker]

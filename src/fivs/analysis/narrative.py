"""Turns the analysis into plain language for readers who are not finance experts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import pandas as pd

from fivs.analysis.ratios import cagr

if TYPE_CHECKING:
    from fivs.pipeline import Analysis

STRONG, OKAY, WEAK, NO_DATA = "Strong", "Okay", "Weak", "No data"

GLOSSARY = {
    "Model value": "Our estimate of one share's worth: 60% from the company's own cash flows or book value, 40% from how similar companies are priced.",
    "DCF": "Discounted cash flow. Forecasts the cash a business will generate, then converts it to today's money.",
    "WACC": "The yearly return investors expect for funding this company. A higher WACC makes future cash worth less today.",
    "Terminal growth": "The rate the business is assumed to grow forever after the 10-year forecast. It should stay near long-run economic growth.",
    "Terminal value share": "How much of the DCF value comes from years after the forecast. A high share means the result leans heavily on long-term assumptions.",
    "Implied growth": "The starting revenue growth needed for the DCF to equal today's share price.",
    "Monte Carlo": "Runs the DCF 10,000 times with slightly different assumptions to show a range of outcomes rather than one number.",
    "P10 / P50 / P90": "P50 is the middle outcome. 10% of scenarios fall below P10 and 10% rise above P90.",
    "Residual income": "Values a bank or insurer as its book equity plus future profits above what shareholders require.",
    "ROE": "Return on equity: profit earned on each rupee or dollar of shareholders' money.",
    "ROIC": "Return on invested capital: profit earned on all the money invested in the business, from lenders and owners.",
    "EBIT": "Earnings before interest and tax, also called operating profit.",
    "Free cash flow": "Cash left after running the business and paying for equipment and buildings.",
    "F-Score": "Piotroski F-Score: nine yes-or-no checks on profitability, debt and efficiency. Higher is better.",
    "Altman Z''": "Altman Z''-Score: estimates financial distress risk. Above 2.6 is safe and below 1.1 signals distress.",
    "P/E": "Share price divided by earnings per share: what investors pay for one unit of annual profit.",
    "EV/EBITDA": "Total business value divided by operating profit before depreciation. It allows for differences in debt.",
    "P/B": "Share price divided by book value per share. Commonly used for banks.",
}


@dataclass
class Scorecard:
    name: str
    rating: str
    figure: str
    meaning: str


def currency_symbol(currency: str) -> str:
    return {"USD": "$", "INR": "₹"}.get(currency, f"{currency} ")


def compact_money(value: float, currency: str) -> str:
    """Human-scale amount: crore for rupees, millions or billions otherwise."""
    symbol = currency_symbol(currency)
    if currency == "INR":
        return f"{symbol}{value / 1e7:,.1f} crore"
    if abs(value) >= 1e12:
        return f"{symbol}{value / 1e12:,.2f} trillion"
    if abs(value) >= 1e9:
        return f"{symbol}{value / 1e9:,.1f} billion"
    return f"{symbol}{value / 1e6:,.1f} million"


def _rate(value: Optional[float], strong: float, okay: float) -> str:
    if value is None or pd.isna(value):
        return NO_DATA
    return STRONG if value >= strong else OKAY if value >= okay else WEAK


def _revenue_growth(a: "Analysis") -> Optional[float]:
    years = min(3, len(a.company.financials) - 1)
    return cagr(a.company.financials["revenue"], years) if years > 0 else None


def scorecards(a: "Analysis") -> List[Scorecard]:
    latest = a.ratios.iloc[-1]
    fin = a.company.financials.iloc[-1]
    growth = _revenue_growth(a)
    growth_card = Scorecard(
        "Growth", _rate(growth, 0.10, 0.03), "n/a" if growth is None else f"{growth:.1%} a year",
        "How fast revenue has been rising.",
    )
    if a.company.is_balance_sheet_financial:
        equity_share = fin["equity"] / fin["total_assets"]
        return [
            Scorecard("Profitability", _rate(latest["roe"], 0.12, 0.08), f"{latest['roe']:.1%} ROE", "Profit earned on shareholders' money."),
            growth_card,
            Scorecard("Capital cushion", _rate(equity_share, 0.08, 0.05), f"{equity_share:.1%} of assets", "Owners' money available to absorb losses."),
            Scorecard("Asset returns", _rate(latest["roa"], 0.01, 0.005), f"{latest['roa']:.2%} ROA", "Profit earned on everything the bank holds."),
        ]

    margin, leverage, conversion = latest["ebit_margin"], latest["net_debt_to_ebitda"], latest["cash_conversion"]
    if pd.isna(leverage):
        debt_rating, debt_figure = NO_DATA, "n/a"
    elif latest["ebitda"] <= 0:
        debt_rating, debt_figure = WEAK, "Operating losses"
    else:
        debt_rating = STRONG if leverage < 1 else OKAY if leverage < 3 else WEAK
        debt_figure = "More cash than debt" if leverage < 0 else f"{leverage:.1f}x EBITDA"
    return [
        Scorecard("Profitability", _rate(margin, 0.15, 0.05), "n/a" if pd.isna(margin) else f"{margin:.1%} margin", "Operating profit kept from each unit of sales."),
        growth_card,
        Scorecard("Debt safety", debt_rating, debt_figure, "How easily profits could repay net debt."),
        Scorecard("Cash quality", _rate(conversion, 1.0, 0.7), "n/a" if pd.isna(conversion) else f"{conversion:.0%} of profit", "How much reported profit arrives as cash."),
    ]


def plain_summary(a: "Analysis") -> List[str]:
    """Three or four sentences a non-specialist can read."""
    c, v = a.company, a.valuation
    symbol = currency_symbol(c.currency)
    cards = scorecards(a)
    sentences: List[str] = []

    known = [card for card in cards if card.rating != NO_DATA]
    strong = [card.name.lower() for card in known if card.rating == STRONG]
    weak = [card.name.lower() for card in known if card.rating == WEAK]
    if known:
        if len(strong) >= 3 and not weak:
            overall = "looks financially strong"
        elif len(weak) >= 2:
            overall = "shows real financial weaknesses"
        else:
            overall = "has a mixed financial picture"
        detail = []
        if strong:
            detail.append(f"it scores well on {_join(strong)}")
        if weak:
            detail.append(f"it is weak on {_join(weak)}")
        sentences.append(f"{c.name} {overall}" + (f": {', and '.join(detail)}." if detail else "."))

    if v.fair_value is None:
        sentences.append("There was not enough data to estimate a value.")
        return sentences

    if c.price is None:
        equity = v.fair_value * c.shares_outstanding
        sentences.append(
            f"The model values the whole company at about {compact_money(equity, c.currency)}, "
            f"or {symbol}{v.fair_value:,.2f} per share."
        )
    else:
        gap = v.upside
        if gap < -0.15:
            sentences.append(f"At {symbol}{c.price:,.2f}, the share price is well above the model's estimate of {symbol}{v.fair_value:,.2f}.")
        elif gap > 0.15:
            sentences.append(f"At {symbol}{c.price:,.2f}, the share price is well below the model's estimate of {symbol}{v.fair_value:,.2f}.")
        else:
            sentences.append(f"At {symbol}{c.price:,.2f}, the share price is close to the model's estimate of {symbol}{v.fair_value:,.2f}.")

    if v.implied_growth is not None and v.dcf is not None:
        sentences.append(
            f"To justify today's price, revenue would need to start growing about {v.implied_growth:.0%} a year; "
            f"the model assumes {v.dcf.assumptions.growth_initial:.0%}."
        )
    return sentences


def _join(items: List[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]

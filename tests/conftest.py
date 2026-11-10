import pandas as pd
import pytest

from fivs.models import Company, normalize_frame


def make_financials(years=4):
    """A simple, internally consistent industrial company growing 10% a year."""
    rows = {}
    for i in range(years):
        growth = 1.10 ** i
        revenue = 1000.0 * growth
        assets = 800.0 * growth
        equity = 500.0 * growth
        rows[f"{2021 + i}-12-31"] = {
            "revenue": revenue, "cogs": 0.6 * revenue, "gross_profit": 0.4 * revenue,
            "operating_income": 0.2 * revenue, "pretax_income": 0.19 * revenue, "income_tax": 0.19 * revenue * 0.25,
            "net_income": 0.19 * revenue * 0.75, "interest_expense": 0.01 * revenue,
            "depreciation_amortization": 0.05 * revenue, "total_assets": assets, "current_assets": 0.5 * assets,
            "total_liabilities": assets - equity, "current_liabilities": 0.2 * assets, "equity": equity,
            "minority_interest": 0.0, "cash": 100.0, "short_term_investments": 20.0, "total_debt": 200.0,
            "long_term_debt": 180.0, "retained_earnings": 0.5 * equity, "receivables": 0.1 * revenue,
            "inventory": 0.1 * revenue, "accounts_payable": 0.05 * revenue,
            "operating_cash_flow": 0.2 * revenue, "capex": 0.07 * revenue, "dividends_paid": 0.05 * revenue,
            "diluted_shares": 100.0,
        }
    return normalize_frame(pd.DataFrame.from_dict(rows, orient="index"))


def make_company(financials=None, sector="Industrials", industry="Machinery", market="US", beta=1.0):
    return Company(
        ticker="TEST", name="Test Co", market=market, currency="USD", sector=sector, industry=industry,
        price=20.0, shares_outstanding=100.0, market_cap=2000.0,
        financials=make_financials() if financials is None else financials,
        source="fixture", beta=beta, high_52w=25.0, low_52w=15.0,
    )


@pytest.fixture
def company():
    return make_company()


@pytest.fixture
def bank():
    fin = make_financials()
    fin["total_assets"] *= 10
    fin["total_liabilities"] = fin["total_assets"] - fin["equity"]
    fin[["operating_income", "current_assets", "current_liabilities", "gross_profit", "cogs"]] = float("nan")
    return make_company(fin, sector="Financial Services", industry="Banks - Regional")

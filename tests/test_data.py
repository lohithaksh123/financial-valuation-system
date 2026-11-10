import time
from datetime import datetime

import pandas as pd
import pytest

from conftest import make_company, make_financials
from fivs.data.cache import Cache
from fivs.data.edgar import parse_company_facts
from fivs.data.quality import check_quality
from fivs.data.repository import CompanyRepository, candidate_symbols
from fivs.data.yahoo import rank_search_results, statements_to_frame
from fivs.models import Company, DataError, SymbolNotFound


def flow(value, year, filed=None, form="10-K", start=None, end=None):
    return {
        "start": start or f"{year}-01-01", "end": end or f"{year}-12-31", "val": value,
        "form": form, "filed": filed or f"{year + 1}-02-01",
    }


def instant(value, year, filed=None):
    return {"end": f"{year}-12-31", "val": value, "form": "10-K", "filed": filed or f"{year + 1}-02-01"}


def facts(**tags):
    return {"facts": {"us-gaap": {tag: {"units": {"USD": rows}} for tag, rows in tags.items()}}}


def test_latest_filing_wins_for_restated_numbers():
    payload = facts(NetIncomeLoss=[flow(100, 2023, "2024-02-01"), flow(90, 2023, "2025-02-01")])
    assert parse_company_facts(payload).loc["2023-12-31", "net_income"] == 90


def test_quarterly_and_multi_year_durations_are_ignored():
    payload = facts(NetIncomeLoss=[
        flow(10, 2023, start="2023-10-01"),
        flow(300, 2023, start="2021-01-01"),
        flow(100, 2023),
    ])
    frame = parse_company_facts(payload)
    assert len(frame) == 1 and frame["net_income"].iloc[0] == 100


def test_fifty_three_week_year_is_annual():
    payload = facts(NetIncomeLoss=[flow(100, 2023, start="2022-12-25", end="2023-12-30")])
    assert parse_company_facts(payload)["net_income"].iloc[0] == 100


def test_quarterly_forms_are_ignored():
    payload = facts(NetIncomeLoss=[flow(100, 2023), flow(999, 2022, form="10-Q")])
    assert len(parse_company_facts(payload)) == 1


def test_tag_change_merges_across_years():
    payload = facts(
        NetIncomeLoss=[flow(1, 2022), flow(1, 2023)],
        Revenues=[flow(100, 2022)],
        RevenueFromContractWithCustomerExcludingAssessedTax=[flow(120, 2023)],
    )
    assert list(parse_company_facts(payload)["revenue"]) == [100, 120]


def test_debt_components_are_not_double_counted():
    payload = facts(
        NetIncomeLoss=[flow(1, 2023)],
        DebtCurrent=[instant(30, 2023)],
        LongTermDebtCurrent=[instant(20, 2023)],
        CommercialPaper=[instant(10, 2023)],
        LongTermDebtNoncurrent=[instant(100, 2023)],
    )
    assert parse_company_facts(payload)["total_debt"].iloc[0] == 130


def test_total_debt_stays_missing_when_nothing_is_reported():
    frame = parse_company_facts(facts(NetIncomeLoss=[flow(1, 2023)]))
    assert pd.isna(frame["total_debt"].iloc[0])


def test_missing_total_liabilities_is_derived():
    payload = facts(
        NetIncomeLoss=[flow(1, 2023)], Assets=[instant(100, 2023)],
        StockholdersEquity=[instant(30, 2023)], MinorityInterest=[instant(5, 2023)],
    )
    assert parse_company_facts(payload)["total_liabilities"].iloc[0] == 65


def test_bank_gets_no_synthetic_operating_income():
    payload = facts(
        NetIncomeLoss=[flow(1, 2023)],
        IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest=[flow(10, 2023)],
        InterestExpense=[flow(50, 2023)],
    )
    assert pd.isna(parse_company_facts(payload)["operating_income"].iloc[0])


def test_foreign_filer_raises_data_error():
    with pytest.raises(DataError):
        parse_company_facts({"facts": {"ifrs-full": {}}})


def test_yahoo_normalizes_capex_and_drops_padded_years():
    columns = pd.to_datetime(["2024-03-31", "2023-03-31", "2022-03-31"])
    income = pd.DataFrame({columns[0]: [200.0, 20.0], columns[1]: [100.0, 10.0], columns[2]: [None, None]},
                          index=["Total Revenue", "Net Income"])
    balance = pd.DataFrame({columns[0]: [500.0], columns[1]: [400.0], columns[2]: [None]}, index=["Total Assets"])
    cash = pd.DataFrame({columns[0]: [-30.0], columns[1]: [-20.0], columns[2]: [None]}, index=["Capital Expenditure"])
    frame = statements_to_frame(income, balance, cash)
    assert len(frame) == 2
    assert frame.index.is_monotonic_increasing
    assert frame["capex"].iloc[-1] == 30


def test_company_round_trips_through_json_dict(company):
    restored = Company.from_dict(company.to_dict())
    pd.testing.assert_frame_equal(restored.financials, company.financials, check_freq=False)
    assert restored.ticker == company.ticker


def test_cache_expires(tmp_path):
    cache = Cache(tmp_path / "c.sqlite3", ttl_seconds=1)
    cache.set("k", {"a": 1})
    assert cache.get("k") == {"a": 1}
    time.sleep(1.1)
    assert cache.get("k") is None


def test_quality_passes_clean_data(company):
    report = check_quality(company, today=datetime(2025, 3, 1))
    assert report.score == 100 and not report.issues


def test_quality_flags_unbalanced_short_and_stale_data():
    fin = make_financials(years=2)
    fin["equity"] *= 0.5
    report = check_quality(make_company(fin), today=datetime(2030, 1, 1))
    assert report.score < 60
    text = " ".join(report.issues)
    assert "history" in text and "differ" in text and "days old" in text


class FakeYahoo:
    def __init__(self, reporting="USD"):
        self.reporting = reporting

    def market_snapshot(self, ticker):
        return dict(name="X", sector="Technology", industry="Software", currency="USD", financial_currency=self.reporting,
                    price=10.0, shares_outstanding=100.0, market_cap=1000.0, beta=1.0, high_52w=12.0, low_52w=8.0)

    def annual_financials(self, ticker):
        return make_financials()

    def fx_rate(self, source, target):
        return 0.5


class FailingEdgar:
    def annual_financials(self, ticker):
        raise DataError("no filing.")


class WorkingEdgar:
    calls = 0

    def annual_financials(self, ticker):
        WorkingEdgar.calls += 1
        return make_financials()


def test_repository_falls_back_to_yahoo_and_warns(tmp_path):
    repo = CompanyRepository(Cache(tmp_path / "c.sqlite3"), FailingEdgar(), FakeYahoo())
    company = repo.get("ABC")
    assert company.source == "Yahoo Finance" and company.warnings


def test_repository_serves_second_request_from_cache(tmp_path):
    WorkingEdgar.calls = 0
    repo = CompanyRepository(Cache(tmp_path / "c.sqlite3"), WorkingEdgar(), FakeYahoo())
    repo.get("ABC")
    repo.get("ABC")
    assert WorkingEdgar.calls == 1


def test_repository_converts_statements_to_trading_currency(tmp_path):
    repo = CompanyRepository(Cache(tmp_path / "c.sqlite3"), FailingEdgar(), FakeYahoo(reporting="INR"))
    company = repo.get("ADR")
    original = make_financials()
    assert company.latest["revenue"] == pytest.approx(original["revenue"].iloc[-1] * 0.5)
    assert company.latest["diluted_shares"] == original["diluted_shares"].iloc[-1]
    assert any("converted" in warning for warning in company.warnings)


def test_edgar_statements_are_treated_as_usd(tmp_path):
    repo = CompanyRepository(Cache(tmp_path / "c.sqlite3"), WorkingEdgar(), FakeYahoo(reporting="INR"))
    assert repo.get("ABC").latest["revenue"] == pytest.approx(make_financials()["revenue"].iloc[-1])


def test_repository_fails_clearly_without_exchange_rate(tmp_path):
    yahoo = FakeYahoo(reporting="INR")
    yahoo.fx_rate = lambda source, target: None
    repo = CompanyRepository(Cache(tmp_path / "c.sqlite3"), FailingEdgar(), yahoo)
    with pytest.raises(DataError, match="exchange rate"):
        repo.get("ADR")


def test_candidate_symbols():
    assert candidate_symbols(" tcs ", "IN") == ["TCS.NS", "TCS"]
    assert candidate_symbols("nvda", "US") == ["NVDA", "NVDA.NS"]
    assert candidate_symbols("reliance.bo", "US") == ["RELIANCE.BO"]
    assert candidate_symbols("bf.b", "US") == ["BF-B"]
    assert candidate_symbols("tata motors", "IN") == []
    with pytest.raises(DataError):
        candidate_symbols("  ")


class ListedOnly(FakeYahoo):
    """A Yahoo stand-in where only some symbols exist."""

    def __init__(self, listed, matches=()):
        super().__init__()
        self.listed, self.matches, self.searched = listed, list(matches), []

    def market_snapshot(self, ticker):
        if ticker not in self.listed:
            raise SymbolNotFound(ticker)
        return super().market_snapshot(ticker)

    def search(self, query, market=None):
        self.searched.append(query)
        return self.matches


def repo_with(tmp_path, yahoo):
    return CompanyRepository(Cache(tmp_path / "c.sqlite3"), FailingEdgar(), yahoo)


def test_us_ticker_is_found_while_india_is_selected(tmp_path):
    assert repo_with(tmp_path, ListedOnly({"NVDA"})).find("NVDA", "IN").ticker == "NVDA"


def test_indian_ticker_is_found_while_us_is_selected(tmp_path):
    assert repo_with(tmp_path, ListedOnly({"TCS.NS"})).find("TCS", "US").ticker == "TCS.NS"


def test_company_name_uses_search(tmp_path):
    yahoo = ListedOnly({"NVDA"}, matches=["NVDA"])
    assert repo_with(tmp_path, yahoo).find("Nvidia Corporation", "US").ticker == "NVDA"


def test_mistyped_exchange_suffix_searches_on_the_base_name(tmp_path):
    yahoo = ListedOnly({"RELIANCE.NS"}, matches=["RELIANCE.NS"])
    assert repo_with(tmp_path, yahoo).find("RELIANCE.B", "IN").ticker == "RELIANCE.NS"
    assert yahoo.searched == ["RELIANCE"]


def test_unknown_company_gives_a_helpful_error(tmp_path):
    with pytest.raises(SymbolNotFound, match="Could not find"):
        repo_with(tmp_path, ListedOnly(set())).find("Zzzqqqx")


def test_failed_search_is_reported_as_not_found(tmp_path):
    yahoo = ListedOnly(set())
    yahoo.search = lambda query, market=None: 1 / 0
    with pytest.raises(SymbolNotFound):
        repo_with(tmp_path, yahoo).find("Some Company")


def test_search_ranking_filters_and_prefers_selected_market():
    quotes = [
        {"symbol": "INFY", "quoteType": "EQUITY", "exchange": "NYQ"},
        {"symbol": "INFY.NS", "quoteType": "EQUITY", "exchange": "NSI"},
        {"symbol": "INFY-BL.NS", "quoteType": "EQUITY", "exchange": "NSI"},
        {"symbol": "IOY.F", "quoteType": "EQUITY", "exchange": "FRA"},
        {"symbol": "INFYX", "quoteType": "ETF", "exchange": "NYQ"},
    ]
    assert rank_search_results(quotes, "IN") == ["INFY.NS", "INFY"]
    assert rank_search_results(quotes, "US") == ["INFY", "INFY.NS"]

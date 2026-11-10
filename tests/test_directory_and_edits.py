from dataclasses import replace

import numpy as np
import pandas as pd

from fivs.data.cache import Cache
from fivs.data.directory import COLUMNS, parse_nse_list, parse_sec_list
from fivs.data.edits import (
    EditsStore,
    apply_edits,
    count_edits,
    describe_edits,
    financials_to_grid,
    grid_differences,
    grid_scale,
    validate_company,
)
from fivs.data.repository import CompanyRepository
from fivs.models import DataError
from fivs.valuation.engine import value_company

DATE = "2024-12-31"


# --- Directory ---

def test_sec_list_keeps_exchange_listed_companies_only():
    table = parse_sec_list({
        "fields": ["cik", "name", "ticker", "exchange"],
        "data": [[1, "Apple Inc.", "AAPL", "Nasdaq"], [2, "Pink Co", "PINK", "OTC"], [3, "No Venue", "NV", None], [1, "Apple Inc.", "AAPL", "Nasdaq"]],
    })
    assert list(table["Ticker"]) == ["AAPL"] and table["Market"].iloc[0] == "US"


def test_nse_list_handles_padded_headers_and_adds_suffix():
    csv = b"SYMBOL,NAME OF COMPANY, SERIES\nTCS,Tata Consultancy Services Limited,EQ\nM&M,Mahindra & Mahindra Limited,EQ\n"
    table = parse_nse_list(csv)
    assert list(table["Ticker"]) == ["TCS.NS", "M&M.NS"]
    assert table["Exchange"].iloc[0] == "NSE" and table["Market"].iloc[0] == "India"


class StubLists:
    def __init__(self, fail_india=False):
        self.fail_india, self.calls = fail_india, 0

    def us(self):
        self.calls += 1
        return pd.DataFrame([["zeta corp", "Z", "NYSE", "US"], ["Alpha Inc", "A", "Nasdaq", "US"]], columns=COLUMNS)

    def india(self):
        if self.fail_india:
            raise DataError("offline")
        return pd.DataFrame([["Mid Ltd", "MID.NS", "NSE", "India"]], columns=COLUMNS)


def repo(tmp_path, lists):
    return CompanyRepository(Cache(tmp_path / "c.sqlite3"), object(), object(), lists)


def test_directory_combines_and_sorts_without_regard_to_case(tmp_path):
    assert list(repo(tmp_path, StubLists()).directory().table["Company"]) == ["Alpha Inc", "Mid Ltd", "zeta corp"]


def test_directory_survives_one_market_failing(tmp_path):
    result = repo(tmp_path, StubLists(fail_india=True)).directory()
    assert len(result.table) == 2 and "India" in result.problems[0]


def test_directory_is_cached(tmp_path):
    lists = StubLists()
    repository = repo(tmp_path, lists)
    repository.directory()
    repository.directory()
    assert lists.calls == 1


# --- Saving ---

def test_edits_persist_across_store_instances_and_can_be_undone(tmp_path):
    path = tmp_path / "e.sqlite3"
    EditsStore(path).save("AAPL", {"price": 10.0})
    assert EditsStore(path).get("AAPL") == {"price": 10.0}
    assert EditsStore(path).tickers() == {"AAPL"}
    EditsStore(path).delete("AAPL")
    assert EditsStore(path).get("AAPL") is None


# --- Applying ---

def test_apply_changes_only_the_edited_cell_and_leaves_source_intact(company):
    original = company.financials.loc[DATE, "revenue"]
    edited = apply_edits(company, {"financials": {DATE: {"revenue": 5000.0}}})
    assert edited.financials.loc[DATE, "revenue"] == 5000.0
    assert company.financials.loc[DATE, "revenue"] == original
    assert edited.financials.loc[DATE, "net_income"] == company.financials.loc[DATE, "net_income"]


def test_no_edits_returns_the_company_unchanged(company):
    assert apply_edits(company, None) is company
    assert apply_edits(company, {}) is company


def test_cleared_cell_becomes_missing(company):
    assert np.isnan(apply_edits(company, {"financials": {DATE: {"capex": None}}}).financials.loc[DATE, "capex"])


def test_price_and_shares_update_market_cap(company):
    edited = apply_edits(company, {"price": 30.0, "shares": 200.0})
    assert edited.market_cap == 6000.0


def test_edit_for_period_no_longer_reported_is_ignored(company):
    edited = apply_edits(company, {"financials": {"1999-12-31": {"revenue": 1.0}}})
    pd.testing.assert_frame_equal(edited.financials, company.financials)


def test_edited_results_are_labelled(company):
    edited = apply_edits(company, {"price": 30.0, "financials": {DATE: {"revenue": 1.0, "cogs": 1.0}}})
    assert count_edits({"price": 30.0, "financials": {DATE: {"revenue": 1.0, "cogs": 1.0}}}) == 3
    assert "3 values" in edited.warnings[0] and "with your edits" in edited.source


def test_edit_flows_through_to_valuation(company):
    base = value_company(company, [], 0.04, runs=100)
    margin = {date.strftime("%Y-%m-%d"): {"operating_income": company.financials.loc[date, "revenue"] * 0.3} for date in company.financials.index}
    raised = value_company(apply_edits(company, {"financials": margin}), [], 0.04, runs=100)
    assert raised.dcf.value_per_share > base.dcf.value_per_share


# --- Comparing the grid with the source ---

def test_untouched_grid_produces_no_changes(company):
    scale = 1e6 / 7  # an awkward divisor, to expose rounding noise
    assert grid_differences(company.financials, financials_to_grid(company.financials, scale), scale) == {}


def test_grid_records_changes_in_full_units(company):
    grid = financials_to_grid(company.financials, 10.0)
    grid.loc["Revenue (sales)", DATE] = 999.0
    assert grid_differences(company.financials, grid, 10.0) == {DATE: {"revenue": 9990.0}}


def test_grid_records_cleared_cells_as_none(company):
    grid = financials_to_grid(company.financials, 1.0)
    grid.loc["Inventory", DATE] = np.nan
    assert grid_differences(company.financials, grid, 1.0) == {DATE: {"inventory": None}}


def test_grid_stores_outflows_as_positive(company):
    grid = financials_to_grid(company.financials, 1.0)
    grid.loc["Capital expenditure", DATE] = -55.0
    assert grid_differences(company.financials, grid, 1.0)[DATE]["capex"] == 55.0


def test_grid_scale_uses_crore_for_rupees(company):
    assert grid_scale(replace(company, currency="INR")) == (1e7, "₹ crore")
    assert grid_scale(company) == (1e9, "USD billions") if company.latest["revenue"] >= 1e9 else (1e6, "USD millions")


def test_validation_blocks_unusable_values(company):
    assert validate_company(company) == []
    broken = apply_edits(company, {"financials": {DATE: {"revenue": 0.0}}, "shares": 0.0})
    problems = " ".join(validate_company(broken))
    assert "revenue" in problems and "shares" in problems


def test_change_list_shows_before_and_after(company):
    original = company.financials.loc[DATE, "revenue"]
    table = describe_edits(company, {"price": 30.0, "financials": {DATE: {"revenue": original + 10}}}, 1.0)
    assert list(table["Item"]) == ["Share price", "Revenue (sales)"]
    assert table["Original"].iloc[0] == "20.00" and table["Yours"].iloc[0] == "30.00"

import io

import numpy as np
import pandas as pd
import pytest

from conftest import make_company
from fivs.analysis.narrative import STRONG, WEAK, compact_money, plain_summary, scorecards
from fivs.data.cache import Cache
from fivs.data.manual import (
    CompanyDetails,
    blank_table,
    build_company,
    example_table,
    parse_table,
    read_upload,
    resize_table,
    table_to_financials,
    validate_financials,
)
from fivs.data.quality import check_quality
from fivs.data.repository import CompanyRepository, convert_company
from fivs.models import DataError, PeerMultiples
from fivs.pipeline import Analysis, analyze_company
from fivs.valuation.engine import value_company

PEERS = [
    PeerMultiples("A", "A", pe=10, ev_ebitda=8, price_book=1),
    PeerMultiples("B", "B", pe=20, ev_ebitda=10, price_book=2),
    PeerMultiples("C", "C", pe=30, ev_ebitda=12, price_book=3),
]


class StubYahoo:
    def fx_rate(self, source, target):
        return 80.0


class StubRepository(CompanyRepository):
    def __init__(self, tmp_path):
        super().__init__(Cache(tmp_path / "c.sqlite3"), object(), StubYahoo())

    def peer_multiples(self, tickers):
        return PEERS


# --- Currency display ---

def test_conversion_scales_every_per_share_value_exactly(company):
    base = value_company(company, PEERS, 0.04, runs=200)
    converted = convert_company(company, 80.0, "INR")
    result = value_company(converted, PEERS, 0.04, runs=200)
    assert result.fair_value == pytest.approx(base.fair_value * 80)
    assert result.upside == pytest.approx(base.upside)
    assert result.dcf.assumptions.wacc == pytest.approx(base.dcf.assumptions.wacc)
    assert result.monte_carlo.p50 == pytest.approx(base.monte_carlo.p50 * 80)


def test_conversion_keeps_shares_and_market(company):
    converted = convert_company(company, 80.0, "INR")
    assert converted.currency == "INR" and converted.market == company.market
    assert converted.shares_outstanding == company.shares_outstanding
    assert converted.price == company.price * 80


def test_repository_fx_rate_same_currency_needs_no_lookup(tmp_path):
    assert StubRepository(tmp_path).fx_rate("INR", "INR") == 1.0
    assert StubRepository(tmp_path).fx_rate("USD", "INR") == 80.0


def test_compact_money():
    assert compact_money(5e9, "INR") == "₹500.0 crore"
    assert compact_money(2.5e12, "USD") == "$2.50 trillion"
    assert compact_money(4e8, "USD") == "$400.0 million"


# --- Reading uploads ---

def test_upload_understands_common_statement_names_and_formats():
    raw = pd.DataFrame({
        "Particulars": ["Turnover", "PAT", "Net worth", "Borrowings", "Sundry debtors", "Unknown row"],
        "FY24": ["1,23,456", "10,000", "₹50,000", "(5,000)", "-", "9"],
        "2025": [200000, 20000, 60000, 6000, 700, 9],
    })
    parsed = parse_table(raw)
    frame = table_to_financials(parsed.table, 1.0, 3)
    assert list(frame["revenue"]) == [123456, 200000]
    assert frame["net_income"].iloc[0] == 10000
    assert frame["equity"].iloc[0] == 50000
    assert frame["total_debt"].iloc[0] == -5000
    assert pd.isna(frame["receivables"].iloc[0])
    assert any("Unknown row" in note for note in parsed.notes)


def test_upload_reports_unreadable_cells():
    parsed = parse_table(pd.DataFrame({"Item": ["Revenue"], "2024": ["abc"], "2025": ["10"]}))
    assert any('"abc"' in note for note in parsed.notes)


def test_upload_needs_years_and_known_rows():
    with pytest.raises(DataError, match="year columns"):
        parse_table(pd.DataFrame({"Item": ["Revenue"], "Amount": [1]}))
    with pytest.raises(DataError, match="matched"):
        parse_table(pd.DataFrame({"Item": ["Nonsense"], "2024": [1]}))


def test_template_round_trips_through_csv_and_excel():
    table = example_table()
    csv = read_upload("t.csv", table.to_csv().encode()).table
    pd.testing.assert_frame_equal(csv, table)
    buffer = io.BytesIO()
    table.to_excel(buffer)
    excel = read_upload("t.xlsx", buffer.getvalue()).table
    pd.testing.assert_frame_equal(excel, table)


def test_unsupported_file_type():
    with pytest.raises(DataError, match="csv"):
        read_upload("t.pdf", b"x")


def test_resize_keeps_entered_numbers():
    table = blank_table([2023, 2024])
    table.iloc[0, 1] = 5.0
    resized = resize_table(table, [2024, 2025])
    assert list(resized.columns) == ["FY2024", "FY2025"]
    assert resized.iloc[0, 0] == 5.0


# --- From the grid to a company ---

def example_company(price=None, sector="Industrials"):
    frame = table_to_financials(example_table(), 1e7, 3)
    return build_company(CompanyDetails("Sample Pvt Ltd", "INR", sector, 1_000_000, price), frame)


def test_units_and_fiscal_year_end():
    frame = table_to_financials(example_table(), 1e7, 3)
    assert frame["revenue"].iloc[-1] == 720e7
    assert frame.index[-1] == pd.Timestamp("2025-03-31")


def test_gaps_are_derived_and_outflow_sign_is_normalized():
    table = blank_table([2024, 2025])
    table.loc["Revenue (sales) *"] = [100.0, 120.0]
    table.loc["Cost of goods sold"] = [60.0, 70.0]
    table.loc["Net profit *"] = [10.0, 12.0]
    table.loc["Income tax"] = [3.0, 4.0]
    table.loc["Capital expenditure"] = [-5.0, 6.0]
    frame = table_to_financials(table, 1.0, 12)
    assert frame["gross_profit"].iloc[-1] == 50
    assert frame["pretax_income"].iloc[-1] == 16
    assert frame["capex"].iloc[0] == 5


def test_blank_years_are_dropped():
    table = example_table()
    table["FY2026"] = np.nan
    assert len(table_to_financials(table, 1.0, 3)) == 4


def test_validation_names_missing_fields():
    frame = table_to_financials(example_table(), 1.0, 3)
    frame.loc[frame.index[-1], "cash"] = np.nan
    assert "Cash and equivalents" in validate_financials(frame, False)[0]


def test_validation_requires_two_years():
    frame = table_to_financials(example_table(), 1.0, 3).tail(1)
    assert "2 years" in validate_financials(frame, False)[0]


def test_banks_do_not_need_operating_profit():
    frame = table_to_financials(example_table(), 1.0, 3)
    frame["operating_income"] = np.nan
    assert validate_financials(frame, True) == []
    assert validate_financials(frame, False)


def test_example_data_is_clean():
    company = example_company()
    assert validate_financials(company.financials, False) == []
    assert not any("differ" in issue for issue in check_quality(company).issues)


def test_build_company_rejects_bad_details():
    frame = table_to_financials(example_table(), 1.0, 3)
    with pytest.raises(DataError, match="name"):
        build_company(CompanyDetails(" ", "INR", "Technology", 100), frame)
    with pytest.raises(DataError, match="shares"):
        build_company(CompanyDetails("X", "INR", "Technology", 0), frame)


# --- Unlisted companies through the whole pipeline ---

def test_unlisted_company_is_valued_without_a_price(tmp_path):
    analysis = analyze_company(example_company(), StubRepository(tmp_path))
    v = analysis.valuation
    assert v.fair_value > 0
    assert v.upside is None and v.implied_growth is None
    assert v.monte_carlo.prob_above_price is None
    assert v.verdict == "Model value only"


def test_given_price_enables_price_comparison(tmp_path):
    analysis = analyze_company(example_company(price=12000.0), StubRepository(tmp_path))
    assert analysis.valuation.upside is not None
    assert analysis.valuation.implied_growth is not None


def test_bank_entered_by_hand_uses_residual_income(tmp_path):
    analysis = analyze_company(example_company(sector="Bank or insurer"), StubRepository(tmp_path))
    assert analysis.valuation.residual_income is not None and analysis.valuation.dcf is None


def test_unlisted_company_converts_to_dollars(tmp_path):
    company = convert_company(example_company(), 0.0125, "USD")
    assert company.price is None
    assert analyze_company(company, StubRepository(tmp_path)).valuation.fair_value > 0


# --- Plain-language output ---

def test_summary_for_unlisted_company_mentions_total_value(tmp_path):
    text = " ".join(plain_summary(analyze_company(example_company(), StubRepository(tmp_path))))
    assert "values the whole company" in text and "crore" in text


def test_summary_mentions_implied_growth_for_listed_company(tmp_path, company):
    text = " ".join(plain_summary(analyze_company(company, StubRepository(tmp_path), peer_tickers=[], risk_free=0.04)))
    assert "justify today's price" in text


def test_scorecards_rate_a_healthy_company(tmp_path, company):
    cards = {c.name: c for c in scorecards(analyze_company(company, StubRepository(tmp_path), risk_free=0.04))}
    assert cards["Profitability"].rating == STRONG
    assert cards["Growth"].rating == STRONG


def test_scorecards_flag_operating_losses(tmp_path):
    company = make_company()
    company.financials["operating_income"] = -0.3 * company.financials["revenue"]
    cards = {c.name: c for c in scorecards(analyze_company(company, StubRepository(tmp_path), risk_free=0.04))}
    assert cards["Debt safety"].rating == WEAK and cards["Profitability"].rating == WEAK


def test_scorecards_use_bank_measures(tmp_path, bank):
    names = [c.name for c in scorecards(analyze_company(bank, StubRepository(tmp_path), risk_free=0.04))]
    assert "Capital cushion" in names and "Debt safety" not in names

import numpy as np
import pandas as pd
import pytest

from conftest import make_company, make_financials
from fivs.analysis.insights import generate_insights
from fivs.analysis.ratios import cagr, compute_ratios
from fivs.analysis.scores import altman_z_double_prime, piotroski


def test_ratios_match_hand_calculation():
    latest = compute_ratios(make_financials()).iloc[-1]
    assert latest.revenue_growth == pytest.approx(0.10)
    assert latest.ebit_margin == pytest.approx(0.20)
    assert latest.fcf_margin == pytest.approx(0.13)
    assert latest.current_ratio == pytest.approx(2.5)


def test_roe_uses_average_equity():
    fin = make_financials(years=2)
    expected = fin["net_income"].iloc[-1] / fin["equity"].mean()
    assert compute_ratios(fin)["roe"].iloc[-1] == pytest.approx(expected)


def test_zero_denominators_become_nan_not_infinity():
    fin = make_financials()
    fin["interest_expense"] = 0.0
    assert compute_ratios(fin)["interest_coverage"].isna().all()


def test_cagr():
    series = pd.Series([100.0, 110.0, 121.0])
    assert cagr(series, 2) == pytest.approx(0.10)
    assert cagr(series, 5) is None
    assert cagr(pd.Series([-1.0, 5.0]), 1) is None


def test_piotroski_all_nine_signals_pass():
    fin = make_financials(years=2)
    cur = fin.index[-1]
    fin.loc[cur, "net_income"] = 200.0
    fin.loc[cur, "operating_cash_flow"] = 300.0
    fin.loc[cur, "long_term_debt"] = 50.0
    fin.loc[cur, "current_assets"] = 800.0
    fin.loc[cur, "revenue"] *= 1.5
    fin.loc[cur, "gross_profit"] = 0.5 * fin.loc[cur, "revenue"]
    result = piotroski(fin)
    assert (result.score, result.tested, result.label) == (9, 9, "Strong")


def test_piotroski_counts_missing_signals_as_untested():
    fin = make_financials(years=2)
    fin["gross_profit"] = np.nan
    result = piotroski(fin)
    assert result.tested == 8


def test_piotroski_share_issuance_tolerance():
    fin = make_financials(years=2)
    fin.loc[fin.index[-1], "diluted_shares"] = 100.5
    assert next(s for s in piotroski(fin).signals if "share" in s.name).passed is True
    fin.loc[fin.index[-1], "diluted_shares"] = 105.0
    assert next(s for s in piotroski(fin).signals if "share" in s.name).passed is False


def test_altman_matches_hand_calculation():
    fin = make_financials(years=1)
    # X1 = (400-160)/800 = .3, X2 = 250/800 = .3125, X3 = 200/800 = .25, X4 = 500/300
    expected = 6.56 * 0.3 + 3.26 * 0.3125 + 6.72 * 0.25 + 1.05 * (500 / 300)
    result = altman_z_double_prime(fin)
    assert result.z == pytest.approx(expected)
    assert result.zone == "Safe"


def test_altman_needs_complete_data():
    fin = make_financials(years=1)
    fin["retained_earnings"] = np.nan
    assert altman_z_double_prime(fin) is None


def test_insights_flag_weak_cash_conversion():
    fin = make_financials()
    fin["operating_cash_flow"] = fin["net_income"] * 0.5
    titles = [i.title for i in generate_insights(make_company(fin), compute_ratios(fin))]
    assert "Weak earnings quality" in titles


def test_insights_use_bank_rules_for_banks(bank):
    insights = generate_insights(bank, compute_ratios(bank.financials))
    titles = [i.title for i in insights]
    assert "Equity cushion" in titles and "Negative free cash flow" not in titles

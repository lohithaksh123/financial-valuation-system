from dataclasses import replace

import numpy as np
import pytest

from conftest import make_company, make_financials
from fivs.models import PeerMultiples
from fivs.valuation.assumptions import (
    DCFAssumptions,
    ResidualIncomeAssumptions,
    adjusted_beta,
    build_dcf_assumptions,
    cost_of_capital,
)
from fivs.valuation.comps import value_by_comps
from fivs.valuation.dcf import market_implied_growth, project, run_dcf, sensitivity_table
from fivs.valuation.engine import value_company
from fivs.valuation.monte_carlo import Uncertainty, simulate
from fivs.valuation.residual_income import run_residual_income

BASE = DCFAssumptions(
    revenue_base=1000.0, growth_initial=0.08, growth_terminal=0.03, margin_initial=0.20, margin_target=0.20,
    tax_rate=0.25, net_capex_pct=0.02, nwc_pct=0.10, wacc=0.10, roic_spread=0.05, years=5,
)


def test_dcf_matches_independent_loop():
    """Re-derive the vectorized model year by year."""
    a = BASE
    revenue, pv = a.revenue_base, 0.0
    sustainable_rate = a.growth_terminal / (a.wacc + a.roic_spread)
    sustainable_capex = sustainable_rate * a.margin_target * (1 - a.tax_rate)
    for year in range(1, a.years + 1):
        growth = a.growth_initial + (a.growth_terminal - a.growth_initial) * (year - 1) / a.years
        previous, revenue = revenue, revenue * (1 + growth)
        capex = a.net_capex_pct + (sustainable_capex - a.net_capex_pct) * year / a.years
        fcff = revenue * a.margin_target * (1 - a.tax_rate) - capex * revenue - a.nwc_pct * (revenue - previous)
        pv += fcff / (1 + a.wacc) ** year
    terminal = revenue * (1 + a.growth_terminal) * a.margin_target * (1 - a.tax_rate) * (1 - sustainable_rate) / (a.wacc - a.growth_terminal)
    expected_ev = pv + terminal / (1 + a.wacc) ** a.years

    result = run_dcf(a, net_debt=100.0, minority_interest=20.0, shares=10.0)
    assert result.enterprise_value == pytest.approx(expected_ev)
    assert result.equity_value == pytest.approx(expected_ev - 120.0)
    assert result.value_per_share == pytest.approx((expected_ev - 120.0) / 10)


def test_terminal_value_requires_wacc_above_growth():
    with pytest.raises(ValueError, match="WACC"):
        run_dcf(replace(BASE, wacc=0.03), 0, 0, 1)


def test_value_moves_in_the_expected_direction():
    value = lambda a: run_dcf(a, 0, 0, 1).value_per_share  # noqa: E731
    assert value(replace(BASE, wacc=0.09)) > value(BASE) > value(replace(BASE, wacc=0.11))
    assert value(replace(BASE, margin_target=0.25)) > value(BASE)
    assert value(replace(BASE, growth_initial=0.12)) > value(BASE)


def test_vectorized_and_scalar_projections_agree():
    arrays = project(1000.0, np.array([0.05, 0.08]), 0.03, 0.2, 0.2, 0.25, 0.02, 0.1, np.array([0.1, 0.1]), 0.05, 5)
    single = run_dcf(BASE, 0, 0, 1)
    assert arrays.enterprise_value[1] == pytest.approx(single.enterprise_value)


def test_sensitivity_center_equals_base_case():
    table = sensitivity_table(BASE, 100.0, 20.0, 10.0)
    assert table.iloc[2, 2] == pytest.approx(run_dcf(BASE, 100.0, 20.0, 10.0).value_per_share)


def test_sensitivity_blanks_invalid_cells():
    table = sensitivity_table(replace(BASE, wacc=0.05, growth_terminal=0.04), 0, 0, 1)
    assert np.isnan(table.iloc[0, 4])


def test_reverse_dcf_recovers_the_input_growth():
    price = run_dcf(BASE, 100.0, 20.0, 10.0).value_per_share
    assert market_implied_growth(BASE, 100.0, 20.0, 10.0, price) == pytest.approx(BASE.growth_initial, abs=1e-6)


def test_reverse_dcf_returns_none_when_unreachable():
    assert market_implied_growth(BASE, 0, 0, 1, 1e12) is None


def test_monte_carlo_without_uncertainty_equals_dcf():
    none = Uncertainty(growth_sd=0, margin_sd=0, wacc_range=0, terminal_growth_range=0)
    result = simulate(BASE, 100.0, 20.0, 10.0, price=1.0, uncertainty=none, runs=50)
    assert result.p50 == pytest.approx(run_dcf(BASE, 100.0, 20.0, 10.0).value_per_share)


def test_monte_carlo_is_reproducible_and_ordered():
    first = simulate(BASE, 0, 0, 1, price=100.0, runs=2000, seed=7)
    second = simulate(BASE, 0, 0, 1, price=100.0, runs=2000, seed=7)
    assert first.p50 == second.p50
    assert first.p10 < first.p50 < first.p90
    assert 0 <= first.prob_above_price <= 1


def test_monte_carlo_floors_equity_at_zero():
    result = simulate(BASE, net_debt=1e7, minority_interest=0, shares=1, price=1, runs=100)
    assert (result.values == 0).all()


def test_beta_adjustment_is_bounded():
    assert adjusted_beta(1.0) == pytest.approx(1.0)
    assert adjusted_beta(0.05) == 0.70
    assert adjusted_beta(5.0) == 2.0
    assert adjusted_beta(None) == 1.0


def test_wacc_matches_hand_calculation(company):
    capital = cost_of_capital(company, 0.04)
    equity_weight = 2000 / 2200
    cost_of_equity = 0.04 + 1.0 * 0.045
    # Median interest / debt across years falls inside the allowed band.
    assert capital.cost_of_equity == pytest.approx(cost_of_equity)
    expected = equity_weight * cost_of_equity + (1 - equity_weight) * capital.pre_tax_cost_of_debt * (1 - capital.tax_rate)
    assert capital.wacc == pytest.approx(expected)
    assert capital.tax_rate == pytest.approx(0.25)


def test_assumptions_clamp_extreme_history():
    fin = make_financials()
    fin["revenue"] = [100.0, 400.0, 1600.0, 6400.0]
    company = make_company(fin)
    assumptions = build_dcf_assumptions(company, cost_of_capital(company, 0.04))
    assert assumptions.growth_initial == 0.25


def test_residual_income_equals_book_when_roe_equals_cost_of_equity():
    a = ResidualIncomeAssumptions(1000.0, 0.10, 0.10, 0.4, roe_terminal_spread=0.0)
    assert run_residual_income(a, 10.0).value_per_share == pytest.approx(100.0)


def test_residual_income_premium_when_roe_beats_cost_of_equity():
    a = ResidualIncomeAssumptions(1000.0, 0.15, 0.10, 0.4)
    assert run_residual_income(a, 10.0).value_per_share > 100.0


PEERS = [
    PeerMultiples("A", "A", pe=10, ev_ebitda=8, price_sales=1, price_book=1),
    PeerMultiples("B", "B", pe=20, ev_ebitda=10, price_sales=2, price_book=2),
    PeerMultiples("C", "C", pe=30, ev_ebitda=12, price_sales=3, price_book=3),
]


def test_comps_equity_and_enterprise_multiples(company):
    result = value_by_comps(company, PEERS)
    rows = {row.multiple: row for row in result.rows}
    latest = company.latest
    assert rows["P/E"].median_value == pytest.approx(20 * latest["net_income"] / 100)
    ebitda = latest["operating_income"] + latest["depreciation_amortization"]
    assert rows["EV/EBITDA"].median_value == pytest.approx((10 * ebitda - company.net_debt) / 100)
    assert result.value_per_share == pytest.approx((rows["P/E"].median_value + rows["EV/EBITDA"].median_value) / 2)


def test_comps_excludes_outliers_and_requires_three_peers(company):
    peers = PEERS[:2] + [PeerMultiples("D", "D", pe=500)]
    assert "P/E" not in [row.multiple for row in value_by_comps(company, peers).rows]


def test_comps_uses_book_multiple_for_banks(bank):
    assert [row.multiple for row in value_by_comps(bank, PEERS).rows] == ["P/E", "P/B"]


def test_engine_routes_banks_to_residual_income(bank):
    result = value_company(bank, PEERS, 0.04, runs=100)
    assert result.residual_income is not None and result.dcf is None


def test_engine_blends_models_and_labels_reading(company):
    result = value_company(company, PEERS, 0.04, runs=100)
    assert result.fair_value == pytest.approx(0.6 * result.dcf.value_per_share + 0.4 * result.comps.value_per_share)
    assert result.upside == pytest.approx(result.fair_value / company.price - 1)
    assert result.verdict in {"Price below model", "Price above model", "Near model value"}


def test_engine_applies_overrides(company):
    base = value_company(company, [], 0.04, runs=100)
    raised = value_company(company, [], 0.04, overrides={"margin_target": 0.30}, runs=100)
    assert raised.dcf.value_per_share > base.dcf.value_per_share
    assert base.notes  # warns that peers were unavailable


def test_engine_excludes_dcf_for_loss_makers():
    fin = make_financials()
    fin["operating_income"] = -0.1 * fin["revenue"]
    result = value_company(make_company(fin), PEERS, 0.04, runs=100)
    assert result.dcf is not None
    assert result.fair_value == pytest.approx(result.comps.value_per_share)
    assert result.method == "Peer multiples only"

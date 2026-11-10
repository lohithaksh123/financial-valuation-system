"""Command-line interface: `fivs analyze AAPL` and `fivs screen AAPL MSFT TCS.NS`."""
from __future__ import annotations

import argparse
import sys
from typing import List, Optional

import pandas as pd

from fivs.data.repository import CompanyRepository
from fivs.models import DataError
from fivs.pipeline import Analysis, analyze_company, screen_row


def _print_analysis(a: Analysis) -> None:
    c, v = a.company, a.valuation
    print(f"\n{c.name} ({c.ticker}) | {c.sector} / {c.industry}")
    print(f"Source: {c.source} | Data quality: {a.quality.score}/100 ({a.quality.grade})")
    print(f"Price: {c.price:,.2f} {c.currency} | Method: {v.method}")
    if v.fair_value is not None:
        print(f"Fair value: {v.fair_value:,.2f} {c.currency} ({v.upside:+.1%}) -> {v.verdict}")
    if a.piotroski:
        print(f"Piotroski F-Score: {a.piotroski.score}/{a.piotroski.tested} ({a.piotroski.label})")
    if a.altman:
        print(f"Altman Z'': {a.altman.z:.2f} ({a.altman.zone})")
    if v.dcf:
        print(f"DCF: {v.dcf.value_per_share:,.2f} | WACC {v.capital.wacc:.2%} | Terminal share {v.dcf.terminal_share:.0%}")
        print(f"Monte Carlo P10/P50/P90: {v.monte_carlo.p10:,.2f} / {v.monte_carlo.p50:,.2f} / {v.monte_carlo.p90:,.2f}"
              f" | P(value > price) {v.monte_carlo.prob_above_price:.0%}")
        if v.implied_growth is not None:
            print(f"Market-implied starting growth: {v.implied_growth:.1%} (model uses {v.dcf.assumptions.growth_initial:.1%})")
    if v.residual_income:
        print(f"Residual income: {v.residual_income.value_per_share:,.2f} | Cost of equity {v.capital.cost_of_equity:.2%}")
    if v.comps:
        for row in v.comps.rows:
            print(f"{row.multiple}: median {row.median_multiple:.1f}x -> {row.median_value:,.2f} ({row.peers_used} peers)")
    for insight in a.insights:
        print(f"  [{insight.severity:5}] {insight.title}: {insight.detail}")
    for note in c.warnings + a.quality.issues + v.notes:
        print(f"  note: {note}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="fivs", description="Financial Intelligence & Valuation System")
    commands = parser.add_subparsers(dest="command", required=True)
    analyze = commands.add_parser("analyze", help="Full analysis of one company")
    analyze.add_argument("ticker")
    screen = commands.add_parser("screen", help="Rank several companies")
    screen.add_argument("tickers", nargs="+")
    screen.add_argument("--csv", help="Write results to this CSV file")
    args = parser.parse_args(argv)

    repository = CompanyRepository()
    if args.command == "analyze":
        try:
            _print_analysis(analyze_company(repository.find(args.ticker), repository))
        except DataError as error:
            print(f"Error: {error}", file=sys.stderr)
            return 1
        return 0

    rows = []
    for ticker in args.tickers:
        try:
            rows.append(screen_row(analyze_company(repository.find(ticker), repository)))
        except DataError as error:
            print(f"Skipped {ticker}: {error}", file=sys.stderr)
    if not rows:
        return 1
    table = pd.DataFrame(rows).sort_values("Upside", ascending=False)
    print(table.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
    if args.csv:
        table.to_csv(args.csv, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

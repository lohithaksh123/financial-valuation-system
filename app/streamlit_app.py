"""Financial Intelligence & Valuation System: interactive dashboard."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import charts  # noqa: E402
from fivs.analysis.narrative import (  # noqa: E402
    GLOSSARY,
    NO_DATA,
    OKAY,
    STRONG,
    WEAK,
    compact_money,
    currency_symbol,
    plain_summary,
    scorecards,
)
from fivs.data import edits as company_edits  # noqa: E402
from fivs.data import manual  # noqa: E402
from fivs.data.repository import CompanyRepository, convert_company  # noqa: E402
from fivs.models import Company, DataError  # noqa: E402
from fivs.peers import default_peers  # noqa: E402
from fivs.pipeline import Analysis, analyze_company, screen_row  # noqa: E402
from fivs.valuation.assumptions import (  # noqa: E402
    build_dcf_assumptions,
    build_residual_income_assumptions,
    cost_of_capital,
)
from fivs.valuation.engine import value_company  # noqa: E402

st.set_page_config(page_title="Financial Intelligence & Valuation", page_icon="📊", layout="wide")

SCREENER_PRESETS = {
    "US": "AAPL, MSFT, GOOGL, JPM, WMT, KO, CAT, XOM",
    "IN": "TCS.NS, INFY.NS, RELIANCE.NS, HDFCBANK.NS, ITC.NS, LT.NS, SUNPHARMA.NS, MARUTI.NS",
}
SEVERITY_BADGE = {
    "good": ":green-badge[✓ Strength]",
    "watch": ":orange-badge[! Watch]",
    "risk": ":red-badge[✕ Risk]",
}
RATING_BADGE = {
    STRONG: ":green-badge[✓ Strong]",
    OKAY: ":orange-badge[~ Okay]",
    WEAK: ":red-badge[✕ Weak]",
    NO_DATA: ":gray-badge[- No data]",
}
DISPLAY_OWN, DISPLAY_INR, DISPLAY_USD = "Company's own currency", "Indian rupees (₹)", "US dollars ($)"
NAV_LISTED, NAV_BROWSE, NAV_OWN = "Look up a company", "Browse all companies", "My own company"
DISPLAY_TARGET = {DISPLAY_INR: "INR", DISPLAY_USD: "USD"}
PLOT = dict(use_container_width=True, config={"displayModeBar": False})
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]


@st.cache_resource
def repository() -> CompanyRepository:
    return CompanyRepository()


@st.cache_resource
def edits_store() -> company_edits.EditsStore:
    return company_edits.EditsStore()


@st.cache_data(ttl=3600, show_spinner=False)
def load_company(query: str, market: str) -> Company:
    return repository().find(query, market)


def with_saved_edits(company: Company) -> Company:
    return company_edits.apply_edits(company, edits_store().get(company.ticker))


def md(text: str) -> str:
    """Escape dollar signs, which Streamlit markdown would treat as LaTeX."""
    return text.replace("$", r"\$")


def scale_for(company: Company) -> Tuple[float, str]:
    """Table units: crore for rupees, otherwise billions or millions by company size."""
    if company.currency == "INR":
        return 1e7, "Cr"
    return (1e9, "B") if company.financials["revenue"].iloc[-1] >= 1e9 else (1e6, "M")


def learn(*terms: str) -> None:
    with st.expander("What do these terms mean?"):
        for term in terms:
            st.markdown(f"**{term}:** {md(GLOSSARY[term])}")


def to_display_currency(company: Company, choice: str) -> Tuple[Company, Optional[str]]:
    target = DISPLAY_TARGET.get(choice)
    if target is None or target == company.currency:
        return company, None
    rate = repository().fx_rate(company.currency, target)
    if rate is None:
        return company, f"Could not get today's {company.currency}/{target} rate, so amounts stay in {company.currency}."
    note = (
        f"Converted from {company.currency} to {target} at today's rate: 1 {company.currency} = "
        f"{currency_symbol(target)}{rate:,.4f}. Every year uses this one rate, so growth and ratios do not change."
    )
    return convert_company(company, rate, target), note


# --- Scenario controls -------------------------------------------------------

@dataclass(frozen=True)
class SliderSpec:
    field: str
    label: str
    start: float  # percent
    low: float
    high: float
    cautious: float
    optimistic: float
    help: str


def slider_specs(company: Company, risk_free: float) -> List[SliderSpec]:
    capital = cost_of_capital(company, risk_free)

    def spec(field, label, default, low, high, cautious, optimistic, help_text):
        start = float(np.clip(round(default * 100, 1), low, high))
        return SliderSpec(field, label, start, low, high, cautious, optimistic, help_text)

    if company.is_balance_sheet_financial:
        a = build_residual_income_assumptions(company, capital)
        return [
            spec("roe_initial", "Starting return on equity", a.roe_initial, 0.0, 40.0, -2.0, 2.0, GLOSSARY["ROE"]),
            spec("cost_of_equity", "Return shareholders require", a.cost_of_equity, 5.0, 20.0, 1.0, -1.0, "Higher values make future profits worth less today."),
            spec("payout_ratio", "Share of profit paid as dividends", a.payout_ratio, 0.0, 90.0, 0.0, 0.0, "Profit not paid out is reinvested in the business."),
        ]
    a = build_dcf_assumptions(company, capital)
    return [
        spec("growth_initial", "Revenue growth to start", a.growth_initial, -10.0, 40.0, -3.0, 3.0, "Fades to the long-term rate over 10 years."),
        spec("margin_target", "Long-run operating margin", a.margin_target, -20.0, 70.0, -2.0, 2.0, "Operating profit as a share of sales in the long run."),
        spec("wacc", "Required return (WACC)", a.wacc, 4.0, 20.0, 1.0, -1.0, GLOSSARY["WACC"]),
        spec("growth_terminal", "Growth after year 10", a.growth_terminal, 0.0, 8.0, 0.0, 0.0, GLOSSARY["Terminal growth"]),
    ]


def slider_key(ticker: str, field: str) -> str:
    return f"{ticker}-{field}"


def apply_scenario(specs: Sequence[SliderSpec], ticker: str) -> None:
    choice = st.session_state.get(f"{ticker}-scenario") or "Base"
    for s in specs:
        delta = {"Cautious": s.cautious, "Optimistic": s.optimistic}.get(choice, 0.0)
        st.session_state[slider_key(ticker, s.field)] = float(np.clip(round(s.start + delta, 1), s.low, s.high))


def scenario_controls(company: Company, risk_free: float) -> Dict[str, Any]:
    t = company.ticker
    specs = slider_specs(company, risk_free)
    for s in specs:
        st.session_state.setdefault(slider_key(t, s.field), s.start)

    st.sidebar.subheader("What if?")
    st.sidebar.segmented_control(
        "Scenario", ["Cautious", "Base", "Optimistic"], default="Base", key=f"{t}-scenario",
        on_change=apply_scenario, args=(specs, t), help="Sets the sliders below. You can then adjust any of them yourself.",
    )
    overrides: Dict[str, Any] = {}
    for s in specs:
        chosen = st.sidebar.slider(s.label, s.low, s.high, step=0.1, format="%.1f%%", key=slider_key(t, s.field), help=s.help)
        if chosen != s.start:
            overrides[s.field] = chosen / 100
    return overrides


# --- Own company data --------------------------------------------------------

def _set_table(table: pd.DataFrame) -> None:
    years = sorted(int(column[2:]) for column in table.columns)
    st.session_state.own_table = table
    st.session_state.own_edited = table
    st.session_state.own_latest = years[-1]
    st.session_state.own_years = min(len(years), 15)
    st.session_state.own_version = st.session_state.get("own_version", 0) + 1


def _load_example() -> None:
    _set_table(manual.example_table())
    st.session_state.own_name = "Sample Manufacturing Pvt Ltd"
    st.session_state.own_currency = "INR"
    st.session_state.own_sector = "Industrials"
    st.session_state.own_unit = "Crore"
    st.session_state.own_month = "March"
    st.session_state.own_shares = 10_000_000.0
    st.session_state.own_notes = []


def _clear_table() -> None:
    latest, count = st.session_state.own_latest, st.session_state.own_years
    _set_table(manual.blank_table(list(range(latest - count + 1, latest + 1))))
    st.session_state.own_notes = []


def _resize_years() -> None:
    latest, count = st.session_state.own_latest, st.session_state.own_years
    _set_table(manual.resize_table(st.session_state.own_edited, list(range(latest - count + 1, latest + 1))))


def _read_uploaded_file() -> None:
    uploaded = st.session_state.get("own_file")
    if uploaded is None:
        return
    try:
        parsed = manual.read_upload(uploaded.name, uploaded.getvalue())
    except DataError as error:
        st.session_state.own_notes = []
        st.session_state.own_upload_error = str(error)
        return
    st.session_state.own_upload_error = None
    _set_table(parsed.table)
    st.session_state.own_notes = parsed.notes


def _forget_scenarios(ticker: str) -> None:
    for key in [k for k in st.session_state if isinstance(k, str) and k.startswith(f"{ticker}-")]:
        del st.session_state[key]


def own_company_form() -> Optional[Company]:
    if "own_table" not in st.session_state:
        latest = date.today().year - 1
        st.session_state.own_latest, st.session_state.own_years = latest, 4
        _set_table(manual.blank_table(list(range(latest - 3, latest + 1))))
        st.session_state.own_notes = []
        st.session_state.update(own_shares=1_000_000.0, own_price=0.0, own_beta=1.0, own_unit="Crore", own_month="March")
    current = st.session_state.get("own_company")

    with st.expander("Your company's data" if current is None else f"Edit data for {current.name}", expanded=current is None):
        st.markdown("**Step 1: About the company**")
        left, middle, right = st.columns(3)
        name = left.text_input("Company name", key="own_name", placeholder="For example, Sharma Textiles Pvt Ltd")
        currency = middle.selectbox("Currency of your figures", ["INR", "USD"], key="own_currency")
        sector = right.selectbox("Sector", manual.SECTORS, key="own_sector", help="Sets the valuation method and peer group. Banks and insurers use a different model.")

        left, middle, right = st.columns(3)
        shares = left.number_input("Number of shares", min_value=1.0, step=1000.0, format="%.0f", key="own_shares")
        price = middle.number_input("Share price (leave 0 if unlisted)", min_value=0.0, step=1.0, key="own_price",
                                    help="With a price, the app also compares it with the model value.")
        beta = right.number_input("Beta (risk compared with the market)", min_value=0.3, max_value=3.0, step=0.1, key="own_beta",
                                  help="1.0 means average risk. Higher means more volatile than the market.")

        st.markdown("**Step 2: Add your figures** (upload a file, or type straight into the table)")
        left, middle, right = st.columns([2, 1, 1])
        left.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"], key="own_file", on_change=_read_uploaded_file,
                           label_visibility="collapsed")
        years_shown = sorted(int(c[2:]) for c in st.session_state.own_table.columns)
        middle.download_button("Download blank template", manual.blank_table(years_shown).to_csv(), "company_template.csv", "text/csv", use_container_width=True)
        middle.button("Clear the table", on_click=_clear_table, use_container_width=True)
        right.button("Fill in an example", on_click=_load_example, use_container_width=True)
        st.caption("Files need line items in the first column and years as headers (2024, FY2024 or FY24). Common names like Turnover, PAT, Net worth and Borrowings are recognized.")

        if st.session_state.get("own_upload_error"):
            st.error(md(st.session_state.own_upload_error))
        for note in st.session_state.get("own_notes", []):
            st.warning(md(note))

        columns = st.columns(4)
        unit = columns[0].selectbox("Amounts are in", list(manual.UNITS), key="own_unit")
        month = columns[1].selectbox("Financial year ends in", MONTHS, key="own_month")
        columns[2].number_input("Latest financial year", min_value=2000, max_value=2100, step=1, key="own_latest", on_change=_resize_years)
        columns[3].number_input("Years of history", min_value=2, max_value=15, step=1, key="own_years", on_change=_resize_years)

        grid = st.session_state.own_table
        edited = st.data_editor(
            grid, key=f"own_editor_{st.session_state.own_version}", height=420, use_container_width=True, num_rows="fixed",
            column_config={column: st.column_config.NumberColumn(column, format="%.2f") for column in grid.columns},
        )
        st.session_state.own_edited = edited
        st.caption("Rows marked * are needed for the latest year. Fill in more rows for a fuller analysis.")

        st.markdown("**Step 3: Analyze**")
        if st.button("Analyze this company", type="primary"):
            try:
                frame = manual.table_to_financials(edited, manual.UNITS[unit], MONTHS.index(month) + 1)
                problems = manual.validate_financials(frame, sector == manual.BANK_SECTOR) if len(frame) else ["The table is empty."]
                if problems:
                    for problem in problems:
                        st.error(md(problem))
                else:
                    company = manual.build_company(manual.CompanyDetails(name, currency, sector, shares, price or None, beta), frame)
                    _forget_scenarios(company.ticker)
                    st.session_state.own_company = company
                    st.rerun()
            except DataError as error:
                st.error(md(str(error)))
    return st.session_state.get("own_company")


# --- Browse every company and edit figures -----------------------------------

def _open_company(ticker: str, market: str, edit: bool) -> None:
    st.session_state.nav = NAV_LISTED
    st.session_state.market_choice = "India" if market == "India" else "United States"
    st.session_state["ticker-IN" if market == "India" else "ticker-US"] = ticker
    st.session_state.open_editor = ticker if edit else None


def render_directory() -> None:
    st.title("All listed companies")
    st.caption("Every company on Nasdaq, NYSE and India's NSE, from the SEC and NSE lists. Pick one to open or edit it.")
    with st.spinner("Loading the company lists"):
        directory = repository().directory()
    for problem in directory.problems:
        st.warning(md(problem))
    table = directory.table.copy()
    edited = edits_store().tickers()
    table["Your changes"] = table["Ticker"].map(lambda ticker: "Edited" if ticker in edited else "")

    columns = st.columns(3)
    columns[0].metric("US companies", f"{(table['Market'] == 'US').sum():,}")
    columns[1].metric("Indian companies", f"{(table['Market'] == 'India').sum():,}")
    columns[2].metric("Edited by you", f"{table['Your changes'].eq('Edited').sum():,}")

    left, middle, right = st.columns([3, 2, 1])
    search = left.text_input("Search by name or ticker", placeholder="For example, Tata or NVDA").strip()
    market = middle.segmented_control("Market", ["All", "US", "India"], default="All", key="directory_market") or "All"
    only_edited = right.checkbox("Only edited", key="directory_edited")

    shown = table
    if search:
        names = shown["Company"].str.contains(search, case=False, regex=False, na=False)
        tickers = shown["Ticker"].str.contains(search, case=False, regex=False, na=False)
        shown = shown[names | tickers]
    if market != "All":
        shown = shown[shown["Market"] == market]
    if only_edited:
        shown = shown[shown["Your changes"] == "Edited"]
    shown = shown.reset_index(drop=True)
    st.caption(f"Showing {len(shown):,} of {len(table):,} companies. Tick a row to select it.")

    event = st.dataframe(
        shown, hide_index=True, use_container_width=True, height=460, on_select="rerun", selection_mode="single-row",
        key=f"directory-{search}-{market}-{only_edited}",
    )
    rows = event.selection.rows
    if not rows:
        st.info("Tick the box beside a company to open it or edit its figures.")
        return
    picked = shown.iloc[rows[0]]
    with st.container(border=True):
        st.markdown(md(f"**{picked['Company']}** ({picked['Ticker']}, {picked['Exchange']})"))
        left, right, _ = st.columns([1, 1, 3])
        left.button("Open company", type="primary", on_click=_open_company, args=(picked["Ticker"], picked["Market"], False), use_container_width=True)
        right.button("Edit its figures", on_click=_open_company, args=(picked["Ticker"], picked["Market"], True), use_container_width=True)


def render_company_editor(base: Company) -> None:
    """Edit a listed company's figures. Always works in the company's own currency."""
    store = edits_store()
    ticker = base.ticker
    saved = store.get(ticker)
    current = company_edits.apply_edits(base, saved)
    count = company_edits.count_edits(saved)
    version = st.session_state.get(f"edit-version-{ticker}", 0)
    scale, unit = company_edits.grid_scale(base)
    title = f"Edit figures ({count} changed)" if count else "Edit this company's figures"

    with st.expander(title, expanded=st.session_state.get("open_editor") == ticker):
        st.caption(f"Change any number, then save. Your changes are kept on this computer and every tab recalculates. Amounts are in {unit}.")
        left, middle, right = st.columns(3)
        price = left.number_input(f"Share price ({base.currency})", min_value=0.01, value=float(current.price), step=1.0, format="%.2f", key=f"edit-price-{ticker}-{version}")
        shares = middle.number_input("Number of shares", min_value=1.0, value=float(current.shares_outstanding), step=1000.0, format="%.0f", key=f"edit-shares-{ticker}-{version}")
        beta = right.number_input("Beta", min_value=0.0, max_value=5.0, value=float(current.beta if current.beta is not None else 1.0), step=0.05, format="%.2f", key=f"edit-beta-{ticker}-{version}")

        table = company_edits.financials_to_grid(current.financials, scale)
        grid = st.data_editor(
            table, key=f"edit-grid-{ticker}-{version}", height=420, use_container_width=True, num_rows="fixed",
            column_config={column: st.column_config.NumberColumn(column, format="%.2f") for column in table.columns},
        )

        save_column, undo_column, _ = st.columns([1, 1, 3])
        if save_column.button("Save changes", type="primary", key=f"edit-save-{ticker}", use_container_width=True):
            changes: Dict[str, Any] = {}
            differences = company_edits.grid_differences(base.financials, grid, scale)
            if differences:
                changes["financials"] = differences
            if abs(price - base.price) >= 0.005:
                changes["price"] = price
            if abs(shares - base.shares_outstanding) >= 0.5:
                changes["shares"] = shares
            original_beta = base.beta if base.beta is not None else 1.0
            if abs(beta - original_beta) >= 0.005:
                changes["beta"] = beta
            problems = company_edits.validate_company(company_edits.apply_edits(base, changes))
            if problems:
                for problem in problems:
                    st.error(md(problem))
            else:
                if changes:
                    store.save(ticker, changes)
                else:
                    store.delete(ticker)
                st.session_state[f"edit-version-{ticker}"] = version + 1
                st.session_state.open_editor = ticker
                _forget_scenarios(ticker)
                st.rerun()
        if undo_column.button("Undo all my changes", disabled=not saved, key=f"edit-undo-{ticker}", use_container_width=True):
            store.delete(ticker)
            st.session_state[f"edit-version-{ticker}"] = version + 1
            _forget_scenarios(ticker)
            st.rerun()

        if saved:
            st.markdown("**Your saved changes**")
            st.dataframe(company_edits.describe_edits(base, saved, scale), hide_index=True, use_container_width=True)


# --- Tabs --------------------------------------------------------------------

def render_overview(a: Analysis, base_value: Optional[float]) -> None:
    c, v, s = a.company, a.valuation, currency_symbol(a.company.currency)
    with st.container(border=True):
        st.markdown("#### In plain words")
        st.markdown(md(" ".join(plain_summary(a))))

    delta = None
    if base_value and v.fair_value is not None:
        delta = f"{v.fair_value / base_value - 1:+.1%} vs base case"
    columns = st.columns(3)
    model_text = f"{s}{v.fair_value:,.2f}" if v.fair_value is not None else "n/a"
    if c.price is not None:
        columns[0].metric("Share price", f"{s}{c.price:,.2f}", help="The latest market price of one share.")
        columns[1].metric("Model value per share", model_text, delta, help=GLOSSARY["Model value"])
        columns[2].metric("Model vs price", f"{v.upside:+.1%}" if v.upside is not None else "n/a",
                          help="Positive means the model value is above the share price.")
    else:
        equity = compact_money(v.fair_value * c.shares_outstanding, c.currency) if v.fair_value is not None else "n/a"
        columns[0].metric("Share price", "Not listed")
        columns[1].metric("Model value per share", model_text, delta, help=GLOSSARY["Model value"])
        columns[2].metric("Estimated company value", equity, help="Model value per share multiplied by the number of shares.")
    st.caption(md(f"Reading: {v.verdict}. Method: {v.method}. Data: {c.source}."))
    for note in c.warnings + v.notes:
        st.info(md(note))
    if c.price is None:
        st.info("Listed-company peer multiples are used as they are. Unlisted shares are harder to sell, so a real deal would usually apply a discount.")

    st.markdown("#### Health at a glance")
    for column, card in zip(st.columns(4), scorecards(a)):
        with column.container(border=True):
            st.markdown(f"**{card.name}**  \n{RATING_BADGE[card.rating]}")
            st.markdown(f"##### {md(card.figure)}")
            st.caption(md(card.meaning))

    if v.ranges:
        st.plotly_chart(charts.football_field(v.ranges, c.price, s), **PLOT)
        st.caption("Each bar shows a method's range and the dot its central estimate. Hover for exact values.")

    left, right = st.columns(2)
    with left:
        st.markdown("#### Key findings")
        if not a.insights:
            st.write("No notable strengths or red flags were triggered.")
        for insight in a.insights:
            st.markdown(md(f"{SEVERITY_BADGE[insight.severity]} **{insight.title}.** {insight.detail}"))
    with right:
        st.markdown("#### How reliable is the data?")
        st.progress(a.quality.score / 100, f"{a.quality.grade} quality: {a.quality.score} out of 100")
        st.markdown(f"{a.quality.years} financial years, {a.quality.completeness:.0%} of key fields present.")
        for issue in a.quality.issues:
            st.markdown(f"- {md(issue)}")
    learn("Model value", "DCF", "P/E", "EV/EBITDA")


def render_financials(a: Analysis) -> None:
    c = a.company
    scale, unit_letter = scale_for(c)
    symbol = currency_symbol(c.currency)
    fin = c.financials
    available = {
        "Revenue": fin["revenue"], "Gross profit": fin["gross_profit"], "Operating profit": fin["operating_income"],
        "Net profit": fin["net_income"], "Operating cash flow": fin["operating_cash_flow"],
        "Free cash flow": a.ratios["free_cash_flow"], "Cash": fin["cash"], "Total debt": fin["total_debt"],
    }
    colors = dict(zip(available, charts.PALETTE))  # fixed per item, so colors never shift
    usable = [name for name, series in available.items() if series.notna().any()]

    st.markdown("#### Explore the numbers")
    left, middle, right = st.columns([3, 2, 1])
    picked = left.multiselect("Show up to four items", usable, default=[n for n in ("Revenue", "Net profit") if n in usable], max_selections=4, key=f"{c.ticker}-items")
    view = middle.segmented_control("View as", ["Amount", "Yearly growth", "% of revenue"], default="Amount", key=f"{c.ticker}-view") or "Amount"
    kind = right.segmented_control("Chart", ["Bars", "Lines"], default="Bars", key=f"{c.ticker}-kind") or "Bars"

    years = list(fin.index)
    if len(years) > 2:
        start, end = st.select_slider(
            "Years", options=years, value=(years[0], years[-1]), format_func=lambda d: str(d.year), key=f"{c.ticker}-years"
        )
    else:
        start, end = years[0], years[-1]

    if picked:
        table = pd.DataFrame({name: available[name] for name in picked})
        if view == "Yearly growth":
            table = table.pct_change(fill_method=None).where(table.shift(1) > 0)
            value_format, axis_format, title = ".1%", ".0%", "Yearly growth"
        elif view == "% of revenue":
            table = table.div(fin["revenue"], axis=0)
            value_format, axis_format, title = ".1%", ".0%", "As a share of revenue"
        else:
            table = table / scale
            value_format, axis_format, title = ",.1f", ",.0f", f"Amounts ({symbol} {unit_letter})"
        st.plotly_chart(charts.explorer(table.loc[start:end], colors, kind, value_format, axis_format, title), **PLOT)
    else:
        st.info("Pick at least one item to chart.")

    with st.expander(f"See the full statements ({symbol} {unit_letter})"):
        statement = (fin / scale).drop(columns="diluted_shares").dropna(axis=1, how="all").T
        statement.columns = [str(index.date()) for index in fin.index]
        statement.index = [manual.LABELS.get(name, name) for name in statement.index]
        st.dataframe(statement.style.format("{:,.1f}", na_rep="-"), use_container_width=True)
        st.download_button("Download as CSV", statement.to_csv(), f"{c.ticker}_financials.csv", "text/csv")


RATIO_LABELS = {
    "revenue_growth": ("Revenue growth", "pct"), "gross_margin": ("Gross margin", "pct"), "ebit_margin": ("Operating margin", "pct"),
    "net_margin": ("Net margin", "pct"), "fcf_margin": ("Free cash flow margin", "pct"), "roe": ("Return on equity", "pct"),
    "roa": ("Return on assets", "pct"), "roic": ("Return on invested capital", "pct"), "asset_turnover": ("Asset turnover", "x"),
    "current_ratio": ("Current ratio", "x"), "debt_to_equity": ("Debt / equity", "x"), "net_debt_to_ebitda": ("Net debt / EBITDA", "x"),
    "interest_coverage": ("Interest coverage", "x"), "cash_conversion": ("Cash flow / net profit", "x"),
}


def render_health(a: Analysis) -> None:
    c = a.company
    if c.is_balance_sheet_financial:
        st.info("Piotroski and Altman scores were built for industrial companies, so they are not shown for banks and insurers.")
    left, right = st.columns(2)
    if a.piotroski:
        with left.container(border=True):
            st.markdown("#### Quality checklist (Piotroski F-Score)")
            st.progress(a.piotroski.score / 9, f"{a.piotroski.score} of {a.piotroski.tested} checks passed: {a.piotroski.label}")
            marks = {True: ":green-badge[✓ Pass]", False: ":red-badge[✕ Fail]", None: ":gray-badge[- No data]"}
            for signal in a.piotroski.signals:
                st.markdown(f"{marks[signal.passed]} {signal.name}")
    if a.altman:
        with right.container(border=True):
            st.markdown("#### Distress risk (Altman Z'')")
            st.metric("Score", f"{a.altman.z:.2f}", help=GLOSSARY["Altman Z''"])
            st.markdown(f"This company sits in the **{a.altman.zone.lower()} zone**.")
            st.plotly_chart(charts.altman_scale(a.altman.z), **PLOT)
            st.caption("Buybacks shrink retained earnings, which can lower this score for otherwise healthy companies.")

    recent = a.ratios.tail(6)
    usable = {label: column for column, (label, _) in RATIO_LABELS.items() if recent[column].notna().any()}
    st.markdown("#### Ratio trends")
    choice = st.selectbox("Pick a ratio to chart", list(usable), key=f"{c.ticker}-ratio")
    column = usable[choice]
    is_percent = RATIO_LABELS[column][1] == "pct"
    st.plotly_chart(charts.single_line(recent[column], choice, ".1%" if is_percent else ".2f", ".0%" if is_percent else ".1f"), **PLOT)
    rows = {
        label: ["-" if pd.isna(x) else (f"{x:.1%}" if RATIO_LABELS[col][1] == "pct" else f"{x:.2f}x") for x in recent[col]]
        for label, col in usable.items()
    }
    with st.expander("See all ratios"):
        st.dataframe(pd.DataFrame.from_dict(rows, orient="index", columns=[str(i.date()) for i in recent.index]), use_container_width=True)
    learn("F-Score", "Altman Z''", "ROE", "ROIC", "Free cash flow")


def render_dcf(a: Analysis) -> None:
    c, v, s = a.company, a.valuation, currency_symbol(a.company.currency)
    dcf, capital, mc = v.dcf, v.capital, v.monte_carlo
    scale, unit_letter = scale_for(c)
    st.info("Move the **What if?** controls in the sidebar and every number here updates instantly.")

    columns = st.columns(4)
    columns[0].metric("DCF value per share", f"{s}{dcf.value_per_share:,.2f}", help=GLOSSARY["DCF"])
    columns[1].metric("Required return", f"{dcf.assumptions.wacc:.2%}", help=GLOSSARY["WACC"])
    columns[2].metric("From after year 10", f"{dcf.terminal_share:.0%}", help=GLOSSARY["Terminal value share"])
    if c.price is not None:
        columns[3].metric("Growth the price needs", f"{v.implied_growth:.1%}" if v.implied_growth is not None else "n/a",
                          help=f"{GLOSSARY['Implied growth']} The model starts at {dcf.assumptions.growth_initial:.1%}.")
    else:
        columns[3].metric("Starting growth used", f"{dcf.assumptions.growth_initial:.1%}")

    pv_forecast = float(dcf.forecast["pv_fcff"].sum()) / scale
    pv_terminal = dcf.enterprise_value / scale - pv_forecast
    steps = [("Next 10 years", pv_forecast, "relative"), ("After year 10", pv_terminal, "relative"), ("Business value", 0.0, "total")]
    if c.net_debt:
        steps.append(("Net cash" if c.net_debt < 0 else "Less net debt", -c.net_debt / scale, "relative"))
    if c.minority_interest:
        steps.append(("Less minority share", -c.minority_interest / scale, "relative"))
    steps.append(("Shareholders' value", 0.0, "total"))
    st.plotly_chart(charts.value_bridge(steps, s, unit_letter), **PLOT)

    st.markdown("#### What could the value be?")
    st.plotly_chart(charts.distribution(mc.values, c.price, {"P10": mc.p10, "P50": mc.p50, "P90": mc.p90}, s), **PLOT)
    message = "10,000 scenarios varying growth, margin, required return and long-term growth."
    if mc.prob_above_price is not None:
        message += f" In {mc.prob_above_price:.0%} of them the shares are worth more than today's price."
    st.caption(message)

    reference, label = (c.price, "price") if c.price is not None else (dcf.value_per_share, "base case")
    st.plotly_chart(charts.sensitivity_heatmap(v.sensitivity, reference, label, s, "required return", "long-term growth"), **PLOT)

    with st.expander("See the calculation details"):
        left, right = st.columns(2)
        raw = "n/a" if capital.raw_beta is None else f"{capital.raw_beta:.2f}"
        left.markdown("**Required return (WACC) build-up**")
        left.dataframe(pd.DataFrame(
            [("Risk-free rate", f"{capital.risk_free:.2%}"), ("Equity risk premium", f"{capital.equity_risk_premium:.2%}"),
             ("Raw beta", raw), ("Adjusted beta", f"{capital.beta:.2f}"), ("Cost of equity", f"{capital.cost_of_equity:.2%}"),
             ("Pre-tax cost of debt", f"{capital.pre_tax_cost_of_debt:.2%}"), ("Tax rate", f"{capital.tax_rate:.1%}"),
             ("Equity weight", f"{capital.equity_weight:.1%}"), ("Computed WACC", f"{capital.wacc:.2%}")],
            columns=["Input", "Value"]), hide_index=True, use_container_width=True)
        right.markdown("**From business value to share value**")
        right.dataframe(pd.DataFrame(
            [("Business value", compact_money(dcf.enterprise_value, c.currency)), ("Net debt", compact_money(c.net_debt, c.currency)),
             ("Minority interest", compact_money(c.minority_interest, c.currency)), ("Shareholders' value", compact_money(dcf.equity_value, c.currency)),
             ("Shares", f"{c.shares_outstanding:,.0f}"), ("Value per share", f"{s}{dcf.value_per_share:,.2f}")],
            columns=["Item", "Value"]), hide_index=True, use_container_width=True)
        forecast = dcf.forecast.copy()
        for column in ("revenue", "nopat", "reinvestment", "fcff", "pv_fcff"):
            forecast[column] /= scale
        forecast = forecast.rename(columns={"revenue": "Revenue", "ebit_margin": "Op. margin", "nopat": "After-tax profit",
                                            "reinvestment": "Reinvestment", "fcff": "Free cash flow", "pv_fcff": "Value today"})
        st.markdown(f"**10-year forecast ({s} {unit_letter})**")
        numeric = {name: "{:,.1f}" for name in forecast.columns}
        numeric["Op. margin"] = "{:.1%}"
        st.dataframe(forecast.style.format(numeric), use_container_width=True)
    learn("DCF", "WACC", "Terminal growth", "Monte Carlo", "P10 / P50 / P90")


def render_residual_income(a: Analysis) -> None:
    c, v, s = a.company, a.valuation, currency_symbol(a.company.currency)
    ri = v.residual_income
    scale, unit_letter = scale_for(c)
    st.info("Move the **What if?** controls in the sidebar and every number here updates instantly.")
    columns = st.columns(4)
    columns[0].metric("Value per share", f"{s}{ri.value_per_share:,.2f}", help=GLOSSARY["Residual income"])
    columns[1].metric("Return shareholders require", f"{ri.assumptions.cost_of_equity:.2%}")
    columns[2].metric("Starting ROE", f"{ri.assumptions.roe_initial:.1%}", help=GLOSSARY["ROE"])
    columns[3].metric("Book value per share", f"{s}{ri.assumptions.book_value / c.shares_outstanding:,.2f}")

    book = ri.assumptions.book_value / scale
    steps = [("Book equity", book, "relative"), ("Next 10 years' excess profit", ri.pv_excess_returns / scale, "relative"),
             ("After year 10", ri.pv_terminal / scale, "relative"), ("Shareholders' value", 0.0, "total")]
    st.plotly_chart(charts.value_bridge(steps, s, unit_letter), **PLOT)

    reference, label = (c.price, "price") if c.price is not None else (ri.value_per_share, "base case")
    st.plotly_chart(charts.sensitivity_heatmap(v.sensitivity, reference, label, s, "required return", "starting ROE"), **PLOT)
    with st.expander(f"See the 10-year forecast ({s} {unit_letter})"):
        forecast = ri.forecast.copy()
        for column in ("opening_book", "residual_income", "pv_residual"):
            forecast[column] /= scale
        forecast = forecast.rename(columns={"opening_book": "Opening book", "roe": "ROE", "residual_income": "Excess profit", "pv_residual": "Value today"})
        st.dataframe(forecast.style.format({"Opening book": "{:,.1f}", "ROE": "{:.1%}", "Excess profit": "{:,.1f}", "Value today": "{:,.1f}"}), use_container_width=True)
    learn("Residual income", "ROE", "P/B")


def render_peers(a: Analysis) -> None:
    c, v, s = a.company, a.valuation, currency_symbol(a.company.currency)
    st.caption("Change the companies used here with the **Peers** box in the sidebar.")
    if not v.comps or v.comps.peer_table.empty:
        st.info("No peer data available. Add peer tickers in the sidebar.")
        return
    if v.comps.rows:
        rows = [{
            "Measure": r.multiple, "Peers used": r.peers_used, "Typical peer": f"{r.median_multiple:.1f}x",
            "Peer range": f"{r.low_multiple:.1f}x to {r.high_multiple:.1f}x",
            "Implied value per share": f"{s}{r.median_value:,.2f}", "Counted in model value": "Yes" if r.primary else "No",
        } for r in v.comps.rows]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    else:
        st.warning("Fewer than three usable peer multiples, so no comparable-company value was calculated.")
    st.markdown("#### The peer group")
    st.dataframe(v.comps.peer_table.style.format(
        {"P/E": "{:.1f}", "EV/EBITDA": "{:.1f}", "P/S": "{:.1f}", "P/B": "{:.1f}"}, na_rep="-"
    ), hide_index=True, use_container_width=True)
    st.caption("Peer multiples are trailing twelve months from Yahoo Finance, applied to the company's latest annual results.")
    learn("P/E", "EV/EBITDA", "P/B")


def render_compare(market: str, display_choice: str) -> None:
    st.write("Rank several companies with the same models and default assumptions. Lists can mix US and Indian companies.")
    text = st.text_input("Company names or tickers, comma separated", SCREENER_PRESETS[market], key=f"screen-{market}")
    if st.button("Compare", type="primary"):
        names = [t for t in (x.strip() for x in text.split(",")) if t]
        progress, rows, failures = st.progress(0.0), [], []
        for i, raw in enumerate(names, start=1):
            progress.progress(i / len(names), f"Analysing {raw}")
            try:
                company, _ = to_display_currency(with_saved_edits(load_company(raw, market)), display_choice)
                rows.append(screen_row(analyze_company(company, repository())))
            except (DataError, ValueError) as error:
                failures.append(f"{raw}: {error}")
        progress.empty()
        st.session_state["screen_result"] = (rows, failures)

    if "screen_result" in st.session_state:
        rows, failures = st.session_state["screen_result"]
        if rows:
            table = pd.DataFrame(rows).sort_values("Upside", ascending=False, na_position="last")
            ranked = table.dropna(subset=["Upside"])
            if len(ranked):
                st.plotly_chart(charts.ranking(ranked), **PLOT)
            st.dataframe(table.style.format({
                "Price": "{:,.2f}", "Fair value": "{:,.2f}", "Upside": "{:+.1%}", "Altman Z''": "{:.2f}",
                "ROE": "{:.1%}", "Op. margin": "{:.1%}",
            }, na_rep="-"), hide_index=True, use_container_width=True)
            st.download_button("Download as CSV", table.to_csv(index=False), "comparison.csv", "text/csv")
        for failure in failures:
            st.warning(md(failure))


def main() -> None:
    st.sidebar.title("Valuation system")
    source = st.sidebar.radio("What would you like to do?", [NAV_LISTED, NAV_BROWSE, NAV_OWN], key="nav")
    if source == NAV_BROWSE:
        render_directory()
        return
    display_choice = st.sidebar.selectbox("Show amounts in", [DISPLAY_OWN, DISPLAY_INR, DISPLAY_USD], key="display_currency")

    base_company = None
    if source == NAV_LISTED:
        market = "IN" if st.sidebar.radio("Market", ["United States", "India"], horizontal=True, key="market_choice") == "India" else "US"
        st.session_state.setdefault(f"ticker-{market}", "AAPL" if market == "US" else "TCS")
        query = st.sidebar.text_input(
            "Company name or ticker", key=f"ticker-{market}",
            help="Type a name or a symbol, for example Nvidia, NVDA, Tata Motors or RELIANCE.BO. Press Enter to load it.",
        ).strip()
        try:
            with st.spinner(f"Loading {query}"):
                base_company = load_company(query, market)
            company = with_saved_edits(base_company)
        except DataError as error:
            st.error(md(str(error)))
            render_compare(market, display_choice)
            return
        except Exception as error:  # network failures should not show a stack trace
            st.error(f"Could not load {query}. The data source may be unavailable. ({type(error).__name__})")
            return
    else:
        st.title("Analyze your own company")
        st.caption("For private companies, or any company you have figures for. Nothing leaves this app, and data is kept only until you close it.")
        company = own_company_form()
        if company is None:
            return
        market = company.market

    repo = repository()
    risk_free = repo.risk_free(company.market)
    original_price = company.price
    original_currency = company.currency
    company, conversion_note = to_display_currency(company, display_choice)

    overrides = scenario_controls(company, risk_free)
    if overrides:
        st.sidebar.caption("Showing your adjusted scenario.")
    peers_text = st.sidebar.text_input("Peers", ", ".join(default_peers(company)), key=f"{company.ticker}-peers")
    peers = [t.strip().upper() for t in peers_text.split(",") if t.strip()]

    st.title(md(company.name))
    st.caption(md(f"{company.ticker} · {company.sector} · {company.industry}"))
    if base_company is not None:
        render_company_editor(base_company)
    if conversion_note:
        extra = ""
        if original_price is not None and company.currency != original_currency:
            extra = f" Original share price: {currency_symbol(original_currency)}{original_price:,.2f}."
        st.caption(md(conversion_note + extra))

    try:
        analysis = analyze_company(company, repo, peer_tickers=peers, risk_free=risk_free, overrides=overrides)
        base_value = None
        if overrides:
            base_value = value_company(company, repo.peer_multiples(peers), risk_free, runs=500).fair_value
    except ValueError as error:
        st.error(f"Those assumptions cannot be valued: {error}")
        return

    tabs = st.tabs(["Overview", "Financials", "Health", "Valuation", "Peers", "Compare"])
    with tabs[0]:
        render_overview(analysis, base_value)
    with tabs[1]:
        render_financials(analysis)
    with tabs[2]:
        render_health(analysis)
    with tabs[3]:
        (render_residual_income if company.is_balance_sheet_financial else render_dcf)(analysis)
    with tabs[4]:
        render_peers(analysis)
    with tabs[5]:
        render_compare(market, display_choice)
    st.divider()
    st.caption("Educational project, not investment advice. Model values depend on the assumptions shown and can differ widely from market prices.")


main()

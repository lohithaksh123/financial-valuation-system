"""Company data supplied by the user, typed in or uploaded as CSV or Excel."""
from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from fivs.models import Company, DataError, normalize_frame

# Canonical field -> label shown to the user, in statement order.
FIELDS: List[Tuple[str, str]] = [
    ("revenue", "Revenue (sales)"),
    ("cogs", "Cost of goods sold"),
    ("gross_profit", "Gross profit"),
    ("operating_income", "Operating profit (EBIT)"),
    ("interest_expense", "Interest expense"),
    ("pretax_income", "Profit before tax"),
    ("income_tax", "Income tax"),
    ("net_income", "Net profit"),
    ("depreciation_amortization", "Depreciation and amortization"),
    ("total_assets", "Total assets"),
    ("current_assets", "Current assets"),
    ("total_liabilities", "Total liabilities"),
    ("current_liabilities", "Current liabilities"),
    ("equity", "Shareholders' equity"),
    ("minority_interest", "Minority interest"),
    ("cash", "Cash and equivalents"),
    ("short_term_investments", "Short-term investments"),
    ("total_debt", "Total debt"),
    ("long_term_debt", "Long-term debt"),
    ("retained_earnings", "Retained earnings"),
    ("receivables", "Receivables"),
    ("inventory", "Inventory"),
    ("accounts_payable", "Accounts payable"),
    ("operating_cash_flow", "Operating cash flow"),
    ("capex", "Capital expenditure"),
    ("dividends_paid", "Dividends paid"),
]
LABELS = dict(FIELDS)
REQUIRED = ["revenue", "operating_income", "net_income", "total_assets", "total_liabilities", "equity", "cash", "total_debt"]
REQUIRED_MARK = " *"

UNITS = {"Rupees or dollars (as is)": 1.0, "Thousands": 1e3, "Lakh": 1e5, "Millions": 1e6, "Crore": 1e7, "Billions": 1e9}
BANK_SECTOR = "Bank or insurer"
SECTORS = [
    "Technology", "Healthcare", "Consumer Cyclical", "Consumer Defensive", "Industrials", "Basic Materials",
    "Energy", "Utilities", "Communication Services", "Real Estate", BANK_SECTOR,
]


def _key(text: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


# Common names for each line item in Indian and US statements.
_ALIASES = {
    "revenue": ["revenue", "sales", "netsales", "turnover", "totalrevenue", "revenuefromoperations", "totalincome"],
    "cogs": ["cogs", "costofsales", "costofrevenue", "costofmaterialsconsumed"],
    "operating_income": ["ebit", "operatingprofit", "operatingincome"],
    "interest_expense": ["interest", "financecost", "financecosts", "interestpaid"],
    "pretax_income": ["pbt", "ebt", "pretaxprofit", "pretaxincome"],
    "income_tax": ["tax", "taxexpense", "incometaxexpense"],
    "net_income": ["pat", "netincome", "profitaftertax", "netprofitloss", "profitfortheyear"],
    "depreciation_amortization": ["depreciation", "da", "dna", "depreciationandamortisation"],
    "equity": ["networth", "shareholdersfunds", "totalequity", "stockholdersequity"],
    "minority_interest": ["noncontrollinginterest", "nci"],
    "cash": ["cash", "cashandbank", "cashandbankbalances", "cashandcashequivalents"],
    "short_term_investments": ["currentinvestments", "marketablesecurities"],
    "total_debt": ["debt", "borrowings", "totalborrowings"],
    "long_term_debt": ["longtermborrowings", "noncurrentborrowings"],
    "retained_earnings": ["reserves", "reservesandsurplus"],
    "receivables": ["accountsreceivable", "tradereceivables", "debtors", "sundrydebtors"],
    "inventory": ["inventories", "stock"],
    "accounts_payable": ["payables", "tradepayables", "creditors", "sundrycreditors"],
    "operating_cash_flow": ["cfo", "cashfromoperations", "cashflowfromoperations", "netcashfromoperatingactivities"],
    "capex": ["capitalexpenses", "purchaseoffixedassets", "purchaseofppe", "purchaseofpropertyplantandequipment"],
    "dividends_paid": ["dividend", "dividends"],
}
_LOOKUP: Dict[str, str] = {}
for _field, _label in FIELDS:
    _LOOKUP[_key(_label)] = _field
    _LOOKUP[_key(_field)] = _field
for _field, _names in _ALIASES.items():
    _LOOKUP.update({name: _field for name in _names})


def blank_table(years: List[int]) -> pd.DataFrame:
    """The entry grid: one row per line item, one column per fiscal year."""
    labels = [label + (REQUIRED_MARK if field in REQUIRED else "") for field, label in FIELDS]
    table = pd.DataFrame(np.nan, index=labels, columns=[f"FY{year}" for year in sorted(years)], dtype=float)
    table.index.name = "Line item"
    return table


def resize_table(table: pd.DataFrame, years: List[int]) -> pd.DataFrame:
    """Change the fiscal years shown, keeping any numbers already entered."""
    resized = blank_table(years)
    shared = [column for column in resized.columns if column in table.columns]
    resized[shared] = table[shared].to_numpy()
    return resized


def example_table() -> pd.DataFrame:
    """A consistent sample company, in crore, so people can see how the grid works."""
    data = {
        "revenue": [420, 510, 610, 720], "cogs": [260, 312, 366, 425], "operating_income": [52, 68, 88, 110],
        "interest_expense": [6, 7, 7, 8], "pretax_income": [46, 61, 81, 102], "income_tax": [12, 16, 21, 26],
        "net_income": [34, 45, 60, 76], "depreciation_amortization": [14, 16, 19, 22],
        "total_assets": [380, 440, 520, 610], "current_assets": [190, 225, 270, 325],
        "total_liabilities": [210, 240, 270, 295], "current_liabilities": [95, 110, 128, 146],
        "equity": [170, 200, 250, 315], "cash": [40, 46, 62, 85], "total_debt": [90, 98, 100, 104],
        "long_term_debt": [60, 65, 66, 68], "retained_earnings": [120, 155, 205, 270], "receivables": [62, 75, 90, 106],
        "inventory": [48, 58, 70, 82], "accounts_payable": [38, 45, 54, 62], "operating_cash_flow": [44, 56, 72, 92],
        "capex": [30, 36, 42, 50], "dividends_paid": [6, 8, 10, 12],
    }
    table = blank_table([2022, 2023, 2024, 2025])
    for field, values in data.items():
        table.loc[_label_in(table, field)] = np.asarray(values, dtype=float)
    return table


def _label_in(table: pd.DataFrame, field: str) -> str:
    return next(label for label in table.index if _LOOKUP.get(_key(label)) == field)


def _header_year(header: Any) -> Optional[int]:
    if isinstance(header, (pd.Timestamp, np.datetime64)) or hasattr(header, "year"):
        return pd.Timestamp(header).year
    text = str(header)
    match = re.search(r"(?:19|20)\d{2}", text)
    if match:
        return int(match.group())
    match = re.search(r"FY\s*'?(\d{2})\b", text, re.IGNORECASE)
    return 2000 + int(match.group(1)) if match else None


def _number(value: Any) -> Optional[float]:
    """Read a statement figure: handles 1,23,456, currency signs and (negatives)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)
    text = re.sub(r"[,\s₹$]", "", str(value))
    if text in ("", "-", "--", "nan", "NA", "N/A"):
        return np.nan
    negative = text.startswith("(") and text.endswith(")")
    try:
        number = float(text.strip("()"))
    except ValueError:
        return None
    return -number if negative else number


@dataclass
class ParsedUpload:
    table: pd.DataFrame
    notes: List[str]


def read_upload(filename: str, content: bytes) -> ParsedUpload:
    """Read a CSV or Excel statement file into the entry grid."""
    try:
        if filename.lower().endswith((".xlsx", ".xlsm")):
            raw = pd.read_excel(io.BytesIO(content), dtype=object)
        elif filename.lower().endswith(".csv"):
            raw = pd.read_csv(io.BytesIO(content), dtype=object)
        else:
            raise DataError("Please upload a .csv or .xlsx file.")
    except DataError:
        raise
    except Exception as error:
        raise DataError(f"Could not read that file: {error}") from error
    return parse_table(raw)


def parse_table(raw: pd.DataFrame) -> ParsedUpload:
    if raw.shape[1] < 2:
        raise DataError("The file needs line items in the first column and one column per year.")
    label_column, value_columns = raw.columns[0], raw.columns[1:]
    years = {column: _header_year(column) for column in value_columns}
    years = {column: year for column, year in years.items() if year is not None}
    if not years:
        raise DataError("No year columns found. Name them like 2024, FY2024 or FY24.")

    table = blank_table(sorted(set(years.values())))
    notes: List[str] = []
    matched = 0
    unmatched = []
    for _, row in raw.iterrows():
        label = row[label_column]
        field = _LOOKUP.get(_key(label))
        if field is None:
            if not pd.isna(label):
                unmatched.append(str(label))
            continue
        matched += 1
        for column, year in years.items():
            value = _number(row[column])
            if value is None:
                notes.append(f'Could not read "{row[column]}" for {label} in {year}.')
                continue
            table.loc[_label_in(table, field), f"FY{year}"] = value
    if not matched:
        raise DataError("None of the rows matched a known line item. Download the template to see the expected names.")
    if unmatched:
        notes.append(f"Ignored rows not recognized: {', '.join(unmatched[:8])}" + (" and more." if len(unmatched) > 8 else "."))
    return ParsedUpload(table, notes)


def table_to_financials(table: pd.DataFrame, unit: float, year_end_month: int) -> pd.DataFrame:
    """Convert the entry grid into the canonical annual table."""
    rows = {}
    for column in table.columns:
        year = _header_year(column)
        end = pd.Timestamp(year=year, month=year_end_month, day=1) + pd.offsets.MonthEnd(0)
        values = {_LOOKUP[_key(label)]: table.loc[label, column] * unit for label in table.index}
        if not all(pd.isna(value) for value in values.values()):
            rows[end] = values
    frame = pd.DataFrame.from_dict(rows, orient="index", dtype=float).reindex(columns=[field for field, _ in FIELDS])
    frame["gross_profit"] = frame["gross_profit"].fillna(frame["revenue"] - frame["cogs"])
    frame["pretax_income"] = frame["pretax_income"].fillna(frame["net_income"] + frame["income_tax"])
    # People enter outflows with either sign; the canonical table stores magnitudes.
    frame[["capex", "dividends_paid"]] = frame[["capex", "dividends_paid"]].abs()
    return normalize_frame(frame)


def validate_financials(frame: pd.DataFrame, is_bank: bool) -> List[str]:
    """Problems that must be fixed before the company can be analysed."""
    errors: List[str] = []
    if len(frame) < 2:
        return ["Enter at least 2 years of figures so growth can be measured."]
    latest = frame.iloc[-1]
    needed = [field for field in REQUIRED if not (is_bank and field == "operating_income")]
    missing = [LABELS[field] for field in needed if pd.isna(latest[field])]
    if missing:
        errors.append(f"Latest year is missing: {', '.join(missing)}.")
    if not pd.isna(latest["revenue"]) and latest["revenue"] <= 0:
        errors.append("Latest year's revenue must be above zero.")
    if not pd.isna(latest["total_assets"]) and latest["total_assets"] <= 0:
        errors.append("Latest year's total assets must be above zero.")
    return errors


@dataclass
class CompanyDetails:
    name: str
    currency: str
    sector: str
    shares: float
    price: Optional[float] = None
    beta: float = 1.0


def build_company(details: CompanyDetails, frame: pd.DataFrame) -> Company:
    if not details.name.strip():
        raise DataError("Enter a company name.")
    if details.shares <= 0:
        raise DataError("Number of shares must be above zero.")
    is_bank = details.sector == BANK_SECTOR
    price = details.price if details.price and details.price > 0 else None
    equity = frame["equity"].iloc[-1]
    warnings = ["Figures were entered by the user and have not been audited or checked against filings."]
    if price is None:
        warnings.append("No share price was given, so book equity stands in for market value when weighting WACC.")
    return Company(
        ticker="OWN-" + re.sub(r"[^A-Z0-9]", "", details.name.upper())[:12],
        name=details.name.strip(),
        market="IN" if details.currency == "INR" else "US",
        currency=details.currency,
        sector="Financial Services" if is_bank else details.sector,
        industry="Banks - Regional" if is_bank else "Unknown",
        price=price,
        shares_outstanding=float(details.shares),
        market_cap=price * details.shares if price else max(float(equity), 1.0),
        financials=frame,
        source="Your own data",
        beta=details.beta,
        warnings=warnings,
    )

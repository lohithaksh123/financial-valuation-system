"""User corrections to listed companies, saved separately from the source data.

Only differences are stored, so refreshed source data still flows through for
every figure the user has not changed, and undoing is simply deleting the record.
"""
from __future__ import annotations

import json
import math
import sqlite3
import time
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from fivs.config import EDITS_PATH
from fivs.data.manual import FIELDS, LABELS
from fivs.models import Company

_FIELD_FOR_LABEL = {label: field for field, label in FIELDS}
_OUTFLOWS = {"capex", "dividends_paid"}


class EditsStore:
    """SQLite store that, unlike the cache, never expires."""

    def __init__(self, path: Path = EDITS_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as db, db:
            db.execute("CREATE TABLE IF NOT EXISTS edits (ticker TEXT PRIMARY KEY, data TEXT NOT NULL, saved_at REAL NOT NULL)")

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=30)

    def get(self, ticker: str) -> Optional[Dict[str, Any]]:
        with closing(self._connect()) as db:
            row = db.execute("SELECT data FROM edits WHERE ticker = ?", (ticker,)).fetchone()
        return json.loads(row[0]) if row else None

    def save(self, ticker: str, edits: Dict[str, Any]) -> None:
        with closing(self._connect()) as db, db:
            db.execute("INSERT OR REPLACE INTO edits (ticker, data, saved_at) VALUES (?, ?, ?)", (ticker, json.dumps(edits), time.time()))

    def delete(self, ticker: str) -> None:
        with closing(self._connect()) as db, db:
            db.execute("DELETE FROM edits WHERE ticker = ?", (ticker,))

    def tickers(self) -> Set[str]:
        with closing(self._connect()) as db:
            return {row[0] for row in db.execute("SELECT ticker FROM edits")}


def count_edits(edits: Optional[Dict[str, Any]]) -> int:
    if not edits:
        return 0
    cells = sum(len(fields) for fields in edits.get("financials", {}).values())
    return cells + sum(1 for key in ("price", "shares", "beta") if key in edits)


def apply_edits(company: Company, edits: Optional[Dict[str, Any]]) -> Company:
    """A copy of the company with the user's changes laid over the source data."""
    if not edits:
        return company
    frame = company.financials.copy()
    for date_text, fields in edits.get("financials", {}).items():
        date = pd.Timestamp(date_text)
        if date not in frame.index:
            continue  # the source no longer reports this period
        for field, value in fields.items():
            if field in frame.columns:
                frame.loc[date, field] = np.nan if value is None else float(value)

    price = edits.get("price", company.price)
    shares = edits.get("shares", company.shares_outstanding)
    market_cap = price * shares if ("price" in edits or "shares" in edits) and price else company.market_cap
    count = count_edits(edits)
    return replace(
        company,
        financials=frame,
        price=price,
        shares_outstanding=shares,
        market_cap=market_cap,
        beta=edits.get("beta", company.beta),
        source=f"{company.source}, with your edits",
        warnings=[f"You have changed {count} value{'' if count == 1 else 's'}, so results no longer match the original data."] + company.warnings,
    )


def grid_scale(company: Company) -> Tuple[float, str]:
    if company.currency == "INR":
        return 1e7, "₹ crore"
    return (1e9, f"{company.currency} billions") if company.financials["revenue"].iloc[-1] >= 1e9 else (1e6, f"{company.currency} millions")


def financials_to_grid(frame: pd.DataFrame, scale: float) -> pd.DataFrame:
    """The editable table: line items down the side, fiscal year ends across the top."""
    grid = pd.DataFrame(
        {date.strftime("%Y-%m-%d"): [frame.loc[date, field] / scale for field, _ in FIELDS] for date in frame.index},
        index=[label for _, label in FIELDS], dtype=float,
    )
    grid.index.name = "Line item"
    return grid


def _same(first: float, second: float) -> bool:
    if pd.isna(first) or pd.isna(second):
        return pd.isna(first) and pd.isna(second)
    return math.isclose(first, second, rel_tol=1e-9, abs_tol=1e-6)


def grid_differences(original: pd.DataFrame, grid: pd.DataFrame, scale: float) -> Dict[str, Dict[str, Optional[float]]]:
    """Cells where the grid differs from the source. A cleared cell is stored as None."""
    changes: Dict[str, Dict[str, Optional[float]]] = {}
    for date_text in grid.columns:
        date = pd.Timestamp(date_text)
        for label in grid.index:
            field = _FIELD_FOR_LABEL[label]
            value = pd.to_numeric(grid.loc[label, date_text], errors="coerce")
            amount = value * scale
            if field in _OUTFLOWS and not pd.isna(amount):
                amount = abs(amount)
            if not _same(amount, original.loc[date, field]):
                changes.setdefault(date_text, {})[field] = None if pd.isna(amount) else float(amount)
    return changes


def validate_company(company: Company) -> List[str]:
    """Problems that would make an edited company impossible to value."""
    latest, errors = company.latest, []
    if pd.isna(latest["revenue"]) or latest["revenue"] <= 0:
        errors.append("Latest year's revenue must be above zero.")
    if pd.isna(latest["total_assets"]) or latest["total_assets"] <= 0:
        errors.append("Latest year's total assets must be above zero.")
    if company.shares_outstanding <= 0:
        errors.append("Number of shares must be above zero.")
    if company.price is not None and company.price <= 0:
        errors.append("Share price must be above zero.")
    return errors


def describe_edits(base: Company, edits: Optional[Dict[str, Any]], scale: float) -> pd.DataFrame:
    """A readable before-and-after list of every saved change."""
    def show(value: Optional[float], divisor: float = 1.0) -> str:
        return "(blank)" if value is None or pd.isna(value) else f"{value / divisor:,.2f}"

    rows = []
    for key, label, original in (("price", "Share price", base.price), ("shares", "Number of shares", base.shares_outstanding), ("beta", "Beta", base.beta)):
        if edits and key in edits:
            rows.append({"Item": label, "Year end": "-", "Original": show(original), "Yours": show(edits[key])})
    for date_text, fields in sorted((edits or {}).get("financials", {}).items()):
        for field, value in fields.items():
            date = pd.Timestamp(date_text)
            original = base.financials.loc[date, field] if date in base.financials.index else None
            rows.append({"Item": LABELS.get(field, field), "Year end": date_text, "Original": show(original, scale), "Yours": show(value, scale)})
    return pd.DataFrame(rows, columns=["Item", "Year end", "Original", "Yours"])

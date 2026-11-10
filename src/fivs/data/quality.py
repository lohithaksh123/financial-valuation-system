"""Automated data-quality checks, run on every company before it is analysed."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

import numpy as np

from fivs.models import Company

CORE_FIELDS = ["revenue", "net_income", "total_assets", "total_liabilities", "equity", "operating_cash_flow"]
INDUSTRIAL_FIELDS = ["operating_income", "current_assets", "current_liabilities", "capex", "depreciation_amortization"]
BALANCE_TOLERANCE = 0.02
MAX_FILING_AGE_DAYS = 550


@dataclass
class QualityReport:
    score: int  # 0-100
    issues: List[str] = field(default_factory=list)
    years: int = 0
    completeness: float = 0.0

    @property
    def grade(self) -> str:
        return "High" if self.score >= 85 else "Medium" if self.score >= 60 else "Low"


def check_quality(company: Company, today: Optional[datetime] = None) -> QualityReport:
    frame = company.financials
    today = today or datetime.now()
    issues: List[str] = []
    penalty = 0

    if len(frame) < 3:
        issues.append(f"Only {len(frame)} fiscal year(s) of history; trends and growth are unreliable.")
        penalty += 25

    fields = CORE_FIELDS + ([] if company.is_balance_sheet_financial else INDUSTRIAL_FIELDS)
    completeness = float(frame[fields].notna().to_numpy().mean())
    if completeness < 0.95:
        missing = sorted(f for f in fields if frame[f].isna().any())
        issues.append(f"Field coverage is {completeness:.0%}. Gaps in: {', '.join(missing)}.")
        penalty += round((1 - completeness) * 60)

    gap = (
        frame["total_assets"] - frame["total_liabilities"] - frame["equity"] - frame["minority_interest"].fillna(0)
    ).abs() / frame["total_assets"]
    unbalanced = gap[gap > BALANCE_TOLERANCE]
    if len(unbalanced):
        years = ", ".join(str(index.year) for index in unbalanced.index)
        issues.append(f"Assets differ from liabilities + equity by more than {BALANCE_TOLERANCE:.0%} in: {years}.")
        penalty += min(25, 10 * len(unbalanced))

    if (frame["revenue"].dropna() <= 0).any():
        issues.append("Non-positive revenue reported in at least one year.")
        penalty += 15

    jumps = frame["revenue"].pct_change(fill_method=None).abs()
    if (jumps > 3).any():
        issues.append("Revenue moved by more than 300% in one year (acquisition, split or a data error).")
        penalty += 10

    age = (today - frame.index[-1].to_pydatetime()).days
    if age > MAX_FILING_AGE_DAYS:
        issues.append(f"Latest annual data is {age} days old.")
        penalty += 20

    return QualityReport(
        score=int(np.clip(100 - penalty, 0, 100)), issues=issues, years=len(frame), completeness=completeness
    )

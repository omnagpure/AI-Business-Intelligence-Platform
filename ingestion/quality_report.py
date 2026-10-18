"""Turn a profile (+ the DataFrame) into explicit, evidence-backed data-quality issues."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

import pandas as pd
from pandas.api import types as pdt

from config.analytics_config import DEFAULT_THRESHOLDS, QualityThresholds
from ingestion.profiler import DatasetProfile

SEVERITIES = ("high", "medium", "low")
_PII = re.compile(r"(e-?mail|password|passwd|phone|mobile|ssn|passport|street|address|"
                  r"first.?name|last.?name|fname|lname|birth|dob)", re.IGNORECASE)


@dataclass(frozen=True)
class QualityIssue:
    severity: str
    code: str
    column: str | None
    message: str
    affected_rows: int = 0
    suggestion: str = ""


@dataclass
class QualityReport:
    issues: list[QualityIssue] = field(default_factory=list)
    completeness_pct: float = 100.0
    status: str = "good"            # good | review | poor

    def by_severity(self, sev: str) -> list[QualityIssue]:
        return [i for i in self.issues if i.severity == sev]

    def to_records(self) -> list[dict]:
        return [asdict(i) for i in self.issues]


def _label_variants(s: pd.Series) -> int:
    """Number of distinct raw labels that collapse into another label after trim/case-folding."""
    u = pd.Series(s.dropna().astype(str).unique())
    if len(u) > 2000:
        return 0
    norm = u.str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
    return int(len(u) - norm.nunique())


def build_quality_report(df: pd.DataFrame, profile: DatasetProfile,
                         load_warnings: list[str] | None = None,
                         t: QualityThresholds = DEFAULT_THRESHOLDS) -> QualityReport:
    issues: list[QualityIssue] = []
    add = issues.append
    for w in load_warnings or []:
        add(QualityIssue("low", "load_warning", None, w))

    if profile.rows < t.min_rows_warn:
        add(QualityIssue("medium", "few_rows", None, f"Only {profile.rows} rows; statistics and models will be unreliable."))
    if profile.duplicate_rows:
        sev = "medium" if profile.duplicate_pct >= t.duplicate_warn_pct else "low"
        add(QualityIssue(sev, "duplicate_rows", None,
                         f"{profile.duplicate_rows} exact duplicate row(s) ({profile.duplicate_pct}%).",
                         profile.duplicate_rows, "Review before removing; repeats may be legitimate records."))

    for c in profile.columns_profile:
        if c.kind == "empty":
            add(QualityIssue("high", "empty_column", c.name, "Column is entirely empty.", profile.rows,
                             "Exclude it from analysis."))
            continue
        if c.missing_pct >= t.missing_high_pct:
            add(QualityIssue("high", "high_missing", c.name, f"{c.missing_pct}% of values are missing.",
                             c.missing, "Check data collection or exclude the column."))
        elif c.missing_pct >= t.missing_warn_pct:
            add(QualityIssue("medium", "missing", c.name, f"{c.missing_pct}% of values are missing.",
                             c.missing, "Decide on an imputation or exclusion strategy."))
        if c.is_constant:
            add(QualityIssue("low", "constant_column", c.name, "Column has a single constant value.",
                             profile.rows, "It carries no information for analysis."))
        if c.stored_as_text and c.kind == "numeric":
            add(QualityIssue("medium", "numeric_as_text", c.name, "Numbers are stored as text.",
                             profile.rows - c.missing, "Convert to a numeric type."))
        if c.stored_as_text and c.kind == "temporal":
            add(QualityIssue("low", "date_as_text", c.name, "Dates are stored as text.",
                             profile.rows - c.missing, "Parse into a datetime type."))
        if c.temporal and c.temporal.get("unparsed"):
            add(QualityIssue("medium", "unparseable_dates", c.name,
                             f"{c.temporal['unparsed']} value(s) could not be parsed as dates.",
                             c.temporal["unparsed"], "Check for mixed or invalid date formats."))
        if c.numeric and c.numeric.get("negatives") and re.search(r"(qty|quantity|units|count|age|price)", c.name, re.IGNORECASE):
            add(QualityIssue("low", "negative_values", c.name,
                             f"{c.numeric['negatives']} negative value(s) in a column that is usually non-negative.",
                             c.numeric["negatives"], "Verify whether these are returns, errors or valid."))
        for note in c.suspicious:
            add(QualityIssue("medium", "suspicious_values", c.name, note))
        if _PII.search(c.name):
            add(QualityIssue("medium", "possible_pii", c.name,
                             "Column name suggests personal or sensitive data.", 0,
                             "Exclude it from reports and exports unless it is required."))

        s = df[c.name]
        is_text = pdt.is_object_dtype(s) or pdt.is_string_dtype(s)
        if c.kind in ("categorical", "boolean") and is_text:
            variants = _label_variants(s)
            if variants:
                add(QualityIssue("medium", "inconsistent_labels", c.name,
                                 f"{variants} label(s) differ only by case or spacing.", 0,
                                 "Standardise category labels."))
        if is_text:
            sample = s.dropna().head(5000)
            strs = sample[sample.map(lambda v: isinstance(v, str))]
            if len(strs) and (strs != strs.str.strip()).any():
                add(QualityIssue("low", "whitespace", c.name,
                                 "Some values have leading or trailing spaces.", int((strs != strs.str.strip()).sum())))
            if len(sample) and sample.map(type).nunique() > 1:
                add(QualityIssue("medium", "mixed_types", c.name, "Column mixes different value types."))

    order = {s: i for i, s in enumerate(SEVERITIES)}
    issues.sort(key=lambda i: (order[i.severity], i.code, i.column or ""))
    highs = sum(i.severity == "high" for i in issues)
    meds = sum(i.severity == "medium" for i in issues)
    status = "poor" if highs else ("review" if meds else "good")
    return QualityReport(issues=issues, completeness_pct=round(100 - profile.missing_cells_pct, 2), status=status)

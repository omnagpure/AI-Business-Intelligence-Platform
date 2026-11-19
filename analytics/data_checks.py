"""Small, pure data-validity helpers shared by the KPI engine. Nothing here changes data."""
from __future__ import annotations

import pandas as pd

from semantic.column_lexicon import token_set

_TRUE = {"1", "1.0", "true", "yes", "y", "t"}
_FALSE = {"0", "0.0", "false", "no", "n", "f"}
SENSITIVE_TOKENS = frozenset({"password", "passwd", "pwd", "email", "phone", "mobile", "ssn", "street",
                              "address", "fname", "lname", "firstname", "lastname", "zipcode", "passport"})


def parse_flag(series: pd.Series) -> tuple[pd.Series | None, str]:
    """Read a Yes/No, True/False or 0/1 column as booleans (missing stays missing).

    Returns (flags, "") when every non-missing value is a recognised flag value, otherwise
    (None, reason). Placeholder labels such as 'on_time_delivery_12' are refused, never guessed.
    """
    s = series.dropna().astype("string").str.strip().str.lower()
    if s.empty:
        return None, "the column has no values"
    unknown = sorted(set(s) - _TRUE - _FALSE)
    if unknown:
        shown = ", ".join(f"'{u}'" for u in unknown[:3])
        return None, (f"its values are not Yes/No or 0/1 ({len(unknown)} other value(s), for example {shown})")
    raw = series.astype("string").str.strip().str.lower()
    out = pd.Series(pd.NA, index=series.index, dtype="boolean")
    out[raw.isin(_TRUE)] = True
    out[raw.isin(_FALSE)] = False
    return out, ""


def sensitive_columns(columns) -> list[str]:
    """Column names that look like personal data (password, email, street, names...)."""
    return [c for c in columns if token_set(str(c)) & SENSITIVE_TOKENS]

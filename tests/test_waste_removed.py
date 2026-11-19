"""Repeated work removed in Batch 2 does not change results."""
import pandas as pd

from analytics.kpis import DataFilter, _Ctx
from config.analytics_config import DEFAULT_ANALYTICS
from ingestion.profiler import profile_dataframe
from ingestion.schema_detector import detect_column
from semantic.semantic_model import build_semantic_model


def _ctx():
    df = pd.DataFrame({"lead_time_days": [3.0, 4, 5] * 20, "notes": ["alpha beta"] * 60, "customer_email": ["a@b.c"] * 60, "units": range(60)})
    m = build_semantic_model(df, profile_dataframe(df), {"lead_time_days": "unknown", "notes": "unknown", "customer_email": "unknown"})
    return _Ctx(df, m, DEFAULT_ANALYTICS, DataFilter())


def test_name_filter_gives_the_same_list_as_filtering_afterwards():
    ctx = _ctx()
    pred = lambda t: "lead" in t and "time" in t                      # noqa: E731
    assert ctx.rank("unknown", numeric=True, name_filter=pred) == [c for c in ctx.rank("unknown", numeric=True) if pred(ctx.tokens(c))]


def test_columns_the_name_filter_rejects_are_never_parsed_as_numbers():
    ctx = _ctx()
    seen = []
    real = ctx.num
    ctx.num = lambda c: (seen.append(c), real(c))[1]
    ctx.rank("unknown", numeric=True, name_filter=lambda t: "lead" in t)
    assert seen == ["lead_time_days"]


def test_detect_column_gives_the_same_answer_with_or_without_a_precomputed_dropna():
    for s in (pd.Series(["1", "2", None, "3"] * 30, name="a"), pd.Series(["2024-01-01", None] * 40, name="d"),
              pd.Series(["red", "blue", None] * 30, name="c"), pd.Series(range(100), name="n")):
        assert detect_column(s) == detect_column(s, nn=s.dropna())

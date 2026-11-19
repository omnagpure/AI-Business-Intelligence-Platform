"""Insights page rendered with real Streamlit (AppTest)."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = str(Path(__file__).resolve().parent.parent)


def _page(with_data: bool) -> None:
    import numpy as np
    import pandas as pd

    from ingestion.pipeline import ingest_bytes
    from ui import state
    from ui.insights_page import render_insights_page
    if with_data and state.get_result() is None:
        rng = np.random.default_rng(1)
        n = 200
        df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=n),
                           "units_sold": np.array([10, 12, 14, 30, 50, 80, 20])[np.arange(n) % 7] + rng.normal(0, 2, n),
                           "stockout_flag": ([1] * 20 + [0] * 180), "note": [None] * 120 + ["x"] * 80})
        state.set_result(ingest_bytes(df.to_csv(index=False).encode(), "inv.csv", max_bytes=10_000_000))
    render_insights_page()


def _run(with_data=True):
    return AppTest.from_function(_page, args=(with_data,), default_timeout=180).run()


def test_page_without_data_shows_notice():
    at = _run(False)
    assert not at.exception


def test_page_shows_findings_with_evidence_and_advanced_button():
    at = _run()
    assert not at.exception
    assert [m.label for m in at.metric] == ["High", "Medium", "Low", "Info"]
    assert any("missing values" in str(m.value) for m in at.markdown)               # 'note' is 40% empty
    assert any(b.label == "Include advanced analyses" for b in at.button)


def test_advanced_analyses_add_forecast_findings_and_never_crash():
    at = _run()
    [b for b in at.button if b.label == "Include advanced analyses"][0].click().run()
    assert not at.exception
    groups = " ".join(str(m.value) for m in at.markdown if "eyebrow" in str(m.value))
    assert "Anomalies ·" in groups                                                   # the advanced anomaly run added a finding

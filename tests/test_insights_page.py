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


def _text(at) -> str:
    return " ".join(str(m.value) for m in at.markdown) + " ".join(h.proto.body for h in at.get("html"))


def test_page_runs_everything_on_open_and_reads_like_a_briefing():
    at = _run()
    assert not at.exception, [e.value for e in at.exception]
    assert not any(b.label == "Include advanced analyses" for b in at.button)          # no button: all analyses run on open
    text = _text(at)
    assert "We found" in text and "findings in your data" in text                      # the summary sentence
    for label in ("Needs attention", "Opportunities", "Patterns", "Data notes"):       # summary cards
        assert label in text
    assert "missing values" in text                                                    # 'note' is 40% empty
    assert "Action plan" in text and "Why it matters" in text and "What to do" in text
    assert "Anomalies" in text                                                         # the advanced anomaly run added a finding
    assert any(b.label == "Download action plan" for b in at.download_button)


def test_filter_shows_only_the_chosen_kind():
    at = _run()
    ctl = {c.key: c for c in at.segmented_control}["ins_filter"]
    ctl.set_value("data").run()
    assert not at.exception, [e.value for e in at.exception]
    cards = [m.value for m in at.markdown if "ins-head" in str(m.value) and "ins-n" not in str(m.value)]
    assert cards and all("Data note" in c for c in cards)

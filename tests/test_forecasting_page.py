"""Forecasting page rendered with real Streamlit (AppTest)."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = str(Path(__file__).resolve().parent.parent)


def _page(days: int) -> None:
    import numpy as np
    import pandas as pd

    from ingestion.pipeline import ingest_bytes
    from ui import state
    from ui.forecasting_page import render_forecasting_page
    if state.get_result() is None:
        rng = np.random.default_rng(1)
        d = pd.date_range("2024-01-01", periods=days)
        csv = pd.DataFrame({"date": d, "units_sold": np.array([10, 12, 14, 30, 50, 80, 20])[np.arange(days) % 7]
                            + rng.normal(0, 2, days)}).to_csv(index=False).encode()
        state.set_result(ingest_bytes(csv, "inv.csv", max_bytes=10_000_000))
    render_forecasting_page()


def _run(days):
    at = AppTest.from_function(_page, args=(days,), default_timeout=120)
    at.session_state["_unused"] = 0
    return at.run()


def _html(at) -> str:
    return " ".join(str(getattr(h, "body", "")) for h in at.get("html")) + " ".join(m.value for m in at.markdown)


def test_page_forecasts_on_open_and_shows_real_numbers():
    at = _run(200)
    assert not at.exception
    text = _html(at)
    assert "is forecast at" in text and "Typical error in testing" in text and "Next day" in text
    assert [t.label for t in at.tabs] == ["Forecast values", "Model comparison", "How this forecast was made"]
    values = at.dataframe[0].value
    assert list(values.columns) == ["Period", "Forecast", "Low", "High"] and len(values) == 14
    assert (values["Low"] <= values["Forecast"]).all() and (values["Forecast"] <= values["High"]).all()


def test_page_explains_when_data_is_too_short():
    at = _run(20)
    assert not at.exception and not at.dataframe
    assert any("at least" in m.value for m in at.markdown)

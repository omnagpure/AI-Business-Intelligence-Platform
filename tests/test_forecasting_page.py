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


def test_page_runs_a_forecast_and_shows_real_numbers():
    at = _run(200)
    assert not at.exception
    next(b for b in at.button if b.key == "fc_run").click().run()
    assert not at.exception
    m = {x.label: x.value for x in at.metric}
    assert {"Selected model", "Validation MAE", "Baseline MAE"} <= set(m)
    assert [t.label for t in at.tabs] == ["Model comparison", "Forecast values", "Method and limitations"]


def test_page_explains_when_data_is_too_short():
    at = _run(20)
    next(b for b in at.button if b.key == "fc_run").click().run()
    assert not at.exception and not at.metric
    assert any("at least" in m.value for m in at.markdown)

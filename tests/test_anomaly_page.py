"""Anomaly page rendered with real Streamlit (AppTest)."""
from streamlit.testing.v1 import AppTest


def _page() -> None:
    import numpy as np
    import pandas as pd

    from ingestion.pipeline import ingest_bytes
    from ui import state
    from ui.anomaly_page import render_anomaly_page
    if state.get_result() is None:
        rng = np.random.default_rng(1)
        n = 300
        units = np.array([10, 12, 14, 30, 50, 80, 20])[np.arange(n) % 7] + 100 + rng.normal(0, 2, n)
        units[120] += 200
        csv = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=n), "units_sold": units,
                            "unit_price": rng.normal(20, 2, n)}).to_csv(index=False).encode()
        state.set_result(ingest_bytes(csv, "inv.csv", max_bytes=10_000_000))
    render_anomaly_page()


def test_page_screens_records_and_periods_on_open():
    at = AppTest.from_function(_page, default_timeout=120).run()
    assert not at.exception
    text = " ".join(m.value for m in at.markdown) + " ".join(str(getattr(h, "body", "")) for h in at.get("html"))
    assert "records" in text and "High agreement" in text
    # the planted spike (day 121) is flagged as an unusual period
    assert "fall outside the expected range" in text
    periods = at.dataframe[-1].value
    assert str(periods.iloc[0]["Period"]) == "2024-04-30" and periods.iloc[0]["Direction"] == "High"

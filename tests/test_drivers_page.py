"""Driver analysis page rendered with real Streamlit (AppTest)."""
from streamlit.testing.v1 import AppTest


def _page() -> None:
    import numpy as np
    import pandas as pd

    from ingestion.pipeline import ingest_bytes
    from ui import state
    from ui.drivers_page import render_drivers_page
    if state.get_result() is None:
        rng = np.random.default_rng(1)
        n = 300
        price, promo = rng.normal(20, 3, n), rng.choice(["yes", "no"], n)
        units = 100 - 2 * price + 15 * (promo == "yes") + rng.normal(0, 2, n)
        csv = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=n), "unit_price": price, "promotion": promo,
                            "temperature": rng.normal(15, 5, n), "units_sold": units}).to_csv(index=False).encode()
        state.set_result(ingest_bytes(csv, "demand.csv", max_bytes=10_000_000))
    render_drivers_page()


def test_page_runs_only_on_request_then_shows_the_model_check():
    at = AppTest.from_function(_page, default_timeout=180).run()
    assert not at.exception
    assert not any("Model check" in m.value for m in at.markdown)
    {b.key: b for b in at.button}["dr_run"].click().run()
    assert not at.exception
    assert any("Model check on held-out rows" in m.value for m in at.markdown)

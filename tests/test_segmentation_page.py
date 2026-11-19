"""Segmentation page rendered with real Streamlit (AppTest)."""
from streamlit.testing.v1 import AppTest


def _page() -> None:
    import pandas as pd

    from ingestion.pipeline import ingest_bytes
    from ui import state
    from ui.segmentation_page import render_segmentation_page
    if state.get_result() is None:
        rows = []
        for i in range(14):
            good = i < 7
            for j in range(10):
                o = pd.Timestamp("2024-01-01") + pd.Timedelta(days=7 * j)
                late = (not good) and j < 7
                rows.append({"supplier_id": f"SUP-{i:03d}", "ordered_at": o, "expected_at": o + pd.Timedelta(days=14),
                             "received_at": o + pd.Timedelta(days=17 if late else 12), "quantity_ordered": 100,
                             "quantity_received": 90 if late else 100, "order_value": 1000 + 40 * i})
        state.set_result(ingest_bytes(pd.DataFrame(rows).to_csv(index=False).encode(), "po.csv", max_bytes=10_000_000))
    render_segmentation_page()


def test_page_runs_only_on_request_then_shows_clusters():
    at = AppTest.from_function(_page, default_timeout=120).run()
    assert not at.exception
    assert not any("clusters" in m.value for m in at.markdown)                  # nothing runs before the button
    {b.key: b for b in at.button}["seg_run"].click().run()
    assert not at.exception
    assert any("clusters" in m.value and "silhouette" in m.value for m in at.markdown)

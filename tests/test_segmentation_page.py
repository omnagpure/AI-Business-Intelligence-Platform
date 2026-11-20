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


def test_page_segments_on_open_and_names_the_groups():
    at = AppTest.from_function(_page, default_timeout=120).run()
    assert not at.exception
    lead = " ".join(m.value for m in at.markdown)
    assert "The 14 suppliers fall into 2 segments" in lead and "The split is clear" in lead
    cards = " ".join(str(h.proto.body) for h in at.get("html"))
    assert "Lower on-time rate" in cards and "Higher on-time rate" in cards       # named by their real difference
    profile = at.dataframe[0].value
    assert list(profile["Segment"])[-1] == "All" and "On-time rate" in profile.columns

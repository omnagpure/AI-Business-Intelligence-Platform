import pandas as pd

from ingestion.pipeline import ingest_bytes, ingest_sample
from ingestion.sample import make_sample_sales


def test_ingest_bytes_end_to_end():
    csv = ("id,amount,when\n" + "\n".join(f"{i},{i * 2.5},2024-01-{(i % 28) + 1:02d}" for i in range(1, 61))).encode()
    r = ingest_bytes(csv, "sales.csv", max_bytes=10**6)
    assert r.profile.rows == 60 and r.profile.columns == 3
    assert r.profile.column("amount").numeric["max"] == 150.0
    assert r.quality.completeness_pct == 100.0


def test_sample_is_labelled_deterministic_and_messy():
    a, b = make_sample_sales(), make_sample_sales()
    pd.testing.assert_frame_equal(a, b)
    r = ingest_sample()
    assert r.dataset.is_sample and r.profile.duplicate_rows == 5
    assert any(i.code == "inconsistent_labels" for i in r.quality.issues)

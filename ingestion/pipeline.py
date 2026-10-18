"""Single entry point: bytes -> loaded dataset + profile + quality report. No Streamlit."""
from __future__ import annotations

from dataclasses import dataclass

from ingestion.file_loader import LoadedDataset, fingerprint_bytes, load_dataset
from ingestion.profiler import DatasetProfile, profile_dataframe
from ingestion.quality_report import QualityReport, build_quality_report
from ingestion.sample import make_sample_sales


@dataclass
class IngestionResult:
    dataset: LoadedDataset
    profile: DatasetProfile
    quality: QualityReport


def ingest_bytes(data: bytes, filename: str, *, max_bytes: int, sheet: str | None = None) -> IngestionResult:
    ds = load_dataset(data, filename, max_bytes=max_bytes, sheet=sheet)
    profile = profile_dataframe(ds.df)
    return IngestionResult(ds, profile, build_quality_report(ds.df, profile, ds.warnings))


def ingest_sample() -> IngestionResult:
    df = make_sample_sales()
    ds = LoadedDataset(df=df, filename="sample_sales (synthetic)", file_type="text",
                       fingerprint=fingerprint_bytes(b"sample_sales_v1"), is_sample=True)
    profile = profile_dataframe(df)
    return IngestionResult(ds, profile, build_quality_report(df, profile, ds.warnings))

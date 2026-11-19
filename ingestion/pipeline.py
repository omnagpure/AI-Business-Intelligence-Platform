"""Single entry point: bytes -> loaded dataset + profile + quality report. No Streamlit."""
from __future__ import annotations

from dataclasses import dataclass

from core.progress import ProgressFn, notify
from ingestion.file_loader import LoadedDataset, fingerprint_bytes, load_dataset
from ingestion.profiler import DatasetProfile, profile_dataframe
from ingestion.quality_report import QualityReport, build_quality_report
from ingestion.sample import make_sample_sales


@dataclass
class IngestionResult:
    dataset: LoadedDataset
    profile: DatasetProfile
    quality: QualityReport


def ingest_bytes(data: bytes, filename: str, *, max_bytes: int, sheet: str | None = None,
                 progress: ProgressFn | None = None) -> IngestionResult:
    """Read, profile and quality-check an upload. `progress` (core.progress) hears three stages; column profiling reports per column."""
    notify(progress, 0, 3, "Reading the file (no partial progress)")
    ds = load_dataset(data, filename, max_bytes=max_bytes, sheet=sheet)
    notify(progress, 1, 3, "Profiling columns")
    profile = profile_dataframe(ds.df, progress=None if progress is None else (lambda i, n, m: notify(progress, 1, 3, m)))
    notify(progress, 2, 3, "Checking data quality")
    quality = build_quality_report(ds.df, profile, ds.warnings)
    notify(progress, 3, 3, "Done")
    return IngestionResult(ds, profile, quality)


def ingest_sample() -> IngestionResult:
    df = make_sample_sales()
    ds = LoadedDataset(df=df, filename="sample_sales (synthetic)", file_type="text",
                       fingerprint=fingerprint_bytes(b"sample_sales_v1"), is_sample=True)
    profile = profile_dataframe(df)
    return IngestionResult(ds, profile, build_quality_report(df, profile, ds.warnings))

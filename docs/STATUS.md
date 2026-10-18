# Project status

| Phase | State |
|---|---|
| 1 Foundation | Done |
| 2 Ingestion + profiling | Done (59 automated tests passed in author's sandbox; re-run locally) |
| 3 Cleaning | Not started |

## Phase 2 files
| Path | Purpose |
|---|---|
| ingestion/file_loader.py | CSV/XLSX validation + parsing, encoding fallback, comment-line skipping, sheet listing |
| ingestion/schema_detector.py | Column kind inference (numeric, date, id, text, ...), fast date parsing |
| ingestion/profiler.py | Column and dataset profile (never mutates input) |
| ingestion/quality_report.py | Evidence-backed quality issues with severity |
| ingestion/pipeline.py | bytes -> dataset + profile + quality |
| ingestion/sample.py | Synthetic sample data |
| config/analytics_config.py | Quality thresholds |
| ui/state.py, upload_page.py, profile_page.py, data_quality_page.py | Streamlit pages |

Design change: session-state wrapper lives in `ui/state.py` (not `core/`) so `core/` stays Streamlit-free.

## Verified on real files (sandbox)
Manufacturing sample (comment header), DataCo (cp1252, 180,519 x 53), inventory xlsx (18,000 x 13),
supply chain CSV (91,250 x 15): all load and profile; originals unchanged.

## Not verified
Browser appearance on Windows, upload through the real file widget, pandas 2.x.

## Update: broader input support
.tsv/.txt/.xls/.xlsm accepted; title rows above headers skipped; decimal commas; UTF-16 text. Legacy .xls verified via xlwt-generated file in sandbox only.

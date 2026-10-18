# Project status

| Phase | State |
|---|---|
| 1 Foundation | Done |
| 2 Ingestion + profiling | Done (59 automated tests passed in author's sandbox; re-run locally) |
| 3 Cleaning | Implemented; 22 tests pass in author's sandbox (UI not run, see below) |
| 4 Semantics (batches A+B) | Implemented; 15 tests pass in author's sandbox (UI not run). Batch C (trained model) not started |
| UI redesign | Applied (theme, CSS design system, grouped navigation, Semantics page) |

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

## Phase 3 files
| Path | Purpose |
|---|---|
| cleaning/actions.py | CleaningAction (what/why/rows/consequence/risk), LogEntry, helpers |
| cleaning/type_inference.py | text -> number / date / boolean proposals and operations |
| cleaning/labels.py | inconsistent labels, whitespace, column-name normalisation |
| cleaning/missing_values.py | median / 'Unknown' fill, drop rows/columns (never zero-fill, never default-on) |
| cleaning/duplicate_handler.py | exact and near-duplicate rows (keep first, original index kept) |
| cleaning/outlier_handler.py | IQR cap/remove, negatives-to-missing (never default-on) |
| cleaning/cleaner.py | propose_actions, apply_actions, select_with_dependencies, actions_fingerprint |
| ui/state.py, ui/data_quality_page.py, ui/sidebar.py | Cleaning + History tabs, active-dataset switch |
| tests/test_cleaning.py | 22 tests |

Later phases should read data via `ui.state.get_analysis_df()` (cleaned data if applied, else original).

## Phase 3 not verified
Streamlit rendering of the Cleaning/History tabs, data_editor behaviour, download buttons, pandas 2.x.
Known limits: Profile page still shows the original data's profile; outliers use a fixed IQR rule (k=3);
near-duplicate scan skipped above 3M cells.

## Phase 4 files (batches A and B)
| Path | Purpose |
|---|---|
| semantic/column_lexicon.py | roles, name vocabulary, domain vocabulary, tokeniser |
| semantic/column_classifier.py | rule scoring: name + kind + value distribution + column relationships; optional NameModel hook |
| semantic/business_domain.py | domain detection with confidence, evidence, alternatives; falls back to "general" |
| semantic/semantic_model.py | SemanticModel, user overrides (roles recompute domain), fingerprint for cache keys |
| ui/semantic_page.py, ui/state.py, app.py | Semantics page, cached model keyed by data + cleaning + overrides |
| ui/layout.py, .streamlit/config.toml, assets/logo.svg | design system and theme |
| tests/test_semantics.py | 15 tests |

Roles added beyond the spec list: `price` (unit price) and `duration` (days/hours/tenure).
Downstream code should call `ui.state.get_semantic_model()` and use `columns_with_role()`.

## Phase 4 / redesign not verified
Streamlit rendering of the Semantics page and the new CSS (stat strips, sidebar, tabs), the Inter web font
(needs internet, falls back to system fonts), `st.logo` with SVG, data_editor save flow, narrow-screen layout.
Known limits: scores are heuristics; domain for supply-chain data may read as Retail or Inventory
(override available); dark theme only.

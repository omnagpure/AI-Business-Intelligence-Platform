# Project status

| Phase | State |
|---|---|
| 1 Foundation | Done |
| 2 Ingestion + profiling | Done (59 automated tests passed in author's sandbox; re-run locally) |
| 3 Cleaning | Implemented; 22 tests pass in author's sandbox (UI not run, see below) |
| 4 Semantics (A+B+C) | Implemented. Rules, domain, overrides, optional trained name model with held-out evaluation. 25 semantic/training tests pass in author's sandbox (UI not run in a browser) |
<<<<<<< HEAD
=======
| 5 Descriptive analytics + KPIs | Implemented. 37 new tests; logic checked against raw pandas on all four real datasets (see below). Dashboard/Explorer not yet run in a real browser |
>>>>>>> 1687b64 (Phase 5 analytics parts and some minor changes)
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

## Phase 4 batch C (trained name model) and UI update
| Path | Purpose |
|---|---|
| training/make_data.py | hand-written seed names, variants, train/test split by base name, hand-labelled real headers |
| training/train_semantic_model.py | TF-IDF (word + char n-grams) + logistic regression; C chosen by grouped CV on training data only |
| training/evaluate_semantic_model.py | held-out evaluation, name-only rules baseline, optional end-to-end check on your files |
| semantic/trained_name_model.py | trusted loader: checksum + version checks, lazy sklearn import, safe fallback |
| ui/overview_page.py | new home page: counts, quality, workflow strip, detected roles (no invented KPIs) |
| ui/semantic_page.py | tabs: Column roles, Review queue (one-click fixes), Role mix, Trained model |
| ui/settings_page.py, ui/state.py, ui/layout.py, app.py | model status/toggle, cache key includes model version, new CSS helpers, Overview is the default page |
| tests/test_semantic_training.py | 10 tests |

<<<<<<< HEAD
### Evaluation result in the author's sandbox (seed 42; your retrain may differ slightly)
Held-out synthetic names: model 65.8% vs rules-by-name 67.9%. Hand-labelled real headers (n=73): both 80.8%.
End to end on the four supply-chain files: rules 95.9% vs rules+model 97.3% (one column changed). Conclusion: the
model is about as good as the rules, not clearly better, so it ships OFF by default. The baseline vocabulary was
written by the same author who labelled the real headers, which flatters the rules.
=======
### Evaluation result after expanding the training names (author's sandbox, seed 42)
Training names: 1,246 distinct base names (was 367), with short forms, prefixes, suffixes and varied separators.
| Test set | Trained model | Rules by name only |
|---|---|---|
| Held-out synthetic names (n=5,507) | 86.0% | 57.9% |
| Names from industries not in training (n=1,685) | 85.8% | 59.5% |
| Hand-labelled real headers (n=73) | 93.2% | 80.8% |
| End to end on 4 supply-chain files (rules+values vs +model) | 97.3% | 95.9% (1 column changed) |

Read with care: (1) the synthetic test names are generated by the same variation process as the training names
(same abbreviation list and affixes), so the model's lead there is flattered, while the rules were not built for
abbreviated names. (2) The seed names were written by the same author who labelled the real headers and had seen
those headers, so the real-header score is optimistic. (3) The blend weight is 0.30, so the model mostly shifts
confidence and rarely overturns a confident rule. The honest test is your own new files.
>>>>>>> 1687b64 (Phase 5 analytics parts and some minor changes)

## Phase 4 batch C not verified
Streamlit rendering of the new pages (only a stubbed-Streamlit smoke test was run), tests/test_app_startup.py
(needs Streamlit; the sample-data test now relies on the Overview page), training on Windows, scikit-learn versions other than the sandbox's.
<<<<<<< HEAD
=======


## Phase 5 files
| Path | Purpose |
|---|---|
| analytics/descriptive.py | numeric/categorical/date summaries, time aggregation (partial periods flagged), grouped summaries, cross-tabs, Top-N + Other |
| analytics/correlation.py | correlation of suitable numeric columns; constant and sparse columns excluded and listed; top pairs |
| analytics/kpis.py | role-driven KPI engine (KPIResult, DataFilter, cache_key); a KPI exists only when its roles exist |
| analytics/formatting.py | KPI value display (no currency symbol is assumed) |
| visualization/charts.py | Plotly builders; NaN/inf cleaned, categories capped, scatter sampled |
| ui/analytics_state.py | session cache keyed on dataset + cleaning + roles + overrides + parameters |
| ui/dashboard_page.py, ui/explorer_page.py | Dashboard (KPIs, trend, breakdown, distribution, correlation) and Data explorer (100 rows per page) |
| config/analytics_config.py | AnalyticsConfig added |
| tests/test_descriptive.py, test_kpis.py, test_charts.py | hand-calculated expectations |

KPI rules: margin = SUM(profit)/SUM(revenue); zero denominator gives value None with a "Not calculable" limitation;
growth compares only complete periods and is skipped when the comparison total is zero/negative or missing;
identifier, customer, product, category, location and date columns are never summed or averaged;
inventory KPIs use the latest snapshot only; estimated revenue (units x unit price) appears only if no revenue column exists.

## Phase 5 verified (sandbox, Python 3.12, pandas 3.0)
Full suite except tests/test_app_startup.py: 141 passed, 2 skipped (plotly, xlwt not installed there).
KPI values cross-checked against independent pandas on DataCo (revenue, orders, customers, profit, margin, Jan-2018 growth),
supply_chain_dataset1 (stock, stock value, estimated revenue) and the inventory xlsx (growth, category share).
Dashboard and Explorer page code was executed against a stand-in for Streamlit on all four real files, including a role
override and filters (filtered totals matched raw pandas).

## Phase 5 not verified
Real Streamlit rendering, real Plotly figures (builders ran only against a recording stand-in), tests/test_app_startup.py,
pandas 2.x. Known limits: roles come from heuristics (e.g. 'on_time_delivery' in the manufacturing sample is labelled
identifier); 'Top category' uses whichever category column ranks first (for the inventory file that is 'status').
>>>>>>> 1687b64 (Phase 5 analytics parts and some minor changes)

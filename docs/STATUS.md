# Project status

| Phase | State |
|---|---|
| 1 Foundation | Done |
| 2 Ingestion + profiling | Done (59 automated tests passed in author's sandbox; re-run locally) |
| 3 Cleaning | Implemented; 22 tests pass in author's sandbox (UI not run, see below) |
| 4 Semantics (A+B+C) | Implemented. Rules, domain, overrides, optional trained name model with held-out evaluation. 25 semantic/training tests pass in author's sandbox (UI not run in a browser) |
| 5 Descriptive analytics + KPIs | Implemented. 37 new tests; logic checked against raw pandas on all four real datasets (see below). Dashboard/Explorer not yet run in a real browser |
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
| ui/layout.py, .streamlit/config.toml, assets/logo-light.svg, assets/logo-dark.svg | design system and theme |
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

## Phase 4 batch C not verified
Streamlit rendering of the new pages (only a stubbed-Streamlit smoke test was run), tests/test_app_startup.py
(needs Streamlit; the sample-data test now relies on the Overview page), training on Windows, scikit-learn versions other than the sandbox's.


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

## Batch A: supply-chain semantics (after Phase 5)
- New roles: supplier, warehouse, flag (yes/no status such as on-time, stockout, delayed). New domain: supply_chain.
- Yes/No status columns are no longer labelled identifier; two-valued 0/1 columns are not read as percentages.
- New roles are never summed or averaged (descriptive + KPI engine).
- Name model retrained (seed 42). Held-out synthetic acc 0.844 / macro-F1 0.818 (was 0.860 / 0.864: new roles have few seed names, warehouse recall is low). Real headers (n=73) acc 0.932, unchanged. Synthetic and hand-labelled names only.
- Tests: 152 passed, 1 skipped (was 144). Not verified: the four real datasets (not available in the sandbox), real Streamlit rendering.
- Next: Batch B, supply-chain KPIs (OTIF, fill rate, lead time, stockout, defect, turnover).

## Batch B: supply-chain KPIs and data-validity guards
New: `analytics/supply_chain_kpis.py` (on-time rate, average delay, lead time, fill rate, OTIF, stockout rate,
share of stock rows below reorder point, days of cover, inventory turnover, defect rate, top-1 and top-3 supplier
share, purchase spend), `analytics/kpi_relevance.py` (headline KPIs by domain; hidden KPIs stay listed),
`analytics/data_checks.py` (flag parsing, personal-data column names), `tests/test_supply_chain_kpis.py` (21 tests,
hand-calculated).
Changed: `analytics/kpis.py` (runs the new builders; procurement data shows spend instead of revenue, cancelled
orders excluded; status columns are no longer a revenue "category"; growth skipped when one period has more than
10x the rows of the other; report notes list personal-data columns and "Data check" findings),
`config/analytics_config.py` (`growth_max_row_ratio`), `ui/dashboard_page.py` (top-6 cards, more/unavailable groups).
Checked on the four real files against raw pandas: PO workbook on-time 22.28%, delay 5.49 days, lead time 37.02
days, fill rate 96.15%, OTIF 20.49%.
Not verified: Streamlit rendering of the new dashboard section; `tests/test_app_startup.py`; run `pytest` locally.
Known data problems found (not code bugs): the manufacturing sample's `on_time_delivery` holds placeholder text and
12 of 25 rows have defects above units; the inventory CSV's `Stockout_Flag` is all zero.
Next: Batch C (supply-chain charts, suggested views with captions, primary-column picker).

## Batch C: suggested views and primary-column picker
New: `analytics/supply_chain_views.py` (up to 4 views chosen from roles: lowest on-time rate by supplier or shipping
mode, lead-time distribution, stock vs reorder point, defect rate over time or by group, supplier share, plus general
trend and category share; each caption computed from the plotted table), `analytics/primary_columns.py`
(candidates and override editing), `tests/test_supply_chain_views.py` (13 tests, hand-calculated captions).
Changed: `visualization/charts.py` (`multi_line_chart`), `analytics/supply_chain_kpis.py` (`lead_days` and
`supplier_shares` helpers shared with the views), `ui/dashboard_page.py` (Suggested views section, Primary columns
picker that records a role override so KPIs and charts refresh).
Checked: the dashboard KPI and views sections run on all four real files against a stand-in Streamlit and Plotly.
Not verified: real browser rendering, `tests/test_app_startup.py`, real Plotly figures. Run `pytest` and
`streamlit run app.py` locally.

## Phase 6A: Forecasting
| Path | Purpose |
|---|---|
| analytics/forecasting.py | Series preparation, seasonality check, 8 candidate models, rolling-origin validation, simplest-within-margin selection, empirical intervals |
| ui/forecasting_page.py | Forecasting page (Analyze group) |
| visualization/charts.py | `forecast_chart` added |
| core/exceptions.py, config/analytics_config.py, requirements.txt, app.py | `ForecastingError`, forecast settings, statsmodels, page registered |
| tests/test_forecasting.py, tests/test_forecasting_page.py | 19 tests incl. real Streamlit AppTest |

Run in author's sandbox (Python 3.12, pandas 3, statsmodels 0.15, Streamlit, Plotly): full suite 203 passed + the new page tests. Not run on the four real datasets (files not available in the sandbox) or on Windows.

## Phase 6B: Anomaly detection
| Path | Purpose |
|---|---|
| analytics/anomaly_detection.py | Median/MAD, IQR and z-score rules (z-score only where roughly normal), Isolation Forest on suitable numeric features (fixed seed, missing rows left out), unusual-period detection against a robust baseline, feature comparison |
| ui/anomaly_page.py | Anomalies page (sensitivity 1-5; Unusual values and Unusual periods tabs) |
| visualization/charts.py | `anomaly_series_chart` added |
| core/exceptions.py, config/analytics_config.py, app.py | `AnomalyDetectionError`, anomaly settings, page registered |
| tests/test_anomaly_detection.py, tests/test_anomaly_page.py | hand-calculated rules, planted outliers, sensitivity, reproducibility, independence, real Streamlit AppTest |

Nothing is deleted or changed; every finding carries method, score, threshold, direction and a plain explanation with the "not necessarily an error" note. Not run on the four real datasets (not available in the author's sandbox).

## Phase 6C: Segmentation
| Path | Purpose |
|---|---|
| analytics/segmentation.py | Entity candidates (supplier, customer, product, warehouse), one-row-per-entity features (on-time rate, delay, lead time, fill rate, defect rate, stockout rate, value, volume, other measures by role), explicit missing/constant/duplicate handling, KMeans with elbow and silhouette, smallest-cluster guard, plain-words cluster traits, fixed seed |
| ui/segmentation_page.py | Segmentation page (runs only on button press; feature, scaling and K choices) |
| visualization/charts.py | `segment_scatter` added |
| tests/test_segmentation.py, tests/test_segmentation_page.py | hand-calculated entity features, cluster recovery, reproducibility, guard rails, AppTest |

## Phase 6D: Driver analysis
| Path | Purpose |
|---|---|
| analytics/drivers.py | Random Forest on a held-out split (chronological when a date exists), trivial-baseline comparison, permutation importance on held-out rows, leakage rules (name overlap, near-copy correlation, categorical copy, profit/revenue ties), reliability flag, dominance warning, group tables, association-not-causation note |
| ui/drivers_page.py | Drivers page (target and column choice; runs on button press) |
| ui/analysis_runner.py | Shared cached wrapper that turns failures into messages |
| core/exceptions.py, config/analytics_config.py, app.py | `SegmentationError`, `DriverAnalysisError`, settings, pages registered |
| tests/test_drivers.py, tests/test_drivers_page.py | importance ranking, leakage, chronological split, noise flagged unreliable, classification, reproducibility, AppTest |

Run on the four real files in the author's sandbox (no Streamlit or statsmodels there): PO workbook supplier segmentation gives
weak structure (silhouette 0.23), which the page reports as weak; driver analysis on the real files runs in 1 to 7 seconds.
Not verified: the two new pages in real Streamlit (tests/test_*_page.py need it), real Plotly figures.

## Phase 7: Insights and recommendations
| Path | Purpose |
|---|---|
| insights/evidence.py | `Evidence` and `Insight` types; an insight cannot exist without evidence |
| insights/insight_rules.py | 14 deterministic rules over KPI results, the data profile and Phase 6 results (growth, category share, supply-chain KPI levels, margin, missing KPIs, missing values, duplicates, tiny data, forecast direction and reliability, anomaly share, small high-value segments, drivers) |
| insights/prioritization.py | priority = 10 x severity weight + confidence weight; stable sort |
| insights/insight_engine.py | runs every rule independently; rules without inputs or that fail are listed as skipped with a reason |
| insights/default_runs.py | default-setting runs of forecast, anomalies, segmentation, drivers (used by the page) |
| analytics/recommendations.py | template recommendations, each linked to an insight id |
| config/analytics_config.py | `InsightConfig` (all thresholds, shown on each insight; adjustable reference levels, not industry standards) |
| ui/insights_page.py, app.py | Insights page (severity/group filters, evidence, limitations, skipped rules); advanced analyses run on button press |
| tests/test_insights.py, tests/test_insights_page.py | hand-calculated rule outcomes, determinism, evidence and recommendation links, failure isolation, AppTest |

Tests: 275 passed, 1 skipped (Python 3.12, pandas 3.0.6, Streamlit 1.67 in the author's sandbox). Run on supply_chain_dataset1.csv:
forecast +13% (checked by hand: 5,332.7 vs 4,729.6), no KPI crossed a threshold, drivers insight flags that Demand_Forecast
holds about 100% of importance for Units_Sold (a near-copy of the target, shown as low confidence). Not run on the other three datasets.
No LLM layer (optional, off by default, not built). Known limit: insights use default analysis settings, not the settings chosen on the other pages.

### Phase 7 addendum: data-pattern insights
`analytics/data_findings.py` reads patterns straight from the data (group gaps, flag effects, weekday pattern, latest-month trend,
top-10% concentration, accuracy of a forecast column in the file) and 7 more rules turn them into insights (`group_gap`, `flag_effect`,
`weekday`, `trend`, `concentration`, `forecast_accuracy`, `cover`). `tests/test_data_findings.py` has hand-calculated cases (12 tests).
On supply_chain_dataset1.csv: promotion rows +28% units (pandas check 27.7%), lead time varies 85% across SKUs (check 85.2%),
Nov vs Oct units +23.2% (pandas check 23.25%; December is excluded as a partial month because the data ends on Dec 30). Group differences and flag effects are associations, not tests of significance.

## Phase 8: Reports and exports
| Path | Purpose |
|---|---|
| reporting/theme.py | Three-colour palette in a 70 / 25 / 5 ratio (paper / ink navy / accent orange); greys are mixes of these |
| reporting/report_builder.py | `build_report(...)` turns computed results into one renderer-neutral `ReportData`; skipped analyses carry their reason; each section fails on its own |
| reporting/charts.py | Matplotlib charts in the palette (bars, forecast with interval) |
| reporting/html_report.py | One self-contained HTML file (charts embedded, all text escaped, print styles) |
| reporting/pdf_report.py | ReportLab PDF: cover, contents with page numbers, running header/footer, bookmarks, DejaVu fonts bundled with Matplotlib |
| ui/reports_page.py, app.py | Reports page (title, prepared-for, options, downloads for PDF/HTML, analysed-data CSV on request, log CSV); reuses cached Insights-page results |
| core/exceptions.py, requirements.txt | `ReportGenerationError`; matplotlib and reportlab added |
| tests/test_reporting.py | 13 tests: values match KPI results and pandas, hand-calculated forecast lead, escaping, skipped sections, failure isolation |

Verified in the author's sandbox (no Streamlit, no statsmodels, no pytest, so tests ran through a small shim): 13/13 reporting tests pass; PDF and HTML
generated on all four real files (DataCo: PDF 1 s). Forecast sections were checked with a hand-built result
because statsmodels was missing. Not verified: the Reports page in real Streamlit, PDF in Acrobat.

## Phase 9, Batch 1: fixes and semantic layer
- **Vocabulary:** supply-chain terms (inventory, procurement, logistics, warehouse, manufacturing, quality) added to `semantic/column_lexicon.py`; short forms (supp, vndr, wh, shpmt, dlvry, qty ...), glued names (vendorid, supplierkey), and one-letter misspellings (suppllier, delivary, warehose) are repaired by `semantic_tokens`. The classifier and domain detection use it; other modules still use the plain `token_set`.
- **Generic headers (Amount, Value, Total, Code, Rate, Cost, Location, Reference, Time):** confidence capped at 45% so they are flagged for review. Neighbouring columns decide the meaning: purchasing/logistics neighbours point to cost, sales neighbours to revenue, none leaves it low-confidence.
- **Fixes:** inventory turnover is no longer revenue; image/URL columns are text, not products; first/last-name columns are treated as personal data and excluded; weight, distance and capacity are not forced into "quantity" (that would put them in the units KPI); duplicate concentration insights (Product Image / Product Name) are merged.
- **Insights and Reports reuse your own settings:** analyses run on the Forecasting, Anomalies, Segmentation and Drivers pages are reused when they exist for the current data; others use defaults. The report states which.
- **Model:** training data now has labelled supply-chain headers (half trained on, half held out), misspelling augmentation and more short forms. `evaluate_semantic_model` reports a supply-chain held-out line. Retrain locally: `python -m training.make_data`, `python -m training.train_semantic_model`, `python -m training.evaluate_semantic_model`.
- **Tests:** `tests/test_supply_chain_headers.py` (13 tests).

## Phase 9, Batch 3: supply-chain rules and the automatic name-model switch
**Decision:** the project targets supply-chain data, where the rules beat the name model (supply-chain held-out names:
model 72%, rules 97%). The model is therefore used automatically only where it helps.

- **Supply-chain detection** (`semantic/business_domain.py`): three or more supply-chain-only terms in the column names
  (`SUPPLY_CHAIN_TERMS` in `semantic/column_lexicon.py`; shared commerce words such as order/product/customer excluded)
  make the domain supply chain even when retail, operations or inventory vocabulary outnumbers it. **DataCo now reads as
  supply chain (75%, was retail 36%)**, which reverses the Batch 2 decision below; Superstore-style retail data (ship date
  and ship mode only) stays retail. Confidence for supply chain grows with the number of terms and supplier/warehouse
  roles (other three real files: 90%, 72%, 69%, were 37%, 32%, 52%). `assess_supply_chain` is the single definition used
  by the switch and by the supply-chain KPIs (`analytics/kpis.py`); a domain chosen by the user wins.
- **Supply-chain roles:** "order value"/"PO amount" on purchasing data is cost (spend), not revenue; `product_sku` is a
  product (sku counts as an id-style token next to an entity word); two-valued "... Status" columns are status flags
  (DataCo `Product Status`). The golden roles were refreshed on purpose; these were the only role changes on the four
  real files. Note: the hand label for `Product Status` in `training/make_data.py` (category) was left unchanged.
- **Automatic switch** (`semantic/semantic_model.py`, `MODEL_MODES`): Automatic (default) / Always on / Off. Rules run
  first; supply-chain data gets the model only for columns the rules left Unknown, other data gets it on every column.
  Where the rules found nothing the model's weight is 0.6 (else 0.3), and its vote for a role the values rule out (text
  as revenue, numbers as dates) is cut to a tenth (`_kind_allows`). Rule scoring runs once for both passes. The Semantics
  page shows the decision; Settings and the Trained model tab have the three-way choice.
- **Model:** 218 names added from marketing, SaaS, HR and finance; retrained (1-8a332044). `evaluate_semantic_model`
  now runs end to end as the app does (rules only vs Automatic) on labelled HR/marketing/SaaS/finance tables
  (`training/domain_frames.py`) and on the real supply-chain files. Results: other domains 68% -> 91% (n=44; headers not
  in the training names 60% -> 73%, n=15; the old model reached 70%); real supply-chain files 95.9% both (the model
  changed no role). On names alone the new model scores lower on real supply-chain headers (92%, was 97%), which
  Automatic mode does not use. `scikit-learn` pinned to 1.5.x so the saved model keeps loading.
- **Tests:** `tests/test_model_switch.py` (19). `tests/test_role_stability.py` now finds the real files in sub-folders
  and under slightly different names; with `BI_REAL_FILES_DIR` set, all four real files are checked.

## Phase 9, Batch 4: dashboard numbers and redesign
- **Cancelled shipments:** status values are matched by word ("Shipping canceled", "PENDING_PAYMENT"), and a delivery
  status is preferred. Cancelled shipments have no delivery outcome, so on-time and delay leave them out (DataCo on-time
  42.7%, the 7,754 cancelled rows were counted before). New KPI `cancel_rate` (DataCo 4.3%), only for supply-chain data.
- **New and renamed KPIs:** `avg_late_delay` (average delay of late rows only; DataCo 1.6 days). `avg_lead_time` falls
  back to an actual shipping-days column and is then called "Average shipping time" (DataCo 3.5 days). Day values use a
  new `days` unit. Stock KPIs say whether they read the latest date or all dates ("Items below reorder point (latest
  date)" 14 vs "Stock records below reorder point (all dates)" 5.2%), which used to look contradictory.
- **KPI history** (`analytics/kpi_trend.kpi_history`): every headline KPI is recomputed on each of the last 12 complete
  periods, so a sparkline means exactly what its card means (on-time rate per month, stock on each month's last date).
  Rates change in percentage points; changes under 0.5% (0.1 pts) read "No change". Growth KPIs get no history.
  Cost: about 1.5 s on DataCo, cached per filter.
- **Data-shape check** (`analytics/data_shifts.py`): a lasting x1.6 change near the end in rows per period or rows per
  order is reported above the cards. DataCo: from Oct 2017 rows per order go from 3.0 to 1.0, which explains its -34%
  revenue month.
- **Views:** stock vs reorder now shows the share of items below their own reorder point per date (summing stock and
  reorder points across items hid every shortage: "0 of 365 dates" became 5.2% on an average date, 14% at the peak).
  New on-time by market/region view; on-time bars carry the overall rate as a reference line and the caption says when
  groups hardly differ (under 2 pts). The trend view is not repeated (the Explore tab has it).
- **Design:** header with file, rows, date coverage and domain badge; comparison and primary-column settings in one
  popover; eight uniform KPI cards (value, change chip coloured by whether up is good, grey for neutral KPIs such as
  stock or supplier concentration, sparkline, footnote saying what period it covers); other KPIs grouped by area; one
  "How these numbers are calculated" panel; bar value labels, readable axis titles, hollow partial-period markers.
  Breakdown defaults to shipping mode / market for supply-chain data and formats its table. Light and dark themes.
- **Tests:** `tests/test_dashboard_numbers.py` (9); the stock-vs-reorder view test now checks that one item above its
  reorder point cannot hide another below it.

## Phase 9, Batch 6: segmentation page
- `analytics/segmentation.py`:
  - Entity columns that group the rows the same way (Customer Id / Order Customer Id, Product Card Id / Product Name) are offered once; the name is kept over a numeric id.
  - The count feature is "Orders" (distinct order ids) when an order spans several rows, else "Rows".
  - Delay and lead time carry a "days" unit, and lead time uses the shipping-time wording where that fits.
  - Traits need a clear gap (0.5 standard deviations) and a meaningful one (5% of the overall value, or 2 points for rates), so gaps such as +4% between near-identical warehouses are no longer reported as traits.
  - New `names` (a short name per segment from its strongest difference, preferring supply-chain measures; "more"/"fewer" for counts).
  - New `separation` (the share of each measure's variance that lies between the segments).
- `ui/segmentation_page.py` (rewrite):
  - Runs on open; setup card (what to group, number of segments, Options for measures and scaling).
  - Lede sentence with an honest fit rating. It says when the split is weak, when the segments are all close to average, or when they split by size rather than delivery performance.
  - One card per segment with its name, size and share of value.
  - Two charts: "how much each measure differs" and "share of entities vs share of value".
  - Formatted profile table with an "All" row.
  - Scatter with segment averages; large tables are thinned to about 4,000 points, and the chart says so.
  - Tabs: entities (with CSV download), choosing the number of segments, and measures used.
- Home card names the largest segment. The insight rule no longer treats days as a total.
- Tests: `tests/test_segmentation_numbers.py` (5). `test_segmentation_page.py` was rewritten for auto-run.

## Phase 9, Batch 5: forecasting and anomaly pages
- **KPI targets** (`analytics/derived_targets.py`): orders (distinct, not lines), on-time rate, average delay, shipping or
  lead time and cancelled rate can be forecast and checked per period like columns. The per-period values use the KPI
  engine's own definitions (DataCo Dec 2017 on-time: 42.294% both ways). Supply-chain measures only on supply-chain data.
- **Forecasting:** runs on open and on every setting change (cached). One-sentence answer, then four cards: next period
  with its 80% range, the horizon against the same span just before it (pts for rates), typical validation error as a
  share of the level (Reliable under 10%, Rough guide over 25%) and the model against the naive baseline
  (`summarise_forecast`). Rates are kept inside 0-100 (`bounds`). The chart joins the forecast to the last actual, marks
  where the forecast starts, draws each validation window, and marks a data change. Forecast tables are rounded.
- **Data changes:** `data_shifts` now searches about 8 months back whatever the period (it only looked 8 periods back,
  so weekly and daily data missed DataCo's Oct 2017 change) and requires the change to last 2 months / 4 weeks / 21 days.
  Forecasts that cross a change say how many periods come after it.
- **Unusual records:** every flag is counted per record before the display cap (`AnomalyReport.votes`, `agreed`,
  `column_votes`, `methods_run`), so the headline is records flagged by 3+ methods (DataCo 3,606 of 180,519, 2.0%)
  instead of a mix of 5,977 capped findings. Default columns go by role (money, volumes, times) and skip copies of a
  column already chosen (`default_columns`; also used by Insights and the Home card). The review table shows each
  record's most unusual value against the typical one; long-tail notes are combined into one.
- **Unusual periods:** the expected range is drawn as a band (it equals the flag rule exactly), high and low flags are
  marked separately, a flag in the month of a data change is labelled as such, and the summary gives the size of the
  differences (DataCo monthly on-time: 7 of 37 months, 1.1 to 1.9 pts, called out as small).
- Shared page blocks in `ui/layout.py` (`section`, `callout`, `lede`, `meta_line`, `stat_cards`); light and dark themes.
- **Tests:** `tests/test_forecast_anomaly_numbers.py` (8); page tests now check the shown numbers instead of clicking
  Run; two chart tests check trace names instead of counts.

## Phase 9, Batch 2: performance, report options, progress bars
**Excel export was deliberately removed** (module, button, tests, documentation). Reports are PDF and HTML only. `openpyxl` stays in
`requirements.txt` because it reads uploaded .xlsx files.

**Settings / behaviour**
- Reports: PDF is the main button; "Also build HTML" is unchecked by default. The analysed-data CSV is built only when you press "Prepare CSV" (3.5 s and 97 MB on DataCo); the transformation-log CSV stays a normal download.
- Report steps: Metrics, Insights, Forecast, Anomalies, Segments, Drivers, PDF, HTML (if ticked), each shown as "Step k of n: name" with elapsed time. A step that fails or cannot run is listed as skipped with the reason and the run continues. Insights now follows the data patterns step; the insight text itself is assembled inside the PDF step because it needs every analysis result.
- Bars: Reports, Insights (four advanced analyses), Forecasting ("Comparing models, 3 of 7"), Anomalies, Segmentation, Drivers, and Upload for files of 5 MB or more. A single model fit or clustering shows its name, not a percentage. Estimated time appears only when the run is expected to take 8 s or more, from the row count (`core/estimates.py`, rates from `docs/PERFORMANCE.md`).
- `BI_WORKERS=<n>` sets the thread count for drivers (default: CPU count, at most 4; 1 turns threading off). Results never depend on it.
- Semantics page: "Generic header, please confirm" badge (review queue and table); a "Competing columns" tab shows which column is used for revenue, category and date, and why.

**Files changed:** `analytics/{anomaly_detection,drivers,forecasting,kpis,primary_columns,segmentation,supply_chain_kpis}.py`, `core/{estimates,parallel,progress}.py` (new), `ingestion/{pipeline,profiler,schema_detector}.py`, `insights/default_runs.py`, `reporting/{assembly (new),report_builder,theme}.py`, `semantic/column_classifier.py`, `scripts/{benchmark,snapshot_roles}.py` (new), `ui/{reports_page,insights_page,forecasting_page,anomaly_page,segmentation_page,drivers_page,dashboard_page,upload_page,semantic_page,user_runs}.py`, `ui/{progress_bar,shared_results}.py` (new), tests, docs. **Deleted:** `reporting/excel_export.py`.

**Domain label for DataCo ("retail", 36%) - investigated, not changed.** Retail scores 7 matched terms (order, item, category, customer, discount, price, product) plus 5 supported roles; supply chain scores 3 (order, delivery, shipping) and 17% after normalising. The generic commerce words outvote the few logistics words. Re-weighting generic words would be tuned on one file, and the rule was to change it only if the other three real files keep their domain; only one of them (supply_chain_dataset1 -> supply_chain, 37%) could be checked here. The label affects wording only: supply-chain KPIs are chosen by columns, not by domain. `tests/test_role_stability.py` pins both domains.

**Tests run (sandbox, stand-in runner, no pytest):** 330 passed, 1 failed, 19 skipped with the two available real files on the path (328 passed, 21 skipped without them). The failure (`test_manual_model_choice_is_honoured_and_reported`) needs statsmodels, which the sandbox lacks, and failed the same way before any change. Skipped: Streamlit page tests, plotly tests, and two real-file tests whose files were not available. New: `test_progress.py` (14), `test_report_assembly.py` (10), `test_drivers_permutation.py` (5), `test_role_stability.py` (7 + 2 skipped), `test_waste_removed.py` (3). A bug in the first version of the faster permutation code (an all-missing numeric column shifted the column layout) was caught by `test_unexpected_layout_falls_back_to_scikit_learn` and fixed.

**Known issues / not verified**
- No Streamlit, plotly, statsmodels, pytest or ruff in the sandbox. The new and changed pages (Reports, Insights, Dashboard KPI sharing, the five progress bars, Upload bar, Semantics additions) compile but were **never run**. Run `python -m pytest -q` and click through them.
- Thread speed-ups for drivers were not measured (1 CPU). Forecast timings exclude statsmodels models.
- The manufacturing sample and the inventory workbook were not available: their role-stability test is skipped here. Put them in `tests/real_files/` (or set `BI_REAL_FILES_DIR`) and run `python -m pytest tests/test_role_stability.py -q`; it compares the new code with the original algorithm directly, so no golden is needed.
- Progress callbacks fire only when a result is computed, not when it comes from the cache.

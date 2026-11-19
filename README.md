# Tessaly

Business intelligence for operational datasets (formerly "AI Business Intelligence Platform").

Local Streamlit app for uploading business data (CSV / Excel), profiling and cleaning it, working out what each
column means, computing only the KPIs the data supports, and then forecasting, screening for anomalies, segmenting,
explaining drivers, generating evidence-linked insights and exporting finished reports.
Every number shown is computed from your uploaded data. No external AI API, language model or paid service is required.
Supply chain is the showcase domain (OTIF, fill rate, lead time, stockout, inventory cover), but the core stays general.

**Status:** Phases 1-8 complete: upload, profiling, cleaning, semantics, dashboard and KPIs, forecasting, anomaly detection,
segmentation, driver analysis, insights and recommendations, and PDF / HTML reports. Remaining: Phase 9 (hardening,
docs) and Phase 10 (release). See `docs/STATUS.md` and `docs/LIMITATIONS.md`.

## Features
- **Data:** CSV, TSV, TXT, XLSX, XLSM, XLS upload with encoding and delimiter detection, profiling and a data-quality report.
- **Cleaning:** proposed actions you review and apply; the original data is never changed and every change is logged.
- **Semantics:** column roles (revenue, date, quantity, supplier and so on) and business domain, editable, with an optional trained column-name model.
- **Dashboard and Data explorer:** KPIs with formula, columns and limitations; trends, breakdowns, distributions, correlations; paged, searchable table.
- **Forecasting:** chronological (rolling-origin) validation of naive, seasonal naive, moving average, exponential smoothing, Holt-Winters, ARIMA/SARIMA; the simplest model within 5% of the best is chosen; intervals are empirical. Too-short or constant data is refused with a reason.
- **Anomaly detection:** median/MAD, IQR, z-score where valid, Isolation Forest on suitable columns, plus unusual periods in a time series. Records are flagged, never deleted.
- **Segmentation:** KMeans on suppliers, customers, products or warehouses with elbow/silhouette checks and plain-words traits. Weak structure is reported as weak.
- **Drivers:** Random Forest and permutation importance on a held-out split, with leakage rules. Associations, not causes.
- **Insights:** deterministic rules over computed results; every insight carries its evidence, threshold, limitations and a template recommendation.
- **Reports (Phase 8):** a PDF report (cover, contents, executive summary, quality, KPIs, insights, recommendations, forecast, anomalies, segments, drivers, transformation log, methodology), and an optional self-contained HTML report. The analysed data and the transformation log download as CSV. Excel export was deliberately removed. Analyses that cannot run are listed as "Not included" with the reason. Design: paper / navy / orange in a 70 / 25 / 5 ratio.

## Setup (Windows PowerShell)
```powershell
cd "C:\Users\Om Nagpure\Projects\AI-Business-Intelligence"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```
If activation is blocked: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`

macOS/Linux: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`

## Run
```powershell
streamlit run app.py
```
Open http://localhost:8501. Visitors see the public homepage; **Get started** creates an account and opens the app.

`app.py` wraps the pages (`tessaly_app.py`) with two HTTP routes that set and clear the HttpOnly session cookie, plus
security headers, so sign-in survives page reloads and "Remember me" works. `streamlit run server.py` is an alias.

## Accounts and data privacy
- **Accounts** live in SQLite at `data_store/app.db` (set `BI_DATA_DIR` to move it; the folder is git-ignored).
  Passwords are stored only as Argon2id hashes. Duplicate emails are refused, and inputs are validated server-side.
- **Sign-in throttling:** 5 failed attempts for an email lock that email for 15 minutes. Messages and timing are the same
  whether or not the email is registered.
- **Sessions:** a random token in an HttpOnly, SameSite=Lax cookie; only its SHA-256 is stored. Browser-session cookies
  last up to 12 hours, and "Remember me" cookies 30 days. Logout, a password change, or **Account > Sign out of all devices**
  revokes sessions server-side.
- **Per-user data:** uploaded files and built reports are saved under `data_store/users/<user id>/`. Every query in
  `storage/repository.py` filters on the signed-in user's id, so another user's dataset or report id behaves exactly
  like one that does not exist. Signing in or out clears the whole in-memory session.
- **Password recovery:** there is no email service, so self-service reset is not offered (the sign-in page says so).
  An administrator with server access resets a password with:
  ```powershell
  python -m scripts.reset_password user@example.com
  ```
  It prompts for the new password without echoing it, and signs the user out everywhere.

## Test and lint
```powershell
python -m pytest -q
python -m ruff check .
```

## Using it
1. **Overview / Upload:** load the synthetic sample (labelled as such) or upload a file.
2. **Profile, Data quality, Semantics:** review the profile, apply cleaning you agree with, check column roles and domain.
3. **Analyze:** Dashboard, Data explorer, Forecasting, Anomalies, Segmentation, Drivers, Insights.
4. **Deliver > Reports:** set a title and an optional "prepared for", choose whether to include the advanced analyses and whether to also build HTML,
   press **Build PDF report**, then download the PDF (and HTML if you ticked it). If you change the data, roles, cleaning or options, build again.

## Appearance and app shell
- **Light, dark or system.** Switch from the account menu (top right), the Account page, or the icons on the homepage and sign-in
  pages. The choice is saved in the browser (a `bi_theme` cookie plus Streamlit's own theme setting), and applies to every page,
  charts included. Colours live in one place: `ui/theme.py`, mirrored in `.streamlit/config.toml` (a test checks they match).
- **Top bar.** Switch between your saved datasets, and open the account menu (Account, Settings, appearance, Sign out).
- **Picks up where you left off.** After you sign in, the dataset you used last opens by itself. Closing a dataset keeps it saved.
- **KPI tiles.** A sparkline and the change on the previous period appear only for KPIs that are a plain sum, average or count
  of one column with no extra filter, computed from complete periods (at least three). Other KPIs show the value alone.
- **Brand.** Logo, mark and favicon are in `assets/` (light and dark variants). The homepage picture is a real screenshot of the
  Dashboard with the synthetic sample data (`assets/screen-dashboard-*.webp`); retake it if the dashboard changes.

## Optional: train the semantic name model
The app works without it (rules only). The model reads only column *names* and is blended in when you switch it on
(Semantics > Trained model, or Settings). It is off by default.
```powershell
python -m training.make_data                 # writes training/data/*.csv (synthetic + hand-labelled headers)
python -m training.train_semantic_model      # writes training/models/semantic_model.joblib + .meta.json
python -m training.evaluate_semantic_model   # held-out evaluation, compared with the rules using names alone
```
Optionally score it end to end on your own files:
```powershell
python -m training.evaluate_semantic_model --real-files "path\to\file1.csv" "path\to\file2.xlsx"
```
Safety: only `training/models/` is ever loaded, and only if the file's SHA-256 matches its metadata. A model trained
with a different scikit-learn version is refused (retrain). Never place files from untrusted sources there.

## Layout
`config/` settings · `core/` exceptions, logging, validation · `ingestion/` loading, profiling, quality ·
`cleaning/` proposals and transformations · `semantic/` roles and domain · `analytics/` descriptive stats, KPIs, forecasting,
anomalies, segmentation, drivers (pure pandas, no Streamlit) · `insights/` rule engine and evidence ·
`reporting/` report builder and assembly, charts, PDF, HTML (no Streamlit) · `visualization/` Plotly charts ·
`training/` offline model workflow · `ui/` Streamlit pages (no core logic) · `tests/`

## Troubleshooting
- **Reports page says a file could not be created:** the other one still downloads; check the terminal log for the reason.
- **PDF text shows boxes:** the PDF uses fonts bundled with Matplotlib; reinstall it (`pip install --force-reinstall matplotlib`).
- **Charts do not render:** check the installed Streamlit version (`pip show streamlit`); it should be 1.50 or newer.
- **Forecasting or reports fail on import:** make sure `statsmodels`, `matplotlib` and `reportlab` installed (`pip install -r requirements.txt`).

## Limitations
- Role detection is heuristic; check the Semantics page, especially when several columns share a role.
- Forecasts need enough history (56 days, 26 weeks or 24 months) and only extend patterns in the data. Intervals are approximate.
- Insights and reports reuse analyses you ran on their own pages; the rest use default settings (the report says which).
- Insight thresholds are adjustable reference levels of this app, not industry standards. Drivers and segments show association, not cause.
- Files load fully into memory; files around 1 GB or larger are not supported.
- No persistence layer and no authentication.
- The pages were exercised with Streamlit's test harness; check them in your browser and report anything odd.

## Future work
Phase 9 hardening (cache-invalidation tests, large-file strategy, security review, full docs), reusable cleaning recipes,
a "suggested views" panel on the dashboard, and an optional, off-by-default LLM explanation layer.

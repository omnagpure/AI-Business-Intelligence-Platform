# AI Business Intelligence Platform

Local Streamlit app for uploading business data (CSV / Excel), profiling it, cleaning it, and working out
what each column means and what kind of business the data describes. Forecasting, KPIs, insights and reports
come in later phases. No external AI API is required.

**Status:** Phases 1-4 complete (upload, profiling, cleaning, semantics incl. the optional trained name model).
Next: Phase 5 (descriptive analytics and KPIs). See `docs/STATUS.md`.

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

## Test and lint
```powershell
python -m pytest -q
python -m ruff check .
```

## Optional: train the semantic name model
The app works without it (rules only). The model reads only column *names* and is blended in when you switch
it on (Semantics > Trained model, or Settings). It is off by default.
```powershell
python -m training.make_data                 # writes training/data/*.csv (synthetic + hand-labelled headers)
python -m training.train_semantic_model      # writes training/models/semantic_model.joblib + .meta.json
python -m training.evaluate_semantic_model   # held-out evaluation, compared with the rules using names alone
```
Optionally score it end to end on your own files (labels exist for the supply-chain files used in development):
```powershell
python -m training.evaluate_semantic_model --real-files "path\to\file1.csv" "path\to\file2.xlsx"
```
Safety: only `training/models/` is ever loaded, and only if the file's SHA-256 matches its metadata. A model trained
with a different scikit-learn version is refused (retrain). Never place files from untrusted sources there.

## Layout
`config/` settings · `core/` exceptions, logging, validation · `ingestion/` loading, profiling, quality ·
`cleaning/` proposals and transformations · `semantic/` roles and domain · `training/` offline model workflow ·
`ui/` Streamlit pages (no core logic) · `tests/`

## Using it
Open Overview, load sample data or upload a file, then work down the sidebar: Profile, Data quality (cleaning),
Semantics (check roles, review low-confidence columns, override the domain).

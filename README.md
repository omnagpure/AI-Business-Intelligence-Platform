# AI Business Intelligence Platform

Local Streamlit app for uploading business data (CSV / XLSX), profiling it and, in later phases,
cleaning, analysing, forecasting and reporting. No external AI API is required.

**Status:** Phases 1-2 complete (upload, profiling, data quality). Next: cleaning. See `docs/STATUS.md`.

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
## Layout
`config/` settings · `core/` exceptions, logging, validation · `ui/` Streamlit pages (no core logic) · `tests/`

## Using it
Upload a CSV, TSV, TXT, XLSX, XLSM or XLS file (or click "Load sample data"), then open Profile and Data quality in the sidebar.
Test and lint: `python -m pytest -q` and `python -m ruff check .`

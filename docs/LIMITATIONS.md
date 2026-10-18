# Known limitations (through Phase 2)

- Supported: .csv, .tsv, .txt (delimited text) and .xlsx, .xlsm, .xls. Other formats (JSON, Parquet, databases, PDF) are rejected. .xls is read with xlrd; .xlsm macros are ignored.
- Header detection skips title rows above the header (a row must fill ~80% of the widest row and contain no numbers or dates). Headers below row 30, or multi-row headers, are not detected.
- Decimal commas are auto-detected only for ';' or tab-delimited files.
- The app has no domain restriction: any tabular dataset loads. Supply-chain-specific analysis arrives with domain detection (Phase 4).
- Non-UTF-8 CSVs are assumed to be Windows-1252 (Western European). Other legacy encodings (e.g. Cyrillic, CJK)
  may show wrong characters; the detected encoding is displayed.
- Text dates are parsed with one inferred format per column. Ambiguous day/month order (e.g. 03/04/2024)
  resolves to month-first when both fit.
- Identifier, target and "possible primary key" detection are name- and uniqueness-based heuristics.
- The profile and preview hold the full file in memory. A 96 MB CSV (180k x 53) profiled in ~6 s here;
  larger files are untested.
- Only the first worksheet of a multi-sheet workbook is profiled at a time (selectable).
- The quality page reports issues; applying cleaning actions is Phase 3.
- Tested with pandas 3.0 / Python 3.12 / Streamlit 1.66 in the author's sandbox only.

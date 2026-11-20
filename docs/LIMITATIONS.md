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

# Added with Phase 4
- Semantic roles are heuristics. Supply-chain data is recognised from supply-chain-only terms in the column names
  (three or more of supplier, warehouse, shipping, delivery, lead, reorder, stockout, freight, late, scheduled ...) or a
  supplier/warehouse column. Order data with fewer such terms may still read as Retail (override available).
- The name model switch (Automatic) decides from the rules-only result. A wrong supply-chain decision means the model is
  used too much or too little; choosing the domain by hand, or "Always on"/"Off", overrides it.
- No role exists for generic measures (weight, temperature) or ledger debit/credit amounts; such columns stay Unknown.
- The optional name model sees names only, is trained on hand-written and synthetic names, and was evaluated on
  held-out and unseen-industry names plus 73 hand-labelled real headers. It beats the name-only rules on abbreviated
  names, but those test names share a generator with the training names and the real headers overlap in vocabulary
  with the seeds, so the scores are optimistic. The end-to-end check on HR/marketing/SaaS/finance tables uses small
  synthetic tables whose headers partly overlap the training names (the non-overlapping subset is reported separately,
  n=15). None of this is evidence of accuracy on your own data.
- A saved model only loads with the scikit-learn minor version it was trained with; otherwise retrain.
- Columns whose names suggest personal data (email, phone, password, ...) are excluded from role detection.
- The Overview page reports counts and detected roles only. KPIs, charts of results and insights arrive in Phase 5+.

## Accounts and authentication
- No email service: there is no self-service password reset or email verification. Administrators reset passwords
  with `python -m scripts.reset_password`.
- Throttling is per email address. There is no per-IP rate limit; put a reverse proxy with rate limiting in front of an
  internet-facing deployment.
- Serve over HTTPS in production. The session cookie is marked Secure automatically when the request scheme is https;
  behind a TLS-terminating proxy, make sure the proxy forwards the scheme.
- SQLite suits a single server. Multiple app servers would need a shared database and shared file storage.
- There are no per-user storage quotas. Users delete their own datasets and reports from the Upload and Reports pages.

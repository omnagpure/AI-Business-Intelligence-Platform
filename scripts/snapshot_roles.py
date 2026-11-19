"""Write the semantic roles and business domain of real files to tests/golden/roles.json.

    python -m scripts.snapshot_roles path1.csv path2.xlsx ...

The file is keyed by file name. tests/test_role_stability.py compares against it whenever the file is found
(in BI_REAL_FILES_DIR or tests/real_files/). Only refresh a golden on purpose, after checking that a role change is wanted.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "tests" / "golden" / "roles.json"


def snapshot(path: Path) -> dict:
    from config.settings import Settings
    from ingestion.pipeline import ingest_bytes
    from semantic.semantic_model import build_semantic_model

    r = ingest_bytes(path.read_bytes(), path.name, max_bytes=Settings().max_upload_bytes)
    m = build_semantic_model(r.dataset.df, r.profile, {}, None, None)
    return {"domain": m.domain.domain, "domain_confidence": m.domain.confidence,
            "roles": {n: c.role for n, c in m.columns.items()},
            "confidence": {n: c.confidence for n, c in m.columns.items()}}


def main(argv: list[str]) -> int:
    sys.path.insert(0, str(ROOT))
    data = json.loads(GOLDEN.read_text(encoding="utf-8")) if GOLDEN.exists() else {}
    for f in argv:
        p = Path(f)
        data[p.name] = snapshot(p)
        print(f"{p.name}: domain={data[p.name]['domain']} ({data[p.name]['domain_confidence']}), {len(data[p.name]['roles'])} columns")
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN.write_text(json.dumps(data, indent=1, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

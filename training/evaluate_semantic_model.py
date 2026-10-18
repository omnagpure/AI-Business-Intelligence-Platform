"""Evaluate the trained name model on data it never saw, and compare it with a rules-only baseline.

Run:  python -m training.evaluate_semantic_model [--real-files a.csv b.xlsx ...]

Three checks, reported separately so nothing is flattered:
1. Held-out synthetic names (base names disjoint from training).
2. Hand-labelled real headers (never used for training), plus the subset whose normalised name is absent
   from the training names.
3. Optional end-to-end: the full classifier (names + values + relationships) on files you pass in, with
   and without the model, scored against the hand labels for the columns they contain.
The baseline for 1 and 2 is the lexicon's name-only matching, so model and baseline see the same input.
Results are written next to the model, tied to its checksum. Nothing here is a real-world accuracy claim.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from core.exceptions import ModelTrainingError
from semantic.column_classifier import _ROLES, _Feat, _name_match, classify_columns
from semantic.column_lexicon import ROLES, token_set, tokenize
from semantic.trained_name_model import EVAL_FILE, MODEL_DIR, load_name_model
from training.make_data import REAL_HEADERS, SEED, build_datasets, normalise, real_headers_frame


def rules_name_only(name: str) -> str:
    """Lexicon name-match baseline: best role by name evidence alone, else 'unknown'."""
    f = _Feat(name, token_set(name), " ".join(tokenize(name)), "", 0, 0.0, 1, None)
    scored = sorted(((_name_match(r, f)[0], r) for r in _ROLES), key=lambda t: (-t[0], t[1]))
    return scored[0][1] if scored[0][0] > 0 else "unknown"


def _score(y_true: list[str], y_pred: list[str]) -> dict:
    from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
    labels = [r for r in ROLES if r in set(y_true) | set(y_pred)]
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {"n": len(y_true), "accuracy": round(float(accuracy_score(y_true, y_pred)), 3),
            "macro_f1": round(float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)), 3),
            "per_role": {l: {"precision": round(float(a), 2), "recall": round(float(b), 2),
                             "f1": round(float(c), 2), "support": int(d)} for l, a, b, c, d in zip(labels, p, r, f, s)}}


def _confusion(y_true: list[str], y_pred: list[str]) -> pd.DataFrame:
    labels = [r for r in ROLES if r in set(y_true) | set(y_pred)]
    return pd.DataFrame(pd.crosstab(pd.Series(y_true, name="actual"), pd.Series(y_pred, name="predicted"))
                        ).reindex(index=labels, columns=labels, fill_value=0)


def _compare(names: list[str], truth: list[str], model) -> dict:
    mp = [max(model.predict_proba(n).items(), key=lambda kv: kv[1])[0] for n in names]
    rp = [rules_name_only(n) for n in names]
    return {"model": _score(truth, mp), "rules_name_only": _score(truth, rp),
            "confusion_model": _confusion(truth, mp).to_dict()}


def _end_to_end(files: list[Path], model) -> dict | None:
    from ingestion.pipeline import ingest_bytes
    hits = {"rules": 0, "blended": 0}
    total = changed = 0
    for path in files:
        res = ingest_bytes(path.read_bytes(), path.name, max_bytes=500 * 1024 * 1024)
        df = res.dataset.df
        if len(df) > 20000:
            df = df.sample(20000, random_state=SEED)
        prof = res.profile
        plain = classify_columns(df, prof, None)
        blend = classify_columns(df, prof, model)
        for col, truth in REAL_HEADERS.items():
            if col in plain:
                total += 1
                hits["rules"] += plain[col].role == truth
                hits["blended"] += blend[col].role == truth
                changed += plain[col].role != blend[col].role
    if not total:
        return None
    return {"columns_scored": total, "rules_correct": int(hits["rules"]), "blended_correct": int(hits["blended"]),
            "rules_accuracy": round(hits["rules"] / total, 3), "blended_accuracy": round(hits["blended"] / total, 3),
            "columns_where_model_changed_the_role": int(changed)}


def evaluate(model_dir: Path = MODEL_DIR, real_files: list[Path] | None = None) -> dict:
    model, status = load_name_model(model_dir)
    if model is None:
        raise ModelTrainingError(f"Cannot evaluate: {status.reason}")
    train, test = build_datasets(SEED)
    assert not set(train["base"]) & set(test["base"])
    synth = _compare(test["name"].tolist(), test["role"].tolist(), model)
    real = real_headers_frame()
    real_all = _compare(real["name"].tolist(), real["role"].tolist(), model)
    seen = {normalise(n) for n in train["name"]}
    unseen = real[[normalise(n) not in seen for n in real["name"]]]
    real_unseen = _compare(unseen["name"].tolist(), unseen["role"].tolist(), model)
    e2e = _end_to_end(real_files, model) if real_files else None
    meta = json.loads((model_dir / "semantic_model.meta.json").read_text(encoding="utf-8"))
    out = {"model_version": status.version, "model_sha256": meta["sha256"], "seed": SEED,
           "label_counts_test": test["role"].value_counts().sort_index().to_dict(),
           "held_out_synthetic": synth, "real_headers_all": real_all,
           "real_headers_unseen_names": real_unseen, "end_to_end_real_files": e2e,
           "headline": {"held_out_accuracy": synth["model"]["accuracy"], "held_out_n": synth["model"]["n"],
                        "baseline_accuracy": synth["rules_name_only"]["accuracy"],
                        "real_headers_accuracy": real_all["model"]["accuracy"], "real_headers_n": real_all["model"]["n"],
                        "real_baseline_accuracy": real_all["rules_name_only"]["accuracy"]}}
    (model_dir / EVAL_FILE).write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return out


def _line(tag: str, r: dict) -> str:
    return (f"{tag:<28} model acc {r['model']['accuracy']:.3f}  macro-F1 {r['model']['macro_f1']:.3f}   |   "
            f"rules-name-only acc {r['rules_name_only']['accuracy']:.3f}  macro-F1 {r['rules_name_only']['macro_f1']:.3f}"
            f"   (n={r['model']['n']})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--real-files", nargs="*", type=Path, default=[])
    ap.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    a = ap.parse_args()
    out = evaluate(a.model_dir, a.real_files)
    print(f"Model {out['model_version']}  (seed {out['seed']})\n")
    print(_line("Held-out synthetic names", out["held_out_synthetic"]))
    print(_line("Real headers (all)", out["real_headers_all"]))
    print(_line("Real headers (unseen names)", out["real_headers_unseen_names"]))
    if out["end_to_end_real_files"]:
        e = out["end_to_end_real_files"]
        print(f"\nEnd-to-end on your files ({e['columns_scored']} labelled columns): rules {e['rules_accuracy']:.3f} "
              f"vs rules+model {e['blended_accuracy']:.3f}; model changed the role of "
              f"{e['columns_where_model_changed_the_role']} column(s).")
    print("\nPer-role F1 on held-out synthetic names (model):")
    for role, m in out["held_out_synthetic"]["model"]["per_role"].items():
        print(f"  {role:<11} P {m['precision']:.2f}  R {m['recall']:.2f}  F1 {m['f1']:.2f}  n={m['support']}")
    print(f"\nSaved to {a.model_dir / EVAL_FILE}")
    print("Caution: synthetic and hand-labelled names only. Not a measure of accuracy on your own data.")


if __name__ == "__main__":
    main()

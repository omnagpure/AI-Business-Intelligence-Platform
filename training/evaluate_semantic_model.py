"""Evaluate the trained name model on data it never saw, and compare it with a rules-only baseline.

Run:  python -m training.evaluate_semantic_model [--real-files a.csv b.xlsx ...]

Three checks, reported separately so nothing is flattered:
1. Held-out synthetic names (base names disjoint from training).
2. Hand-labelled real headers (never used for training), plus the subset whose normalised name is absent
   from the training names.
3. End-to-end, the way the app runs it: the full classifier (names + values + relationships) with the model off
   versus in automatic mode, on (a) small labelled tables from HR, marketing, SaaS and finance
   (training/domain_frames.py), reported for all columns and for the headers absent from the training names, and
   (b) optionally the real supply-chain files you pass in, scored against the hand labels for their columns.
The baseline for 1 and 2 is the lexicon's name-only matching, so model and baseline see the same input.
Results are written next to the model, tied to its checksum. Nothing here is a real-world accuracy claim.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from core.exceptions import ModelTrainingError
from semantic.column_classifier import _ROLES, _Feat, _name_match
from semantic.column_lexicon import ENTITY_ROLES, ROLES, semantic_tokens, tokenize
from semantic.semantic_model import build_semantic_model
from semantic.trained_name_model import EVAL_FILE, MODEL_DIR, load_name_model
from training.domain_frames import domain_frames
from training.make_data import (REAL_HEADERS, SEED, build_datasets, build_unseen_domain, normalise, real_headers_frame,
                                sc_all_frame, sc_holdout_frame)


def rules_name_only(name: str) -> str:
    """Lexicon name-match baseline: best role by name evidence alone, else 'unknown'."""
    f = _Feat(name, semantic_tokens(name), " ".join(tokenize(name)), "", 0, 0.0, 1, None)
    # ties go to the specific entity roles, as the full classifier does for id-style names (it penalises "identifier")
    scored = sorted(((_name_match(r, f)[0], r) for r in _ROLES), key=lambda t: (-t[0], t[1] not in ENTITY_ROLES, t[1]))
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
    """Real supply-chain files: rules only vs automatic mode, scored on the hand-labelled columns."""
    from ingestion.pipeline import ingest_bytes
    hits = {"rules": 0, "auto": 0}
    total = changed = 0
    applied: dict[str, str] = {}
    for path in files:
        res = ingest_bytes(path.read_bytes(), path.name, max_bytes=500 * 1024 * 1024)
        df = res.dataset.df
        if len(df) > 20000:
            df = df.sample(20000, random_state=SEED)
        prof = res.profile
        plain = build_semantic_model(df, prof, model_mode="off").columns
        auto_model = build_semantic_model(df, prof, name_model=model, model_mode="auto")
        applied[path.name] = auto_model.name_model_use.applied
        for col, truth in REAL_HEADERS.items():
            if col in plain:
                total += 1
                hits["rules"] += plain[col].role == truth
                hits["auto"] += auto_model.columns[col].role == truth
                changed += plain[col].role != auto_model.columns[col].role
    if not total:
        return None
    return {"columns_scored": total, "rules_correct": int(hits["rules"]), "auto_correct": int(hits["auto"]),
            "rules_accuracy": round(hits["rules"] / total, 3), "auto_accuracy": round(hits["auto"] / total, 3),
            "columns_where_model_changed_the_role": int(changed), "model_applied": applied}


def _other_domains(model, seen_names: set[str]) -> dict:
    """Labelled HR / marketing / SaaS / finance tables: rules only vs automatic mode (names + values)."""
    from ingestion.profiler import profile_dataframe
    rows = []
    for title, df, labels in domain_frames(SEED):
        prof = profile_dataframe(df)
        plain = build_semantic_model(df, prof, model_mode="off")
        auto_model = build_semantic_model(df, prof, name_model=model, model_mode="auto")
        for col, truth in labels.items():
            rows.append({"table": title, "column": col, "truth": truth, "rules": plain.columns[col].role,
                         "auto": auto_model.columns[col].role, "unseen": normalise(col) not in seen_names,
                         "model_applied": auto_model.name_model_use.applied})
    t = pd.DataFrame(rows)

    def acc(frame: pd.DataFrame) -> dict:
        n = len(frame)
        return {"n": n, "rules_accuracy": round(float((frame["rules"] == frame["truth"]).mean()), 3) if n else None,
                "auto_accuracy": round(float((frame["auto"] == frame["truth"]).mean()), 3) if n else None}
    return {"all_columns": acc(t), "headers_not_in_training": acc(t[t["unseen"]]),
            "by_table": {k: acc(g) for k, g in t.groupby("table")},
            "model_applied": t.groupby("table")["model_applied"].first().to_dict(),
            "columns": t.to_dict(orient="records")}


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
    unseen_dom = build_unseen_domain(SEED)
    dom = _compare(unseen_dom["name"].tolist(), unseen_dom["role"].tolist(), model)
    sch = sc_holdout_frame()
    sc_hold = _compare(sch["name"].tolist(), sch["role"].tolist(), model)
    e2e = _end_to_end(real_files, model) if real_files else None
    other = _other_domains(model, seen)
    meta = json.loads((model_dir / "semantic_model.meta.json").read_text(encoding="utf-8"))
    out = {"model_version": status.version, "model_sha256": meta["sha256"], "seed": SEED,
           "label_counts_test": test["role"].value_counts().sort_index().to_dict(),
           "held_out_synthetic": synth, "real_headers_all": real_all,
           "real_headers_unseen_names": real_unseen, "unseen_domains": dom, "supply_chain_holdout": sc_hold, "end_to_end_real_files": e2e,
           "end_to_end_other_domains": other,
           "headline": {"held_out_accuracy": synth["model"]["accuracy"], "held_out_n": synth["model"]["n"],
                        "baseline_accuracy": synth["rules_name_only"]["accuracy"],
                        "real_headers_accuracy": real_all["model"]["accuracy"], "real_headers_n": real_all["model"]["n"],
                        "real_baseline_accuracy": real_all["rules_name_only"]["accuracy"],
                        "unseen_domain_accuracy": dom["model"]["accuracy"], "unseen_domain_n": dom["model"]["n"],
                        "unseen_domain_baseline_accuracy": dom["rules_name_only"]["accuracy"],
                        "other_domains_n": other["all_columns"]["n"],
                        "other_domains_rules_accuracy": other["all_columns"]["rules_accuracy"],
                        "other_domains_auto_accuracy": other["all_columns"]["auto_accuracy"],
                        "other_domains_unseen_n": other["headers_not_in_training"]["n"],
                        "other_domains_unseen_rules_accuracy": other["headers_not_in_training"]["rules_accuracy"],
                        "other_domains_unseen_auto_accuracy": other["headers_not_in_training"]["auto_accuracy"],
                        "supply_chain_holdout_model_accuracy": sc_hold["model"]["accuracy"],
                        "supply_chain_holdout_rules_accuracy": sc_hold["rules_name_only"]["accuracy"],
                        **({"real_files_n": e2e["columns_scored"], "real_files_rules_accuracy": e2e["rules_accuracy"],
                            "real_files_auto_accuracy": e2e["auto_accuracy"]} if e2e else {})}}
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
    print(_line("Unseen industries", out["unseen_domains"]))
    print(_line("Real headers (all)", out["real_headers_all"]))
    print(_line("Real headers (unseen names)", out["real_headers_unseen_names"]))
    print(_line("Supply-chain held-out", out["supply_chain_holdout"]))
    if out["end_to_end_real_files"]:
        e = out["end_to_end_real_files"]
        print(f"\nEnd-to-end on your supply-chain files ({e['columns_scored']} labelled columns): rules only "
              f"{e['rules_accuracy']:.3f} vs automatic {e['auto_accuracy']:.3f}; model changed the role of "
              f"{e['columns_where_model_changed_the_role']} column(s). Model applied: {e['model_applied']}")
    o = out["end_to_end_other_domains"]
    print(f"\nEnd-to-end, HR/marketing/SaaS/finance tables: rules only {o['all_columns']['rules_accuracy']:.3f} vs "
          f"automatic {o['all_columns']['auto_accuracy']:.3f} (n={o['all_columns']['n']}); headers not in the training "
          f"names: {o['headers_not_in_training']['rules_accuracy']} vs {o['headers_not_in_training']['auto_accuracy']} "
          f"(n={o['headers_not_in_training']['n']})")
    for k, v in o["by_table"].items():
        print(f"  {k:<10} rules {v['rules_accuracy']:.3f}  automatic {v['auto_accuracy']:.3f}  (n={v['n']})")
    print("\nPer-role F1 on held-out synthetic names (model):")
    for role, m in out["held_out_synthetic"]["model"]["per_role"].items():
        print(f"  {role:<11} P {m['precision']:.2f}  R {m['recall']:.2f}  F1 {m['f1']:.2f}  n={m['support']}")
    print(f"\nSaved to {a.model_dir / EVAL_FILE}")
    print("Caution: synthetic and hand-labelled names only. Not a measure of accuracy on your own data.")


if __name__ == "__main__":
    main()

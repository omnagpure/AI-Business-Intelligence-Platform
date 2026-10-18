"""Train the optional column-name classifier (TF-IDF + logistic regression) and save it.

Run from the project root:  python -m training.train_semantic_model
Writes training/models/semantic_model.joblib and semantic_model.meta.json (checksum, versions, seed).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from core.exceptions import ModelTrainingError
from semantic.trained_name_model import (META_FILE, MODEL_DIR, MODEL_FILE, SCHEMA_VERSION, normalise_name,
                                         sha256_file)
from training.make_data import SEED, build_datasets, label_distribution

C_GRID = (1.0, 3.0, 10.0, 30.0)


def build_pipeline(c: float):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import FeatureUnion, Pipeline
    feats = FeatureUnion([
        ("word", TfidfVectorizer(preprocessor=normalise_name, analyzer="word", token_pattern=r"\S+",
                                 ngram_range=(1, 2), sublinear_tf=True)),
        ("char", TfidfVectorizer(preprocessor=normalise_name, analyzer="char_wb", ngram_range=(2, 5),
                                 sublinear_tf=True)),
    ])
    clf = LogisticRegression(C=c, max_iter=3000, class_weight="balanced", random_state=SEED)
    return Pipeline([("features", feats), ("clf", clf)])


def choose_c(train, seed: int) -> tuple[float, dict[float, float]]:
    """Pick C by cross-validation grouped by base name, using the TRAINING data only."""
    from sklearn.metrics import f1_score
    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=4)
    scores: dict[float, float] = {}
    for c in C_GRID:
        fold = []
        for tr, va in gkf.split(train["name"], train["role"], groups=train["base"]):
            p = build_pipeline(c).fit(train["name"].iloc[tr], train["role"].iloc[tr])
            fold.append(f1_score(train["role"].iloc[va], p.predict(train["name"].iloc[va]), average="macro"))
        scores[c] = round(sum(fold) / len(fold), 4)
    return max(scores, key=lambda k: (scores[k], -k)), scores


def train(out_dir: Path = MODEL_DIR, seed: int = SEED) -> dict:
    try:
        import joblib
        import sklearn
    except ImportError as exc:
        raise ModelTrainingError("scikit-learn and joblib are required for training. "
                                 "Install them with: pip install scikit-learn", detail=str(exc)) from exc
    train_df, _ = build_datasets(seed)          # the held-out test split is never touched here
    best_c, cv = choose_c(train_df, seed)
    pipe = build_pipeline(best_c).fit(train_df["name"], train_df["role"])
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / MODEL_FILE
    joblib.dump(pipe, model_path)
    digest = sha256_file(model_path)
    data_hash = hashlib.sha256(train_df.to_csv(index=False).encode()).hexdigest()[:12]
    meta = {
        "schema_version": SCHEMA_VERSION, "model_version": f"1-{digest[:8]}", "sha256": digest,
        "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "seed": seed,
        "sklearn_version": sklearn.__version__, "C": best_c, "cv_macro_f1_by_C": {str(k): v for k, v in cv.items()},
        "train_rows": int(len(train_df)), "train_base_names": int(train_df["base"].nunique()),
        "label_counts": label_distribution(train_df), "train_data_hash": data_hash,
        "note": "Trained on synthetic and hand-written column names only. Evaluate before relying on it.",
    }
    (out_dir / META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--out", type=Path, default=MODEL_DIR)
    a = p.parse_args()
    meta = train(a.out, a.seed)
    print(f"Saved model {meta['model_version']} to {a.out}")
    print(f"Chosen C={meta['C']} (grouped CV macro-F1 by C: {meta['cv_macro_f1_by_C']})")
    print("Next: python -m training.evaluate_semantic_model")


if __name__ == "__main__":
    main()

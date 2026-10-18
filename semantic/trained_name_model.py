"""Loader for the OPTIONAL trained column-name model (see training/).

Safety: only the fixed artifact in training/models/ is ever loaded, never a user upload, and only after its
SHA-256 matches the value recorded in the metadata file next to it. joblib/pickle can run code, so
treat this folder as trusted. Anything wrong (missing file, hash mismatch, different scikit-learn version,
import error) yields "unavailable" and the app carries on with rules only. scikit-learn is imported lazily.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from config.settings import PROJECT_ROOT
from core.logging_config import get_logger
from semantic.column_lexicon import tokenize

log = get_logger("semantic.model")

MODEL_DIR = PROJECT_ROOT / "training" / "models"
MODEL_FILE = "semantic_model.joblib"
META_FILE = "semantic_model.meta.json"
EVAL_FILE = "semantic_model.eval.json"
SCHEMA_VERSION = 1


def normalise_name(name: str) -> str:
    """'OrderItemTotal' / 'order_item_total' -> 'order item total'. Referenced by the saved pipeline."""
    return " ".join(tokenize(name))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class ModelStatus:
    available: bool
    reason: str                      # human-readable; why it is (not) usable
    version: str = ""
    trained_at: str = ""
    n_roles: int = 0
    metrics: dict | None = None      # headline numbers from the evaluation file, if it matches this artifact


class TrainedNameModel:
    """Implements the NameModel protocol used by semantic.column_classifier."""

    def __init__(self, pipeline, version: str) -> None:
        self._pipe = pipeline
        self.version = version
        self._classes = [str(c) for c in pipeline.classes_]
        self._cache: dict[str, dict[str, float]] = {}

    def predict_proba(self, name: str) -> dict[str, float]:
        key = str(name)
        if key not in self._cache:
            probs = self._pipe.predict_proba([key])[0]
            self._cache[key] = {c: float(p) for c, p in zip(self._classes, probs)}
        return self._cache[key]


def _read_json(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def load_name_model(model_dir: Path | None = None) -> tuple[TrainedNameModel | None, ModelStatus]:
    """Return (model, status). The model is None whenever it cannot be used safely."""
    d = Path(model_dir) if model_dir is not None else MODEL_DIR
    mf, metaf = d / MODEL_FILE, d / META_FILE
    if not mf.is_file() or not metaf.is_file():
        return None, ModelStatus(False, "No trained model found. Optional: run the training steps in the README.")
    meta = _read_json(metaf)
    if meta is None or meta.get("schema_version") != SCHEMA_VERSION or "sha256" not in meta:
        return None, ModelStatus(False, "Model metadata is unreadable or from an incompatible version.")
    if sha256_file(mf) != meta["sha256"]:
        log.warning("Model artifact hash mismatch; refusing to load it")
        return None, ModelStatus(False, "Model file does not match its recorded checksum, so it was not loaded.")
    try:
        import joblib
        import sklearn
    except ImportError:
        return None, ModelStatus(False, "scikit-learn is not installed, so the trained model cannot be used.")
    if str(meta.get("sklearn_version", "")).split(".")[:2] != sklearn.__version__.split(".")[:2]:
        return None, ModelStatus(False, f"Model was trained with scikit-learn {meta.get('sklearn_version')} but "
                                        f"{sklearn.__version__} is installed. Retrain the model.")
    try:
        pipe = joblib.load(mf)
        model = TrainedNameModel(pipe, str(meta.get("model_version", "unknown")))
    except Exception:           # corrupt or incompatible artifact: never take the app down
        log.exception("Could not load the trained name model")
        return None, ModelStatus(False, "The model file could not be loaded. Retrain it or delete it.")
    ev = _read_json(d / EVAL_FILE)
    metrics = ev.get("headline") if ev and ev.get("model_sha256") == meta["sha256"] else None
    return model, ModelStatus(True, "Trained model loaded.", model.version, str(meta.get("trained_at", "")),
                              len(model._classes), metrics)

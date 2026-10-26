"""
Model artifacts and the metadata that must stay in sync with them.

src/model_config.json sits next to the saved models and records what the app and
predict.py need to interpret their outputs: the label mapping, the meaning and
direction of each score, both decision thresholds, and a fingerprint (SHA-256)
of the model files each threshold was calibrated for. If a model is retrained
after its threshold was chosen, the fingerprints no longer match and the
threshold is reported as stale instead of being used silently.
"""
import hashlib
import json
import os

SRC = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SRC, "model_pipeline.joblib")
DL_MODEL_PATH = os.path.join(SRC, "dl_model.keras")
CONFIG_PATH = os.path.join(SRC, "model_config.json")
METRICS_PATH = os.path.join(SRC, "..", "Reports", "metrics.json")

LABELS = {0: "Computer-generated", 1: "Original"}

DEFAULT_CONFIG = {
    "labels": {"0": "Computer-generated (CG)", "1": "Original (OR, human-written)"},
    "stage_a": {
        "score": "P(Original): logistic regression predict_proba[:, 1], averaged with the BiLSTM sigmoid output "
                 "when the BiLSTM is available",
        "rule": "flag as computer-generated if score < threshold",
        "threshold": None,
        "calibrated_for": None,
    },
    "stage_b": {
        "score": "softmax probability of output index 0 (machine-generated, MAGE convention) from the detector",
        "rule": "flag as computer-generated if score >= threshold",
        "threshold": None,
        "calibrated_for": None,
    },
    "fusion": "computer-generated if Stage A flags OR Stage B flags",
}


def sha256(path):
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(include_dl=True):
    """SHA-256 of the Stage A model files currently on disk."""
    out = {"model_pipeline.joblib": sha256(MODEL_PATH)}
    if include_dl:
        out["dl_model.keras"] = sha256(DL_MODEL_PATH)
    return out


def load_config():
    if not os.path.exists(CONFIG_PATH):
        return json.loads(json.dumps(DEFAULT_CONFIG))
    with open(CONFIG_PATH) as f:
        return json.load(f)


def save_config(config):
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)


def update_stage(stage, **values):
    """Merge values into config[stage] and save."""
    config = load_config()
    config.setdefault(stage, dict(DEFAULT_CONFIG[stage])).update(values)
    save_config(config)
    return config


def load_metrics():
    if not os.path.exists(METRICS_PATH):
        return {}
    with open(METRICS_PATH) as f:
        return json.load(f)


def save_metrics(metrics):
    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)


def stage_a_threshold(dl_available=True):
    """
    (threshold, warning). The Stage A threshold is only valid for the exact models it
    was calibrated for. Returns (0.5, reason) when there is no valid calibrated value,
    so callers can show the reason instead of silently using an uncalibrated cut-off.
    """
    stage = load_config().get("stage_a", {})
    value, calibrated = stage.get("threshold"), stage.get("calibrated_for")
    if value is None:
        return 0.5, "no calibrated Stage A threshold found (run choose_threshold.py); using 0.5"
    current = fingerprint(include_dl=dl_available)
    if calibrated is not None and calibrated != current:
        if not dl_available and "dl_model.keras" in calibrated:
            return float(value), (f"the threshold {value} was calibrated for the LR + BiLSTM ensemble, but the "
                                  "BiLSTM is unavailable: false-positive rate is not controlled")
        return float(value), (f"the threshold {value} was calibrated for different model files; rerun "
                              "choose_threshold.py")
    return float(value), None


def stage_b_threshold():
    value = load_config().get("stage_b", {}).get("threshold")
    return None if value is None else float(value)

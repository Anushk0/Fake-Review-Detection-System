import json

import numpy as np
import pytest

import artifacts
from choose_threshold import pick_threshold, rates
from features import decision_breakdown, explain, to_frame


def test_contributions_reconstruct_the_logit(toy_pipeline):
    text, rating = "Arrived late but works fine. The battery lasts about two days.", 4
    b = decision_breakdown(toy_pipeline, text, rating)
    logit = toy_pipeline.decision_function(to_frame([text], [rating]))[0]
    assert b["logit"] == pytest.approx(logit, abs=1e-6)
    assert b["prob_original"] == pytest.approx(toy_pipeline.predict_proba(to_frame([text], [rating]))[0, 1])
    words, signals = explain(toy_pipeline, text, rating, top_k=10_000)
    assert words.contribution.sum() + signals.contribution.sum() + b["intercept"] == pytest.approx(logit, abs=1e-6)


def test_explanation_direction_matches_label_orientation(toy_pipeline):
    # a review made of machine-template phrases should have net negative contributions (toward class 0)
    b = decision_breakdown(toy_pipeline, "the only thing very good and I will keep it very good and", 5)
    assert b["contributions"] + b["intercept"] < 0


def test_pick_threshold_respects_budget_with_ties():
    rng = np.random.default_rng(1)
    p = np.r_[np.round(rng.uniform(0.3, 1, 500), 2), rng.uniform(0, 0.4, 500)]
    y = np.r_[np.ones(500, int), np.zeros(500, int)]
    t = pick_threshold(y, p, 0.01)
    assert rates(y, p, t)[0] <= 0.01
    assert rates(y, p, round(t + 0.01, 2))[0] > 0.01 or t >= 0.99


def test_stale_threshold_is_reported(tmp_path, monkeypatch):
    model = tmp_path / "model_pipeline.joblib"; model.write_bytes(b"v1")
    dl = tmp_path / "dl_model.keras"; dl.write_bytes(b"dl")
    monkeypatch.setattr(artifacts, "MODEL_PATH", str(model))
    monkeypatch.setattr(artifacts, "DL_MODEL_PATH", str(dl))
    monkeypatch.setattr(artifacts, "CONFIG_PATH", str(tmp_path / "model_config.json"))
    assert artifacts.stage_a_threshold() == (0.5, artifacts.stage_a_threshold()[1])
    artifacts.update_stage("stage_a", threshold=0.22, calibrated_for=artifacts.fingerprint())
    assert artifacts.stage_a_threshold() == (0.22, None)
    _, warning = artifacts.stage_a_threshold(dl_available=False)
    assert "BiLSTM is unavailable" in warning
    model.write_bytes(b"v2 retrained")
    value, warning = artifacts.stage_a_threshold()
    assert value == 0.22 and "different model files" in warning
    assert json.loads((tmp_path / "model_config.json").read_text())["stage_a"]["rule"].startswith("flag")

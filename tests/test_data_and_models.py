"""Tests that need the project's dataset or saved models; skipped when those are not present."""
import os

import numpy as np
import pytest

import train
from artifacts import DL_MODEL_PATH, MODEL_PATH


def test_dedup_key_ignores_case_punctuation_and_spacing():
    assert train.dedup_key("Great  product!!") == train.dedup_key("great product.")


needs_data = pytest.mark.skipif(not os.path.exists(train.DATA), reason="Data/Raw/fakeReviewData.csv not present")


@needs_data
def test_no_review_text_crosses_split_boundaries():
    X_train, _, X_fit, _, X_val, _, X_test, _ = train.load_splits(include_mage=False)
    keys = {name: set(X["text"].map(train.dedup_key)) for name, X in
            [("fit", X_fit), ("val", X_val), ("test", X_test)]}
    assert not keys["fit"] & keys["val"] and not keys["fit"] & keys["test"] and not keys["val"] & keys["test"]
    assert len(X_fit) + len(X_val) == len(X_train)


@needs_data
def test_splits_are_stratified_and_reproducible():
    a = train.load_splits(include_mage=False)
    b = train.load_splits(include_mage=False)
    assert (a[6]["text"].to_numpy() == b[6]["text"].to_numpy()).all()
    y_test = a[7]
    assert abs(y_test.mean() - 0.5) < 0.01


needs_models = pytest.mark.skipif(not (os.path.exists(MODEL_PATH)), reason="model_pipeline.joblib not present")


@needs_models
def test_saved_pipeline_single_and_batch_inference():
    from predict import load_dl_model, load_pipeline, predict_texts
    pipe = load_pipeline()
    dl = load_dl_model()
    texts = ["I bought this for my son. He loves it, though the battery could be better.",
             "This is a great product and the quality is very good and the price is very good and the"]
    out = predict_texts(pipe, dl, texts, [5, 5], threshold=0.5)
    assert out["prob_original"].between(0, 1).all()
    assert set(out["prediction"]) <= {"Original", "Computer-generated"}
    single = predict_texts(pipe, dl, texts[:1], [5], threshold=0.5)
    assert single.loc[0, "prob_original"] == pytest.approx(out.loc[0, "prob_original"])


@pytest.mark.skipif(not os.path.exists(DL_MODEL_PATH), reason="dl_model.keras not present")
def test_bilstm_outputs_probability_of_original():
    from predict import load_dl_model
    import predict
    dl = load_dl_model()
    if dl is None:
        pytest.skip(f"BiLSTM could not be loaded: {predict.DL_LOAD_ERROR}")
    p = dl(["Works fine, arrived on time.", ""])
    assert p.shape == (2,) and ((p >= 0) & (p <= 1)).all()

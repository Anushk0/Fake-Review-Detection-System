import numpy as np
import pandas as pd
import pytest

from predict import LABEL_NAMES, clean_ratings, ensemble_probability, evaluate, fuse, predict_texts


def test_label_mapping():
    assert LABEL_NAMES == {0: "Computer-generated", 1: "Original"}


def test_ensemble_averages_probabilities_not_votes():
    np.testing.assert_allclose(ensemble_probability([0.9, 0.2], [0.5, 0.4]), [0.7, 0.3])
    np.testing.assert_allclose(ensemble_probability([0.9, 0.2]), [0.9, 0.2])


def test_threshold_direction_and_or_fusion():
    p_original = [0.10, 0.30, 0.90, 0.90]
    p_ai = [0.10, 0.10, 0.999, np.nan]
    a, b, flagged = fuse(p_original, p_ai, threshold=0.22, modern_threshold=0.99)
    assert a.tolist() == [True, False, False, False]      # Stage A: P(Original) < 0.22
    assert b.tolist() == [False, False, True, False]      # Stage B: P(AI) >= 0.99; NaN never flags
    assert flagged.tolist() == [True, False, True, False]


def test_fusion_without_stage_b_is_stage_a():
    a, b, flagged = fuse([0.1, 0.9], None, 0.5, None)
    assert not b.any() and (flagged == a).all()


def test_threshold_equality_is_not_flagged_by_stage_a():
    a, _, _ = fuse([0.22], None, 0.22)
    assert not a[0]


def test_clean_ratings_fills_and_clips():
    np.testing.assert_allclose(clean_ratings([5, None, "x", 9, 0], 4), [5, 4, 4, 5, 1])


def test_evaluate_orientation():
    m = evaluate(["CG", "CG", "OR", "OR"], ["Computer-generated", "Original", "Original", "Original"])
    # rows = true CG/OR, columns = predicted CG/OR; positive class = Original
    assert m["confusion_matrix"].tolist() == [[1, 1], [0, 2]]
    assert m["accuracy"] == pytest.approx(0.75)
    assert m["precision"] == pytest.approx(2 / 3) and m["recall"] == pytest.approx(1.0)
    assert evaluate(["?", "x"], ["Original", "Original"]) is None


class FakeModern:
    def predict(self, texts, progress=None):
        return [0.999 if "chatgpt" in t else 0.01 for t in texts]


def test_batch_inference_end_to_end(toy_pipeline):
    texts = pd.Series(["Arrived late but works fine. The battery lasts about two days because I use it a lot.",
                       "This is a great product and the quality is very good and the price is very good and item 1",
                       "", "written by chatgpt: Arrived late but works fine."])
    dl = lambda t: np.full(len(t), 0.5)
    out = predict_texts(toy_pipeline, dl, texts, [5, 5, 5, 5], threshold=0.5, modern=FakeModern(),
                        modern_threshold=0.99)
    assert out.loc[2, "prediction"] == ""                    # empty text is skipped, not guessed
    assert out.loc[0, "prediction"] == "Original"
    assert out.loc[1, "prediction"] == "Computer-generated"
    assert out.loc[3, "flagged_by"] in ("modern AI style", "both checks")
    np.testing.assert_allclose(out.loc[[0, 1], "prob_original"],
                               (out.loc[[0, 1], "prob_original_lr"] + 0.5) / 2)

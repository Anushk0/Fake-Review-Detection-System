import importlib.util

import pytest

import modern_detector
from modern_detector import AI_INDEX, ModernAIDetector


def test_machine_is_output_zero():
    # MAGE's reference detect(): label2decisions {0: "machine-generated", 1: "human-written"}
    assert AI_INDEX == 0


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        ModernAIDetector("nope")


torch_ok = importlib.util.find_spec("torch") is not None


@pytest.mark.skipif(not torch_ok, reason="torch not installed")
def test_score_is_softmax_of_output_zero():
    import torch

    class Tok:
        def __call__(self, texts, **kw):
            return {"input_ids": torch.zeros((len(texts), 3), dtype=torch.long)}

    class Model:
        def __call__(self, input_ids):
            class Out:
                logits = torch.tensor([[2.0, 0.0]] * input_ids.shape[0])
            return Out()

    det = ModernAIDetector.__new__(ModernAIDetector)
    det.torch, det.tokenizer, det.model = torch, Tok(), Model()
    det.preprocess, det.max_tokens = False, 512
    probs, logits = det.predict_detailed(["a", "b"])
    assert probs[0] == pytest.approx(torch.softmax(torch.tensor([2.0, 0.0]), 0)[0].item())
    assert logits == [2.0, 2.0]


@pytest.mark.skipif(importlib.util.find_spec("cleantext") is None, reason="clean-text not installed")
def test_mage_preprocessing_removes_line_breaks_and_tokenisation_spaces():
    from mage_preprocess import preprocess
    assert preprocess("Great product !\n\nWorks well , I think .") == "Great product! Works well, I think."


def test_load_failure_is_reported_not_silent(monkeypatch):
    def boom(mode):
        raise ImportError("no torch")
    monkeypatch.setattr(modern_detector, "ModernAIDetector", boom)
    assert modern_detector.load_modern_detector() is None
    assert "no torch" in modern_detector.LOAD_ERROR

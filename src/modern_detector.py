"""
Second check for reviews written by modern AI (ChatGPT, GPT-4, LLaMA and similar).

The project's own models were trained on GPT-2-era fake reviews and do not
recognise fluent modern AI text. This module runs a pretrained transformer
detector for that: the Longformer released with the MAGE benchmark
("MAGE: Machine-generated Text Detection in the Wild", Li et al., ACL 2024),
https://huggingface.co/nealcly/detection-longformer, trained on human text and
text from 27 language models.

Provenance (checked 2026-10 on the Hugging Face hub): LongformerForSequenceClassification,
longformer-base-4096, two unnamed outputs (id2label {0: 0, 1: 1}), commit REVISION below,
no model card. The meaning of the outputs comes from MAGE's reference code
(deployment/utils.py, detect()): output 0 = machine-generated, 1 = human-written.

Two ways to run it:
  - "project" (the original setting of this project): raw text, truncated to 512
    tokens, score = softmax probability of output 0.
  - "mage" (MAGE's reference usage): MAGE's text preprocessing (mage_preprocess.py),
    up to the model's 4,096-token limit, and the same softmax score. MAGE's own
    demo instead thresholds the raw output-0 logit at 3.0858; `logit_machine` is
    returned too so that rule can be evaluated.

The model (about 600 MB) is downloaded from Hugging Face on first use and cached.
It needs the optional packages torch and transformers (and clean-text for "mage").
It is slow on a CPU, about 1 second per review.
"""
import os

REPO = "nealcly/detection-longformer"
REVISION = "61d3aab2e0e5afcce6cb00b92661b0f9ee18ac2b"
AI_INDEX = 0          # output 0 = machine-generated, 1 = human-written (MAGE's convention)
MODES = {"project": {"preprocess": False, "max_tokens": 512},
         "mage": {"preprocess": True, "max_tokens": 4096}}
MAGE_LOGIT_THRESHOLD = 3.08583984375   # MAGE demo: machine if logit[0] > this (their th = -3.0858 on -logit[0])


class ModernAIDetector:
    def __init__(self, mode="project"):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {list(MODES)}")
        self.mode = mode
        self.preprocess = MODES[mode]["preprocess"]
        self.max_tokens = MODES[mode]["max_tokens"]
        if self.preprocess:
            import cleantext  # noqa: F401  fail now, not on the first review
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, logging
        logging.set_verbosity_error()
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(REPO, revision=REVISION)
        self.model = AutoModelForSequenceClassification.from_pretrained(REPO, revision=REVISION).eval()

    def _prepare(self, texts):
        texts = [t if isinstance(t, str) else "" for t in texts]
        if self.preprocess:
            from mage_preprocess import preprocess
            texts = [preprocess(t) for t in texts]
        return texts

    def predict_detailed(self, texts, batch_size=8, progress=None):
        """For each text: (probability of AI = softmax output 0, raw output-0 logit)."""
        texts = self._prepare(texts)
        probs, logits = [], []
        for start in range(0, len(texts), batch_size):
            enc = self.tokenizer(texts[start:start + batch_size], truncation=True, max_length=self.max_tokens,
                                 padding=True, return_tensors="pt")
            with self.torch.no_grad():
                out = self.model(**enc).logits
            probs.extend(self.torch.softmax(out, dim=-1)[:, AI_INDEX].tolist())
            logits.extend(out[:, AI_INDEX].tolist())
            if progress is not None:
                progress(min(start + batch_size, len(texts)), len(texts))
        return probs, logits

    def predict(self, texts, batch_size=8, progress=None):
        """Probability that each text was written by a modern AI model."""
        return self.predict_detailed(texts, batch_size, progress)[0]


LOAD_ERROR = None


def load_modern_detector(mode="project"):
    """The detector, or None if torch/transformers are missing or the model can't be downloaded
    (the reason is kept in LOAD_ERROR)."""
    global LOAD_ERROR
    try:
        return ModernAIDetector(mode)
    except Exception as e:
        LOAD_ERROR = f"{type(e).__name__}: {e}"
        return None

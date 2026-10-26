"""
Feature extraction shared by training (train.py) and the Streamlit app (app.py).

Everything the model needs is built from two inputs: the review text and its
star rating. Keeping the code in one module means the app computes features
exactly the way the model was trained on them.
"""
import re
import string

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob

WORD_RE = re.compile(r"[A-Za-z']+")
SENTENCE_RE = re.compile(r"[.!?]+")
VOWEL_GROUP_RE = re.compile(r"[aeiouy]+")


def _syllables(word):
    """Rough syllable count: groups of vowels, minus a silent trailing 'e'."""
    word = word.lower()
    count = len(VOWEL_GROUP_RE.findall(word))
    if word.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


def _repeated_bigram_ratio(words):
    """Share of word bigrams in the review that occur more than once in it."""
    bigrams = list(zip(words, words[1:]))
    if not bigrams:
        return 0.0
    counts = pd.Series(bigrams).value_counts()
    return float(counts[counts > 1].sum()) / len(bigrams)


STAT_FEATURES = [
    "word_count",
    "avg_word_length",
    "lexical_diversity",
    "repeated_bigram_ratio",
    "punctuation_ratio",
    "uppercase_ratio",
    "exclamation_count",
    "avg_sentence_length",
    "flesch_reading_ease",
    "sentiment_polarity",
    "subjectivity",
    "rating",
    "sentiment_rating_mismatch",
]


def clean_rating(rating):
    """
    A star rating as a float in [1, 5]. Missing or unreadable ratings raise an error
    instead of becoming NaN, which would otherwise crash the scaler or, worse, be
    silently imputed somewhere downstream. Callers that want a default (predict.py,
    the app) substitute it before calling the model.
    """
    try:
        value = float(rating)
    except (TypeError, ValueError):
        raise ValueError(f"star rating must be a number from 1 to 5, got {rating!r}") from None
    if not np.isfinite(value):
        raise ValueError("star rating is missing; pass a number from 1 to 5")
    return min(max(value, 1.0), 5.0)


def text_stats(text, rating):
    """Hand-crafted style, sentiment and repetition features for one review."""
    text = text if isinstance(text, str) else ""
    rating = clean_rating(rating)
    words = WORD_RE.findall(text)
    lower_words = [w.lower() for w in words]
    n_words = len(words)
    n_chars = max(len(text), 1)
    n_sentences = max(len([s for s in SENTENCE_RE.split(text) if s.strip()]), 1)
    syllables = sum(_syllables(w) for w in words)

    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    # Map 1-5 stars onto the same -1..1 scale as TextBlob polarity.
    rating_scaled = (float(rating) - 3.0) / 2.0

    return [
        n_words,
        (sum(len(w) for w in words) / n_words) if n_words else 0.0,
        (len(set(lower_words)) / n_words) if n_words else 0.0,
        _repeated_bigram_ratio(lower_words),
        sum(c in string.punctuation for c in text) / n_chars,
        sum(c.isupper() for c in text) / n_chars,
        text.count("!"),
        n_words / n_sentences,
        (206.835 - 1.015 * (n_words / n_sentences) - 84.6 * (syllables / n_words)) if n_words else 0.0,
        polarity,
        blob.sentiment.subjectivity,
        float(rating),
        abs(polarity - rating_scaled),
    ]


class TextStats(BaseEstimator, TransformerMixin):
    """Turns a DataFrame with 'text' and 'rating' columns into STAT_FEATURES."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([text_stats(t, r) for t, r in zip(X["text"], X["rating"])], dtype=np.float64)

    def get_feature_names_out(self, input_features=None):
        return np.array(STAT_FEATURES)


SIMILARITY_FEATURES = ["max_similarity", "near_duplicate_count"]


class CorpusSimilarity(BaseEstimator, TransformerMixin):
    """
    Compares each review with the reviews seen in training.

    max_similarity       - cosine similarity to the closest training review
    near_duplicate_count - training reviews with similarity >= near_duplicate_threshold

    A review is never compared with itself: training reviews (fit_transform) and
    any later query whose text is *identical* to a training review (transform)
    have the training entries with that exact text removed before scoring. The
    same rule runs at training and inference time, so a training review scored
    again gets exactly the features it had during training.

    Earlier versions instead zeroed every similarity >= 0.999 at inference only.
    That rule was not used during training (where identical copies of a review
    still matched at 1.0) and it also discarded different reviews that merely
    have the same TF-IDF vector (e.g. differing only in case or punctuation),
    creating a train/serve mismatch. Pipelines pickled before this change fall
    back to the old rule; retrain to use the new one.
    """

    EXACT_MATCH = 0.999  # only used by pipelines pickled before the identity rule

    def __init__(self, near_duplicate_threshold=0.5, chunk_size=500):
        self.near_duplicate_threshold = near_duplicate_threshold
        self.chunk_size = chunk_size

    def fit(self, X, y=None):
        texts = _as_texts(X)
        self.vectorizer_ = TfidfVectorizer(sublinear_tf=True, min_df=2, stop_words="english", dtype=np.float32)
        self.corpus_ = self.vectorizer_.fit_transform(texts).tocsr()
        self.corpus_text_ = texts
        # labels of the training reviews, so explanations can say what the neighbours are
        self.corpus_labels_ = None if y is None else np.asarray(y)
        self.text_index_ = {}
        for i, t in enumerate(texts):
            self.text_index_.setdefault(t, []).append(i)
        return self

    def _self_matches(self, texts):
        """For each query text, the training indices holding that exact text (empty if none)."""
        index = getattr(self, "text_index_", None)
        return [index.get(t, []) for t in texts] if index is not None else None

    def _scores(self, query, texts):
        corpus_t = self.corpus_.T.tocsc()
        matches = self._self_matches(texts)
        rows = []
        for start in range(0, query.shape[0], self.chunk_size):
            sims = (query[start:start + self.chunk_size] @ corpus_t).toarray()
            if matches is not None:
                for i, own in enumerate(matches[start:start + self.chunk_size]):
                    sims[i, own] = 0.0
            else:  # legacy pickles: the old inference-only rule
                sims[sims >= self.EXACT_MATCH] = 0.0
            rows.append(np.column_stack([
                sims.max(axis=1),
                (sims >= self.near_duplicate_threshold).sum(axis=1),
            ]))
        return np.vstack(rows).astype(np.float64)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self._scores(self.corpus_, self.corpus_text_)

    def transform(self, X):
        texts = _as_texts(X)
        return self._scores(self.vectorizer_.transform(texts), texts)

    def most_similar(self, text, k=3):
        """
        The k closest training reviews to `text`, as dicts with keys similarity,
        text, label (0/1, or None for pipelines trained before labels were stored)
        and exact_copy (True when the training review has exactly this text).
        Similarity is lexical overlap only; it is evidence, not proof of authorship.
        """
        if not isinstance(text, str) or not text.strip() or self.corpus_.shape[0] == 0:
            return []
        sims = (self.vectorizer_.transform([text]) @ self.corpus_.T).toarray().ravel()
        top = [i for i in np.argsort(sims)[::-1][:k] if sims[i] > 0]
        labels = getattr(self, "corpus_labels_", None)
        return [{"similarity": float(sims[i]), "text": self.corpus_text_[i],
                 "label": None if labels is None else int(labels[i]),
                 "exact_copy": self.corpus_text_[i] == text} for i in top]

    def get_feature_names_out(self, input_features=None):
        return np.array(SIMILARITY_FEATURES)


def _as_texts(X):
    """A list of strings; missing values become empty strings."""
    return [t if isinstance(t, str) else "" for t in X]


def to_frame(texts, ratings):
    """Build the two-column input the pipeline expects."""
    return pd.DataFrame({"text": list(texts), "rating": list(ratings)})


def decision_breakdown(pipeline, text, rating):
    """
    The logistic-regression decision for one review, split into its parts:
    intercept + sum of feature contributions = logit of P(Original).
    Contributions are additive on the logit scale only; their effect on the
    probability depends on all the others (the sigmoid is non-linear).
    """
    features = pipeline.named_steps["features"]
    clf = pipeline.named_steps["clf"]
    x = sparse.csr_matrix(features.transform(to_frame([text], [rating])))
    contributions = float(x.multiply(clf.coef_[0]).sum())
    intercept = float(clf.intercept_[0])
    logit = intercept + contributions
    return {"intercept": intercept, "contributions": contributions, "logit": logit,
            "prob_original": float(1 / (1 + np.exp(-logit)))}


def explain(pipeline, text, rating, top_k=8):
    """
    Per-feature contributions to the model's decision for one review.

    The classifier is logistic regression, so each feature's contribution is
    simply its (transformed) value times its coefficient: TF-IDF weight times
    coefficient for words, standardised value times coefficient for the signals.
    Positive values push the logit toward class 1 (Original), negative values
    toward class 0 (Computer-generated). These explain the logistic-regression
    half of Stage A only; the BiLSTM and the modern-AI detector are not explained.
    Returns (word_contributions, feature_contributions) as DataFrames.
    """
    features = pipeline.named_steps["features"]
    clf = pipeline.named_steps["clf"]
    x = features.transform(to_frame([text], [rating]))
    x = sparse.csr_matrix(x)
    names = features.get_feature_names_out()
    idx = x.indices
    df = pd.DataFrame({"feature": names[idx], "contribution": x.data * clf.coef_[0][idx]})
    is_word = df["feature"].str.startswith("tfidf__")
    words = df[is_word].copy()
    words["feature"] = words["feature"].str.replace("tfidf__", "", regex=False)
    words = words.reindex(words["contribution"].abs().sort_values(ascending=False).index).head(top_k)

    raw = dict(zip(STAT_FEATURES, TextStats().transform(to_frame([text], [rating]))[0]))
    if "similarity" in features.named_transformers_:
        sim = features.named_transformers_["similarity"].named_steps["sim"]
        raw.update(zip(SIMILARITY_FEATURES, sim.transform([text])[0]))
    other = df[~is_word].copy()
    other["feature"] = other["feature"].str.split("__").str[-1]
    other["value"] = other["feature"].map(raw)
    other = other.reindex(other["contribution"].abs().sort_values(ascending=False).index)
    return words.reset_index(drop=True), other.reset_index(drop=True)

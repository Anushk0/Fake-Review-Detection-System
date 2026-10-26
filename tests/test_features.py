import math

import numpy as np
import pandas as pd
import pytest

from features import STAT_FEATURES, TextStats, _repeated_bigram_ratio, clean_rating, text_stats


def stats(text, rating=5):
    return dict(zip(STAT_FEATURES, text_stats(text, rating)))


def test_thirteen_named_features():
    assert len(STAT_FEATURES) == 13
    assert len(text_stats("Nice.", 4)) == 13


@pytest.mark.parametrize("text", ["", "   ", None, "!!!", "a"])
def test_empty_and_tiny_reviews_give_finite_values(text):
    values = text_stats(text, 3)
    assert all(math.isfinite(v) for v in values)


def test_empty_review_has_zero_text_statistics():
    s = stats("", 3)
    assert s["word_count"] == 0 and s["lexical_diversity"] == 0 and s["flesch_reading_ease"] == 0


@pytest.mark.parametrize("rating", [None, float("nan"), "n/a", ""])
def test_missing_rating_is_an_error_not_nan(rating):
    with pytest.raises(ValueError):
        text_stats("Works well.", rating)


def test_rating_is_clipped_to_scale():
    assert clean_rating(7) == 5.0 and clean_rating(0) == 1.0 and clean_rating("4") == 4.0


def test_repeated_bigram_ratio_counts_every_occurrence():
    # bigrams: (a,b) (b,a) (a,b) -> the two (a,b) occurrences are repeated -> 2/3
    assert _repeated_bigram_ratio(["a", "b", "a", "b"]) == pytest.approx(2 / 3)
    assert _repeated_bigram_ratio(["one", "two", "three"]) == 0.0
    assert _repeated_bigram_ratio(["single"]) == 0.0


def test_sentiment_rating_mismatch_maps_stars_to_polarity_scale():
    neutral = stats("The box contains a cable.", 3)       # polarity 0, 3 stars -> 0 on the [-1, 1] scale
    assert neutral["sentiment_rating_mismatch"] == pytest.approx(abs(neutral["sentiment_polarity"] - 0.0))
    happy_one_star = stats("Excellent, wonderful, perfect!", 1)
    assert happy_one_star["sentiment_rating_mismatch"] == pytest.approx(happy_one_star["sentiment_polarity"] + 1.0)


def test_basic_counts():
    s = stats("Great product. Works WELL!", 5)
    assert s["word_count"] == 4
    assert s["exclamation_count"] == 1
    assert s["avg_sentence_length"] == pytest.approx(2.0)
    assert s["lexical_diversity"] == 1.0


def test_transformer_matches_function_row_by_row():
    X = pd.DataFrame({"text": ["Good.", "Bad bad bad.", ""], "rating": [5, 1, 3]})
    out = TextStats().transform(X)
    assert out.shape == (3, 13)
    np.testing.assert_allclose(out[1], text_stats("Bad bad bad.", 1))

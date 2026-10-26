import numpy as np

from features import CorpusSimilarity

# every content word appears in at least two reviews, so it survives the vectoriser's min_df=2
CORPUS = ["battery lasts long charges fast",
          "battery lasts long charges slowly",
          "soft blanket warm cozy",
          "blanket warm cozy soft gift wrapped nicely gift wrapped nicely",
          "service terrible refund never slowly",
          "service terrible refund never fast"]
LABELS = [1, 0, 1, 1, 0, 1]


def fitted():
    sim = CorpusSimilarity(near_duplicate_threshold=0.5)
    train_features = sim.fit_transform(CORPUS, LABELS)
    return sim, train_features


def test_training_features_compare_each_review_with_the_others_only():
    sim, train_features = fitted()
    V = sim.vectorizer_.transform(CORPUS)
    S = (V @ V.T).toarray()
    np.fill_diagonal(S, 0.0)
    np.testing.assert_allclose(train_features[:, 0], S.max(axis=1), atol=1e-6)
    np.testing.assert_allclose(train_features[:, 1], (S >= 0.5).sum(axis=1))


def test_rescoring_a_training_review_reproduces_its_training_features():
    # regression: the train/serve rule must be the same for a review already in the corpus
    sim, train_features = fitted()
    np.testing.assert_allclose(sim.transform(CORPUS), train_features)


def test_new_review_with_same_tfidf_vector_is_not_discarded():
    # regression: the old rule zeroed every similarity >= 0.999 at inference, so a different review
    # that only differs in case/punctuation lost its closest match
    sim, _ = fitted()
    out = sim.transform(["Battery lasts long, charges FAST!"])
    assert out[0, 0] > 0.999


def test_near_duplicate_count_uses_threshold():
    sim, _ = fitted()
    out = sim.transform(["battery lasts long"])
    assert out[0, 1] == 2


def test_most_similar_reports_labels_and_exact_copies():
    sim, _ = fitted()
    top = sim.most_similar(CORPUS[1], k=2)
    assert top[0]["exact_copy"] and top[0]["label"] == 0 and top[0]["similarity"] > 0.999
    assert not top[1]["exact_copy"] and top[1]["label"] == 1
    assert sim.most_similar("", k=3) == []
    assert all(n["similarity"] > 0 for n in sim.most_similar("blanket", k=6))


def test_missing_texts_are_treated_as_empty():
    sim, _ = fitted()
    out = sim.transform([None, np.nan])
    assert out.shape == (2, 2) and (out == 0).all()

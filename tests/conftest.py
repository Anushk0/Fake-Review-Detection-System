import os
import sys

import numpy as np
import pandas as pd
import pytest

SRC = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.abspath(SRC))

HUMAN = ["I bought this for my daughter and she uses it every day, even though the strap is a bit short.",
         "Arrived late but works fine. The battery lasts about two days because I use it a lot.",
         "Not what I expected at all. The color was off and the fabric felt cheap, so I returned it.",
         "Great value for the money. My dog loves these, but they do not last long with him.",
         "The book started slow, though the last third kept me up at night. Would read again."]
MACHINE = ["This is a great product and the quality is very good and the price is very good and",
           "I love it and it is very good. I love it and it is very good. The only thing I",
           "Very good product. Very good quality. I will keep buying this and the only thing",
           "This book is very good and the story is very good and the characters are very good and the",
           "I have one and it is very good and I will keep it and it works very good and"]


@pytest.fixture(scope="session")
def toy_data():
    rng = np.random.default_rng(0)
    texts, labels, ratings = [], [], []
    for i in range(40):
        for t in HUMAN:
            texts.append(f"{t} Order {i}."); labels.append(1); ratings.append(int(rng.integers(1, 6)))
        for t in MACHINE:
            texts.append(f"{t} item {i}"); labels.append(0); ratings.append(int(rng.integers(1, 6)))
    return pd.DataFrame({"text": texts, "rating": ratings}), np.array(labels)


@pytest.fixture(scope="session")
def toy_pipeline(toy_data):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from train import build_features
    X, y = toy_data
    pipe = Pipeline([("features", build_features()), ("clf", LogisticRegression(C=1, solver="liblinear"))])
    return pipe.fit(X, y)

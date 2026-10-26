"""
Train the deep learning model (BiLSTM) on the same splits as train.py, then
score it alone and as an ensemble with the logistic regression pipeline.

    cd src
    python train.py      # first, if model_pipeline.joblib does not exist yet
    python train_dl.py

Writes:
    src/dl_model.keras              - BiLSTM with its tokenizer built in (takes raw text)
    Reports/metrics.json            - adds "deep_learning" and "ensemble" sections
    Reports/dl_training_curve.png
"""
import hashlib
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import keras
import tensorflow as tf
from keras import layers

from artifacts import load_metrics, save_metrics
from train import MODEL_OUT, REPORTS, SEED, load_splits, scores_by_source

DL_MODEL_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dl_model.keras")
VOCAB_SIZE = 20000
SEQUENCE_LENGTH = 200   # words; 93% of reviews fit in full
EMBEDDING_DIM = 64
EPOCHS = 6
BATCH_SIZE = 64


def build_model(train_texts):
    vectorize = layers.TextVectorization(max_tokens=VOCAB_SIZE, output_sequence_length=SEQUENCE_LENGTH,
                                         name="tokenizer")
    vectorize.adapt(train_texts, batch_size=1024)

    inputs = keras.Input(shape=(1,), dtype="string", name="review_text")
    x = vectorize(inputs)
    x = layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM, mask_zero=True, name="embedding")(x)
    x = layers.Bidirectional(layers.LSTM(64), name="bilstm")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="prob_original")(x)

    model = keras.Model(inputs, outputs, name="bilstm_review_classifier")
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def as_input(texts):
    return tf.constant([[t] for t in texts], dtype=tf.string)


def as_dataset(texts, labels, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((as_input(texts), np.asarray(labels, dtype="float32")))
    if shuffle:
        ds = ds.shuffle(len(labels), seed=SEED)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def main():
    t0 = time.time()
    keras.utils.set_random_seed(SEED)
    _, _, X_fit, y_fit, X_val, y_val, X_test, y_test = load_splits()
    print(f"train={len(X_fit)} val={len(X_val)} test={len(X_test)}")

    model = build_model(tf.constant(list(X_fit["text"]), dtype=tf.string))
    model.summary()
    history = model.fit(
        as_dataset(X_fit["text"], y_fit, shuffle=True),
        validation_data=as_dataset(X_val["text"], y_val),
        epochs=EPOCHS, verbose=2,
        callbacks=[keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=1, restore_best_weights=True)],
    )

    dl_proba = model.predict(as_input(X_test["text"]), batch_size=256, verbose=0).ravel()
    lr_proba = joblib.load(MODEL_OUT).predict_proba(X_test)[:, 1]
    ensemble_proba = (dl_proba + lr_proba) / 2

    by_source = {name: scores_by_source(X_test, y_test, p)
                 for name, p in [("lr", lr_proba), ("dl", dl_proba), ("ensemble", ensemble_proba)]}
    dl, ens = by_source["dl"]["all"], by_source["ensemble"]["all"]
    for name, label in [("lr", "LogisticRegression pipeline"), ("dl", "BiLSTM"), ("ensemble", "Ensemble")]:
        print(f"{label}:")
        for source, m in by_source[name].items():
            print(f"  {source:10s} n={m['n']:<6} accuracy {m['accuracy']:.2%}  F1 {m['f1']:.2%}")

    model.save(DL_MODEL_OUT)

    # per-review test probabilities (all P(Original)), for paired tests and confidence intervals
    pd.DataFrame({"text_sha1": [hashlib.sha1(t.encode("utf-8")).hexdigest() for t in X_test["text"]],
                  "rating": X_test["rating"].to_numpy(), "label": y_test, "p_lr": lr_proba, "p_dl": dl_proba,
                  "p_ensemble": ensemble_proba}).to_csv(os.path.join(REPORTS, "test_predictions.csv"), index=False)

    metrics = load_metrics()
    metrics["stale_sections"] = [s for s in metrics.get("stale_sections", []) if s not in ("deep_learning", "ensemble")]
    metrics["deep_learning"] = {
        "model": f"TextVectorization({VOCAB_SIZE} words, {SEQUENCE_LENGTH} tokens) -> Embedding({EMBEDDING_DIM}) "
                 "-> BiLSTM(64) -> Dense(32) -> sigmoid",
        "epochs_trained": len(history.history["loss"]),
        "best_val_accuracy": round(max(history.history["val_accuracy"]), 4),
        "test": dl,
        "test_by_source": by_source["dl"],
    }
    metrics["ensemble"] = {"model": "mean of LogisticRegression pipeline and BiLSTM probabilities", "test": ens,
                           "test_by_source": by_source["ensemble"]}
    save_metrics(metrics)

    fig, ax = plt.subplots(figsize=(6, 4))
    epochs = range(1, len(history.history["loss"]) + 1)
    ax.plot(epochs, history.history["accuracy"], marker="o", label="train accuracy")
    ax.plot(epochs, history.history["val_accuracy"], marker="o", label="validation accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("BiLSTM training")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "dl_training_curve.png"), dpi=120)

    print(f"saved {DL_MODEL_OUT} ({os.path.getsize(DL_MODEL_OUT) / 1e6:.1f} MB) in {time.time() - t0:.0f}s total")


if __name__ == "__main__":
    main()

"""
Predict a whole CSV of reviews at once (also used by the app's batch tab).

    cd src
    python predict.py reviews.csv --text-column text_ --rating-column rating -o predictions.csv

Options:
    --rating-column      column with 1-5 star ratings (omit to use --default-rating for every row)
    --default-rating     rating used when the rating column is missing or empty (default 5)
    --label-column       column with true labels (CG/OR or 0/1); prints accuracy if given
    --threshold          probability of "Original" needed to predict Original (default: the
                         threshold chosen by choose_threshold.py, or 0.5 if it hasn't been run)
    --no-modern          skip the modern-AI check (much faster; it takes ~1.5 s per review on a CPU)

A review is predicted computer-generated if the ensemble's probability of Original is below the
threshold, OR the modern-AI detector (modern_detector.py) is confident it was written by modern AI.

The output CSV is the input with these columns added:
    prob_original_lr, prob_original_dl, prob_original (the ensemble),
    prob_ai_modern (modern-AI detector), flagged_by, prediction
"""
import argparse
import os
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from artifacts import DL_MODEL_PATH, LABELS, MODEL_PATH, load_metrics, stage_a_threshold, stage_b_threshold
from features import to_frame

LABEL_NAMES = LABELS  # 0 = computer-generated, 1 = original; every probability here is P(Original)
LABEL_ALIASES = {"cg": 0, "computer-generated": 0, "computer generated": 0, "0": 0, "0.0": 0,
                 "or": 1, "original": 1, "1": 1, "1.0": 1}
DL_LOAD_ERROR = None  # why the BiLSTM could not be loaded, if it couldn't


def default_threshold(metrics=None, dl_available=True):
    """The Stage A threshold chosen by choose_threshold.py (0.5 if none). See threshold_warning()."""
    return stage_a_threshold(dl_available)[0]


def threshold_warning(dl_available=True):
    """Why the Stage A threshold cannot be trusted for the models on disk, or None."""
    return stage_a_threshold(dl_available)[1]


def default_modern_threshold(metrics=None):
    """The Stage B threshold chosen by calibrate_modern.py, or None if it hasn't been run."""
    return stage_b_threshold()


def ensemble_probability(p_lr, p_dl=None):
    """Stage A score: mean of the two models' P(Original), or the LR probability alone."""
    p_lr = np.asarray(p_lr, dtype=float)
    return p_lr if p_dl is None else (p_lr + np.asarray(p_dl, dtype=float)) / 2


def fuse(p_original, p_ai=None, threshold=0.5, modern_threshold=None):
    """
    The final decision. Returns (stage_a_flag, stage_b_flag, computer_generated) as boolean arrays.
    Stage A flags when P(Original) < threshold; Stage B flags when P(AI) >= modern_threshold.
    A missing Stage B score (NaN, or the check not run) never flags.
    """
    p_original = np.asarray(p_original, dtype=float)
    a = p_original < threshold
    if p_ai is None or modern_threshold is None:
        b = np.zeros_like(a)
    else:
        p_ai = np.asarray(p_ai, dtype=float)
        b = np.nan_to_num(p_ai, nan=-np.inf) >= modern_threshold
    return a, b, a | b


def load_pipeline():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


def load_evidence_index(pipeline=None):
    """
    The training-review index used to show the most similar training reviews: the classifier's own
    similarity step if it has one, otherwise src/similarity_index.joblib, otherwise None.
    """
    if pipeline is not None:
        transformers = pipeline.named_steps["features"].named_transformers_
        if "similarity" in transformers:
            return transformers["similarity"].named_steps["sim"]
    path = os.path.join(os.path.dirname(MODEL_PATH), "similarity_index.joblib")
    return joblib.load(path) if os.path.exists(path) else None


def load_dl_model():
    """
    The BiLSTM is optional: returns a function mapping a list of texts to
    probabilities of "Original", or None if the model or TensorFlow is unavailable
    (the reason is kept in DL_LOAD_ERROR so callers can report it).
    """
    global DL_LOAD_ERROR
    if not os.path.exists(DL_MODEL_PATH):
        DL_LOAD_ERROR = f"{DL_MODEL_PATH} not found"
        return None
    try:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import keras
        import tensorflow as tf
        model = keras.models.load_model(DL_MODEL_PATH)
    except Exception as e:
        DL_LOAD_ERROR = f"{type(e).__name__}: {e}"
        return None

    def predict(texts):
        inputs = tf.constant([[t] for t in texts], dtype=tf.string)
        return model.predict(inputs, batch_size=256, verbose=0).ravel().astype(float)

    return predict


def clean_ratings(ratings, default_rating):
    """Numeric 1-5 ratings; missing or unreadable values become default_rating."""
    r = pd.to_numeric(pd.Series(ratings), errors="coerce").fillna(default_rating)
    return r.clip(1, 5).to_numpy()


def predict_texts(pipeline, dl_predict, texts, ratings, threshold=0.5, chunk_size=1000, progress=None,
                  modern=None, modern_threshold=None):
    """
    Predict many reviews in chunks. `progress(done, total)` is called as work completes
    (for the modern-AI check if it runs, since that takes almost all the time).
    Empty texts get no prediction (NaN). Returns a DataFrame aligned with `texts`.
    """
    texts = pd.Series(texts).fillna("").astype(str).reset_index(drop=True)
    ratings = pd.Series(ratings).reset_index(drop=True)
    out = pd.DataFrame(np.nan, index=texts.index,
                       columns=["prob_original_lr", "prob_original_dl", "prob_original"])
    valid = texts.str.strip().ne("").to_numpy()
    idx = np.flatnonzero(valid)

    for start in range(0, len(idx), chunk_size):
        rows = idx[start:start + chunk_size]
        lr = pipeline.predict_proba(to_frame(texts[rows], ratings[rows]))[:, 1]
        out.loc[rows, "prob_original_lr"] = lr
        dl = dl_predict(list(texts[rows])) if dl_predict is not None else None
        if dl is not None:
            out.loc[rows, "prob_original_dl"] = dl
        out.loc[rows, "prob_original"] = ensemble_probability(lr, dl)
        if progress is not None and modern is None:
            progress(min(start + chunk_size, len(idx)), len(idx))

    p_ai = None
    if modern is not None and modern_threshold is not None:
        out["prob_ai_modern"] = np.nan
        out.loc[idx, "prob_ai_modern"] = modern.predict(list(texts[idx]), progress=progress)
        p_ai = out["prob_ai_modern"]
    classic_flag, modern_flag, flagged = fuse(out["prob_original"], p_ai, threshold, modern_threshold)

    out["flagged_by"] = np.select([classic_flag & modern_flag, classic_flag, modern_flag],
                                  ["both checks", "GPT-2-style patterns", "modern AI style"], "")
    out["prediction"] = np.where(out["prob_original"].isna(), "",
                                 np.where(flagged, LABEL_NAMES[0], LABEL_NAMES[1]))
    if dl_predict is None:
        out = out.drop(columns="prob_original_dl")
    return out


def evaluate(true_labels, predictions):
    """
    Accuracy etc. against true labels given as CG/OR or 0/1.
    Returns None if the labels can't be read.
    """
    y_true = pd.Series(true_labels).astype(str).str.strip().str.lower().map(LABEL_ALIASES)
    y_pred = pd.Series(predictions).map({v: k for k, v in LABEL_NAMES.items()})
    keep = y_true.notna() & y_pred.notna()
    if keep.sum() == 0:
        return None
    y_true, y_pred = y_true[keep].astype(int), y_pred[keep].astype(int)
    return {
        "rows": int(keep.sum()),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]),
    }


def main():
    parser = argparse.ArgumentParser(description="Predict a CSV of reviews.")
    parser.add_argument("csv")
    parser.add_argument("--text-column", required=True)
    parser.add_argument("--rating-column")
    parser.add_argument("--default-rating", type=float, default=5)
    parser.add_argument("--label-column")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--no-modern", action="store_true", help="skip the slow modern-AI check")
    parser.add_argument("-o", "--output", default="predictions.csv")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    t0 = time.time()
    df = pd.read_csv(args.csv)
    for col in (args.text_column, args.rating_column, args.label_column):
        if col and col not in df.columns:
            parser.error(f"column '{col}' not in the CSV; columns are: {', '.join(df.columns)}")

    pipeline = load_pipeline()
    if pipeline is None:
        parser.error("model_pipeline.joblib not found; run train.py first")
    dl_predict = load_dl_model()
    print("models: logistic regression" + (" + BiLSTM (ensemble)" if dl_predict else
                                           f" only (BiLSTM unavailable: {DL_LOAD_ERROR})"))
    if args.threshold is None:
        args.threshold = default_threshold(dl_available=dl_predict is not None)
        warning = threshold_warning(dl_available=dl_predict is not None)
        if warning:
            print(f"WARNING: {warning}")
    print(f"Stage A threshold (flag if P(Original) < threshold): {args.threshold}")
    modern, modern_threshold = None, default_modern_threshold()
    if not args.no_modern and modern_threshold is not None:
        from modern_detector import load_modern_detector
        modern = load_modern_detector()
        if modern is None:
            print("modern-AI check unavailable (install torch and transformers); skipping it")
        else:
            minutes = 1.5 * len(df) / 60
            duration = f"about {minutes:.0f} min" if minutes >= 1 else "under a minute"
            print(f"modern-AI check on (threshold {modern_threshold}): {duration} for {len(df)} reviews; "
                  "use --no-modern to skip")

    ratings = clean_ratings(df[args.rating_column] if args.rating_column else [np.nan] * len(df),
                            args.default_rating)
    result = predict_texts(pipeline, dl_predict, df[args.text_column], ratings, args.threshold,
                           progress=lambda done, total: print(f"  {done}/{total} reviews", flush=True),
                           modern=modern, modern_threshold=modern_threshold)
    out = pd.concat([df.reset_index(drop=True), result], axis=1)
    out.to_csv(args.output, index=False)

    counts = result["prediction"].value_counts()
    print(f"\n{len(df)} rows -> {args.output} in {time.time() - t0:.0f}s")
    for name in LABEL_NAMES.values():
        print(f"  {name}: {counts.get(name, 0)}")
    for reason, n in result.loc[result["flagged_by"] != "", "flagged_by"].value_counts().items():
        print(f"    flagged by {reason}: {n}")
    skipped = (result["prediction"] == "").sum()
    if skipped:
        print(f"  skipped (empty text): {skipped}")

    if args.label_column:
        m = evaluate(df[args.label_column], result["prediction"])
        if m is None:
            print("\nCould not read the labels (expected CG/OR or 0/1).")
        else:
            print(f"\nAgainst '{args.label_column}' ({m['rows']} rows): accuracy {m['accuracy']:.2%}, "
                  f"precision {m['precision']:.2%}, recall {m['recall']:.2%}, F1 {m['f1']:.2%}")
            print("confusion matrix (rows = true CG/OR, columns = predicted CG/OR):")
            print(m["confusion_matrix"])


if __name__ == "__main__":
    main()

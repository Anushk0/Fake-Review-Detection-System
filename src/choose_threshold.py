"""
Choose the decision threshold automatically: the highest threshold that
wrongly flags at most MAX_FALSE_FLAG_RATE of genuine (Original) reviews.

    cd src
    python choose_threshold.py      # after train.py and train_dl.py

The threshold is chosen on the validation split, with a logistic regression
pipeline refit on the training portion only (the deployed pipeline has seen the
validation reviews, so its probabilities there would be overconfident). It is
then checked once on the untouched test set using the deployed models.
The deployed models are not changed.

Writes the "threshold" section of Reports/metrics.json, which the app and
predict.py use as their default threshold.
"""
import hashlib
import math
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from artifacts import fingerprint, load_metrics, save_metrics, update_stage
from predict import load_dl_model, load_pipeline
from train import REPORTS, build_features, load_external, load_splits

MAX_FALSE_FLAG_RATE = 0.01   # at most 1% of genuine reviews flagged as computer-generated


def rates(y, p, threshold):
    """(share of genuine reviews flagged, share of computer-generated reviews caught, accuracy)."""
    flagged = p < threshold
    return (float(flagged[y == 1].mean()), float(flagged[y == 0].mean()),
            float((flagged == (y == 0)).mean()))


def rate_report(y, p, threshold):
    flag, caught, acc = rates(y, p, threshold)
    return {"n": int(len(y)), "genuine_flagged": round(flag, 4), "computer_generated_caught": round(caught, 4),
            "accuracy": round(acc, 4)}


def ensemble_proba(pipeline, dl_predict, X):
    p = pipeline.predict_proba(X)[:, 1]
    if dl_predict is not None:
        p = (p + dl_predict(list(X["text"]))) / 2
    return p


def pick_threshold(y, p, max_rate):
    """Highest threshold (rounded down to 2 decimals) whose false-flag rate on genuine reviews is <= max_rate."""
    genuine = np.sort(p[y == 1])
    allowed = int(math.floor(max_rate * len(genuine)))  # genuine reviews we may flag
    # flagging means p < threshold, so the threshold can rise to the (allowed+1)-th smallest genuine probability
    t = genuine[allowed] if allowed < len(genuine) else 1.0
    t = math.floor(t * 100) / 100
    while t > 0 and rates(y, p, t)[0] > max_rate:
        t = round(t - 0.01, 2)
    return t


def main():
    warnings.filterwarnings("ignore")
    t0 = time.time()
    _, _, X_fit, y_fit, X_val, y_val, X_test, y_test = load_splits()
    metrics = load_metrics()

    print("refitting logistic regression on the training portion only...")
    groups = metrics.get("feature_groups", ["tfidf", "stats", "similarity"])
    lr_fit = Pipeline([("features", build_features(groups)),
                       ("clf", LogisticRegression(C=metrics["C"], solver="liblinear", max_iter=1000))])
    lr_fit.fit(X_fit, y_fit)
    p_val = lr_fit.predict_proba(X_val)[:, 1]

    dl_predict = load_dl_model()
    if dl_predict is not None:
        p_val = (p_val + dl_predict(list(X_val["text"]))) / 2
    else:
        print("BiLSTM unavailable: choosing the threshold for logistic regression alone")

    threshold = pick_threshold(y_val, p_val, MAX_FALSE_FLAG_RATE)
    val_flag, val_caught, val_acc = rates(y_val, p_val, threshold)
    # validation probabilities from models that never saw the validation reviews (used by analysis.py)
    pd.DataFrame({"text_sha1": [hashlib.sha1(t.encode("utf-8")).hexdigest() for t in X_val["text"]],
                  "label": y_val, "p_stage_a": p_val}).to_csv(os.path.join(REPORTS, "validation_predictions.csv"),
                                                             index=False)

    print("checking on the test set with the deployed models...")
    deployed = load_pipeline()
    p_test = ensemble_proba(deployed, dl_predict, X_test)
    test_flag, test_caught, test_acc = rates(y_test, p_test, threshold)
    default_flag, default_caught, default_acc = rates(y_test, p_test, 0.5)

    # the same rates per data source, for ChatGPT alone, and on GPT-4 text that no model saw in training
    breakdown = {}
    for source in sorted(X_test["source"].unique()):
        m = (X_test["source"] == source).to_numpy()
        breakdown[source] = {"chosen": rate_report(y_test[m], p_test[m], threshold),
                             "at_0.5": rate_report(y_test[m], p_test[m], 0.5)}
    chatgpt = (X_test["generator"] == "gpt-3.5-turbo").to_numpy()
    if chatgpt.any():
        breakdown["mage_yelp_chatgpt_only"] = {
            "chosen": {"n": int(chatgpt.sum()), "computer_generated_caught": round(float((p_test[chatgpt] < threshold).mean()), 4)},
            "at_0.5": {"n": int(chatgpt.sum()), "computer_generated_caught": round(float((p_test[chatgpt] < 0.5).mean()), 4)}}
    ood = load_external("mage_ood_gpt4_imdb")
    if ood is not None:
        p_ood = ensemble_proba(deployed, dl_predict, ood[0])
        breakdown["unseen_gpt4_imdb"] = {"chosen": rate_report(ood[1], p_ood, threshold),
                                        "at_0.5": rate_report(ood[1], p_ood, 0.5)}

    metrics["threshold"] = {
        "value": threshold,
        "goal": f"flag at most {MAX_FALSE_FLAG_RATE:.0%} of genuine reviews as computer-generated",
        "chosen_on": "validation split",
        "validation": {"genuine_flagged": round(val_flag, 4), "computer_generated_caught": round(val_caught, 4),
                       "accuracy": round(val_acc, 4)},
        "test": {"genuine_flagged": round(test_flag, 4), "computer_generated_caught": round(test_caught, 4),
                 "accuracy": round(test_acc, 4)},
        "test_at_0.5": {"genuine_flagged": round(default_flag, 4),
                        "computer_generated_caught": round(default_caught, 4), "accuracy": round(default_acc, 4)},
        "breakdown": breakdown,
        "calibrated_for": "ensemble" if dl_predict is not None else "logistic regression only",
        "note": "chosen with a logistic regression refit on the fit split only; applied to the deployed models",
    }
    metrics["stale_sections"] = [s for s in metrics.get("stale_sections", []) if s != "threshold"]
    save_metrics(metrics)
    update_stage("stage_a", threshold=threshold, target_false_positive_rate=MAX_FALSE_FLAG_RATE,
                 chosen_on="validation split (genuine reviews)",
                 calibrated_for=fingerprint(include_dl=dl_predict is not None))

    print(f"\nchosen threshold: {threshold}")
    print(f"  validation: genuine flagged {val_flag:.2%}, computer-generated caught {val_caught:.2%}")
    print(f"  test:       genuine flagged {test_flag:.2%}, computer-generated caught {test_caught:.2%}, "
          f"accuracy {test_acc:.2%}")
    print(f"  test @0.5:  genuine flagged {default_flag:.2%}, computer-generated caught {default_caught:.2%}, "
          f"accuracy {default_acc:.2%}")
    for name, b in breakdown.items():
        c, h = b["chosen"], b["at_0.5"]
        flag = f"genuine flagged {c['genuine_flagged']:.2%}, " if "genuine_flagged" in c else ""
        print(f"  {name:24s} {flag}AI caught {c['computer_generated_caught']:.2%} "
              f"(at 0.5: {h['computer_generated_caught']:.2%})")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

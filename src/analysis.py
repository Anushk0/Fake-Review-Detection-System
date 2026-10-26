"""
Uncertainty, robustness and calibration analyses reported in the paper.

    cd src
    python analysis.py                     # after train.py, train_dl.py, choose_threshold.py
    python analysis.py --modern project    # also analyse Reports/modern_detector_scores_project.csv

Writes Reports/analysis.json. Nothing here changes a model or a deployed threshold; the
alternative Stage B calibrations are recorded for comparison only. Thresholds are always
chosen on validation data and only evaluated on test data.
"""
import argparse
import hashlib
import json
import math
import os
import re
import warnings

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.feature_extraction.text import TfidfVectorizer

from artifacts import load_config
from predict import ensemble_probability, fuse, load_dl_model, load_pipeline
from train import REPORTS, load_splits

ALPHA = 0.01
BOOTSTRAP = 2000


def wilson(k, n, z=1.959964):
    """95% Wilson score interval for k successes out of n."""
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def rate(k, n):
    return {"count": int(k), "n": int(n), "rate": k / n if n else float("nan"), "ci95": wilson(k, n)}


def sha1(t):
    return hashlib.sha1(str(t).encode("utf-8")).hexdigest()


def stage_a_uncertainty(pred, t1):
    """Bootstrap CIs for accuracy/F1, Wilson CIs for the flag rates, exact McNemar tests."""
    y = pred.label.to_numpy()
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(y), size=(BOOTSTRAP, len(y)))
    hard = {"lr@0.5": pred.p_lr >= 0.5, "bilstm@0.5": pred.p_dl >= 0.5, "ensemble@0.5": pred.p_ensemble >= 0.5,
            f"ensemble@{t1}": pred.p_ensemble >= t1}
    out = {}
    for name, h in hard.items():
        h = h.to_numpy().astype(int)

        def acc_f1(p, yy):
            tp = ((p == 1) & (yy == 1)).sum(-1); fp = ((p == 1) & (yy == 0)).sum(-1); fn = ((p == 0) & (yy == 1)).sum(-1)
            return (p == yy).mean(-1), 2 * tp / (2 * tp + fp + fn)
        a, f = acc_f1(h, y)
        ab, fb = acc_f1(h[idx], y[idx])
        out[name] = {"accuracy": float(a), "accuracy_ci95": np.percentile(ab, [2.5, 97.5]).tolist(),
                     "f1_original_positive": float(f), "f1_ci95": np.percentile(fb, [2.5, 97.5]).tolist(),
                     "confusion_rows_true_CG_OR_cols_pred_CG_OR": [[int(((h == 0) & (y == 0)).sum()), int(((h == 1) & (y == 0)).sum())],
                                                                  [int(((h == 0) & (y == 1)).sum()), int(((h == 1) & (y == 1)).sum())]],
                     "genuine_flagged": rate(((h == 0) & (y == 1)).sum(), (y == 1).sum()),
                     "cg_caught": rate(((h == 0) & (y == 0)).sum(), (y == 0).sum())}
    for a_, b_ in [("lr@0.5", "ensemble@0.5"), ("bilstm@0.5", "ensemble@0.5"), ("lr@0.5", "bilstm@0.5")]:
        ca, cb = hard[a_].to_numpy() == y, hard[b_].to_numpy() == y
        n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
        out[f"mcnemar {a_} vs {b_}"] = {"only_first_correct": n01, "only_second_correct": n10,
                                        "exact_p": float(binomtest(n01, n01 + n10, 0.5).pvalue)}
    e_lr, e_ens = int((hard["lr@0.5"] != y).sum()), int((hard["ensemble@0.5"] != y).sum())
    out["relative_error_reduction_lr_to_ensemble"] = {"lr_errors": e_lr, "ensemble_errors": e_ens,
                                                      "reduction": (e_lr - e_ens) / e_lr}
    return out


END = re.compile(r"^(.*[.!?][\"')\]]*)\s+[^.!?]*$", re.S)


def trim_trailing_fragment(text):
    """Drop an unfinished final sentence (text after the last . ! or ?), if there is a finished one before it."""
    s = text.rstrip()
    if not s or s[-1] in ".!?\"')":
        return text
    m = END.match(s)
    return m.group(1) if m else text


def robustness(X_test, y_test, t1):
    """How much Stage A relies on generated reviews being cut off mid-sentence."""
    pipe, dl = load_pipeline(), load_dl_model()
    trimmed = X_test.assign(text=X_test["text"].map(trim_trailing_fragment))
    changed = (trimmed["text"] != X_test["text"]).to_numpy()
    out = {"changed": {"cg": int(changed[y_test == 0].sum()), "or": int(changed[y_test == 1].sum())}}
    for name, X in [("original", X_test), ("trimmed", trimmed)]:
        p = ensemble_probability(pipe.predict_proba(X)[:, 1], None if dl is None else dl(list(X["text"])))
        out[name] = {"accuracy@0.5": float(((p >= 0.5) == (y_test == 1)).mean()),
                     f"cg_caught@{t1}": float((p[y_test == 0] < t1).mean()),
                     f"genuine_flagged@{t1}": float((p[y_test == 1] < t1).mean()),
                     "cg_caught_among_changed@0.5": float((p[changed & (y_test == 0)] < 0.5).mean())}
    return out


def near_duplicate_sensitivity(X_train, X_test, y_test, pred):
    v = TfidfVectorizer(sublinear_tf=True, min_df=2, stop_words="english", dtype=np.float32).fit(X_train["text"])
    A, B = v.transform(X_train["text"]), v.transform(X_test["text"])
    mx = np.concatenate([(B[s:s + 1000] @ A.T).toarray().max(1) for s in range(0, B.shape[0], 1000)])
    hard = (pred.p_ensemble.to_numpy() >= 0.5).astype(int)
    out = {}
    for t in (0.999, 0.9, 0.8):
        keep = mx < t
        out[f"cos>={t}"] = {"test_reviews_with_such_train_neighbour": int((~keep).sum()),
                            "ensemble_accuracy_without_them": float((hard[keep] == y_test[keep]).mean())}
    return out


def lowest_threshold(scores, max_rate):
    scores = np.asarray(scores)
    cands = np.r_[np.unique(scores), np.nextafter(scores.max(), np.inf)]
    return float(min(t for t in cands if (scores >= t).mean() <= max_rate))


def stage_b(path, t1, val_pred):
    rows = pd.read_csv(path)
    rows["words"] = rows.text.astype(str).str.split().str.len()
    val, test = rows[rows.split == "validation"], rows[rows.split == "test"]
    t2 = lowest_threshold(val[val.label == 1].p_ai_longformer, ALPHA)
    out = {"threshold_pooled": t2, "test": {}}
    for g, d in test.groupby("group"):
        a, b, u = fuse(d.p_original_ensemble, d.p_ai_longformer, t1, t2)
        out["test"][g] = {"stage_a": rate(a.sum(), len(d)), "stage_b": rate(b.sum(), len(d)),
                          "fused": rate(u.sum(), len(d))}
    # How much does tau_2 (a 1% tail quantile of 450 validation reviews) move if the validation sample
    # were drawn again? Resample the validation human reviews with replacement and recompute everything.
    rng = np.random.default_rng(0)
    vh_scores = val[val.label == 1].p_ai_longformer.to_numpy()
    boot = {g: [] for g in out["test"]}; taus = []
    for _ in range(1000):
        t = lowest_threshold(rng.choice(vh_scores, size=len(vh_scores), replace=True), ALPHA)
        taus.append(t)
        for g, d in test.groupby("group"):
            boot[g].append(fuse(d.p_original_ensemble, d.p_ai_longformer, t1, t)[2].mean())
    out["threshold_bootstrap"] = {"tau2_2.5_50_97.5": np.percentile(taus, [2.5, 50, 97.5]).tolist(),
                                  "fused_rate_2.5_50_97.5": {g: np.percentile(v, [2.5, 50, 97.5]).tolist()
                                                             for g, v in boot.items()}}
    if "logit_machine" in test:
        from modern_detector import MAGE_LOGIT_THRESHOLD
        out["mage_published_logit_rule"] = {
            g: rate(fuse(d.p_original_ensemble, (d.logit_machine > MAGE_LOGIT_THRESHOLD).astype(float), t1, 0.5)[2].sum(),
                    len(d)) for g, d in test.groupby("group")}
        out["mage_published_logit_rule_validation_humans"] = rate(
            (val[val.label == 1].logit_machine > MAGE_LOGIT_THRESHOLD).sum(), int((val.label == 1).sum()))

    ai = test[test.group.isin(["yelp_ai_chatgpt", "yelp_ai_other", "imdb_gpt4_ai"])]
    bins = pd.cut(ai.words, [0, 20, 40, 80, 10 ** 6], labels=["<=20", "21-40", "41-80", ">80"])
    out["modern_ai_caught_by_length"] = {str(k): rate((d.p_ai_longformer >= t2).sum(), len(d))
                                         for k, d in ai.groupby(bins, observed=False)}

    # Alternative calibrations (recorded only): Stage A validation scores come from choose_threshold.py's
    # models that never saw the validation reviews, so the fused validation rate is not optimistic.
    vp = dict(zip(val_pred.text_sha1, val_pred.p_stage_a))
    vh = val[val.label == 1].assign(p_a=lambda d: d.text.map(lambda t: vp.get(sha1(t), np.nan)))
    # Yelp reviews were never used to train Stage A, so the deployed models' scores are fair for them
    vh.loc[vh.group != "amazon_human", "p_a"] = vh.loc[vh.group != "amazon_human", "p_original_ensemble"]
    missing = int(vh.p_a.isna().sum())
    vh = vh.dropna(subset=["p_a"])
    alt = {"pooled (deployed rule)": {"amazon": t2, "yelp": t2}}
    # (i) joint: the lowest t2 whose *fused* pooled validation false-positive rate is <= 1%
    a_flag = (vh.p_a < t1).to_numpy()
    cands = np.r_[np.unique(vh.p_ai_longformer), 1.0 + 1e-9]
    ok = [t for t in cands if (a_flag | (vh.p_ai_longformer.to_numpy() >= t)).mean() <= ALPHA]
    alt["joint (fused pooled FPR <= 1%)"] = {"amazon": float(min(ok)) if ok else None, "yelp": float(min(ok)) if ok else None}
    # (ii) per-domain Stage B thresholds
    per = {dom: lowest_threshold(vh[vh.group == f"{dom}_human"].p_ai_longformer, ALPHA) for dom in ("amazon", "yelp")}
    alt["per-domain Stage B (<= 1% each)"] = per
    out["alternative_calibrations"] = {"validation_rows_without_stage_a_score": missing, "rules": {}}
    domain_of = {"amazon_human": "amazon", "amazon_gpt2_ai": "amazon", "yelp_human": "yelp", "yelp_ai_chatgpt": "yelp",
                 "yelp_ai_other": "yelp", "imdb_human": "amazon", "imdb_gpt4_ai": "amazon"}
    for name, th in alt.items():
        if th["amazon"] is None:
            continue
        res = {}
        for g, d in test.groupby("group"):
            tt = th[domain_of[g]]   # IMDb has no validation data: it gets the Amazon threshold
            _, _, u = fuse(d.p_original_ensemble, d.p_ai_longformer, t1, tt)
            res[g] = rate(u.sum(), len(d))
        out["alternative_calibrations"]["rules"][name] = {"thresholds": th, "test_fused": res}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modern", nargs="*", default=[], help="Stage B score files to analyse, by mode")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    t1 = load_config()["stage_a"]["threshold"]
    X_train, _, _, _, _, _, X_test, y_test = load_splits()
    pred = pd.read_csv(os.path.join(REPORTS, "test_predictions.csv"))
    assert (pred.text_sha1.to_numpy() == X_test["text"].map(sha1).to_numpy()).all(), "test_predictions.csv is stale"
    result = {"stage_a_threshold": t1, "stage_a": stage_a_uncertainty(pred, t1),
              "near_duplicates": near_duplicate_sensitivity(X_train, X_test, y_test, pred),
              "truncation_robustness": robustness(X_test, y_test, t1)}
    val_pred = pd.read_csv(os.path.join(REPORTS, "validation_predictions.csv"))
    for mode in args.modern:
        result[f"stage_b_{mode}"] = stage_b(os.path.join(REPORTS, f"modern_detector_scores_{mode}.csv"), t1, val_pred)
    with open(os.path.join(REPORTS, "analysis.json"), "w") as f:
        json.dump(result, f, indent=2, default=float)
    print(json.dumps(result, indent=1, default=float)[:6000])


if __name__ == "__main__":
    main()

"""
Choose the threshold for the modern-AI check and measure the combined system.

    cd src
    python calibrate_modern.py --mode project     # original setting: raw text, 512 tokens (~45 min on a CPU)
    python calibrate_modern.py --mode mage        # MAGE's reference preprocessing, up to 4,096 tokens
    python calibrate_modern.py --mode mage --reuse --deploy   # recompute from saved scores, make it the default

A review is flagged as computer-generated if EITHER check flags it:
  - Stage A, the project's ensemble (GPT-2-era patterns): P(Original) below the threshold chosen by
    choose_threshold.py, or
  - Stage B, the modern-AI detector (modern_detector.py): P(AI) at or above the threshold chosen here.

The Stage B threshold is the lowest one that flags at most 1% of the pooled validation human reviews
(300 Amazon + 150 Yelp). With 450 reviews that allows 4 false flags, so the realised rate is uncertain
and is reported per domain as well; IMDb has no validation reviews at all. The threshold is then
checked on test reviews: Amazon, Yelp (MAGE), and GPT-4 IMDb reviews from MAGE's out-of-distribution
set, which no model in this project was trained on. The fused false-positive rate is measured, not
controlled: it is at least Stage B's and can exceed the 1% target.

Writes Reports/modern_detector_scores_<mode>.csv (full texts) and the "modern_ai_<mode>" section of
Reports/metrics.json; with --deploy also the "modern_ai" section and the Stage B entry of
src/model_config.json used by the app and predict.py. The original run's scores are kept unchanged in
Reports/modern_detector_scores.csv.
"""
import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd

from artifacts import load_metrics, save_metrics, update_stage
from modern_detector import MAGE_LOGIT_THRESHOLD, MODES, REPO, REVISION, ModernAIDetector
from predict import default_threshold, fuse, load_dl_model, load_pipeline, predict_texts
from train import REPORTS, load_external, load_splits

MAX_FALSE_FLAG_RATE = 0.01
SEED = 0


def sample_reviews():
    _, _, _, _, X_val, y_val, X_test, y_test = load_splits(include_mage=False)
    val, test = X_val.assign(label=y_val), X_test.assign(label=y_test)
    mv, mt, oo = (load_external(n) for n in ("mage_yelp_valid", "mage_yelp_test", "mage_ood_gpt4_imdb"))
    if mv is None or mt is None or oo is None:
        raise SystemExit("MAGE files missing: run prepare_mage.py first")
    mv, mt, oo = (X.assign(label=y) for X, y in (mv, mt, oo))
    chatgpt = (mt.label == 0) & (mt.generator == "gpt-3.5-turbo")

    def take(df, n, split, group):
        d = df.sample(min(n, len(df)), random_state=SEED) if n else df
        return d.assign(split=split, group=group)[["text", "rating", "label", "split", "group", "generator"]]

    return pd.concat([
        take(val[val.label == 1], 300, "validation", "amazon_human"),
        take(mv[mv.label == 1], 150, "validation", "yelp_human"),
        take(mv[mv.label == 0], 150, "validation", "yelp_ai"),
        take(test[test.label == 1], 300, "test", "amazon_human"),
        take(test[test.label == 0], 150, "test", "amazon_gpt2_ai"),
        take(mt[mt.label == 1], 150, "test", "yelp_human"),
        take(mt[chatgpt], 0, "test", "yelp_ai_chatgpt"),
        take(mt[(mt.label == 0) & ~chatgpt], 100, "test", "yelp_ai_other"),
        take(oo[oo.label == 1], 0, "test", "imdb_human"),
        take(oo[oo.label == 0], 0, "test", "imdb_gpt4_ai"),
    ], ignore_index=True)


def score(rows, mode, path):
    t0 = time.time()
    rows["p_original_ensemble"] = predict_texts(load_pipeline(), load_dl_model(), rows.text,
                                                rows.rating)["prob_original"].to_numpy()
    print(f"loading {REPO}@{REVISION[:7]} in '{mode}' mode ...")
    p, logit = ModernAIDetector(mode).predict_detailed(
        rows.text.tolist(), progress=lambda done, total: done % 200 == 0 and print(
            f"  {done}/{total} reviews ({time.time() - t0:.0f}s)", flush=True))
    rows["p_ai_longformer"], rows["logit_machine"] = p, logit
    rows.to_csv(path, index=False)
    return rows


def lowest_threshold(scores, max_rate):
    """Lowest threshold t (from the observed scores) with mean(scores >= t) <= max_rate."""
    candidates = np.r_[np.unique(scores), np.nextafter(scores.max(), np.inf)]
    return float(min(t for t in candidates if (scores >= t).mean() <= max_rate))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=list(MODES), default="project")
    parser.add_argument("--reuse", action="store_true", help="reuse saved scores instead of re-scoring")
    parser.add_argument("--deploy", action="store_true", help="make this mode's threshold the app default")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    path = os.path.join(REPORTS, f"modern_detector_scores_{args.mode}.csv")

    rows = pd.read_csv(path) if args.reuse else score(sample_reviews(), args.mode, path)
    val, test = rows[rows.split == "validation"], rows[rows.split == "test"]
    human = val[val.label == 1]
    modern_t = lowest_threshold(human.p_ai_longformer.to_numpy(), MAX_FALSE_FLAG_RATE)
    classic_t = default_threshold()

    def group_rates(df):
        out = {}
        for group, d in df.groupby("group"):
            a, b, both = fuse(d.p_original_ensemble, d.p_ai_longformer, classic_t, modern_t)
            out[group] = {"n": int(len(d)), "kind": "genuine" if int(d.label.iloc[0]) == 1 else "computer-generated",
                          "flagged_by_classic": round(float(a.mean()), 4), "flagged_by_modern": round(float(b.mean()), 4),
                          "flagged_combined": round(float(both.mean()), 4), "count_combined": int(both.sum()),
                          "flagged_by_mage_logit_rule": round(float((d.logit_machine > MAGE_LOGIT_THRESHOLD).mean()), 4)
                          if "logit_machine" in d else None}
        return out

    section = {
        "detector": REPO, "revision": REVISION, "mode": args.mode, **MODES[args.mode],
        "threshold": modern_t, "classic_threshold": classic_t,
        "rule": "computer-generated if the ensemble's P(Original) < classic_threshold "
                "OR the detector's P(AI) >= threshold",
        "chosen_on": f"pooled validation human reviews (n={len(human)}): flag at most {MAX_FALSE_FLAG_RATE:.0%}",
        "validation_humans_flagged": round(float((human.p_ai_longformer >= modern_t).mean()), 4),
        "validation_humans_flagged_by_domain": {g: round(float((d.p_ai_longformer >= modern_t).mean()), 4)
                                                for g, d in human.groupby("group")},
        "validation": group_rates(val),
        "test": group_rates(test),
        "note": "Stage A probabilities for validation Amazon reviews come from the deployed models, which were "
                "trained on them; the fused validation rates are therefore optimistic for Stage A.",
    }
    metrics = load_metrics()
    metrics[f"modern_ai_{args.mode}"] = section
    if args.deploy:
        metrics["modern_ai"] = section
        metrics["stale_sections"] = [s for s in metrics.get("stale_sections", []) if s != "modern_ai"]
        update_stage("stage_b", threshold=modern_t, mode=args.mode, detector=REPO, revision=REVISION,
                     target_false_positive_rate=MAX_FALSE_FLAG_RATE, chosen_on=section["chosen_on"],
                     calibrated_for={"detector": REPO, "revision": REVISION, **MODES[args.mode]})
    save_metrics(metrics)

    print(f"Stage B threshold ({args.mode}): {modern_t} (validation humans flagged "
          f"{section['validation_humans_flagged']:.2%}, by domain {section['validation_humans_flagged_by_domain']}); "
          f"Stage A threshold {classic_t}")
    for group, m in section["test"].items():
        verb = "wrongly flagged" if m["kind"] == "genuine" else "caught"
        print(f"  {group:16s} n={m['n']:<4} {verb:16s} {m['flagged_combined']:6.1%}  "
              f"(classic {m['flagged_by_classic']:.1%}, modern {m['flagged_by_modern']:.1%}, "
              f"MAGE logit rule {m['flagged_by_mage_logit_rule']})")


if __name__ == "__main__":
    main()

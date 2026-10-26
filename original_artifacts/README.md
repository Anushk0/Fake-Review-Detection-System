# Original artifacts (as shipped in Fake-Review-detection-System-main.zip)

Unmodified copies kept for reference before the audit retrained the models.

| File | What it is |
|------|------------|
| model_pipeline.joblib | Logistic-regression pipeline (TF-IDF + 13 stats + similarity, C=10) trained by the original train.py |
| dl_model.keras | BiLSTM trained by the original train_dl.py |
| metrics.json | Reports/metrics.json as shipped (written by an earlier version of train.py: no `feature_groups` key) |
| confusion_matrix_pipeline.png, dl_training_curve.png | Report figures as shipped |

Reports/modern_detector_scores.csv (Stage B scores of the original run, texts truncated to 300 characters) is
left in place unchanged. See ../../EXPERIMENT_VERIFICATION.md for how these artifacts were re-scored.

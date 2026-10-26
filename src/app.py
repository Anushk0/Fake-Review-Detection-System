import streamlit as st
st.set_page_config(
    page_title="Fake Review Detection - Prediction",
    page_icon="🤖",
    layout="wide"
)

import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import altair as alt

import predict
from artifacts import load_config
from features import decision_breakdown, explain, to_frame
from modern_detector import load_modern_detector
from predict import (LABEL_NAMES, clean_ratings, default_modern_threshold, default_threshold, ensemble_probability,
                     evaluate, fuse, load_dl_model, load_metrics, load_pipeline, predict_texts, threshold_warning)

SHORT_REVIEW_WORDS = 40   # below this the modern-AI check was much less reliable in our evaluation

BASE_DIR = os.path.dirname(__file__)
COLORS = alt.Scale(domain=["Original", "Computer-generated"], range=["#2a9d8f", "#e76f51"])
NONE = "(none)"

SIGNAL_LABELS = {
    "max_similarity": "Similarity to closest training review",
    "near_duplicate_count": "Near-duplicate training reviews",
    "sentiment_polarity": "Sentiment polarity (-1 to 1)",
    "sentiment_rating_mismatch": "Sentiment vs star-rating mismatch",
    "subjectivity": "Subjectivity (0 to 1)",
    "repeated_bigram_ratio": "Repeated phrases within the review",
    "lexical_diversity": "Lexical diversity (unique / total words)",
    "word_count": "Word count",
    "avg_word_length": "Average word length",
    "avg_sentence_length": "Average sentence length (words)",
    "flesch_reading_ease": "Flesch reading ease",
    "punctuation_ratio": "Punctuation ratio",
    "uppercase_ratio": "Uppercase ratio",
    "exclamation_count": "Exclamation marks",
    "rating": "Star rating",
}


# -------------------------------
# Load Models
# -------------------------------
pipeline = st.cache_resource(load_pipeline)()
if pipeline is None:
    st.error("Model file not found. Run `python train.py` inside the src folder to create 'model_pipeline.joblib'.")
    st.stop()
dl_predict = st.cache_resource(load_dl_model)()
evidence = st.cache_resource(lambda: predict.load_evidence_index(pipeline))()
# the modern-AI detector is large (~600 MB), so it is loaded on first use rather than at startup
MODERN_MODE = load_config().get("stage_b", {}).get("mode", "project")
get_modern = st.cache_resource(show_spinner="Loading the modern-AI detector (first use only)...")(
    lambda: load_modern_detector(MODERN_MODE))
if dl_predict is None:
    st.warning(f"BiLSTM unavailable ({predict.DL_LOAD_ERROR}); Stage A uses logistic regression alone.")


def direction(value):
    return "Original" if value > 0 else "Computer-generated"


def guess_column(columns, candidates):
    lowered = {c.lower(): c for c in columns}
    for name in candidates:
        if name in lowered:
            return lowered[name]
    return None


st.title("Fake Review Detection")
st.caption("Classifies reviews as written by a person (Original) or computer-generated, and explains why.")

metrics = load_metrics()
DEFAULT_THRESHOLD = default_threshold(dl_available=dl_predict is not None)
_warning = threshold_warning(dl_available=dl_predict is not None)
if _warning:
    st.warning(f"Threshold warning: {_warning}.")
if metrics.get("stale_sections"):
    st.warning("Some reported figures describe earlier model files: " + ", ".join(metrics["stale_sections"]))
if "threshold" in metrics:
    t = metrics["threshold"]
    THRESHOLD_HELP = (f"Probability of Original needed to call a review Original. The default, {t['value']}, was "
                      f"chosen automatically on validation data to {t['goal']}. On the test set it flagged "
                      f"{t['test']['genuine_flagged']:.1%} of genuine reviews and caught "
                      f"{t['test']['computer_generated_caught']:.1%} of computer-generated ones. Raise it to catch "
                      "more computer-generated reviews at the cost of flagging more genuine ones.")
else:
    THRESHOLD_HELP = "Probability of Original needed to call a review Original."
if metrics:
    test = metrics["ensemble"]["test"] if dl_predict is not None and "ensemble" in metrics else metrics["test"]
    model_name = "ensemble (logistic regression + BiLSTM)" if dl_predict is not None else "logistic regression"
    st.caption(f"Model: {model_name}. Held-out test set at threshold 0.5: accuracy {test['accuracy']:.1%}, "
               f"precision {test['precision']:.1%}, recall {test['recall']:.1%}, F1 {test['f1']:.1%}")
    if "threshold" in metrics:
        st.caption(f"Default threshold {t['value']} ({t['goal']}): on the test set it flagged "
                   f"{t['test']['genuine_flagged']:.1%} of genuine reviews, caught "
                   f"{t['test']['computer_generated_caught']:.1%} of computer-generated ones, "
                   f"accuracy {t['test']['accuracy']:.1%}.")
MODERN_THRESHOLD = default_modern_threshold(metrics)
if MODERN_THRESHOLD is not None:
    mt = metrics.get("modern_ai", {}).get("test", {})
    if {"imdb_gpt4_ai", "yelp_ai_chatgpt", "amazon_human"} <= set(mt):
        st.caption(f"Second check for modern AI text (ChatGPT, GPT-4, LLaMA): a pretrained transformer detector. "
                   f"With both checks, on test samples: {mt['imdb_gpt4_ai']['flagged_combined']:.0%} of unseen "
                   f"GPT-4 reviews and {mt['yelp_ai_chatgpt']['flagged_combined']:.0%} of ChatGPT reviews caught, "
                   f"{mt['amazon_human']['flagged_combined']:.1%} of genuine Amazon reviews wrongly flagged "
                   "(the combined false-positive rate is measured, not held at 1%).")
else:
    st.info("Limitation: the computer-generated training reviews come from older text generators (GPT-2 era). "
            "Reviews written by modern AI such as ChatGPT are fluent and are usually classified as Original. "
            "Run calibrate_modern.py (needs torch and transformers) to enable the modern-AI check.")

tab_single, tab_batch = st.tabs(["Single review", "Batch (CSV file)"])

# -------------------------------
# Single Prediction
# -------------------------------
with tab_single:
    review_text = st.text_area("Enter review text:", height=150)
    col_rating, col_threshold = st.columns(2)
    rating = col_rating.select_slider("Star rating given with the review", options=[1, 2, 3, 4, 5], value=5)
    threshold = col_threshold.slider("Prediction threshold (probability of Original)", 0.0, 1.0, DEFAULT_THRESHOLD,
                                     0.01, help=THRESHOLD_HELP)

    if st.button("Predict"):
        if not review_text.strip():
            st.error("Please enter some text.")
        else:
            with st.spinner("Analyzing..."):
                lr_probabilities = pipeline.predict_proba(to_frame([review_text], [rating]))[0]
                probabilities = lr_probabilities
                dl_original = float(dl_predict([review_text])[0]) if dl_predict is not None else None
                if dl_original is not None:
                    dl_probabilities = np.array([1 - dl_original, dl_original])
                p_original = float(ensemble_probability([lr_probabilities[1]],
                                                        None if dl_original is None else [dl_original])[0])
                probabilities = np.array([1 - p_original, p_original])
                modern = get_modern() if MODERN_THRESHOLD is not None else None
                p_ai_modern = float(modern.predict([review_text])[0]) if modern is not None else None
                a, b, flagged = fuse([p_original], None if p_ai_modern is None else [p_ai_modern], threshold,
                                     MODERN_THRESHOLD)
                classic_flag, modern_flag = bool(a[0]), bool(b[0])
                prediction = 0 if flagged[0] else 1
                words, signals = explain(pipeline, review_text, rating)
                breakdown = decision_breakdown(pipeline, review_text, rating)
                neighbours = evidence.most_similar(review_text, k=3) if evidence is not None else []

            result = LABEL_NAMES[prediction]
            if modern_flag and not classic_flag:
                st.warning(f"Prediction: {result}, flagged by the modern-AI check only "
                           f"({p_ai_modern:.1%} probability of AI)")
            elif classic_flag:
                reason = "both checks" if modern_flag else "Stage A (GPT-2-style patterns) only"
                st.warning(f"Prediction: {result} (flagged by {reason})")
            else:
                st.success(f"Prediction: {result} (neither check flagged it)")

            stage_rows = [{"Check": "Stage A: LR + BiLSTM ensemble" if dl_original is not None else
                                    "Stage A: logistic regression",
                           "Score": f"P(Original) = {p_original:.3f}", "Rule": f"flag if < {threshold:.2f}",
                           "Flags": "yes" if classic_flag else "no"}]
            if p_ai_modern is not None:
                stage_rows.append({"Check": "Stage B: modern-AI detector", "Score": f"P(AI) = {p_ai_modern:.4f}",
                                   "Rule": f"flag if ≥ {MODERN_THRESHOLD:.4f}", "Flags": "yes" if modern_flag else "no"})
            elif MODERN_THRESHOLD is not None:
                stage_rows.append({"Check": "Stage B: modern-AI detector", "Score": "not run",
                                   "Rule": "unavailable: install requirements-modern.txt", "Flags": "–"})
            st.table(pd.DataFrame(stage_rows))
            n_words = len(review_text.split())
            if p_ai_modern is not None and n_words < SHORT_REVIEW_WORDS:
                st.caption(f"This review has {n_words} words. In our tests the modern-AI check caught far fewer "
                           f"AI reviews under {SHORT_REVIEW_WORDS} words, so a 'not flagged' result is weak evidence.")
            st.caption("Both checks were evaluated on English product, restaurant and movie reviews; results on "
                       "other kinds of text are untested. A flag means the text resembles machine-generated text, "
                       "not that the review is deceptive.")

            left, right = st.columns(2)

            with left:
                model_probs = {"Logistic regression": lr_probabilities}
                if dl_predict is not None:
                    model_probs["BiLSTM (deep learning)"] = dl_probabilities
                    model_probs["Ensemble"] = probabilities
                if p_ai_modern is not None:
                    model_probs["Modern-AI detector"] = np.array([p_ai_modern, 1 - p_ai_modern])
                df_probs = pd.DataFrame([
                    {"Model": name, "Label": label, "Probability": p[i]}
                    for name, p in model_probs.items()
                    for i, label in enumerate(["Computer-generated", "Original"])
                ])
                bar_chart = alt.Chart(df_probs).mark_bar().encode(
                    x=alt.X("Model", axis=alt.Axis(labelAngle=0, labelLimit=160), sort=list(model_probs), title=None),
                    xOffset="Label",
                    y=alt.Y("Probability", scale=alt.Scale(domain=[0, 1])),
                    color=alt.Color("Label", scale=COLORS),
                    tooltip=["Model", "Label", alt.Tooltip("Probability", format=".1%")],
                ).properties(title="Prediction probability by model", height=300)
                st.altair_chart(bar_chart, use_container_width=True)

            with right:
                if not words.empty:
                    words["Pushes toward"] = words["contribution"].apply(direction)
                    word_chart = alt.Chart(words).mark_bar().encode(
                        x=alt.X("contribution", title="Contribution to decision"),
                        y=alt.Y("feature", sort="-x", title=None),
                        color=alt.Color("Pushes toward", scale=COLORS),
                        tooltip=["feature", alt.Tooltip("contribution", format=".3f"), "Pushes toward"],
                    ).properties(title="Words and phrases that influenced the decision", height=300)
                    st.altair_chart(word_chart, use_container_width=True)

            st.subheader("What drove the Stage A score")
            if modern_flag and not classic_flag:
                st.info("This review was flagged by the modern-AI check, which is a neural network without "
                        "per-word explanations. The contributions below explain only the Stage A logistic-regression "
                        "score, which did not flag the review.")
            st.caption(f"Logistic-regression logit = intercept {breakdown['intercept']:+.2f} + feature contributions "
                       f"{breakdown['contributions']:+.2f} = {breakdown['logit']:+.2f} (P(Original) = "
                       f"{breakdown['prob_original']:.3f}). Contributions add up on the logit scale; they describe "
                       "the model's arithmetic, not causes. The BiLSTM half of Stage A is not explained.")
            top = signals.head(5)
            reasons = [f"- **{SIGNAL_LABELS.get(r.feature, r.feature)}** = {r.value:.2f} "
                       f"→ pushes toward *{direction(r.contribution)}*" for r in top.itertuples()]
            if not words.empty:
                top_word = words.iloc[0]
                reasons.append(f"- Strongest phrase: **\"{top_word['feature']}\"** "
                               f"→ pushes toward *{direction(top_word['contribution'])}*")
            st.markdown("\n".join(reasons))

            with st.expander("All signals"):
                table = signals.assign(
                    Signal=signals["feature"].map(SIGNAL_LABELS).fillna(signals["feature"]),
                    Value=signals["value"].round(3),
                    Contribution=signals["contribution"].round(3),
                    **{"Pushes toward": signals["contribution"].apply(direction)},
                )[["Signal", "Value", "Contribution", "Pushes toward"]]
                st.dataframe(table, hide_index=True, use_container_width=True)

            if neighbours:
                with st.expander("Most similar reviews in the training data"):
                    st.caption("Lexical (TF-IDF cosine) similarity to training reviews. A close match is evidence "
                               "of shared wording, not proof of who wrote the review.")
                    for n in neighbours:
                        label = "label unknown" if n["label"] is None else f"labelled {LABEL_NAMES[n['label']]}"
                        exact = " (exact copy of this text)" if n["exact_copy"] else ""
                        st.markdown(f"**Similarity {n['similarity']:.2f}**, {label}{exact}")
                        st.write(n["text"])

# -------------------------------
# Batch Prediction
# -------------------------------
with tab_batch:
    st.write("Upload a CSV with one review per row. The main models score about 100 reviews per second; "
             "the optional modern-AI check takes about 1.5 seconds per review.")
    uploaded = st.file_uploader("CSV file", type=["csv"])

    if uploaded is not None:
        try:
            data = pd.read_csv(uploaded)
        except UnicodeDecodeError:
            uploaded.seek(0)
            data = pd.read_csv(uploaded, encoding="latin-1")
        columns = list(data.columns)
        st.caption(f"{len(data):,} rows, {len(columns)} columns")
        st.dataframe(data.head(5), hide_index=True, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        text_guess = guess_column(columns, ["text_", "text", "review", "review_text", "content", "body"])
        text_col = c1.selectbox("Review text column", columns,
                                index=columns.index(text_guess) if text_guess else 0)
        rating_options = [NONE] + columns
        rating_guess = guess_column(columns, ["rating", "stars", "star_rating", "score"])
        rating_col = c2.selectbox("Star rating column (1-5)", rating_options,
                                  index=rating_options.index(rating_guess) if rating_guess else 0)
        label_guess = guess_column(columns, ["label", "class", "target"])
        label_col = c3.selectbox("True label column, optional (CG/OR or 0/1)", rating_options,
                                 index=rating_options.index(label_guess) if label_guess else 0)

        c4, c5 = st.columns(2)
        default_rating = c4.select_slider("Rating to use when missing", options=[1, 2, 3, 4, 5], value=5,
                                          help="The model uses the star rating; rows without one get this value.")
        batch_threshold = c5.slider("Prediction threshold (probability of Original)", 0.0, 1.0, DEFAULT_THRESHOLD,
                                    0.01, key="batch_threshold", help=THRESHOLD_HELP)
        if rating_col == NONE:
            st.info("No rating column selected: every review is scored as if it had "
                    f"{default_rating} stars, which can change some predictions.")
        use_modern = False
        if MODERN_THRESHOLD is not None:
            minutes = 1.5 * len(data) / 60
            label = (f"Also run the modern-AI check (about {minutes:.0f} min for {len(data):,} rows)"
                     if minutes >= 1 else "Also run the modern-AI check (under a minute)")
            use_modern = st.checkbox(label, value=len(data) <= 300,
                                     help="Catches text from modern AI such as ChatGPT and GPT-4, which the main "
                                          "models miss. Slow on a laptop CPU.")

        if st.button("Run predictions", type="primary"):
            bar = st.progress(0.0, text="Starting...")
            ratings = clean_ratings(data[rating_col] if rating_col != NONE else [np.nan] * len(data),
                                    default_rating)
            modern = get_modern() if use_modern else None
            if use_modern and modern is None:
                st.warning("Modern-AI check unavailable (install torch and transformers); running without it.")
            result = predict_texts(
                pipeline, dl_predict, data[text_col], ratings, batch_threshold,
                progress=lambda done, total: bar.progress(done / total, text=f"{done:,} / {total:,} reviews"),
                modern=modern, modern_threshold=MODERN_THRESHOLD)
            bar.empty()
            st.session_state["batch"] = {
                "file": uploaded.name,
                "output": pd.concat([data.reset_index(drop=True), result], axis=1),
                "text_col": text_col,
                "label_col": label_col,
            }

        batch = st.session_state.get("batch")
        if batch is not None and batch["file"] == uploaded.name:
            output = batch["output"]
            predicted = output["prediction"]
            n_cg = int((predicted == LABEL_NAMES[0]).sum())
            n_or = int((predicted == LABEL_NAMES[1]).sum())
            n_skipped = int((predicted == "").sum())

            st.subheader("Results")
            m1, m2, m3 = st.columns(3)
            m1.metric("Reviews scored", f"{n_cg + n_or:,}")
            m2.metric("Computer-generated", f"{n_cg:,}", f"{n_cg / max(n_cg + n_or, 1):.1%} of scored",
                      delta_color="off")
            m3.metric("Original", f"{n_or:,}", f"{n_or / max(n_cg + n_or, 1):.1%} of scored", delta_color="off")
            if n_skipped:
                st.caption(f"{n_skipped:,} rows had empty text and were skipped.")
            reasons = output.loc[output["flagged_by"] != "", "flagged_by"].value_counts()
            if len(reasons):
                st.caption("Flagged by: " + ", ".join(f"{reason} ({n:,})" for reason, n in reasons.items()))

            hist = alt.Chart(output.dropna(subset=["prob_original"])).mark_bar().encode(
                x=alt.X("prob_original", bin=alt.Bin(maxbins=20), title="Probability of Original (ensemble)"),
                y=alt.Y("count()", title="Reviews"),
                color=alt.Color("prediction", title="Prediction", scale=COLORS),
            ).properties(title="How confident the model is across the file", height=260)
            st.altair_chart(hist, use_container_width=True)

            if batch["label_col"] != NONE:
                m = evaluate(output[batch["label_col"]], predicted)
                if m is None:
                    st.warning(f"Couldn't read the labels in '{batch['label_col']}' (expected CG/OR or 0/1).")
                else:
                    st.subheader("Accuracy against your labels")
                    e1, e2, e3, e4 = st.columns(4)
                    e1.metric("Accuracy", f"{m['accuracy']:.2%}")
                    e2.metric("Precision", f"{m['precision']:.2%}")
                    e3.metric("Recall", f"{m['recall']:.2%}")
                    e4.metric("F1", f"{m['f1']:.2%}")
                    cm = pd.DataFrame(m["confusion_matrix"],
                                      index=["Actually computer-generated", "Actually original"],
                                      columns=["Predicted computer-generated", "Predicted original"])
                    st.dataframe(cm, use_container_width=True)
                    st.caption(f"Based on {m['rows']:,} rows with readable labels. If this file contains reviews "
                               "the model was trained on (such as the project's own dataset), these numbers "
                               "will be higher than on genuinely new reviews.")

            st.subheader("Predictions")
            cols = [batch["text_col"], "prediction", "flagged_by", "prob_original"]
            if "prob_ai_modern" in output:
                cols.append("prob_ai_modern")
            show = output[cols].rename(
                columns={batch["text_col"]: "Review", "prediction": "Prediction", "flagged_by": "Flagged by",
                         "prob_original": "Probability of Original (main models)",
                         "prob_ai_modern": "Probability of AI (modern-AI check)"})
            st.dataframe(show.head(1000), hide_index=True, use_container_width=True)
            if len(show) > 1000:
                st.caption("Showing the first 1,000 rows; the download has all of them.")

            st.download_button(
                "Download predictions as CSV",
                output.to_csv(index=False).encode("utf-8"),
                file_name=os.path.splitext(batch["file"])[0] + "_predictions.csv",
                mime="text/csv",
            )

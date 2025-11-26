# app.py -- TruthLensAI (single-file Streamlit app)
# NOTES:
# - Set MODEL_RAW_URL to your GitHub RAW URL for the Orange-exported .pkl model (see instructions below).
# - If auto-load fails, user can upload the model manually via the uploader.
# - LIME explainability is primary. SHAP attempted if available.

import streamlit as st
import requests, pickle, io, sys
import numpy as np
import pandas as pd
import re
import feedparser
from pathlib import Path

# try imports that might be heavy
try:
    from lime.lime_text import LimeTextExplainer
    LIME_AVAILABLE = True
except Exception:
    LIME_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# ---------- CONFIG ----------
# Replace this placeholder with your model RAW URL on GitHub once you upload it.
MODEL_RAW_URL = "https://raw.githubusercontent.com/YOUR-USERNAME/TruthLensAI/main/truthlens_model.pkl"
# ---------------------------

st.set_page_config(page_title="TruthLensAI", page_icon="🛡️", layout="centered")
st.title("🛡️ TruthLensAI — Fake News Detector")
st.markdown("Upload or auto-load an Orange-exported model (.pkl). Enter a headline or pick a live headline to analyze. Explainability via LIME (primary).")

# ---------- Utilities ----------
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def fetch_rss_headlines(rss_url="https://feeds.feedburner.com/ndtvnews-top-stories", max_items=25):
    feed = feedparser.parse(rss_url)
    titles = [entry.title for entry in feed.entries[:max_items]]
    return titles

def load_model_from_github(raw_url):
    try:
        r = requests.get(raw_url, timeout=15)
        r.raise_for_status()
        model = pickle.loads(r.content)
        return model, None
    except Exception as e:
        return None, str(e)

def load_model_from_file(uploaded_file):
    try:
        bytes_data = uploaded_file.read()
        model = pickle.loads(bytes_data)
        return model, None
    except Exception as e:
        return None, str(e)

# Predict wrapper that accepts raw texts list and returns proba for class 1 if available
def predict_proba_wrapper(model, texts):
    # Attempt to call model.predict_proba directly on raw texts.
    try:
        probs = model.predict_proba(texts)
        # Ensure shape: (n_samples, n_classes) and return second column if binary
        if probs.ndim == 2 and probs.shape[1] >= 2:
            return probs
        # else try wrapper via pipeline that needs cleaned text -> model should handle raw text
    except Exception:
        pass
    # fallback: try to clean and feed
    try:
        clean_texts = [clean_text(t) for t in texts]
        probs = model.predict_proba(clean_texts)
        return probs
    except Exception:
        # last resort: return zeros
        n = len(texts)
        return np.zeros((n,2))

def predict_label_and_confidence(model, text):
    txt = clean_text(text)
    # try predict_proba
    try:
        probs = predict_proba_wrapper(model, [txt])
        prob_fake = float(probs[0][1]) if probs.shape[1] >= 2 else float(probs[0].max())
    except Exception:
        prob_fake = None
    # predict label
    try:
        pred = model.predict([txt])[0]
    except Exception:
        # fallback: use prob threshold
        pred = 1 if (prob_fake is not None and prob_fake >= 0.5) else 0
    return int(pred), prob_fake

# Try to extract top features for linear models (coef * tfidf)
def top_linear_contributors(model, text, top_n=5):
    """
    Try to compute influential features if model is linear and uses vectorizer.
    Returns list of (feature, score) or [].
    """
    try:
        # If model is a pipeline and has named_steps
        vect = None
        clf = None
        if hasattr(model, "named_steps"):
            steps = model.named_steps
            # common names: 'vect', 'vectorizer', 'tfidf', 'preprocessor'
            for name in ["vectorizer","vect","tfidf","tfidfvectorizer","tfidf_vect","preprocess"]:
                if name in steps:
                    vect = steps[name]
                    break
            # classifier common names
            for name in ["clf","classifier","logreg","svc","model"]:
                if name in steps:
                    clf = steps[name]
                    break
            # fallback: pick last estimator as clf
            if clf is None:
                # pick last step
                last = list(steps.items())[-1][1]
                clf = last
        else:
            # not pipeline: if model has coef_ assume it's classifier
            if hasattr(model, "coef_"):
                clf = model

        if vect is None:
            # try to find vectorizer attribute
            if hasattr(model, "vectorizer"):
                vect = model.vectorizer

        if vect is None or clf is None:
            return []
        # vectorize text
        Xv = vect.transform([clean_text(text)])
        coef = None
        if hasattr(clf, "coef_"):
            coef = clf.coef_[0]
        elif hasattr(clf, "feature_log_prob_"):
            # naive bayes: use log prob difference
            # not ideal but gives some signal
            coef = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
        else:
            return []
        import numpy as np
        feature_names = np.array(vect.get_feature_names_out())
        importance = Xv.toarray()[0] * coef
        top_idx = np.argsort(-importance)[:top_n]
        top = []
        for idx in top_idx:
            if importance[idx] <= 0:
                continue
            top.append((feature_names[idx], float(importance[idx])))
        return top
    except Exception:
        return []

def lime_explain(model, text, class_names=["Real","Fake"], n_features=5):
    if not LIME_AVAILABLE:
        return ["LIME not installed on server."]
    try:
        explainer = LimeTextExplainer(class_names=class_names)
        # lime expects function that accepts list of raw texts and returns predict_proba
        def prob_fn(texts):
            probs = predict_proba_wrapper(model, texts)
            return probs
        exp = explainer.explain_instance(text, prob_fn, num_features=n_features)
        # returns list of (feature, weight)
        return exp.as_list()
    except Exception as e:
        return [f"LIME error: {e}"]

# ---------- UI: Model loading ----------
st.sidebar.header("Model Loading")
st.sidebar.markdown("Option A: Auto-load from GitHub RAW URL (preferred for Streamlit Cloud).")
raw_url_input = st.sidebar.text_input("Model RAW URL (GitHub)", value=MODEL_RAW_URL)
load_btn = st.sidebar.button("Load model from RAW URL")

uploaded_model = st.sidebar.file_uploader("Or upload your Orange .pkl model", type=["pkl","sav","joblib"])
model = None
load_error = None

if load_btn and raw_url_input.strip() != "":
    with st.spinner("Downloading model from GitHub..."):
        model, load_error = load_model_from_github(raw_url_input.strip())
        if model is not None:
            st.sidebar.success("Model loaded from GitHub!")
        else:
            st.sidebar.error(f"Failed to load model: {load_error}")

# fallback: if uploaded
if uploaded_model is not None and model is None:
    with st.spinner("Loading uploaded model..."):
        model, load_error = load_model_from_file(uploaded_model)
        if model is not None:
            st.sidebar.success("Model loaded from uploader!")
        else:
            st.sidebar.error(f"Upload failed: {load_error}")

if model is None:
    st.warning("Model not loaded yet. Provide a valid GitHub RAW URL or upload the model file.")
    st.stop()

# ---------- Main UI ----------
mode = st.radio("Mode:", ["Enter Your Own Headline", "Live News Mode"], index=0)

if mode == "Enter Your Own Headline":
    text_in = st.text_area("Paste headline or text to analyze", height=120)
    analyze = st.button("Analyze")
    if analyze:
        if not text_in.strip():
            st.warning("Enter a headline or piece of text.")
        else:
            pred, prob = predict_label_and_confidence(model, text_in)
            st.subheader("Prediction")
            label = "FAKE ❌" if pred == 1 else "REAL ✅"
            st.markdown(f"### {label}")
            if prob is not None:
                st.write(f"Confidence (probability of FAKE): **{prob*100:.2f}%**")
            # LIME
            st.subheader("Top contributing words (LIME)")
            lime_res = lime_explain(model, text_in, class_names=["Real","Fake"])
            if isinstance(lime_res, list):
                for w, wt in lime_res:
                    st.write(f"- **{w}** → {wt:.3f}")
            else:
                st.write(lime_res)
            # linear contributors
            top_lin = top_linear_contributors(model, text_in, top_n=5)
            if top_lin:
                st.subheader("Top linear contributors (coef * tfidf)")
                for feat, sc in top_lin:
                    st.write(f"- **{feat}** → {sc:.4f}")

else:
    st.subheader("Live Headlines (RSS)")
    rss_url = st.text_input("RSS feed URL", value="https://feeds.feedburner.com/ndtvnews-top-stories")
    if st.button("Fetch headlines"):
        with st.spinner("Fetching headlines..."):
            headlines = fetch_rss_headlines(rss_url)
            if not headlines:
                st.warning("No headlines found. Check the RSS URL.")
            else:
                choice = st.selectbox("Pick headline to analyze", headlines)
                if st.button("Analyze selected headline"):
                    pred, prob = predict_label_and_confidence(model, choice)
                    st.subheader("Prediction")
                    label = "FAKE ❌" if pred == 1 else "REAL ✅"
                    st.markdown(f"### {label}")
                    if prob is not None:
                        st.write(f"Confidence (probability of FAKE): **{prob*100:.2f}%**")
                    st.subheader("Top contributing words (LIME)")
                    lime_res = lime_explain(model, choice, class_names=["Real","Fake"])
                    if isinstance(lime_res, list):
                        for w, wt in lime_res:
                            st.write(f"- **{w}** → {wt:.3f}")
                    top_lin = top_linear_contributors(model, choice, top_n=5)
                    if top_lin:
                        st.subheader("Top linear contributors (coef * tfidf)")
                        for feat, sc in top_lin:
                            st.write(f"- **{feat}** → {sc:.4f}")

st.sidebar.info("If your Orange export is a pipeline with vectorizer + classifier, linear contributor extraction will work best.")
st.sidebar.write("Tip: Upload the model to your GitHub repo and paste the RAW URL into the model URL field for auto-loading on Streamlit Cloud.")

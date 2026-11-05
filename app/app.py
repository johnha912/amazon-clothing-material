"""Demo: paste a product listing and see which material the model predicts.

Run from the repo root: streamlit run app/app.py
"""
import sys
from pathlib import Path

import joblib
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.labels import label, strip_composition  # noqa: E402

MODELS = ROOT / "models"


@st.cache_resource
def load(path):
    return joblib.load(path)


st.title("What is it made of?")
st.write("Paste an Amazon clothing listing. The model guesses the main material from the words alone.")

paths = sorted(MODELS.glob("*.joblib"))
if not paths:
    st.error("No model in models/. Train one first, for example: python analysis/material_tfidf.py")
    st.stop()

choice = st.selectbox("Model", paths, format_func=lambda p: p.stem)
text = st.text_area("Listing text", height=200,
                    placeholder="Lightweight hoodie with kangaroo pocket. 80% cotton, 20% polyester...")

if st.button("Predict material") and text.strip():
    model = load(choice)
    clean = strip_composition(text)
    st.subheader(f"Model's answer: {model.predict([clean])[0]}")

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba([clean])[0]
        top = sorted(zip(model.classes_, probs), key=lambda x: -x[1])[:3]
        st.table({"Material": [c for c, _ in top], "Confidence": [f"{p:.0%}" for _, p in top]})

    truth = label(text)
    if truth:
        st.caption(f"The listing's own composition says {truth}. "
                   "That line was removed before the model read the text.")

"""Demo: paste a product listing and see which material the model predicts.

Run from the repo root once the model exists (setup steps are shown in the app
when models/ is empty, and in the README "Getting started"):
    streamlit run app/app.py

Composition strings in the usual forms ("80% cotton", "Polyester 88%") are
removed with the same two cleaning steps the training text went through
(src/labels.strip_composition in make_sample.py, then the clause filter in
analysis/load_data.py), so the demo shows inference, not lookup. Forms neither
step catches ("100 percent cotton", a full-width ％) reach the model here just as
they reached it in training; AGENTS.md records how many training rows had them.
"""
import re
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "analysis"):
    if str(path) not in sys.path:
        sys.path.append(str(path))
from load_data import strip_composition as strip_clauses  # noqa: E402  training-time cleaner
from src.labels import label, strip_composition  # noqa: E402

MODEL = ROOT / "models" / "material_tfidf.joblib"
METRICS = ROOT / "results" / "material_tfidf.csv"
META_URL = ("https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/"
            "raw/meta_categories/meta_Clothing_Shoes_and_Jewelry.jsonl")
SETUP = f"""pip install -r requirements.txt
mkdir -p data
curl -L {META_URL} -o data/meta_Clothing_Shoes_and_Jewelry.jsonl   # about 18 GB
python -m src.make_sample data/meta_Clothing_Shoes_and_Jewelry.jsonl --n 250000   # a few minutes
python analysis/material_tfidf.py   # about 1 to 6 minutes, writes models/material_tfidf.joblib"""

# Real listings from the 250k sample's test split (already cleaned, as the model sees them),
# with the label parsed from each listing's original composition line. Four the model gets
# right and one it misses, so the demo also shows a typical error.
EXAMPLES = {
    "Pea coat": ("wool", "Kenneth Cole REACTION Women's Plush Wool Double Breasted Pea Coat Button closure "
                         "Dry Clean Only , Double breasted Plush wool Professionally dry clean only Made in China "
                         "Plush Wool Double Breasted Pea Coat"),
    "Pump": ("leather", "Clarks Women's Tarah Sofia Platform Pump Imported Rubber sole Heel measures approximately "
                        "2.5\" Patent leather upper Ortholite sheet insole Clarks Plus technology Full grain leather "
                        "and PU sockliner Steel shank"),
    "Kids' jacket": ("polyester", "Carter's Little Boys Green Fleece Lined Jacket (4T) Polyester,Fleece Polyester "
                                  "lining Zipper closure Machine Wash Imported Machine Wash Fleece Lined Jacket With "
                                  "Zipper Closure & 2 Pockets"),
    "Maxi skirt": ("rayon", "RM Fashions Womens Super Soft Fold Over Maxi Skirt (Small-3XL) Viscose,Elastane "
                            "Machine Wash , Machine Washable Women's Foldover Maxi Skirt Super Soft, Maxi Length "
                            "Made In UK"),
    "Sherpa (a miss)": ("cotton", "Southpole - Kids Little Boys' Solid Sherpa Fleece , Machine Wash Sherpa fleece "
                                  "Kids solid sherpa Kids solid sherpa fleece"),
}


@st.cache_resource
def load(path):
    return joblib.load(path)


def clean(text: str) -> str:
    """The exact cleaning the training text received, in the same order."""
    return strip_clauses(strip_composition(text))


def model_caption() -> str:
    """How the saved model scored, read from its results file so it never goes stale."""
    if not METRICS.exists():
        return ""
    scores = pd.read_csv(METRICS).set_index("model")
    row = scores.loc["tfidf_logreg"]
    svm = (f" The linear SVM scores {scores.loc['tfidf_svm', 'macro_f1']:.3f} but gives no probabilities, "
           "so the demo uses logistic regression." if "tfidf_svm" in scores.index else "")
    return (f"TF-IDF + logistic regression, trained on {row.n_train:,} labeled listings; "
            f"test macro-F1 {row.macro_f1:.3f} on {row.n_test:,} listings "
            f"({row.data_version}, seed {row.seed}).{svm}")


st.set_page_config(page_title="What is it made of?", page_icon="🧵")
st.title("What is it made of?")
st.write("Paste an Amazon clothing listing. The model guesses the main material from the words alone, "
         "after the composition line has been removed.")

if not MODEL.exists():
    st.error("No model in models/ yet. From the repo root, set up the data and train the model once:")
    st.code(SETUP, language="bash")
    st.stop()

model = load(MODEL)
if caption := model_caption():
    st.caption(caption)

example = st.radio("Try a real test listing, or paste your own below", ["(none)", *EXAMPLES], horizontal=True)
known_label, example_text = EXAMPLES.get(example, (None, ""))
text = st.text_area("Listing text", value=example_text, height=160,
                    placeholder="Lightweight hoodie with kangaroo pocket. 80% cotton, 20% polyester...")

if st.button("Predict material", type="primary"):
    if not text.strip():
        st.info("Paste a listing first, or pick one of the examples.")
        st.stop()

    cleaned = clean(text)
    with st.expander("What the model reads (composition clauses removed)"):
        st.write(cleaned or "(nothing)")

    # Without a word of two or more letters, only punctuation or stray numbers are left to read.
    if not re.search(r"[A-Za-z]{2,}", cleaned):
        st.warning("Nothing usable is left after removing the composition clauses, so there is nothing to infer from.")
        st.stop()

    top = (pd.Series(model.predict_proba([cleaned])[0], index=model.classes_)
           .nlargest(3))
    st.subheader(f"Model's answer: {top.index[0]}")
    st.table(pd.DataFrame({"Material": top.index, "Confidence": top.map("{:.0%}".format).to_numpy()},
                          index=["1st", "2nd", "3rd"]))

    truth = known_label if text == example_text else label(text)
    if truth:
        st.info(f"The listing's composition line says **{truth}**. "
                "The model read the text with that line removed.")
    else:
        st.caption("No composition the labeling rule can read, so there is no label to check against.")

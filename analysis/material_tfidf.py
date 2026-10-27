"""Task A baselines: predict the dominant material from listing text.

Owner: Nguyen Ha (Issue #1, Week 1).

Run from the repo root:

    python analysis/material_tfidf.py

Trains TF-IDF + logistic regression and TF-IDF + linear SVM on
`split == "train"`, reports macro-F1 on `split == "test"`, writes
`results/material_tfidf.csv`, and saves the logistic-regression pipeline to
`models/material_tfidf.joblib` for the demo app.

Data: `data/sample.parquet` when it exists (built by `src/make_sample.py`,
Rui Song); until then the first 10,000 records of `data/meta_head.jsonl`,
labeled by the shared rule (percentage composition strings only).

Leakage (AGENTS.md): the composition strings the label is parsed from are
removed from `text` before training, and `average_rating` / `rating_number`
are never used as features.
"""

# %% Imports and config
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"
SEED = 42
MIN_CLASS_ROWS = 20  # drop classes too rare to split; src/labels.py will own
                     # the merged class list once it lands

# %% Load the data (sample.parquet preferred, meta_head.jsonl fallback)

from load_data import load_data

df, DATA_VERSION = load_data()
print(f"Data: {DATA_VERSION}; seed={SEED}")

# %% Prepare Task A: labeled rows only, classes with enough support

class_counts = df.material.value_counts()
keep_classes = class_counts[class_counts >= MIN_CLASS_ROWS].index

task_a = (df
          .dropna(subset=["material"])
          .query("material in @keep_classes")
          .assign(text=lambda d: d.text.fillna("")))

train = task_a.query("split == 'train'")
test = task_a.query("split == 'test'")
print(f"Task A rows: train={len(train):,}, test={len(test):,}, "
      f"classes={task_a.material.nunique()}")
print(train.material.value_counts().to_string())

# %% Train both TF-IDF models (identical features, only the classifier differs)

def make_tfidf() -> TfidfVectorizer:
    """A fresh vectorizer per pipeline — sharing one fitted instance across
    pipelines would silently couple the two models."""
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                           max_features=60_000)


models = {
    "tfidf_logreg": Pipeline([("tfidf", make_tfidf()),
                              ("clf", LogisticRegression(C=1.0, max_iter=1000,
                                                         class_weight="balanced",
                                                         random_state=SEED))]),
    "tfidf_svm": Pipeline([("tfidf", make_tfidf()),
                           ("clf", LinearSVC(C=1.0, class_weight="balanced",
                                             random_state=SEED))]),
}

rows = []
preds = {}
for name, model in models.items():
    model.fit(train.text, train.material)
    pred = model.predict(test.text)
    preds[name] = pred
    rows.append({"model": name,
                 "macro_f1": round(f1_score(test.material, pred, average="macro"), 4),
                 "accuracy": round(accuracy_score(test.material, pred), 4),
                 "n_train": len(train), "n_test": len(test),
                 "n_classes": int(task_a.material.nunique()),
                 "command": "python analysis/material_tfidf.py",
                 "seed": SEED, "data_version": DATA_VERSION})
    print(f"\n{name}: macro-F1={rows[-1]['macro_f1']}, accuracy={rows[-1]['accuracy']}")
    if name == "tfidf_logreg":
        print(classification_report(test.material, pred, digits=3))

# %% Save metrics and the logistic-regression pipeline for the demo app

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
(pd.DataFrame(rows)
 .to_csv(RESULTS_DIR / "material_tfidf.csv", index=False))
joblib.dump(models["tfidf_logreg"], MODELS_DIR / "material_tfidf.joblib")
print("\nsaved results/material_tfidf.csv and models/material_tfidf.joblib")

# %% Error analysis for the model note (docs/model-notes/material_tfidf.md)
# Per-class scores, the confusion matrix, and the logreg model's mistakes on
# the test split, saved as files the note (and the Auditor) can cite.

import plotly.express as px

best = preds["tfidf_logreg"]
class_names = sorted(test.material.unique())

report = classification_report(test.material, best, output_dict=True, digits=4)
(pd.DataFrame(report).transpose()
 .to_csv(RESULTS_DIR / "material_tfidf_per_class.csv"))

(pd.DataFrame(confusion_matrix(test.material, best, labels=class_names),
              index=class_names, columns=class_names)
 .to_csv(RESULTS_DIR / "material_tfidf_confusion.csv"))

fig = px.imshow(confusion_matrix(test.material, best, labels=class_names),
                x=class_names, y=class_names, text_auto=True,
                labels={"x": "predicted", "y": "true", "color": "count"},
                title="TF-IDF + logistic regression — confusion matrix (test)")
fig.write_html(RESULTS_DIR / "material_tfidf_confusion.html", include_plotlyjs="cdn")
fig.write_image(RESULTS_DIR / "material_tfidf_confusion.png", width=900, height=750, scale=2)

errors = (test
          .assign(predicted=best)
          .query("predicted != material")
          .loc[:, ["parent_asin", "title", "material", "predicted", "text"]]
          .assign(text=lambda d: d.text.str.slice(0, 300)))
errors.to_csv(RESULTS_DIR / "material_tfidf_errors.csv", index=False)

pairs = (errors
         .groupby(["material", "predicted"])
         .size()
         .sort_values(ascending=False)
         .head(8))
print(f"\nmisclassified test rows: {len(errors):,} of {len(test):,}")
print("top confusion pairs (true -> predicted):")
print(pairs.to_string())
print("saved results/material_tfidf_per_class.csv, "
      "material_tfidf_confusion.csv/.html, material_tfidf_errors.csv")

"""Task A baselines: predict the dominant material from listing text.

Owner: Nguyen Ha (Issue #2, Week 2: the 250k-sample rerun; Week 1 pilot
written for Issue #1).

Run from the repo root:

    python analysis/material_tfidf.py

Trains TF-IDF + logistic regression and TF-IDF + linear SVM on
`split == "train"` and reports macro-F1 on `split == "test"` through the
shared evaluator `src/evaluate.py`. Confidence intervals are still
pending: they will be computed from the saved test-set predictions once
the shared bootstrap helper lands in `src/evaluate.py` (Rui Song's open
Issue #2 task), so no retraining will be needed.
Predictions are merged back onto the test rows by `parent_asin` and the
merge is asserted one-to-one before anything is scored, so a prediction
can never be silently paired with the wrong row. Writes
`results/material_tfidf.csv` and `results/material_tfidf_predictions.csv`,
and saves a sigmoid-calibrated copy of the SVM pipeline to
`models/material_tfidf.joblib` for the demo app (its own scores and its
agreement with the reported SVM go to `results/demo_model.csv`).

Data: `data/sample.parquet` when it exists (built by `src/make_sample.py`,
Rui Song) — for Issue #2 that is the 249,762-row sample; until then the
first 10,000 records of `data/meta_head.jsonl`, labeled by the shared rule
(percentage composition strings only).

Leakage (AGENTS.md): the composition strings the label is parsed from are
removed from `text` before training, and `average_rating` / `rating_number`
are never used as features.
"""

# %% Imports and config
import hashlib
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluate import material_scores  # noqa: E402  shared metrics entry point (AGENTS.md)

DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"
SEED = 42
MIN_CLASS_ROWS = 20  # drop classes too rare to split; matches
                     # analysis/material_bert.py so both Task A models score
                     # the same rows (on sample.parquet the labels already
                     # come from src/labels.py)

# %% Load the data (sample.parquet preferred, meta_head.jsonl fallback)

from load_data import load_data, write_run_json

df, DATA_VERSION = load_data()
print(f"Data: {DATA_VERSION}; seed={SEED}")

# %% Helper: id-aligned predictions

def attach_predictions(test: pd.DataFrame, predicted) -> pd.DataFrame:
    """Merge one model's predictions back onto the test rows by parent_asin.

    Scoring reads the merged frame, never the raw prediction array, and the
    asserts fail loudly if the merge is not an exact one-to-one alignment
    (the team's merge-on-id rule).
    """
    preds = (test
             .loc[:, ["parent_asin"]]
             .assign(predicted=predicted))
    merged = test.merge(preds, on="parent_asin", how="left",
                        validate="one_to_one")
    assert len(merged) == len(test), "prediction merge changed the test row count"
    assert merged.parent_asin.is_unique, "test parent_asin values are not unique"
    assert (merged.parent_asin.to_numpy() == test.parent_asin.to_numpy()).all(), \
        "prediction merge reordered the test rows"
    assert merged.predicted.notna().all(), "some test rows got no prediction"
    return merged


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

# Same digest as fingerprint(test, ["parent_asin"]) in analysis/material_bert.py,
# so results/ can prove both runs scored the same test rows.
test_ids = (test.loc[:, ["parent_asin"]].sort_values("parent_asin")
            .to_dict(orient="records"))
TEST_IDS_SHA256 = hashlib.sha256(json.dumps(
    test_ids, ensure_ascii=False, sort_keys=True, separators=(",", ":")
).encode("utf-8")).hexdigest()
print(f"test_ids_sha256={TEST_IDS_SHA256}")

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
scored = {}
for name, model in models.items():
    model.fit(train.text, train.material)
    frame = attach_predictions(test, model.predict(test.text))
    scored[name] = frame
    rows.append({"model": name,
                 "macro_f1": round(material_scores(frame.material, frame.predicted)["macro_f1"], 4),
                 "accuracy": round(accuracy_score(frame.material, frame.predicted), 4),
                 "n_train": len(train), "n_test": len(test),
                 "n_classes": int(task_a.material.nunique()),
                 "command": "python analysis/material_tfidf.py",
                 "seed": SEED, "data_version": DATA_VERSION,
                 "test_ids_sha256": TEST_IDS_SHA256})
    print(f"\n{name}: macro-F1={rows[-1]['macro_f1']}, "
          f"accuracy={rows[-1]['accuracy']}")
    if name == "tfidf_logreg":
        print(classification_report(frame.material, frame.predicted, digits=3))

# %% Demo model: the same SVM pipeline, probability-calibrated (not a reported result)
# Sigmoid calibration, 5-fold. ensemble=False keeps ONE final SVM fitted on all
# training rows (the cross-validation only fits the sigmoid), so the demo is a
# single model whose scores match the reported tfidf_svm, not an average of 5.
demo = CalibratedClassifierCV(clone(models["tfidf_svm"]), method="sigmoid",
                              cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                              ensemble=False)
demo.fit(train.text, train.material)
demo_frame = attach_predictions(test, demo.predict(test.text))
demo_scores = material_scores(demo_frame.material, demo_frame.predicted)
agreement = float((demo_frame.predicted.to_numpy()
                   == scored["tfidf_svm"].predicted.to_numpy()).mean())
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
pd.DataFrame([{"model": "tfidf_svm_calibrated",
               "macro_f1": round(demo_scores["macro_f1"], 4),
               "accuracy": round(demo_scores["accuracy"], 4),
               "n_train": len(train), "n_test": len(test), "seed": SEED,
               "data_version": DATA_VERSION,
               "top1_agreement_with_tfidf_svm": round(agreement, 4),
               "command": "python analysis/material_tfidf.py"}]
             ).to_csv(RESULTS_DIR.joinpath("demo_model.csv"), index=False)
print(f"\ndemo (calibrated SVM): macro-F1={demo_scores['macro_f1']:.4f}, "
      f"accuracy={demo_scores['accuracy']:.4f}, top-1 agreement with tfidf_svm={agreement:.4f}")

# %% Save metrics, per-row predictions, and the calibrated SVM for the demo app

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
(pd.DataFrame(rows)
 .to_csv(RESULTS_DIR / "material_tfidf.csv", index=False))

# attach_predictions asserted each scored frame is in test order, so the
# positional .to_numpy() columns below are aligned by parent_asin.
predictions = (test
               .loc[:, ["parent_asin", "material"]]
               .assign(tfidf_logreg=scored["tfidf_logreg"].predicted.to_numpy(),
                       tfidf_svm=scored["tfidf_svm"].predicted.to_numpy()))
predictions.to_csv(RESULTS_DIR / "material_tfidf_predictions.csv", index=False)

joblib.dump(demo, MODELS_DIR / "material_tfidf.joblib")
run_json = write_run_json(
    Path(__file__), DATA_VERSION, ["numpy", "pandas", "scikit-learn", "joblib"],
    n_train=len(train), n_test=len(test), n_classes=int(task_a.material.nunique()),
    min_class_rows=MIN_CLASS_ROWS, test_ids_sha256=TEST_IDS_SHA256,
    scores={row["model"]: {"macro_f1": row["macro_f1"], "accuracy": row["accuracy"]} for row in rows},
    hyperparameters={name: {k: v for k, v in m.get_params().items() if "__" in k}
                     for name, m in models.items()})
print("\nsaved results/material_tfidf.csv, results/material_tfidf_predictions.csv, "
      f"{run_json.relative_to(ROOT).as_posix()} and models/material_tfidf.joblib")

# %% Error analysis for the model note (docs/model-notes/material_tfidf.md)
# Per-class scores, the confusion matrix, and the logreg model's mistakes on
# the test split, saved as files the note (and the Auditor) can cite.

import plotly.express as px

best_frame = scored["tfidf_logreg"]
best = best_frame.predicted
class_names = sorted(test.material.unique())

report = classification_report(best_frame.material, best, output_dict=True, digits=4)
(pd.DataFrame(report).transpose()
 .to_csv(RESULTS_DIR / "material_tfidf_per_class.csv"))

(pd.DataFrame(confusion_matrix(best_frame.material, best, labels=class_names),
              index=class_names, columns=class_names)
 .to_csv(RESULTS_DIR / "material_tfidf_confusion.csv"))

fig = px.imshow(confusion_matrix(best_frame.material, best, labels=class_names),
                x=class_names, y=class_names, text_auto=True,
                labels={"x": "predicted", "y": "true", "color": "count"},
                title="TF-IDF + logistic regression — confusion matrix (test)")
fig.write_html(RESULTS_DIR / "material_tfidf_confusion.html", include_plotlyjs="cdn")
fig.write_image(RESULTS_DIR / "material_tfidf_confusion.png", width=900, height=750, scale=2)

errors = (best_frame
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

"""Bootstrap confidence intervals from the saved test-set predictions.

Owner: Nguyen Ha (Issue #3). Run from the repo root:

    python analysis/ci_from_predictions.py

Reads the per-row prediction files the model scripts already save
(`results/material_tfidf_predictions.csv`, `results/rating_predictions.csv`,
`results/review_sentiment_predictions.csv`) and writes bootstrap 95% CIs
for the headline metrics, plus paired-difference CIs between models, to
`results/ci_summary.csv`. DistilBERT's CIs come from
`analysis/material_bert.py --compare-only` (`results/material_bert_ci.csv`).

The bootstrap itself is NOT implemented in this file, on purpose: it is
Rui Song's shared helper in `src/evaluate.py` (`bootstrap_ci` and
`bootstrap_diff_ci`, 1,000 resamples, seed 42), and a second, local
implementation would only invite two different CIs for the same number.
No model is retrained.
"""

# %% Imports and config
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import src.evaluate as evaluate  # noqa: E402

RESULTS_DIR = ROOT / "results"
SEED = 42
N_RESAMPLES = 1000

# Each entry: prediction file, id column, true-label column, metric.
# Every remaining column of the file is one model's predictions.
PREDICTION_FILES = [
    ("material_tfidf_predictions.csv", "parent_asin", "material", "macro_f1"),
    # material_bert_predictions.csv is left out on purpose: it holds the 250k
    # DistilBERT run, and its CIs already live in results/material_bert_ci.csv
    # (made by `python analysis/material_bert.py --compare-only`). Adding it
    # here would give one model two CIs.
    ("rating_predictions.csv", "parent_asin", "average_rating", "rmse"),
    ("rating_predictions.csv", "parent_asin", "average_rating", "mae"),
    ("review_sentiment_predictions.csv", "parent_asin", "material_sentiment", "rmse"),
    ("review_sentiment_predictions.csv", "parent_asin", "material_sentiment", "mae"),
]

# Paired differences: score(model) - score(versus) on the same resampled rows.
# Each entry: prediction file, true-label column, model, versus, metric.
DIFFS = [
    ("material_tfidf_predictions.csv", "material", "tfidf_svm", "tfidf_logreg", "macro_f1"),
    *[("rating_predictions.csv", "average_rating", m, "mean_baseline", metric)
      for m in ("xgboost", "random_forest") for metric in ("rmse", "mae")],
    *[("review_sentiment_predictions.csv", "material_sentiment", "xgboost", "mean_baseline", metric)
      for metric in ("rmse", "mae")],
]

# %% Collect the per-model prediction columns

jobs = []
for filename, id_col, label_col, metric in PREDICTION_FILES:
    path = RESULTS_DIR / filename
    if not path.exists():
        continue
    frame = pd.read_csv(path)
    models = [c for c in frame.columns if c not in (id_col, label_col)]
    for model in models:
        jobs.append({"file": filename, "model": model, "metric": metric,
                     "y_true": frame[label_col], "y_pred": frame[model]})

# %% Compute the CIs and write the summary

rows = []
for job in jobs:
    ci_low, ci_high = evaluate.bootstrap_ci(job["y_true"].to_numpy(),
                             job["y_pred"].to_numpy(),
                             job["metric"],
                             n_resamples=N_RESAMPLES, seed=SEED)
    rows.append({"model": job["model"], "versus": "", "metric": job["metric"],
                 "ci_low": round(float(ci_low), 4),
                 "ci_high": round(float(ci_high), 4),
                 "n_test": len(job["y_true"]),
                 "n_resamples": N_RESAMPLES, "seed": SEED,
                 "source_file": job["file"],
                 "command": "python analysis/ci_from_predictions.py",
                 "excludes_zero": ""})
    print(f"{job['model']} {job['metric']}: 95% CI "
          f"{rows[-1]['ci_low']}–{rows[-1]['ci_high']}")

for filename, label_col, model, versus, metric in DIFFS:
    path = RESULTS_DIR / filename
    if not path.exists():
        continue
    frame = pd.read_csv(path)
    ci_low, ci_high = evaluate.bootstrap_diff_ci(
        frame[label_col].to_numpy(), frame[model].to_numpy(), frame[versus].to_numpy(),
        metric, n_resamples=N_RESAMPLES, seed=SEED)
    rows.append({"model": model, "versus": versus, "metric": metric,
                 "ci_low": round(float(ci_low), 4), "ci_high": round(float(ci_high), 4),
                 "n_test": len(frame), "n_resamples": N_RESAMPLES, "seed": SEED,
                 "source_file": filename,
                 "command": "python analysis/ci_from_predictions.py",
                 "excludes_zero": not (ci_low <= 0 <= ci_high)})
    print(f"{model} - {versus} {metric}: 95% CI "
          f"{rows[-1]['ci_low']}–{rows[-1]['ci_high']} excludes_zero={rows[-1]['excludes_zero']}")

(pd.DataFrame(rows)
 .to_csv(RESULTS_DIR / "ci_summary.csv", index=False))
print("saved results/ci_summary.csv")

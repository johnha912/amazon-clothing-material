"""Bootstrap confidence intervals from the saved test-set predictions.

Owner: Nguyen Ha (Issue #3). Run from the repo root:

    python analysis/ci_from_predictions.py

Reads the per-row prediction files the model scripts already save
(`results/material_tfidf_predictions.csv`, `results/rating_predictions.csv`;
DistilBERT once its 250k run lands) and writes
bootstrap 95% CIs for the headline metrics to `results/ci_summary.csv`.

The bootstrap itself is NOT implemented in this file, on purpose: it is
Rui Song's shared helper in `src/evaluate.py` (open Issue #2 task), and a
second, local implementation would only invite two different CIs for the
same number. This file is the driver that feeds the helper the saved
predictions, so the CIs can be produced the moment the helper lands —
no model is retrained.

Expected helper contract (confirm with Rui Song when the helper lands):
`src.evaluate.bootstrap_ci(y_true, y_pred, metric, n_resamples=1000,
seed=42) -> (ci_low, ci_high)`, with `metric` one of "macro_f1", "rmse",
"mae". The lookup below also accepts the names `bootstrap_material_ci`
and `bootstrap_rating_ci` with the same signature.

Until one of those exists, this script prints every CI as pending and
writes nothing — the report keeps saying "pending" until then.
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
    # material_bert_predictions.csv is left out: today it holds the 10k-pilot
    # run (903 test rows), not the 250k test set. Add it back when the
    # DistilBERT 250k run lands.
    ("rating_predictions.csv", "parent_asin", "average_rating", "rmse"),
    ("rating_predictions.csv", "parent_asin", "average_rating", "mae"),
]

HELPER_NAMES = ("bootstrap_ci", "bootstrap_material_ci", "bootstrap_rating_ci")

# %% Find Rui's helper; if it is not there yet, everything stays pending

helper = next((getattr(evaluate, name) for name in HELPER_NAMES
               if hasattr(evaluate, name)), None)

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

if helper is None:
    print("Bootstrap helper not found in src/evaluate.py yet "
          f"(looked for: {', '.join(HELPER_NAMES)}).")
    print("Confidence intervals stay PENDING for:")
    for job in jobs:
        print(f"  - {job['model']} ({job['metric']}, "
              f"n_test={len(job['y_true'])}, from {job['file']})")
    print("No file written. Run this script again once the helper lands; "
          "the predictions it needs are already saved.")
    sys.exit(0)

# %% Helper present: compute the CIs and write the summary

rows = []
for job in jobs:
    ci_low, ci_high = helper(job["y_true"].to_numpy(),
                             job["y_pred"].to_numpy(),
                             job["metric"],
                             n_resamples=N_RESAMPLES, seed=SEED)
    rows.append({"model": job["model"], "metric": job["metric"],
                 "ci_low": round(float(ci_low), 4),
                 "ci_high": round(float(ci_high), 4),
                 "n_test": len(job["y_true"]),
                 "n_resamples": N_RESAMPLES, "seed": SEED,
                 "source_file": job["file"],
                 "command": "python analysis/ci_from_predictions.py"})
    print(f"{job['model']} {job['metric']}: 95% CI "
          f"{rows[-1]['ci_low']}–{rows[-1]['ci_high']}")

(pd.DataFrame(rows)
 .to_csv(RESULTS_DIR / "ci_summary.csv", index=False))
print("saved results/ci_summary.csv")

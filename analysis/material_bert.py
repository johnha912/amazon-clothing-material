"""Task A: fine-tune DistilBERT on the shared, composition-free listing text.

Owner: Yanglin Hu (Issue #2, Week 2: the 250k-sample run with and without
class weights; the Week 1 pilot was written for Issue #1). Run from the
repository root on a machine with one NVIDIA GPU (a local GPU or a Colab GPU
runtime; never a CPU laptop):

    python analysis/material_bert.py --dry-run         # data and leakage checks only
    python analysis/material_bert.py --overwrite       # train, score, write results/
    python analysis/material_bert.py --error-analysis  # rebuild the error files (CPU; --skip-png without Chrome)
    python analysis/material_bert.py --compare-only    # rebuild the comparison (CPU)

Data: `data/sample.parquet` (249,762 rows, built by `src/make_sample.py`),
read through the shared loader `analysis/load_data.py`, which reapplies the
strict composition filter. The 10k pilot (`--download-pilot`,
`data/meta_head.jsonl`) still runs, but its scores are not comparable with the
250k sample. Labels, the minimum class support (20) and the frozen train/test
split match `analysis/material_tfidf.py`.

Two loss settings, fixed on 2026-11-25 before any 250k run:

    distilbert           unweighted cross-entropy (the pilot setting)
    distilbert_weighted  cross-entropy with class weight c_k = n / (K * n_k),
                         counted on the rows being fitted: the same
                         "balanced" rule the TF-IDF models use

Each setting picks its number of epochs (1 to --epochs) by macro-F1 on a
stratified 20% validation subset of TRAIN, then restarts from the pretrained
weights and refits on ALL training rows for that many epochs. Only that final
model predicts the untouched test split. The setting with the higher
validation macro-F1 is the one the report leads with; both test scores are
reported either way, so the choice never looks at test.

Every score comes from `src/evaluate.py`: `material_scores` for macro-F1 and
accuracy, `material_per_class` for per-class precision, recall and F1,
`bootstrap_ci` for 95% intervals and `bootstrap_diff_ci` for paired gaps
against the TF-IDF models on the same test rows (1,000 resamples, seed 42).
(The 2026-11-25 training run computed accuracy with scikit-learn's
`accuracy_score`, which `material_scores` wraps; `--compare-only` rescores
every saved prediction through `src/evaluate.py` and refuses any mismatch.)

Outputs in results/: material_bert.csv (one row per setting),
material_bert_run.json, material_bert_predictions.csv (parent_asin, material,
distilbert, distilbert_weighted), material_bert_per_class.csv,
material_bert_confusion.csv/.html/.png and material_bert_weighted_confusion.*,
material_bert_errors.csv, material_bert_error_summary.json,
material_bert_ci.csv and material_comparison.csv. Weights, tokenizer,
validation histories and the split manifest stay in models/material_bert/,
which is never committed.
"""

# %% Imports and configuration (GPU libraries are imported only for training)
import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import platform
import re
import shlex
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.load_data import load_data  # noqa: E402
from src.evaluate import bootstrap_ci, bootstrap_diff_ci, material_per_class, material_scores  # noqa: E402

SEED = 42
N_RESAMPLES = 1000
MIN_CLASS_ROWS = 20  # Same as material_tfidf.py, so both Task A scripts score the same rows.
SETTINGS = {"distilbert": False, "distilbert_weighted": True}  # model name -> class-weighted loss
TFIDF_MODELS = ["tfidf_logreg", "tfidf_svm"]
# The eight classes named after one fiber word ("other" and "metal" are not),
# used for the same fiber-word split as docs/model-notes/material_tfidf.md.
FIBER_WORD_CLASSES = ["cotton", "leather", "nylon", "polyester", "rayon", "rubber", "spandex", "wool"]
MODEL_NAME = "distilbert/distilbert-base-uncased"
RESULTS_DIR = ROOT / "results"
MODEL_DIR = ROOT / "models/material_bert"
PILOT_URL = (
    "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/"
    "raw/meta_categories/meta_Clothing_Shoes_and_Jewelry.jsonl"
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--download-pilot", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Check data; never train or write scores.")
    mode.add_argument("--compare-only", action="store_true", help="Rebuild the comparison and CIs from saved files.")
    mode.add_argument("--error-analysis", action="store_true", help="Rebuild the error files from saved predictions.")
    parser.add_argument("--epochs", type=int, default=3, help="Validation search: epochs 1 through N.")
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--revision", default="main", help="Pretrained model revision; resolved SHA is logged.")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace an existing BERT run.")
    parser.add_argument("--skip-png", action="store_true",
                        help="With --error-analysis: write every file except the PNGs (no Chrome needed).")
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or not 1 <= args.max_length <= 512:
        parser.error("epochs/batch-size must be positive; max-length must be 1..512")
    if not 0 < args.validation_size < 1 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("validation-size must be in (0, 1); learning-rate must be finite and positive")
    return args


def download_pilot():
    """Stream exactly 10,000 complete records; never fetch the full category."""
    destination = ROOT / "data/meta_head.jsonl"
    if (ROOT / "data/sample.parquet").exists() or destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".jsonl.part")
    try:
        request = urllib.request.Request(PILOT_URL, headers={"User-Agent": "material-course-project"})
        with urllib.request.urlopen(request, timeout=60) as source, temporary.open("wb") as target:
            for _ in range(10_000):
                line = source.readline()
                if not line:
                    raise ValueError("Dataset stream ended before the pilot was complete.")
                json.loads(line)  # Refuse HTML/error pages and truncated records.
                target.write(line if line.endswith(b"\n") else line + b"\n")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


# %% Shared population and leakage checks
def prepare_task(df: pd.DataFrame) -> pd.DataFrame:
    """Use the baseline population without relabeling or changing its split."""
    required = {"parent_asin", "title", "text", "material", "split"}
    if missing := required.difference(df.columns):
        raise ValueError(f"Shared loader is missing columns: {sorted(missing)}")
    if df.parent_asin.isna().any() or df.parent_asin.duplicated().any():
        raise ValueError("parent_asin must be present and unique; ask the pipeline owner to deduplicate.")
    if df.split.isna().any() or not df.split.isin(["train", "test"]).all():
        raise ValueError("Expected the frozen split values train and test only.")
    # Eligibility counts both splits, the same rule as material_tfidf.py, so both scripts score the
    # same rows. Refuse to run if any class would qualify only because of its test rows.
    counts = df.material.value_counts()
    keep_classes = counts[counts >= MIN_CLASS_ROWS].index
    train_counts = (
        df
        .loc[lambda d: d.split.eq("train"), "material"]
        .value_counts()
    )
    if set(keep_classes) != set(train_counts[train_counts >= MIN_CLASS_ROWS].index):
        raise ValueError("Class eligibility depends on test rows; agree a train-only rule with the team first.")
    task = (
        df
        .dropna(subset=["material"])
        .loc[lambda d: d.material.isin(keep_classes)]
        .assign(text=lambda d: d.text.fillna("").astype(str))
        .reset_index(drop=True)
    )
    if task.empty or task.material.nunique() < 2:
        raise ValueError("Need at least two labeled classes with baseline support.")
    if task.text.str.contains(r"\d\s*%", regex=True).any():
        raise ValueError("Composition percentages remain in shared text. Fix with the loader owner first.")
    train = (
        task
        .loc[lambda d: d.split.eq("train")]
    )
    test = (
        task
        .loc[lambda d: d.split.eq("test")]
    )
    if train.empty or test.empty:
        raise ValueError("Both frozen train and test sets must be nonempty.")
    if set(task.material) != set(train.material):
        raise ValueError("A retained class is absent from train; resolve the shared class policy first.")
    return task


def selection_split(train, validation_size):
    """Validation is drawn only from frozen training rows, with seed 42."""
    try:
        fit, validation = train_test_split(
            train, test_size=validation_size, random_state=SEED, stratify=train.material
        )
    except ValueError as error:
        raise ValueError("Cannot stratify training/validation; coordinate the class policy.") from error
    if set(fit.material) != set(train.material) or set(validation.material) != set(train.material):
        raise ValueError("Every class must appear in both fit and validation; adjust validation-size.")
    return fit, validation


def fingerprint(frame, columns):
    """Order-independent digest of the exact IDs, labels, text and/or splits."""
    records = (
        frame
        .loc[:, columns]
        .sort_values("parent_asin")
        .to_dict(orient="records")
    )
    payload = json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def git_value(*arguments):
    result = subprocess.run(["git", *arguments], cwd=ROOT, text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def scores_of(y_true, y_pred):
    """Macro-F1 and accuracy, both from the shared evaluator src/evaluate.py."""
    scores = material_scores(y_true, y_pred)
    return {"macro_f1": float(scores["macro_f1"]), "accuracy": float(scores["accuracy"])}


def balanced_weights(labels: pd.Series, classes):
    """c_k = n / (K * n_k), sklearn's class_weight="balanced", counted on the rows being fitted."""
    counts = (
        labels
        .value_counts()
        .reindex(classes)
    )
    if counts.isna().any():
        raise ValueError("Every class needs at least one fitted row to get a weight.")
    return (len(labels) / (len(classes) * counts)).tolist()


# %% Saved predictions: every Task A model on the same test rows
def read_predictions(results_dir) -> pd.DataFrame:
    """TF-IDF and DistilBERT test predictions merged one-to-one on parent_asin.

    Rows keep the order of material_tfidf_predictions.csv, so the bootstrap
    resamples the same rows that analysis/ci_from_predictions.py resamples.
    """
    needed = ["material_tfidf_predictions.csv", "material_bert_predictions.csv"]
    if missing := [name for name in needed if not (results_dir / name).exists()]:
        raise ValueError(f"Missing prediction files: {missing}.")
    tfidf = pd.read_csv(results_dir / "material_tfidf_predictions.csv", dtype={"parent_asin": str})
    bert = pd.read_csv(results_dir / "material_bert_predictions.csv", dtype={"parent_asin": str})
    if missing := {"parent_asin", "material", *TFIDF_MODELS}.difference(tfidf.columns):
        raise ValueError(f"material_tfidf_predictions.csv is missing {sorted(missing)}.")
    if missing := {"parent_asin", "material", *SETTINGS}.difference(bert.columns):
        raise ValueError(f"material_bert_predictions.csv is missing {sorted(missing)}; rerun the 250k training.")
    if len(tfidf) != len(bert) or set(tfidf.parent_asin) != set(bert.parent_asin):
        raise ValueError("The TF-IDF and DistilBERT prediction files cover different test rows.")
    merged = tfidf.merge(bert, on="parent_asin", how="left", validate="one_to_one", suffixes=("", "_bert"))
    if not merged.material.eq(merged.material_bert).all():
        raise ValueError("The two prediction files disagree on a true label.")
    return (
        merged
        .drop(columns=["material_bert"])
        .loc[:, ["parent_asin", "material", *TFIDF_MODELS, *SETTINGS]]
    )


# %% Confidence intervals and the comparison with Nguyen's baselines
def write_ci_table(results_dir, predictions, data_version) -> pd.DataFrame:
    """Bootstrap 95% CIs for DistilBERT and paired gaps against the other models."""
    rows = []
    for model in SETTINGS:
        low, high = bootstrap_ci(predictions.material, predictions[model], "macro_f1",
                                 n_resamples=N_RESAMPLES, seed=SEED)
        rows.append({"model": model, "versus": "", "estimate": scores_of(predictions.material,
                                                                          predictions[model])["macro_f1"],
                     "ci_low": low, "ci_high": high})
    pairs = [(model, other) for model in SETTINGS for other in TFIDF_MODELS]
    pairs.append(("distilbert_weighted", "distilbert"))
    for model, other in pairs:
        low, high = bootstrap_diff_ci(predictions.material, predictions[model], predictions[other],
                                      "macro_f1", n_resamples=N_RESAMPLES, seed=SEED)
        estimate = (scores_of(predictions.material, predictions[model])["macro_f1"]
                    - scores_of(predictions.material, predictions[other])["macro_f1"])
        rows.append({"model": model, "versus": other, "estimate": estimate, "ci_low": low, "ci_high": high})
    table = (
        pd.DataFrame(rows)
        .assign(
            quantity=lambda d: np.where(d.versus.eq(""), "macro_f1", "macro_f1(model) - macro_f1(versus)"),
            excludes_zero=lambda d: np.where(d.versus.eq(""), "", np.where((d.ci_low > 0) | (d.ci_high < 0),
                                                                           "yes", "no")),
            estimate=lambda d: d.estimate.round(4),
            ci_low=lambda d: d.ci_low.round(4),
            ci_high=lambda d: d.ci_high.round(4),
            n_test=len(predictions), n_resamples=N_RESAMPLES, seed=SEED, data_version=data_version,
            command="python analysis/material_bert.py --compare-only",
        )
        .loc[:, ["model", "versus", "quantity", "estimate", "ci_low", "ci_high", "excludes_zero",
                 "n_test", "n_resamples", "seed", "data_version", "command"]]
    )
    table.to_csv(results_dir / "material_bert_ci.csv", index=False)
    return table


def write_comparison(results_dir):
    """Every Task A model on the same test rows, each with a bootstrap 95% CI of macro-F1."""
    target = results_dir / "material_comparison.csv"
    target.unlink(missing_ok=True)  # A failed rebuild must not leave an old table looking current.
    needed = ["material_tfidf.csv", "material_bert.csv",
              "material_tfidf_predictions.csv", "material_bert_predictions.csv"]
    if missing := [name for name in needed if not (results_dir / name).exists()]:
        raise ValueError(f"The comparison needs {missing}.")
    table = pd.concat([pd.read_csv(results_dir / "material_tfidf.csv"),
                       pd.read_csv(results_dir / "material_bert.csv")], ignore_index=True)
    expected = sorted([*TFIDF_MODELS, *SETTINGS])
    if sorted(table.model) != expected:
        raise ValueError(f"Expected exactly one row each for {expected}.")
    for column in ["n_train", "n_test", "n_classes", "seed", "data_version", "test_ids_sha256"]:
        if table[column].isna().any() or table[column].nunique() != 1:
            raise ValueError(f"Incomparable {column}: rerun every model on the same frozen data.")
    predictions = read_predictions(results_dir)
    if len(predictions) != int(table.n_test.iloc[0]):
        raise ValueError("The prediction files do not hold one row per test listing.")
    intervals = []
    for row in table.itertuples():
        rescored = scores_of(predictions.material, predictions[row.model])
        # Saved CSVs round to 4 decimals, so a match is within half a unit in the 4th place.
        if any(abs(rescored[key] - getattr(row, key)) > 5.01e-5 for key in ["macro_f1", "accuracy"]):
            raise ValueError(f"{row.model}: the saved scores do not match its saved predictions.")
        low, high = bootstrap_ci(predictions.material, predictions[row.model], "macro_f1",
                                 n_resamples=N_RESAMPLES, seed=SEED)
        intervals.append({"model": row.model, "macro_f1_ci_low": round(low, 4), "macro_f1_ci_high": round(high, 4)})
    comparison = (
        table
        .merge(pd.DataFrame(intervals), on="model", validate="one_to_one")
        .assign(
            macro_f1=lambda d: d.macro_f1.round(4),
            accuracy=lambda d: d.accuracy.round(4),
            n_resamples=N_RESAMPLES,
            ci_command="python analysis/material_bert.py --compare-only",
            comparison_verification=("test_ids_sha256_match; predictions merged one-to-one on parent_asin; "
                                     "saved scores re-derived from the saved predictions"),
        )
        .loc[:, ["model", "macro_f1", "macro_f1_ci_low", "macro_f1_ci_high", "accuracy",
                 "n_train", "n_test", "n_classes", "seed", "data_version", "command",
                 "test_ids_sha256", "n_resamples", "ci_command", "comparison_verification"]]
        .sort_values("macro_f1", ascending=False)
    )
    comparison.to_csv(target, index=False)
    ci_table = write_ci_table(results_dir, predictions, table.data_version.iloc[0])
    print(comparison.loc[:, ["model", "macro_f1", "macro_f1_ci_low", "macro_f1_ci_high", "accuracy"]]
          .to_string(index=False))
    print(ci_table.loc[lambda d: d.versus.ne(""), ["model", "versus", "estimate", "ci_low", "ci_high",
                                                   "excludes_zero"]].to_string(index=False))
    print("saved results/material_comparison.csv and results/material_bert_ci.csv")


# %% Error analysis for the model note (docs/model-notes/material_bert.md)
def per_class_table(frame, models, classes, data_version) -> pd.DataFrame:
    """Per-class precision, recall and F1 for each model (src/evaluate.material_per_class), long format."""
    tables = [
        material_per_class(frame.material, frame[model], classes)
        .assign(model=model,
                support=lambda d: d.support.astype(int),
                n_predicted=lambda d: [int(frame[model].eq(name).sum()) for name in d.material])
        for model in models
    ]
    return (
        pd.concat(tables, ignore_index=True)
        .loc[:, ["model", "material", "precision", "recall", "f1", "support", "n_predicted"]]
        .assign(precision=lambda d: d.precision.round(4), recall=lambda d: d.recall.round(4),
                f1=lambda d: d.f1.round(4), seed=SEED, data_version=data_version,
                command="python analysis/material_bert.py --error-analysis")
    )


def names_word(word, text):
    return bool(re.search(rf"\b{re.escape(word)}\b", text, flags=re.IGNORECASE))


def fiber_word_split(frame, models) -> dict:
    """Error rates and counts with and without the true class's own fiber word in the cleaned text.

    For listings that omit their own fiber word, also count errors whose text names one of the
    other seven fiber words, to separate "names a different fiber" from "names none of them".
    """
    named = (
        frame
        .loc[lambda d: d.material.isin(FIBER_WORD_CLASSES)]
        .assign(
            has_word=lambda d: [names_word(label, text) for label, text in zip(d.material, d.text)],
            other_word=lambda d: [any(names_word(word, text) for word in FIBER_WORD_CLASSES if word != label)
                                  for label, text in zip(d.material, d.text)],
        )
    )
    split = {"rows": len(named), "share_with_word": round(float(named.has_word.mean()), 4),
             "rows_with_word": int(named.has_word.sum()), "rows_without_word": int((~named.has_word).sum())}
    for model in models:
        wrong = named[model].ne(named.material)
        split[model] = {
            "error_with_word": round(float(wrong[named.has_word].mean()), 4),
            "error_without_word": round(float(wrong[~named.has_word].mean()), 4),
            "errors_with_word": int((wrong & named.has_word).sum()),
            "errors_without_word": int((wrong & ~named.has_word).sum()),
            "errors_without_word_naming_other_fiber": int((wrong & ~named.has_word & named.other_word).sum()),
            "errors_without_any_fiber_word": int((wrong & ~named.has_word & ~named.other_word).sum()),
        }
    return split


def write_reports(results_dir, task, data_version, skip_png=False):
    """Per-class scores, confusion matrices, errors and a JSON summary from saved predictions.

    PNG export needs kaleido's Chrome; skip_png writes every other file (for machines without it).
    """
    import plotly.express as px

    predictions = read_predictions(results_dir)
    test = (
        task
        .loc[lambda d: d.split.eq("test"), ["parent_asin", "title", "text", "material"]]
    )
    frame = (
        predictions
        .drop(columns=["material"])
        .merge(test, on="parent_asin", how="inner", validate="one_to_one")
    )
    if len(frame) != len(predictions) or len(frame) != len(test):
        raise ValueError("Saved predictions and the loaded test split cover different rows.")
    if not frame.material.eq(predictions.material).all():
        raise ValueError("Saved predictions carry different true labels from the loaded data.")
    classes = sorted(task.material.unique())
    models = [*TFIDF_MODELS, *SETTINGS]
    per_class_table(frame, models, classes, data_version).to_csv(results_dir / "material_bert_per_class.csv",
                                                                 index=False)
    summary = {"command": "python analysis/material_bert.py --error-analysis", "seed": SEED,
               "data_version": data_version, "n_test": len(frame), "models": {}}
    for model in SETTINGS:
        stem = "material_bert" if model == "distilbert" else "material_bert_weighted"
        matrix = confusion_matrix(frame.material, frame[model], labels=classes)
        (
            pd.DataFrame(matrix, index=classes, columns=classes)
            .to_csv(results_dir / f"{stem}_confusion.csv")
        )
        figure = px.imshow(matrix, x=classes, y=classes, text_auto=True,
                           labels={"x": "predicted", "y": "true", "color": "count"},
                           title=f"{model}: confusion matrix (250k test split)")
        figure.write_html(results_dir / f"{stem}_confusion.html", include_plotlyjs="cdn")
        if not skip_png:
            figure.write_image(results_dir / f"{stem}_confusion.png", width=900, height=750, scale=2)
        pairs = (
            frame
            .loc[lambda d: d[model].ne(d.material)]
            .groupby(["material", model])
            .size()
            .sort_values(ascending=False)
            .head(8)
        )
        summary["models"][model] = {
            "misclassified": int(frame[model].ne(frame.material).sum()),
            "top_confusions": [{"true": true, "predicted": predicted, "count": int(count)}
                               for (true, predicted), count in pairs.items()],
        }
    errors = (
        pd.concat([frame
                   .loc[lambda d, m=model: d[m].ne(d.material), ["parent_asin", "title", "material", "text"]]
                   .assign(model=model, predicted=frame[model]) for model in SETTINGS], ignore_index=True)
        .assign(text=lambda d: d.text.str.slice(0, 300))
        .loc[:, ["model", "parent_asin", "material", "predicted", "title", "text"]]
    )
    errors.to_csv(results_dir / "material_bert_errors.csv", index=False)
    summary["fiber_word_split"] = fiber_word_split(frame, models)
    summary["agreement_with_tfidf_svm"] = {}
    for model in SETTINGS:
        bert_right = frame[model].eq(frame.material)
        svm_right = frame.tfidf_svm.eq(frame.material)
        summary["agreement_with_tfidf_svm"][model] = {
            "same_prediction": int(frame[model].eq(frame.tfidf_svm).sum()),
            "both_right": int((bert_right & svm_right).sum()),
            "only_bert_right": int((bert_right & ~svm_right).sum()),
            "only_svm_right": int((~bert_right & svm_right).sum()),
            "both_wrong": int((~bert_right & ~svm_right).sum()),
        }
    (results_dir / "material_bert_error_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["fiber_word_split"], indent=2))
    print("saved results/material_bert_per_class.csv, material_bert*_confusion.csv/.html/.png, "
          "material_bert_errors.csv and material_bert_error_summary.json")


# %% GPU training, final test evaluation, and reproducible outputs
def resolve_model_revision(model_name, revision):
    """Resolve a tag/branch through the public Hub API before loading any weights."""
    if re.fullmatch(r"[0-9a-f]{40}", revision):
        return revision
    from huggingface_hub import HfApi

    resolved = HfApi().model_info(model_name, revision=revision, timeout=30).sha
    if not isinstance(resolved, str) or not re.fullmatch(r"[0-9a-f]{40}", resolved):
        raise RuntimeError("The Hub did not return a full model commit SHA. Retry or pass --revision with a full SHA.")
    return resolved


def gpu_import_error(error):
    """Expose the original import failure instead of assuming packages are missing."""
    root = error
    while root.__cause__ is not None:
        root = root.__cause__
    if "torchvision::nms" in str(root):
        return RuntimeError(
            "Colab's optional torchvision package cannot load against this PyTorch build. "
            "This text-only task does not need torchvision. In this Colab session run "
            "`python -m pip uninstall -y torchvision`, then rerun the script in a fresh "
            "Python process. Keep the team's torch/transformers pins unchanged."
        )
    return RuntimeError(
        f"GPU training dependencies could not be imported: {type(root).__name__}: {root}. "
        "Check the original exception and the repository requirements in this runtime."
    )


def train_settings(args, task, version, fit, validation):
    try:
        import torch
        from transformers import (
            AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
            PrinterCallback, Trainer, TrainerCallback, TrainingArguments, set_seed,
        )
    except (ImportError, RuntimeError) as error:
        raise gpu_import_error(error) from error
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required: use a machine with an NVIDIA GPU, or in Colab choose "
                           "Runtime > Change runtime type > T4 GPU.")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Use a single visible GPU (set CUDA_VISIBLE_DEVICES) for a reproducible run.")

    has_model_output = MODEL_DIR.exists() and any(MODEL_DIR.iterdir())
    if not args.overwrite and ((RESULTS_DIR / "material_bert.csv").exists() or has_model_output):
        raise FileExistsError("A BERT run already exists. Preserve it before explicitly using --overwrite.")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    train = (
        task
        .loc[lambda d: d.split.eq("train")]
    )
    test = (
        task
        .loc[lambda d: d.split.eq("test")]
    )
    classes = sorted(train.material.unique())
    label2id = {label: index for index, label in enumerate(classes)}
    id2label = dict(enumerate(classes))
    resolved_revision = resolve_model_revision(MODEL_NAME, args.revision)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=resolved_revision)

    class ListingDataset(torch.utils.data.Dataset):
        def __init__(self, frame):
            self.encodings = tokenizer(frame.text.tolist(), truncation=True, max_length=args.max_length)
            self.labels = (
                frame
                .material
                .map(label2id)
                .tolist()
            )

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, index):
            return {**{key: value[index] for key, value in self.encodings.items()}, "labels": self.labels[index]}

    class WeightedTrainer(Trainer):
        """Cross-entropy with optional per-class weights; without weights it is the model's own loss."""

        def __init__(self, *arguments, class_weights=None, **keywords):
            super().__init__(*arguments, **keywords)
            self.class_weights = None if class_weights is None else torch.tensor(class_weights, dtype=torch.float32)

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            outputs = model(**{key: value for key, value in inputs.items() if key != "labels"})
            weight = None if self.class_weights is None else self.class_weights.to(outputs.logits.device)
            loss = torch.nn.functional.cross_entropy(outputs.logits.float(), inputs["labels"], weight=weight)
            return (loss, outputs) if return_outputs else loss

    class TimedPrinter(TrainerCallback):
        """Progress lines with wall-clock time, in place of tqdm bars in the run log."""

        def on_log(self, _args, state, _control, logs=None, **_kwargs):
            print(f"[{time.strftime('%H:%M:%S')}] step {state.global_step}/{state.max_steps} {logs}", flush=True)

    def new_model():
        return AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, revision=resolved_revision, num_labels=len(classes),
            id2label=id2label, label2id=label2id,
        )

    def compute_metrics(prediction):
        predicted = np.argmax(prediction.predictions, axis=-1)
        return scores_of(np.asarray(classes)[prediction.label_ids], np.asarray(classes)[predicted])

    def training_arguments(directory, epochs, selecting):
        return TrainingArguments(
            output_dir=str(directory), num_train_epochs=epochs,
            learning_rate=args.learning_rate, weight_decay=0.01,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            eval_strategy="epoch" if selecting else "no", save_strategy="no",
            logging_strategy="steps", logging_steps=500, disable_tqdm=True,
            seed=SEED, data_seed=SEED, fp16=True, report_to="none", dataloader_num_workers=0,
        )

    def make_trainer(**keywords):
        trainer = WeightedTrainer(processing_class=tokenizer, data_collator=collator,
                                  callbacks=[TimedPrinter()], **keywords)
        trainer.remove_callback(PrinterCallback)
        return trainer

    def truncated_share(frame):
        lengths = np.array([len(ids) for ids in tokenizer(frame.text.tolist(), truncation=False)["input_ids"]])
        return round(float((lengths > args.max_length).mean()), 4)

    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)
    datasets = {name: ListingDataset(frame) for name, frame in
                {"fit": fit, "validation": validation, "train": train, "test": test}.items()}
    truncation = {"train": truncated_share(train), "test": truncated_share(test)}
    print(f"Share of listings longer than {args.max_length} tokens: {truncation}")

    runs = {}
    for name, weighted in SETTINGS.items():
        setting_dir = MODEL_DIR / name
        print(f"\n=== {name}: {'class-weighted' if weighted else 'unweighted'} cross-entropy ===", flush=True)
        clock = time.perf_counter()
        set_seed(SEED)
        selector = make_trainer(
            model=new_model(), args=training_arguments(setting_dir / "selection", args.epochs, True),
            train_dataset=datasets["fit"], eval_dataset=datasets["validation"],
            compute_metrics=compute_metrics,
            class_weights=balanced_weights(fit.material, classes) if weighted else None,
        )
        selector.train()
        history = [entry for entry in selector.state.log_history if "eval_macro_f1" in entry]
        # First maximum wins a tie, favoring fewer epochs without looking at test.
        best = max(history, key=lambda entry: entry["eval_macro_f1"])
        best_epoch = int(round(best["epoch"]))
        (setting_dir / "validation_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        selection_seconds = round(time.perf_counter() - clock, 1)
        print(f"{name}: validation macro-F1 by epoch "
              f"{[round(entry['eval_macro_f1'], 4) for entry in history]}; refitting all "
              f"{len(train):,} training rows for {best_epoch} epoch(s).", flush=True)
        del selector
        gc.collect()
        torch.cuda.empty_cache()

        clock = time.perf_counter()
        refit_weights = balanced_weights(train.material, classes) if weighted else None
        set_seed(SEED)
        final_trainer = make_trainer(
            model=new_model(), args=training_arguments(setting_dir / "refit", best_epoch, False),
            train_dataset=datasets["train"], class_weights=refit_weights,
        )
        final_trainer.train()
        final_trainer.save_model(str(setting_dir / "final"))
        tokenizer.save_pretrained(setting_dir / "final")
        # The only prediction on TEST for this setting. Test is never an eval_dataset.
        prediction = final_trainer.predict(datasets["test"])
        predicted = np.asarray(classes)[np.argmax(prediction.predictions, axis=-1)]
        scores = scores_of(test.material.to_numpy(), predicted)
        if not all(math.isfinite(value) and 0 <= value <= 1 for value in scores.values()):
            raise ValueError("Shared evaluator must return finite macro_f1 and accuracy in [0, 1].")
        runs[name] = {
            "loss": "class-weighted cross-entropy" if weighted else "cross-entropy",
            "class_weights": dict(zip(classes, refit_weights)) if weighted else None,
            "validation_history": [{"epoch": entry["epoch"], "macro_f1": entry["eval_macro_f1"],
                                    "accuracy": entry["eval_accuracy"], "loss": entry["eval_loss"]}
                                   for entry in history],
            "best_epoch": best_epoch, "selection_macro_f1": float(best["eval_macro_f1"]),
            "selection_seconds": selection_seconds,
            "refit_and_predict_seconds": round(time.perf_counter() - clock, 1),
            "test_scores": scores, "predicted": predicted,
        }
        print(f"{name}: test macro-F1={scores['macro_f1']:.4f}, accuracy={scores['accuracy']:.4f}", flush=True)
        del final_trainer
        gc.collect()
        torch.cuda.empty_cache()

    # Pre-declared rule: the report leads with the higher VALIDATION macro-F1 (first setting on a tie).
    selected = max(SETTINGS, key=lambda name: runs[name]["selection_macro_f1"])
    command = shlex.join(["python", "analysis/material_bert.py", *sys.argv[1:]])
    test_ids = fingerprint(test, ["parent_asin"])
    task_hash = fingerprint(task, ["parent_asin", "material", "text", "split"])
    provenance = {
        "command": command, "seed": SEED, "data_version": version,
        "n_train": len(train), "n_test": len(test), "n_classes": len(classes),
        "n_selection_fit": len(fit), "n_validation": len(validation),
        "selected_by_validation": selected,
        "task_fingerprint": task_hash, "test_ids_sha256": test_ids,
        "share_longer_than_max_length": truncation,
        "settings": {name: {key: value for key, value in run.items() if key != "predicted"}
                     for name, run in runs.items()},
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_status": git_value("status", "--short"),
        "model_name": MODEL_NAME, "model_revision": resolved_revision,
        "evaluator": "src.evaluate.material_scores (macro-F1, accuracy)",
        "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
        "machine": platform.node(), "platform": platform.platform(),
        "started_utc": started_utc, "elapsed_seconds": round(time.perf_counter() - started, 2),
        "scope": "shoes in scope (team decision 2026-11-07, AGENTS.md)",
        "hyperparameters": vars(args),
        "versions": {"python": platform.python_version(),
                     **{name: importlib.metadata.version(name) for name in [
                         "numpy", "pandas", "scikit-learn", "torch", "transformers", "accelerate",
                         "pyarrow", "plotly"]}},
    }
    (
        pd.DataFrame([{
            "model": name, "macro_f1": round(run["test_scores"]["macro_f1"], 4),
            "accuracy": round(run["test_scores"]["accuracy"], 4),
            "n_train": len(train), "n_test": len(test), "n_classes": len(classes),
            "command": command, "seed": SEED, "data_version": version, "test_ids_sha256": test_ids,
            "loss": run["loss"], "best_epoch": run["best_epoch"],
            "selection_macro_f1": round(run["selection_macro_f1"], 4),
            "selected_by_validation": name == selected,
            "n_selection_fit": len(fit), "n_validation": len(validation),
            "task_fingerprint": task_hash, "model_revision": resolved_revision,
            "gpu": provenance["gpu"], "elapsed_seconds": provenance["elapsed_seconds"],
        } for name, run in runs.items()])
        .to_csv(RESULTS_DIR / "material_bert.csv", index=False)
    )
    (
        test
        .loc[:, ["parent_asin", "material"]]
        .assign(**{name: run["predicted"] for name, run in runs.items()})
        .to_csv(RESULTS_DIR / "material_bert_predictions.csv", index=False)
    )
    (RESULTS_DIR / "material_bert_run.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    (
        task
        .loc[:, ["parent_asin", "material", "split"]]
        .assign(selection_split=lambda d: np.where(d.parent_asin.isin(validation.parent_asin),
                                                  "validation", d.split))
        .to_csv(MODEL_DIR / "split_manifest.csv", index=False)
    )
    print(f"\nSelected by validation macro-F1: {selected}. Saved results/material_bert.csv, "
          "material_bert_predictions.csv and material_bert_run.json.")
    try:
        write_reports(RESULTS_DIR, task, version)
        write_comparison(RESULTS_DIR)
    except ValueError as error:
        # Keep the valid BERT results even if Nguyen's baseline files are missing or stale.
        print(f"Error analysis or comparison not written: {error}")


# %% Entry point; importing this file never trains or downloads anything
def main(argv=None):
    args = parse_args(argv)
    if args.compare_only:
        write_comparison(RESULTS_DIR)
        return
    if args.download_pilot:
        download_pilot()
    if not (ROOT / "data/sample.parquet").exists() and not (ROOT / "data/meta_head.jsonl").exists():
        raise FileNotFoundError("No sample data. Run with --download-pilot, or build data/sample.parquet.")
    df, version = load_data()
    task = (
        df
        .pipe(prepare_task)
    )
    if args.error_analysis:
        write_reports(RESULTS_DIR, task, version, skip_png=args.skip_png)
        return
    train = (
        task
        .loc[lambda d: d.split.eq("train")]
    )
    test = (
        task
        .loc[lambda d: d.split.eq("test")]
    )
    fit, validation = selection_split(train, args.validation_size)
    print(f"Data: {version}; seed={SEED}")
    print(f"Train={len(train):,}; test={len(test):,}; classes={task.material.nunique()}")
    print(f"Selection fit={len(fit):,}; validation={len(validation):,}; evaluator=src.evaluate.material_scores")
    print(f"Task fingerprint: {fingerprint(task, ['parent_asin', 'material', 'text', 'split'])}")
    print(f"test_ids_sha256={fingerprint(test, ['parent_asin'])}")
    if args.dry_run:
        print("Dry run passed. No model trained and no metric files written.")
        return
    train_settings(args, task, version, fit, validation)


if __name__ == "__main__":
    main()

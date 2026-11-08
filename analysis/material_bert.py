"""Task A: fine-tune DistilBERT on the shared, composition-free listing text.

Owner: Yanglin Hu. Run from the repository root in a Colab NVIDIA GPU runtime:

    python analysis/material_bert.py --download-pilot --dry-run
    python analysis/material_bert.py

The shared loader prefers data/sample.parquet; otherwise it uses the pilot.
Labels, the minimum class support (20), and the frozen train/test split match
material_tfidf.py. Select an epoch on a stratified validation subset of TRAIN,
then restart from the pretrained weights and refit on ALL training rows for
that many epochs. Only the final model predicts the untouched test split.

Outputs: results/material_bert.csv, per-class F1, confusion matrix, predictions,
errors, run provenance, and a guarded three-model comparison. Model weights,
tokenizer, validation history, and the exact split manifest stay in models/.
No training runs on CPU. --dry-run needs only the existing data dependencies.

When src/evaluate.py lands, pass --evaluator src.evaluate:FUNCTION, where
FUNCTION(y_true, y_pred) returns a mapping containing macro_f1 and accuracy.
Until then sklearn.metrics is the repository-approved fallback. Do not guess
the future shared API or silently bypass it.
"""

# %% Imports and configuration (GPU libraries are imported only for training)
import argparse
import gc
import hashlib
import importlib
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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.load_data import load_data  # noqa: E402

SEED = 42
MIN_CLASS_ROWS = 20  # Match material_tfidf.py until the shared class policy lands.
PILOT_URL = (
    "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/"
    "raw/meta_categories/meta_Clothing_Shoes_and_Jewelry.jsonl"
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-pilot", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Check data; never train or write scores.")
    mode.add_argument("--compare-only", action="store_true", help="Combine existing metric CSVs only.")
    parser.add_argument("--epochs", type=int, default=3, help="Validation search: epochs 1 through N.")
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--revision", default="main", help="Pretrained model revision; resolved SHA is logged.")
    parser.add_argument("--evaluator", help="Shared metric callable as module:function (see module docstring).")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace an existing BERT run.")
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
    counts = df.material.value_counts()
    keep_classes = counts[counts >= MIN_CLASS_ROWS].index
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


def metric_function(specification=None):
    """Fail closed if the shared evaluator exists but its API is unspecified."""
    if specification:
        module_name, function_name = specification.split(":", 1)
        if module_name != "src.evaluate":
            raise ValueError("The shared evaluator must come from src.evaluate.")
        evaluator = getattr(importlib.import_module(module_name), function_name)
        return evaluator, specification
    if (ROOT / "src/evaluate.py").exists():
        raise ValueError("src/evaluate.py now exists. Pass --evaluator src.evaluate:FUNCTION after checking its API.")

    def sklearn_metrics(y_true, y_pred):
        return {
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
        }

    return sklearn_metrics, "sklearn.metrics (shared src/evaluate.py not yet available)"


# %% Comparison with Nguyen's baselines; never mix pilot and scaled samples
def write_comparison(results_dir):
    # A failed recomputation must not leave a previous sample's table behind.
    (results_dir / "material_comparison.csv").unlink(missing_ok=True)
    paths = [results_dir / "material_tfidf.csv", results_dir / "material_bert.csv"]
    if not all(path.exists() for path in paths):
        raise ValueError("Both material_tfidf.csv and material_bert.csv are needed for the comparison.")
    frames = [pd.read_csv(path) for path in paths]
    required = {"model", "macro_f1", "accuracy", "n_train", "n_test", "n_classes", "seed", "data_version", "command"}
    for path, frame in zip(paths, frames):
        if not required.issubset(frame.columns):
            raise ValueError(f"Missing comparison provenance in {path.name}.")
    table = pd.concat(frames, ignore_index=True)
    if sorted(table.model) != ["distilbert", "tfidf_logreg", "tfidf_svm"]:
        raise ValueError("Expected exactly one row each for distilbert, tfidf_logreg and tfidf_svm.")
    for column in ["n_train", "n_test", "n_classes", "seed", "data_version"]:
        if table[column].isna().any() or table[column].nunique() != 1:
            raise ValueError(f"Incomparable {column}: rerun every model on the same frozen data.")
    for column in ["macro_f1", "accuracy"]:
        values = pd.to_numeric(table[column], errors="coerce")
        if not values.between(0, 1).all():
            raise ValueError(f"Invalid {column} in saved model results.")
    # Old baseline CSVs do not carry row fingerprints. Say that explicitly.
    verification = "legacy_metadata_match_only; Nguyen must confirm frozen rows"
    if "task_fingerprint" in table and table.task_fingerprint.notna().all():
        if table.task_fingerprint.nunique() != 1:
            raise ValueError("Task fingerprints differ; the listing text/labels/IDs/split changed.")
        verification = "exact_task_fingerprint_match"
    table = (
        table
        .assign(comparison_verification=verification)
        .sort_values("macro_f1", ascending=False)
    )
    table.to_csv(results_dir / "material_comparison.csv", index=False)
    print(table.loc[:, ["model", "macro_f1", "accuracy", "n_train", "n_test"]].to_string(index=False))
    print(f"Comparison verification: {verification}")


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
        "Check the original exception and the repository requirements in this Colab runtime."
    )


def train_model(args, task, version, fit, validation, evaluator, evaluator_name):
    try:
        import torch
        from transformers import (
            AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
            Trainer, TrainingArguments, set_seed,
        )
    except (ImportError, RuntimeError) as error:
        raise gpu_import_error(error) from error
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required. In Colab choose Runtime > Change runtime type > T4 GPU.")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Use a single visible GPU for this reproducible Colab workflow.")

    results_dir = ROOT / "results"
    model_dir = ROOT / "models/material_bert"
    has_model_output = model_dir.exists() and any(model_dir.iterdir())
    if not args.overwrite and ((results_dir / "material_bert.csv").exists() or has_model_output):
        raise FileExistsError("A BERT run already exists. Preserve it before explicitly using --overwrite.")
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    set_seed(SEED)
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
    model_name = "distilbert/distilbert-base-uncased"
    resolved_revision = resolve_model_revision(model_name, args.revision)
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=resolved_revision)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, revision=resolved_revision, num_labels=len(classes),
        id2label=id2label, label2id=label2id,
    )
    class ListingDataset(torch.utils.data.Dataset):
        def __init__(self, frame):
            self.encodings = tokenizer(frame.text.tolist(), truncation=True, max_length=args.max_length)
            self.labels = frame.material.map(label2id).tolist()

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, index):
            return {**{key: value[index] for key, value in self.encodings.items()}, "labels": self.labels[index]}

    def compute_metrics(prediction):
        predicted = np.argmax(prediction.predictions, axis=-1)
        scores = evaluator(np.asarray(classes)[prediction.label_ids], np.asarray(classes)[predicted])
        return {key: float(scores[key]) for key in ["macro_f1", "accuracy"]}

    def training_arguments(directory, epochs, selecting):
        return TrainingArguments(
            output_dir=str(directory), num_train_epochs=epochs,
            learning_rate=args.learning_rate, weight_decay=0.01,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            eval_strategy="epoch" if selecting else "no", save_strategy="no",
            logging_strategy="epoch", seed=SEED, data_seed=SEED,
            fp16=True, report_to="none", dataloader_num_workers=0,
        )

    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)
    selector = Trainer(
        model=model, args=training_arguments(model_dir / "selection", args.epochs, True),
        train_dataset=ListingDataset(fit), eval_dataset=ListingDataset(validation),
        processing_class=tokenizer, data_collator=collator, compute_metrics=compute_metrics,
    )
    selector.train()
    history = [entry for entry in selector.state.log_history if "eval_macro_f1" in entry]
    # First maximum wins a tie, favoring fewer epochs without looking at test.
    best = max(history, key=lambda entry: entry["eval_macro_f1"])
    best_epoch = int(round(best["epoch"]))
    (model_dir / "validation_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Refitting all {len(train):,} training rows for {best_epoch} epoch(s).")
    del selector, model
    gc.collect()
    torch.cuda.empty_cache()

    set_seed(SEED)
    final_model = AutoModelForSequenceClassification.from_pretrained(
        model_name, revision=resolved_revision, num_labels=len(classes),
        id2label=id2label, label2id=label2id,
    )
    final_trainer = Trainer(
        model=final_model, args=training_arguments(model_dir / "refit", best_epoch, False),
        train_dataset=ListingDataset(train), processing_class=tokenizer, data_collator=collator,
    )
    final_trainer.train()
    final_trainer.save_model(str(model_dir / "final"))
    tokenizer.save_pretrained(model_dir / "final")
    # This is the only model prediction on TEST. Never pass test as eval_dataset.
    prediction = final_trainer.predict(ListingDataset(test))
    predicted = np.asarray(classes)[np.argmax(prediction.predictions, axis=-1)]
    scores = evaluator(test.material.to_numpy(), predicted)
    scores = {key: float(scores[key]) for key in ["macro_f1", "accuracy"]}
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in scores.values()):
        raise ValueError("Shared evaluator must return finite macro_f1 and accuracy in [0, 1].")

    command = shlex.join(["python", "analysis/material_bert.py", *sys.argv[1:]])
    task_hash = fingerprint(task, ["parent_asin", "material", "text", "split"])
    provenance = {
        "command": command, "seed": SEED, "data_version": version,
        "n_train": len(train), "n_test": len(test), "n_classes": len(classes),
        "n_selection_fit": len(fit), "n_validation": len(validation),
        "best_epoch": best_epoch, "selection_macro_f1": float(best["eval_macro_f1"]),
        "task_fingerprint": task_hash,
        "test_ids_sha256": fingerprint(test, ["parent_asin"]),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_status": git_value("status", "--short"),
        "model_name": model_name, "model_revision": resolved_revision,
        "evaluator": evaluator_name, "gpu": torch.cuda.get_device_name(0),
        "machine": platform.node(), "platform": platform.platform(),
        "started_utc": started_utc, "elapsed_seconds": round(time.perf_counter() - started, 2),
        "scope": "shared sample including shoes; team scope decision pending",
        "hyperparameters": vars(args),
        "versions": {name: importlib.metadata.version(name) for name in [
            "numpy", "pandas", "scikit-learn", "torch", "transformers", "accelerate", "pyarrow", "plotly"
        ]},
    }
    row = {"model": "distilbert", **scores, **{key: value for key, value in provenance.items()
                                             if not isinstance(value, dict)}}
    (
        pd.DataFrame([row])
        .to_csv(results_dir / "material_bert.csv", index=False)
    )
    (results_dir / "material_bert_run.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    (
        task
        .loc[:, ["parent_asin", "material", "split"]]
        .assign(selection_split=lambda d: np.where(d.parent_asin.isin(validation.parent_asin),
                                                  "validation", d.split))
        .to_csv(model_dir / "split_manifest.csv", index=False)
    )
    per_class = classification_report(test.material, predicted, labels=classes, output_dict=True, zero_division=0)
    (
        pd.DataFrame(per_class)
        .transpose()
        .to_csv(results_dir / "material_bert_per_class.csv")
    )
    matrix = confusion_matrix(test.material, predicted, labels=classes)
    (
        pd.DataFrame(matrix, index=classes, columns=classes)
        .to_csv(results_dir / "material_bert_confusion.csv")
    )
    predictions = (
        test
        .loc[:, ["parent_asin", "title", "material", "text"]]
        .assign(predicted=predicted)
    )
    (
        predictions
        .loc[:, ["parent_asin", "material", "predicted"]]
        .to_csv(results_dir / "material_bert_predictions.csv", index=False)
    )
    (
        predictions
        .loc[lambda d: d.material.ne(d.predicted)]
        .assign(text=lambda d: d.text.str.slice(0, 300))
        .to_csv(results_dir / "material_bert_errors.csv", index=False)
    )
    import plotly.express as px

    figure = px.imshow(matrix, x=classes, y=classes, text_auto=True,
                       labels={"x": "Predicted", "y": "True", "color": "Count"},
                       title="DistilBERT: untouched test confusion matrix")
    figure.write_html(results_dir / "material_bert_confusion.html", include_plotlyjs=True)
    print(f"Test macro-F1={scores['macro_f1']:.4f}, accuracy={scores['accuracy']:.4f}")
    try:
        write_comparison(results_dir)
    except ValueError as error:
        # Preserve valid BERT results even if Nguyen's baseline is unavailable/stale.
        # Remove only our generated table so an old table cannot look current.
        print(f"Comparison not written: {error}")


# %% Entry point; importing this file never trains or downloads anything
def main(argv=None):
    args = parse_args(argv)
    if args.compare_only:
        write_comparison(ROOT / "results")
        return
    if args.download_pilot:
        download_pilot()
    if not (ROOT / "data/sample.parquet").exists() and not (ROOT / "data/meta_head.jsonl").exists():
        raise FileNotFoundError("No sample data. Run with --download-pilot, or upload data/sample.parquet.")
    df, version = load_data()
    task = (
        df
        .pipe(prepare_task)
    )
    train = (
        task
        .loc[lambda d: d.split.eq("train")]
    )
    fit, validation = selection_split(train, args.validation_size)
    evaluator, evaluator_name = metric_function(args.evaluator)
    print(f"Data: {version}; seed={SEED}")
    print(f"Train={len(train):,}; test={int(task.split.eq('test').sum()):,}; classes={task.material.nunique()}")
    print(f"Selection fit={len(fit):,}; validation={len(validation):,}; evaluator={evaluator_name}")
    print(f"Task fingerprint: {fingerprint(task, ['parent_asin', 'material', 'text', 'split'])}")
    if args.dry_run:
        print("Dry run passed. No model trained and no metric files written.")
        return
    train_model(args, task, version, fit, validation, evaluator, evaluator_name)


if __name__ == "__main__":
    main()

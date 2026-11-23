"""Shared data loader for the analysis scripts (Nguyen Ha).

Reads `data/sample.parquet` when it exists (built by `src/make_sample.py`,
Rui Song). Until then it builds the same schema from the first 10,000 records
in `data/meta_head.jsonl`, labeled by the shared rule `explore_sample.dominant()`
(percentage composition strings only), with every composition segment removed
from `text` and an 80/20 train/test split on the project seed. Switch the
import to `src/labels.py` when Rui's move lands.

Self-check: python analysis/load_data.py
"""
import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from explore_sample import DETAIL_KEYS, dominant  # noqa: E402  the one labeling rule

DATA_DIR = ROOT / "data"
SEED = 42
STARTED, STARTED_UTC = time.perf_counter(), datetime.now(timezone.utc)
# Any comma- or sentence-delimited clause holding a percentage, in either order
# ("90% Polyester", "Polyester 88%"), so no composition string survives into the
# model input while the rest of the listing (title, other bullets) is kept.
CLAUSE_DELIM = re.compile(r"((?<!\d)\.|\.(?!\d)|[;,\n◆•|])")  # a decimal point is not a delimiter
PCT = re.compile(r"\d{1,3}(?:\.\d+)?\s*%")


def strip_composition(text: str) -> str:
    """Remove composition segments so the label source never reaches `text`.

    Splits on clause delimiters and blanks every clause holding a digit followed by %.
    Replaces a one-regex version whose backtracking was quadratic on long text."""
    parts = CLAUSE_DELIM.split(str(text))
    kept = "".join(" " if PCT.search(p) else p for p in parts)
    return re.sub(r"\s+", " ", kept).strip(" .")


def from_meta_head(path: Path) -> pd.DataFrame:
    """Build the shared sample schema from raw metadata records."""
    def label_source(row) -> str:
        details = row.get("details") or {}
        parts = [str(details[k]) for k in DETAIL_KEYS if k in details]
        parts += [str(f) for f in (row.get("features") or [])]
        return " ".join(parts)

    def build_text(row) -> str:
        parts = [str(row.get("title") or "")]
        parts += [str(f) for f in (row.get("features") or [])]
        parts += [str(d) for d in (row.get("description") or [])]
        # ". " keeps a clause in one field from swallowing the next field
        return strip_composition(". ".join(parts))

    def department(row) -> str:
        cats = [c for c in (row.get("categories") or []) if isinstance(c, str)]
        return cats[1] if len(cats) > 1 else (row.get("main_category") or "Unknown")

    def label(row):
        material, with_pct = dominant(label_source(row))
        return material if with_pct else np.nan

    raw = pd.read_json(path, lines=True)
    rng = np.random.default_rng(SEED)
    return (raw
            .assign(text=lambda d: d.apply(build_text, axis=1),
                    department=lambda d: d.apply(department, axis=1),
                    material=lambda d: d.apply(label, axis=1),
                    split=lambda d: np.where(rng.random(len(d)) < 0.8, "train", "test"))
            .loc[:, ["parent_asin", "title", "text", "price", "department",
                     "average_rating", "rating_number", "material", "split"]])


def load_data() -> tuple[pd.DataFrame, str]:
    """Return the data and its version string: the row count, plus the
    make_sample.py commit when the official sample is used (AGENTS.md)."""
    parquet = DATA_DIR / "sample.parquet"
    if parquet.exists():
        # src/labels.strip_composition misses "Cotton 100%" / "80%Combed cotton"
        # forms, so reapply the stricter clause filter (14.7% of labeled rows).
        df = (pd.read_parquet(parquet)
              .assign(text=lambda d: d.text.fillna("").map(strip_composition),
                      title=lambda d: d.title.fillna("").map(strip_composition)))
        commit = subprocess.run(["git", "log", "-1", "--format=%h", "--", "src/make_sample.py"],
                                cwd=ROOT, capture_output=True, text=True).stdout.strip()
        return df, f"data/sample.parquet ({len(df):,} rows, make_sample.py {commit or 'uncommitted'})"
    df = from_meta_head(DATA_DIR / "meta_head.jsonl")
    return df, f"data/meta_head.jsonl pilot ({len(df):,} rows)"


LISTING_FEATURES = ["log_price", "has_price", "title_words", "text_words",
                    "department", "material"]


def add_listing_features(df: pd.DataFrame) -> pd.DataFrame:
    """The Task B listing features (same as analysis/rating.py add_features),
    shared with Task C so both targets are predicted from the same inputs."""
    return (df
            .assign(has_price=lambda d: d.price.notna().astype(int),
                    log_price=lambda d: np.log1p(d.price),
                    title_words=lambda d: d.title.fillna("").str.split().str.len(),
                    text_words=lambda d: d.text.fillna("").str.split().str.len(),
                    material=lambda d: d.material.fillna("unknown"),
                    department=lambda d: d.department.fillna("Unknown")))


def git_value(*arguments) -> str:
    result = subprocess.run(["git", *arguments], cwd=ROOT, text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def write_run_json(script: Path, data_version: str, packages: list[str], **fields) -> Path:
    """Write results/<script>_run.json so teammates can rerun or audit a run
    (same idea as results/material_bert_run.json). Elapsed time counts from
    when this module was imported, i.e. the start of the analysis script."""
    run = {
        "command": f"python {script.relative_to(ROOT).as_posix()}",
        "seed": SEED, "data_version": data_version, **fields,
        "source_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_status": git_value("status", "--short"),
        "platform": platform.platform(),
        "started_utc": STARTED_UTC.isoformat(),
        "elapsed_seconds": round(time.perf_counter() - STARTED, 2),
        "versions": {"python": platform.python_version(),
                     **{name: importlib.metadata.version(name) for name in packages}},
    }
    path = ROOT / "results" / f"{script.stem}_run.json"
    path.write_text(json.dumps(run, indent=2, default=str), encoding="utf-8")
    return path


if __name__ == "__main__":
    # Leakage self-check: no labeled row may keep a composition string in `text`.
    assert "%" not in strip_composition("Soft. 90% Polyester, 10% Spandex. Machine wash")
    assert "88%" not in strip_composition("Material: Polyester 88% & Spandex 12% ◆Pattern: solid")
    assert strip_composition("Velvet Dress. 95% Polyester, 5% Spandex. Hand wash").startswith("Velvet Dress")
    df, version = load_data()
    labeled = df.dropna(subset=["material"])
    left = labeled.text.str.contains(r"\d\s*%", regex=True).sum()
    empty = (labeled.text.fillna("").str.len() == 0).sum()
    print(f"{version}: labeled rows with a % left in text = {left}, with empty text = {empty}")
    assert left == 0

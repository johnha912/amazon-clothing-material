"""Build data/sample.parquet from the full clothing metadata file, reading it one line at a time.

Run from the repo root:
    python -m src.make_sample data/meta_Clothing_Shoes_and_Jewelry.jsonl --n 50000
    python -m src.make_sample data/meta_Clothing_Shoes_and_Jewelry.jsonl --n 250000
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

from src.labels import DETAIL_KEYS, label, strip_composition

SEED = 42
TOTAL = 7_200_000  # approximate number of products in the category
TEST_SHARE = 0.2
PARENT = re.compile(r'"parent_asin":\s*"([^"]+)"')

NUMBER = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")


def parse_price(value):
    """Numbers stay numbers. Text like 'from 15.00' becomes 15.0; '—' or None becomes None."""
    if value is None or isinstance(value, (int, float)):
        return value
    found = NUMBER.search(str(value))
    return float(found.group().replace(",", "")) if found else None


def bucket(parent_asin, salt):
    """A fixed number in [0, 1) for each product. Same id and salt give the same number every run."""
    digest = hashlib.md5(f"{SEED}:{salt}:{parent_asin}".encode()).hexdigest()
    return int(digest[:15], 16) / 16**15


def split_of(parent_asin):
    return "test" if bucket(parent_asin, "split") < TEST_SHARE else "train"


def to_row(r):
    details = r.get("details") or {}
    features = [str(x) for x in r.get("features") or []]
    description = [str(x) for x in r.get("description") or []]
    title = r.get("title") or ""
    label_text = " ".join([*(str(details[k]) for k in DETAIL_KEYS if k in details), *features])
    return {
        "parent_asin": r["parent_asin"],
        "title": strip_composition(title),
        "text": strip_composition(" ".join([title, *features, *description])),
        "price": parse_price(r.get("price")),
        "department": details.get("Department"),
        "average_rating": r.get("average_rating"),
        "rating_number": r.get("rating_number"),
        "material": label(label_text),
        "split": split_of(r["parent_asin"]),
    }


def read_sample(path, frac):
    """Yield one row per selected product. Lines that are not selected are never parsed."""
    seen = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            found = PARENT.search(line)
            if not found or bucket(found.group(1), "sample") >= frac or found.group(1) in seen:
                continue
            seen.add(found.group(1))
            yield to_row(json.loads(line))


def clean(df) -> pd.DataFrame:
    return df.assign(price=lambda d: pd.to_numeric(d["price"], errors="coerce"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("--n", type=int, default=50_000, help="target number of products")
    parser.add_argument("--out", default="data/sample.parquet")
    args = parser.parse_args()

    df = pd.DataFrame(read_sample(args.source, args.n / TOTAL)).pipe(clean)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)

    print(f"rows={len(df)} labeled={df['material'].notna().mean():.1%} "
          f"missing_price={df['price'].isna().mean():.1%} seed={SEED} -> {args.out}")
    print(df["split"].value_counts().to_string())
    print(df["material"].value_counts().to_string())


if __name__ == "__main__":
    main()

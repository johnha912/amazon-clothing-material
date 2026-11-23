"""Task C: material sentiment from review text.

Owner: Nguyen Ha (Issue #3, Week 3). Plan and leakage check:
docs/feature-plan.md, Task C.

Run from the repo root (the first run streams the 7.1 GB reviews file once
and caches the sample's reviews in data/reviews_sample.parquet):

    python analysis/review_sentiment.py

Instructor feedback: an item's star average is noisy, because a 1-star review
can be about shipping or sizing, not fabric. This script keeps only review
sentences about the material (fabric or feel words), scores each one with a
small negation-aware lexicon, and turns them into a per-product target,
`material_sentiment` = positive / (positive + negative) material sentences.
It then (1) measures how well the star average tracks that target, (2) breaks
it down by material class, and (3) predicts both targets from the same listing
features on the same rows, so their predictability can be compared.

Leakage (AGENTS.md): review text and review stars build the Task C target
only, never a feature. Features are the Task B listing features.
"""

# %% Imports and config
import gzip
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluate import rating_scores  # noqa: E402  shared metrics entry point (AGENTS.md)

from load_data import LISTING_FEATURES, add_listing_features, load_data, write_run_json  # noqa: E402

DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
SEED = 42
REVIEWS_GZ = DATA_DIR / "reviews_Clothing_Shoes_and_Jewelry.jsonl.gz"
REVIEWS_SAMPLE = DATA_DIR / "reviews_sample.parquet"
MAX_REVIEWS = 20       # per product, first ones in file order (feature-plan.md)
MIN_POLAR = 3          # polar material sentences a product needs for a target

# %% Word lists (the one place they live; see feature-plan.md)

FABRIC = """fabric fabrics material materials cotton polyester poly spandex elastane
lycra nylon rayon viscose modal wool merino cashmere linen silk satin leather suede
denim fleece knit lining acrylic bamboo jersey chiffon mesh textile cloth""".split()
# Feel words that are about fabric on their own.
FEEL = """soft softer softest scratchy itchy itch see-through sheer stretchy breathable
pilling pilled pills silky cozy shrank shrunk shrinks faded fades""".split()
# Not in FEEL: thin thick flimsy smooth rough lightweight sturdy wrinkled static
# stiff. They also describe other things ("thin wrists", "thick socks", "sturdy
# ankle", false positives printed by the first run), so they only count in a
# sentence that already names a fabric word. They still score polarity.
POSITIVE = """soft softer softest smooth silky breathable comfortable comfy cozy durable
sturdy nice great love loved lovely good excellent perfect beautiful thick warm
lightweight stretchy happy""".split()
NEGATIVE = """scratchy itchy itch thin flimsy rough stiff cheap cheaply pilling pilled pills
shrank shrunk shrinks faded fades see-through poor bad terrible awful horrible hate
disappointed disappointing unhappy sweaty static wrinkled wrinkles tore ripped holes unraveled
snagged smells falls-apart fell-apart""".split()
NEGATORS = {"not", "no", "never", "isn't", "wasn't", "doesn't", "don't", "didn't",
            "aren't", "weren't", "nothing", "hardly", "isnt", "wasnt", "doesnt", "dont", "didnt"}

MATERIAL_WORD = re.compile(r"\b(?:" + "|".join(map(re.escape, FABRIC + FEEL)) + r")\b")
PHRASES = [(re.compile(r"\bsee[ -]?thru\b|\bsee through\b"), "see-through"),
           (re.compile(r"\b(falls?|fell) apart\b"), r"\1-apart")]
TOKEN = re.compile(r"[a-z]+(?:[-'][a-z]+)*")
SENTENCE_SPLIT = r"(?<=[.!?])\s+|<br\s*/?>|\n"
POS_SET, NEG_SET = set(POSITIVE), set(NEGATIVE)


def normalize(sentence: str) -> str:
    sentence = sentence.lower()
    for pattern, replacement in PHRASES:
        sentence = pattern.sub(replacement, sentence)
    return sentence


def polarity(sentence: str) -> int:
    """+1 / -1 / 0 for a normalized sentence. A negator within the three
    tokens before a term flips it ("not soft" counts as negative)."""
    tokens = TOKEN.findall(sentence)
    score = 0
    for i, token in enumerate(tokens):
        sign = (token in POS_SET) - (token in NEG_SET)
        if sign and NEGATORS.intersection(tokens[max(0, i - 3):i]):
            sign = -sign
        score += sign
    return int(np.sign(score))

# %% Self-check of the lexicon on hand-written sentences

assert polarity(normalize("So soft and comfy")) == 1
assert polarity(normalize("The fabric is scratchy and thin")) == -1
assert polarity(normalize("The material is not soft at all")) == -1
assert polarity(normalize("Not itchy like other sweaters")) == 1
assert polarity(normalize("It is see through")) == -1
assert polarity(normalize("Fell apart after one wash")) == -1
assert MATERIAL_WORD.search(normalize("The cotton is nice"))
assert not MATERIAL_WORD.search(normalize("Arrived late and too small"))
assert not MATERIAL_WORD.search(normalize("Fortunately my wrists are thin"))
assert polarity(normalize("Not happy with the material")) == -1

# %% Load the sample (for its parent_asins and split)

df, DATA_VERSION = load_data()
print(f"Data: {DATA_VERSION}; seed={SEED}")

# %% Stream-filter the reviews file to the sample's products (cached)


def filter_reviews(asins: set) -> pd.DataFrame:
    """One pass over the gzip file. The parent_asin is read with a string
    search first so only kept lines pay for json.loads."""
    kept, counts, seen = [], {}, 0
    started = time.perf_counter()
    with gzip.open(REVIEWS_GZ, "rt", encoding="utf-8") as lines:
        for line in lines:
            seen += 1
            start = line.find('"parent_asin": "')
            asin = line[start + 16:line.find('"', start + 16)] if start >= 0 else None
            if asin not in asins or counts.get(asin, 0) >= MAX_REVIEWS:
                continue
            counts[asin] = counts.get(asin, 0) + 1
            record = json.loads(line)
            kept.append((record["parent_asin"], record["rating"],
                         record.get("title") or "", record.get("text") or ""))
            if seen % 5_000_000 == 0:
                print(f"  {seen:,} lines read, {len(kept):,} kept")
    print(f"read {seen:,} reviews, kept {len(kept):,} for {len(counts):,} products "
          f"in {time.perf_counter() - started:.0f} s")
    return pd.DataFrame(kept, columns=["parent_asin", "rating", "title", "text"])


if not REVIEWS_SAMPLE.exists():
    filter_reviews(set(df.parent_asin)).to_parquet(REVIEWS_SAMPLE, index=False)
reviews = (pd.read_parquet(REVIEWS_SAMPLE)
           .reset_index(names="review_id"))
REVIEWS_VERSION = (f"{REVIEWS_GZ.name} ({REVIEWS_GZ.stat().st_size if REVIEWS_GZ.exists() else 'n/a'} bytes), "
                   f"{len(reviews):,} reviews kept, at most {MAX_REVIEWS} per product")
print(REVIEWS_VERSION)

# %% Material sentences and their polarity

sentences = (reviews
             .assign(sentence=lambda d: (d.title + ". " + d.text).str.split(SENTENCE_SPLIT, regex=True))
             .explode("sentence")
             .assign(sentence=lambda d: d.sentence.fillna("").map(normalize))
             .loc[lambda d: d.sentence.str.contains(MATERIAL_WORD)]
             .assign(polarity=lambda d: d.sentence.map(polarity))
             .loc[:, ["review_id", "parent_asin", "rating", "sentence", "polarity"]])
print(f"material sentences: {len(sentences):,} from {sentences.review_id.nunique():,} reviews")
print("polarity share:", sentences.polarity.value_counts(normalize=True).round(3).to_dict())

# Printed for the false-positive check in feature-plan.md: fix the word lists
# if these show "material" used for something other than the product's fabric.
print("\n20 random material sentences:")
for row in sentences.sample(20, random_state=SEED).itertuples():
    print(f"  [{row.polarity:+d}] {row.sentence[:140]}")

# Lexicon sanity check: polar material sentences should be mostly positive in
# 5-star reviews and mostly negative in 1-star reviews.
lexicon_check = (sentences
                 .query("polarity != 0")
                 .groupby("rating")
                 .agg(polar_sentences=("polarity", "size"),
                      share_positive=("polarity", lambda p: round(float((p > 0).mean()), 3))))
print("\nshare of polar material sentences that are positive, by review star:")
print(lexicon_check.to_string())

# %% Per-product target

target = (sentences
          .query("polarity != 0")
          .groupby("parent_asin")
          .agg(polar=("polarity", "size"),
               material_sentiment=("polarity", lambda p: float((p > 0).mean())))
          .query("polar >= @MIN_POLAR")
          .reset_index())
task_c = (df
          .merge(target, on="parent_asin", how="inner", validate="one_to_one")
          .dropna(subset=["average_rating"])
          .pipe(add_listing_features))
print(f"\nTask C products (>= {MIN_POLAR} polar material sentences): {len(task_c):,} "
      f"of {df.parent_asin.nunique():,}")

# %% (1) How well does the star average track material sentiment?

spearman = task_c[["average_rating", "material_sentiment"]].corr(method="spearman").iloc[0, 1]
mentions = (reviews
            .assign(material_mention=lambda d: d.review_id.isin(sentences.review_id))
            .groupby("rating")
            .material_mention.mean()
            .round(3))
print(f"Spearman(star average, material sentiment) = {spearman:.3f}")
print("share of reviews with any material sentence, by star:", mentions.to_dict())

# 1-star reviews that never mention the material: the noise the instructor
# described. The keyword filter only picks readable examples for the note.
OFF_TOPIC = r"\b(?:ship|shipping|arrived|delivery|size|small|large|fit|fits|return|refund|seller|package)\b"
noise_examples = (reviews
                  .query("rating == 1")
                  .loc[lambda d: ~d.review_id.isin(sentences.review_id)]
                  .loc[lambda d: d.text.str.lower().str.contains(OFF_TOPIC)]
                  .sample(5, random_state=SEED)
                  .loc[:, ["parent_asin", "rating", "title", "text"]])

# %% (2) Material sentiment by material class (labeled products only)

by_material = (task_c
               .query("material != 'unknown'")
               .groupby("material")
               .agg(products=("material_sentiment", "size"),
                    material_sentiment=("material_sentiment", "mean"),
                    star_average=("average_rating", "mean"))
               .round(3)
               .sort_values("material_sentiment"))
print("\nmaterial sentiment by material class:")
print(by_material.to_string())

# %% (3) Predict both targets from the same listing features on the same rows

NUMERIC = ["log_price", "has_price", "title_words", "text_words"]
CATEGORICAL = ["department", "material"]


def make_model(name: str) -> Pipeline:
    if name == "mean_baseline":
        return Pipeline([("model", DummyRegressor(strategy="mean"))])
    prep = ColumnTransformer([("num", SimpleImputer(strategy="median"), NUMERIC),
                              ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)])
    return Pipeline([("prep", prep),
                     ("model", XGBRegressor(n_estimators=400, learning_rate=0.06, max_depth=6,
                                            subsample=0.9, colsample_bytree=0.9,
                                            random_state=SEED, n_jobs=-1))])


train = task_c.query("split == 'train'")
test = task_c.query("split == 'test'")
rows = []
for target_name in ["material_sentiment", "average_rating"]:
    for name in ["mean_baseline", "xgboost"]:
        model = make_model(name).fit(train[LISTING_FEATURES], train[target_name])
        scores = rating_scores(test[target_name], model.predict(test[LISTING_FEATURES]))
        rows.append({"target": target_name, "model": name,
                     "rmse": round(scores["rmse"], 4), "mae": round(scores["mae"], 4),
                     "n_train": len(train), "n_test": len(test),
                     "command": "python analysis/review_sentiment.py", "seed": SEED,
                     "data_version": DATA_VERSION, "reviews_version": REVIEWS_VERSION})
results = (pd.DataFrame(rows)
           .assign(rmse_vs_baseline=lambda d: (d.rmse / d.groupby("target").rmse.transform("first")).round(4)))
print("\n" + results[["target", "model", "rmse", "mae", "rmse_vs_baseline"]].to_string(index=False))

# %% Save results

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
results.to_csv(RESULTS_DIR / "review_sentiment.csv", index=False)
by_material.to_csv(RESULTS_DIR / "review_sentiment_by_material.csv")
lexicon_check.to_csv(RESULTS_DIR / "review_sentiment_lexicon_check.csv")
noise_examples.to_csv(RESULTS_DIR / "review_sentiment_noise_examples.csv", index=False)
run_json = write_run_json(
    Path(__file__), DATA_VERSION, ["numpy", "pandas", "scikit-learn", "xgboost"],
    reviews_version=REVIEWS_VERSION, max_reviews=MAX_REVIEWS, min_polar=MIN_POLAR,
    material_sentences=len(sentences), task_c_products=len(task_c),
    n_train=len(train), n_test=len(test), features=LISTING_FEATURES,
    spearman_star_vs_material=round(float(spearman), 4),
    material_mention_share_by_star=mentions.to_dict(),
    scores={f"{r['target']}/{r['model']}": {"rmse": r["rmse"], "mae": r["mae"]} for r in rows})
print(f"\nsaved results/review_sentiment*.csv and {run_json.relative_to(ROOT).as_posix()}")

"""Task C v2: agreement between the Task C lexicon and a pretrained SST-2 model.

Owner: Nguyen Ha (Issue #3).

This is the Kaggle notebook taskc_sst2_agreement.ipynb saved as a `# %%`
script (AGENTS.md: no .ipynb in the repo); the code is unchanged. It ran on
Kaggle on 2026-11-23 (GPU Tesla T4, 7.8 min total, 400 s of SST-2
inference) with data/reviews_sample.parquet, the review cache that
analysis/review_sentiment.py writes, attached as a private dataset.
Its two summary outputs are committed as results/taskc_sst2_agreement.csv
and results/taskc_sst2_by_rating.csv; the per-sentence parquet it also
writes (9 MB) is not committed. The model is used as published
(distilbert-base-uncased-finetuned-sst-2-english); nothing is trained.

To rerun outside Kaggle, point REVIEWS_SAMPLE and WORKING at local paths.
"""


# %% Setup: seeds, GPU check, locate the Task C v1 review cache
import glob
import random
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

STARTED = time.perf_counter()
WORKING = Path("/kaggle/working")
WORKING.mkdir(parents=True, exist_ok=True)

if not torch.cuda.is_available():
    raise RuntimeError("No GPU available. Set Settings -> Accelerator -> GPU "
                       "and restart the notebook before running.")
print(f"GPU: {torch.cuda.get_device_name(0)}")

candidates = sorted(glob.glob("/kaggle/input/**/reviews_sample.parquet", recursive=True))
assert candidates, "reviews_sample.parquet not found under /kaggle/input/ - attach the dataset first."
if len(candidates) > 1:
    print("multiple matches, using the first:", candidates)
REVIEWS_SAMPLE = candidates[0]
print(f"reviews cache: {REVIEWS_SAMPLE}")


# %% Load the review cache (the columns analysis/review_sentiment.py reads)
reviews = pd.read_parquet(REVIEWS_SAMPLE)
REQUIRED = ["parent_asin", "rating", "title", "text"]
missing = [c for c in REQUIRED if c not in reviews.columns]
assert not missing, f"reviews_sample.parquet is missing columns: {missing} (found: {list(reviews.columns)})"
# The v1 cache writer stores title/text via `record.get(...) or ""`, so they are
# never null in the real cache; the fillna below is only a safety net. review_id
# is the row index, exactly as in review_sentiment.py.
reviews = (reviews
           .assign(title=lambda d: d.title.fillna("").astype(str),
                   text=lambda d: d.text.fillna("").astype(str),
                   rating=lambda d: pd.to_numeric(d.rating, errors="raise"))
           .reset_index(names="review_id"))
print(f"reviews: {len(reviews):,} for {reviews.parent_asin.nunique():,} products")
print("reviews by star:", reviews.rating.value_counts().sort_index().to_dict())


# %% [markdown]
# ## The v1 lexicon — copied verbatim
#
# The next cell is the word-list / regex / `normalize()` / `polarity()` block of `analysis/review_sentiment.py`, copied character-for-character (including its hand-written self-check asserts at the end). Do not edit the word lists here — they live in the repo file; if they ever change there, this notebook must be rebuilt so the comparison stays honest.


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

print("lexicon self-check passed")


# %%
sentences = (reviews
             .assign(sentence=lambda d: (d.title + ". " + d.text).str.split(SENTENCE_SPLIT, regex=True))
             .explode("sentence")
             .assign(sentence=lambda d: d.sentence.fillna("").map(normalize))
             .loc[lambda d: d.sentence.str.contains(MATERIAL_WORD)]
             .assign(polarity=lambda d: d.sentence.map(polarity))
             .loc[:, ["review_id", "parent_asin", "rating", "sentence", "polarity"]])
print(f"material sentences: {len(sentences):,} from {sentences.review_id.nunique():,} reviews")
print("polarity share:", sentences.polarity.value_counts(normalize=True).round(3).to_dict())

# The v1 run produced exactly 209,830 material sentences from this cache.
# If this fails, the attached parquet is not the v1 cache - stop and check
# before believing any number below.
assert len(sentences) == 209_830, f"expected 209,830 material sentences, rebuilt {len(sentences):,}"
print("sentence count matches Task C v1: 209,830")


# %% [markdown]
# ## SST-2 inference
#
# `distilbert-base-uncased-finetuned-sst-2-english` scores **every** material sentence (not only the polar ones). SST-2 has no neutral output: its label maps to a sign, POSITIVE -> +1, NEGATIVE -> -1. The comparison therefore happens on the sentences where the lexicon itself is polar; for the lexicon-neutral sentences we only report how often SST-2 leans positive.


# %% Run SST-2 over every material sentence (GPU, batched)
from transformers import pipeline

MODEL_ID = "distilbert-base-uncased-finetuned-sst-2-english"
clf = pipeline("sentiment-analysis", model=MODEL_ID, device=0)

texts = sentences.sentence.tolist()
labels, scores = [], []
CHUNK = 8192
infer_started = time.perf_counter()
for start in range(0, len(texts), CHUNK):
    batch = texts[start:start + CHUNK]
    for out in clf(batch, batch_size=64, truncation=True):
        labels.append(out["label"])
        scores.append(float(out["score"]))
    print(f"  {min(start + CHUNK, len(texts)):,}/{len(texts):,} sentences "
          f"({time.perf_counter() - infer_started:.0f} s)")
print(f"SST-2 inference done in {time.perf_counter() - infer_started:.0f} s")

sentences = (sentences
             .assign(sst2_label=labels,
                     sst2_score=scores,
                     sst2_sign=lambda d: np.where(d.sst2_label == "POSITIVE", 1, -1)))


# %% Agreement: SST-2 sign vs lexicon polarity (polar sentences; neutrals separate)
from sklearn.metrics import cohen_kappa_score

polar = sentences.query("polarity != 0")
neutral = sentences.query("polarity == 0")
agree = (polar.sst2_sign == polar.polarity)
agreement = float(agree.mean())
kappa = float(cohen_kappa_score(polar.polarity, polar.sst2_sign))
neutral_pos_share = float((neutral.sst2_sign == 1).mean()) if len(neutral) else float("nan")

print(f"material sentences: {len(sentences):,} (polar {len(polar):,}, lexicon-neutral {len(neutral):,})")
print(f"overall agreement on polar sentences: {agreement:.4f}")
print(f"Cohen's kappa: {kappa:.4f}")
print(f"SST-2 positive share on lexicon-neutral sentences: {neutral_pos_share:.4f}")

# 2x2 confusion table (rows: lexicon, columns: SST-2)
NAME = {-1: "negative", 1: "positive"}
confusion = (pd.crosstab(polar.polarity.map(NAME), polar.sst2_sign.map(NAME))
             .reindex(index=["negative", "positive"], columns=["negative", "positive"], fill_value=0))
confusion.index.name = "lexicon \\ sst2"
confusion.columns.name = "sst2"
print("\nconfusion table (polar sentences):")
print(confusion.to_string())

# Agreement by the review's star rating (agreement is over polar sentences only;
# the NaN agree values of neutral sentences drop out of the mean).
by_rating = (sentences
             .assign(agree=lambda d: np.where(d.polarity != 0,
                                              (d.sst2_sign == d.polarity).astype(float),
                                              np.nan))
             .groupby("rating")
             .agg(sentences=("sentence", "size"),
                  polar_sentences=("polarity", lambda p: int((p != 0).sum())),
                  agreement=("agree", "mean"))
             .reset_index())
print("\nagreement by star rating:")
print(by_rating.to_string(index=False))

# 10 disagreement examples
disagreements = polar.loc[~agree]
print(f"\ndisagreements: {len(disagreements):,} of {len(polar):,} polar sentences")
for row in disagreements.sample(min(10, len(disagreements)), random_state=SEED).itertuples():
    print(f"\n  stars={row.rating:g}  lexicon={NAME[row.polarity]}  sst2={row.sst2_label} ({row.sst2_score:.3f})")
    print(f"  {row.sentence[:300]}")


# %% Save outputs to /kaggle/working
summary = pd.DataFrame([
    ("material_sentences", len(sentences)),
    ("polar_sentences", len(polar)),
    ("lexicon_neutral_sentences", len(neutral)),
    ("agreement_polar", round(agreement, 4)),
    ("cohen_kappa_polar", round(kappa, 4)),
    ("sst2_positive_share_on_lexicon_neutral", round(neutral_pos_share, 4)),
    ("confusion_lexicon_neg_sst2_neg", int(confusion.loc["negative", "negative"])),
    ("confusion_lexicon_neg_sst2_pos", int(confusion.loc["negative", "positive"])),
    ("confusion_lexicon_pos_sst2_neg", int(confusion.loc["positive", "negative"])),
    ("confusion_lexicon_pos_sst2_pos", int(confusion.loc["positive", "positive"])),
    ("seed", SEED),
    ("model", MODEL_ID),
], columns=["metric", "value"])
summary.to_csv(WORKING / "taskc_sst2_agreement.csv", index=False)
by_rating.round({"agreement": 4}).to_csv(WORKING / "taskc_sst2_by_rating.csv", index=False)
(sentences
 .loc[:, ["sentence", "rating", "polarity", "sst2_label", "sst2_score"]]
 .rename(columns={"polarity": "lexicon_polarity"})
 .to_parquet(WORKING / "taskc_sst2_sentences.parquet", index=False))
print(f"saved {WORKING / 'taskc_sst2_agreement.csv'}")
print(f"saved {WORKING / 'taskc_sst2_by_rating.csv'}")
print(f"saved {WORKING / 'taskc_sst2_sentences.parquet'}")
print(f"total runtime: {(time.perf_counter() - STARTED) / 60:.1f} min")

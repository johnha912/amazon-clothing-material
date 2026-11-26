# What Is It Made Of?

**Predicting clothing material, ratings, and material sentiment from Amazon listings and reviews**

![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)
![Course project](https://img.shields.io/badge/CS6140-course%20project-lightgrey)
![Status](https://img.shields.io/badge/status-250k%20results-yellowgreen)

A final project for CS6140 Machine Learning (Fall 2026, Northeastern University — Silicon Valley Campus, Prof. Mohammadhossein Toutiaee), built by a team of three graduate students.

Clothing material drives comfort and durability, yet on Amazon it is buried in free-text bullets instead of a searchable field. We ask three questions about the same listings:

| Task | Question | Type | Metrics |
|---|---|---|---|
| **A — Material** | Can a model infer a product's dominant material from its listing text? | Classification (10 classes) | Macro-F1 |
| **B — Rating** | Do material, price, and department predict the product's star average? | Regression | RMSE, MAE |
| **C — Material sentiment** | What do buyers say about the fabric itself, and how much of the star average does it explain? | Review NLP + regression | Spearman, RMSE, MAE |

Task C was added after instructor feedback that star averages are noisy: a 1-star review can be about shipping or sizing, not fabric. Plan and leakage check: [`docs/feature-plan.md`](docs/feature-plan.md). Pipeline figure: [`report/assets/project-pipeline.mmd`](report/assets/project-pipeline.mmd).

## Data

Clothing, Shoes and Jewelry from [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) (Hou et al., McAuley Lab): about 7.2 million products and 66 million reviews. All results use `data/sample.parquet`: 249,762 products (199,838 train / 49,924 test), built with `python -m src.make_sample data/meta_Clothing_Shoes_and_Jewelry.jsonl --n 250000`, seed 42, `src/make_sample.py` commit 2219c1e. Raw data is never committed.

Labels are extracted, not given: a label is the highest-percentage fiber in a composition string such as "90% Polyester, 10% Spandex" (ties go to the first fiber listed). Only 36.4% of listings carry one (90,979), so Task A has 72,661 train / 18,318 test rows. Cotton (33.0%), polyester (31.3%), and leather (17.3%) dominate; 79.8% of listings have no price. Shoes stay in scope, so the leather class mostly measures footwear.

**Leakage controls.** Composition strings are removed from the model text, so Task A is not a lookup of its own label (`analysis/load_data.py` blanks every clause holding a digit followed by a percent sign, and its self-check passes). `average_rating` and `rating_number` are never features; `rating_number` is only a row filter (at least 5 ratings) and a sample weight in Task B. Review text and review stars only build the Task C target, never a feature. One fixed split (seed 42) for every run. Full rules: [AGENTS.md](AGENTS.md).

## Results (250k sample, test split)

All numbers: `data/sample.parquet (249,762 rows, make_sample.py 2219c1e)`, seed 42, scored through `src/evaluate.py`. 95% confidence intervals (1,000 resamples, seed 42) are in `results/ci_summary.csv` (`python analysis/ci_from_predictions.py`) and, for DistilBERT, `results/material_bert_ci.csv`. Each model has a note in [`docs/model-notes/`](docs/model-notes/) (method, rationale, error analysis with real examples); run provenance is in `results/*_run.json`.

**Task A — Material** (`python analysis/material_tfidf.py`)

| Model | Macro-F1 | Accuracy |
|---|---|---|
| TF-IDF + logistic regression | 0.5799 | 0.7326 |
| TF-IDF + linear SVM | 0.6203 | 0.7872 |
| DistilBERT, unweighted (`python analysis/material_bert.py --overwrite`, GPU) | **0.6758** | **0.8075** |
| DistilBERT, class-weighted | 0.6576 | 0.7733 |

Leather is nearly solved (F1 0.943 for logistic regression); metal (0.224, 27 test rows) and spandex (0.324) are hardest, and polyester is most often confused with rayon, nylon, and cotton. DistilBERT (unweighted, chosen on validation macro-F1 before the test scores) is 0.0555 above the SVM in macro-F1 (paired 95% CI 0.0325 to 0.0763, excludes 0; CIs 0.6512–0.6968 DistilBERT, 0.6027–0.6359 SVM, from `--compare-only`), at a 1,719.8 s GPU run for its two settings (RTX 5090 Laptop GPU); the weighted run is 0.0182 lower than the unweighted one (−0.0343 to −0.0018), one seed each. Details: [`docs/model-notes/material_bert.md`](docs/model-notes/material_bert.md). The old pilot score (0.3563 on 903 pilot rows) is not comparable.

**Task B — Rating** (`python analysis/rating.py`)

| Model | RMSE | MAE |
|---|---|---|
| Mean baseline | 0.4955 | 0.3817 |
| Random forest | 0.5044 | 0.3853 |
| XGBoost | **0.4761** | **0.3620** |

Only products with at least 5 ratings are kept (138,355 train / 34,836 test; seed 42), so these numbers are not comparable with the earlier unfiltered run (0.7512 / 0.7647 / 0.7373). Listing features beat the mean modestly (XGBoost is 3.9% lower in RMSE, our arithmetic; RMSE 95% CI 0.4715–0.4808 against 0.4908–0.5001 for the baseline, paired difference −0.0208 to −0.0180), and the hardest group is listings whose material could not be parsed (RMSE 0.4984).

**Task C — Material sentiment** (`python analysis/review_sentiment.py`, lexicon version 1)

Up to 20 reviews per product (866,008 reviews) are split into sentences; 209,830 sentences that name a fabric or a fabric feel are scored with a negation-aware lexicon, and a product's target is the share of its polar material sentences that are positive (15,397 products with at least 3). Only 18.6% of 1-star reviews contain a lexicon-matched material sentence (`results/review_sentiment_run.json`), and the star average correlates only weakly with material sentiment (Spearman 0.292). Cotton (0.773) and polyester (0.776) draw the most fabric complaints, leather (0.843) the fewest.

| Target (same 3,093 test products) | Mean baseline RMSE | XGBoost RMSE | Change |
|---|---|---|---|
| Material sentiment | 0.2795 | 0.2815 | +0.7% |
| Star average | 0.4695 | 0.4487 | −4.4% |

Listing fields say something about the overall rating but little about how the fabric feels: the paired interval for XGBoost minus baseline RMSE, −0.0008 to +0.0045, includes zero, so the two do not differ measurably on material sentiment.

## Getting started

Requires Python 3.12 or later.

```bash
git clone https://github.com/johnha912/amazon-clothing-material.git
cd amazon-clothing-material
pip install -r requirements.txt
mkdir -p data

# Metadata (18 GB) and the 250k sample (about 3 minutes on a laptop)
curl -L -C - -o data/meta_Clothing_Shoes_and_Jewelry.jsonl https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/meta_categories/meta_Clothing_Shoes_and_Jewelry.jsonl
python -m src.make_sample data/meta_Clothing_Shoes_and_Jewelry.jsonl --n 250000

# Reviews for Task C (7.1 GB; one pass of about 13 minutes, then cached)
curl -L -C - -o data/reviews_Clothing_Shoes_and_Jewelry.jsonl.gz https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Clothing_Shoes_and_Jewelry.jsonl.gz

python analysis/load_data.py         # leakage self-check
python analysis/eda.py               # EDA charts -> results/eda/
python analysis/material_tfidf.py    # Task A
python analysis/rating.py            # Task B (about 21 minutes on a laptop CPU)
python analysis/review_sentiment.py  # Task C
```

Without `data/sample.parquet`, the scripts fall back to a 10k pilot file `data/meta_head.jsonl` (the first 10,000 metadata lines) with the same schema and split.

## Repository layout

```
├── AGENTS.md          # working rules: ownership, labels, leakage, evaluation
├── ai-prompts.md      # AI-assistant prompt log (course requirement)
├── explore_sample.py  # home of the shared labeling rule
├── analysis/          # load_data, eda, material_tfidf, material_bert, rating, review_sentiment
├── docs/              # feature plan, model notes, code style, method sources
├── app/               # Streamlit demo
├── report/            # report sections and figures
├── results/           # metric CSVs, run provenance, charts, error examples
├── src/               # sample builder, labels and tests, shared evaluation
└── data/, models/     # local only, never committed
```

DistilBERT weights and its Colab notebook are attached to the [`yanglin-distilbert-2026-11-02` release](https://github.com/johnha912/amazon-clothing-material/releases/tag/yanglin-distilbert-2026-11-02).

## Status

- ✅ Week 1 ([Issue #1](https://github.com/johnha912/amazon-clothing-material/issues/1)): 250k sample, pilot models, Related Work and Related Implementations drafted
- 🚧 Week 2 ([Issue #2](https://github.com/johnha912/amazon-clothing-material/issues/2)): TF-IDF and rating rerun on 250k done; DistilBERT 250k run done (2026-11-25); 250k demo done
- 🚧 Week 3 ([Issue #3](https://github.com/johnha912/amazon-clothing-material/issues/3)): Task C version 1 done; report sections for Task C next
- 📝 Report (PDF, max 8 pages) and a 5 ± 1 minute recorded presentation with a demo, due **December 6, 2026, 11:55 PM EST**

## Team

| Name | Program | GitHub | Main responsibilities |
|---|---|---|---|
| Nguyen Ha | M.S. in Computer Science | [@johnha912](https://github.com/johnha912) | TF-IDF, rating, and material-sentiment models, EDA, report |
| Yanglin Hu | M.S. in Data Science | [@Elliothu-oss](https://github.com/Elliothu-oss) | DistilBERT modeling (lead) |
| Rui Song | M.S. in Computer Science | [@ruisnow-e](https://github.com/ruisnow-e) | Data pipeline (`src/`), demo app, related work |

## Related work

[`related_work.md`](report/sections/related_work.md) covers six papers (the dataset paper, HFT rating prediction, OpenTag and MAVE attribute extraction, DistilBERT, TF-IDF SVMs vs. transformers); [`related_implementations.md`](report/sections/related_implementations.md) compares three Kaggle notebooks on this dataset family; [`method-analysis-sources.md`](docs/method-analysis-sources.md) lists five more papers used in the Method and Analysis sections. The report's full reference list: [`11_references.md`](report/sections/11_references.md).

## AI assistance

This project is developed with AI assistants under the rules in [AGENTS.md](AGENTS.md): file ownership per member, audits before each push, and no metric in the report unless it was produced on the untouched test split. Prompts are logged in [`ai-prompts.md`](ai-prompts.md) for the report's "AI Prompts Used" section.

## References

- Hou, Yupeng, et al. "Bridging Language and Items for Retrieval and Recommendation." *arXiv*, 2024. [arxiv.org/abs/2403.03952](https://arxiv.org/abs/2403.03952)
- Chen, Tianqi, and Carlos Guestrin. "XGBoost: A Scalable Tree Boosting System." *Proceedings of the 22nd ACM SIGKDD*, ACM, 2016, pp. 785–94.
- Sanh, Victor, et al. "DistilBERT, a Distilled Version of BERT: Smaller, Faster, Cheaper and Lighter." *arXiv*, 2019. [arxiv.org/abs/1910.01108](https://arxiv.org/abs/1910.01108)

---

Dataset courtesy of the McAuley Lab (UC San Diego). This repository is a course project; code and results are shared for learning and review.

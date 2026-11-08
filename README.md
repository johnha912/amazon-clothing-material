# What Is It Made Of?

**Predicting clothing material and buyer ratings from Amazon product listings**

![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)
![Course project](https://img.shields.io/badge/CS6140-course%20project-lightgrey)
![Status](https://img.shields.io/badge/status-pilot%20results-yellow)

A final project for CS6140 Machine Learning (Fall 2026, Northeastern University — Silicon Valley Campus, Prof. Mohammadhossein Toutiaee), built by a team of three graduate students.

Clothing material drives comfort, durability, and how a product ages, yet on Amazon it is usually buried in free-text feature bullets instead of a searchable field. We study two questions on the same listings:

| Task | Question | Type | Primary metrics |
|---|---|---|---|
| **A — Material** | Can a model infer a product's dominant material from its listing text? | Multi-class classification | Macro-F1 (plus per-class F1) |
| **B — Rating** | Do material, price, and department help predict the product's average rating? | Regression | RMSE, MAE |

```mermaid
flowchart LR
    subgraph S0["Source"]
        hf[("Amazon Reviews 2023<br/>Clothing, Shoes & Jewelry<br/>~7.2M products")]
    end

    subgraph S1["Sample & labels"]
        pilot["explore_sample.py · analysis/load_data.py<br/>10k pilot (meta_head.jsonl)<br/>current results"]
        labels["src/labels.py<br/>dominant() · %-string labels only"]
        makesample["src/make_sample.py<br/>sample by parent_asin hash · seed 42"]
        sample[("data/sample.parquet<br/>249,762 rows · 36.4% labeled<br/>split = train / test")]
    end

    subgraph S2["Models & EDA"]
        eda["analysis/eda.py<br/>Plotly charts"]
        tfidf["analysis/material_tfidf.py<br/>Task A · TF-IDF + LogReg / SVM"]
        bert["analysis/material_bert.py<br/>Task A · DistilBERT (GPU)"]
        rating["analysis/rating.py<br/>Task B · mean / RF / XGBoost"]
    end

    subgraph S3["Outputs"]
        evaluate["src/evaluate.py<br/>shared metrics (planned path)"]
        results[("results/*.csv<br/>macro-F1 · RMSE · MAE<br/>+ command, seed, data version")]
        model[("models/<br/>saved pipelines · not committed")]
        app["app/ · Streamlit demo<br/>paste listing → material"]
        report["report/ · report + presentation<br/>due Dec 6"]
    end

    hf -- "first 10k records" --> pilot
    pilot -- "pilot data" --> eda
    pilot -- "pilot data" --> tfidf
    pilot -- "pilot data" --> bert
    pilot -- "pilot data" --> rating
    hf --> makesample
    labels -- "label rule" --> makesample
    makesample -- "composition stripped from text" --> sample
    sample -. "250k rerun" .-> eda
    sample -. "250k rerun" .-> tfidf
    sample -. "250k rerun" .-> bert
    sample -. "250k rerun" .-> rating
    tfidf -- "pilot: sklearn.metrics" --> results
    bert -- "pilot: sklearn.metrics" --> results
    rating -- "pilot: sklearn.metrics" --> results
    tfidf -. "250k rerun" .-> evaluate
    bert -. "250k rerun" .-> evaluate
    rating -. "250k rerun" .-> evaluate
    evaluate -. "scores on test split" .-> results
    tfidf -- "trained pipeline" --> model
    model -- "loads model" --> app
    eda -- "charts + findings" --> report
    results -- "merged tables" --> report
```

Solid arrows are steps already run, and the current pilot numbers come from `data/meta_head.jsonl` scored with `sklearn.metrics`; dashed arrows are the path for the ~250k rerun through `src/evaluate.py`. Source: [`report/assets/project-pipeline.mmd`](report/assets/project-pipeline.mmd).

## Dataset

We use the Clothing, Shoes and Jewelry metadata of [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) (Hou et al., McAuley Lab): about 7.2 million products, each with a title, feature bullets, description, price, department, and average rating. The current results use a **pilot of the first 10,000 records**. The reproducible ~250k-record sample (`python -m src.make_sample data/meta_Clothing_Shoes_and_Jewelry.jsonl --n 250000`, seed 42, `src/make_sample.py` commit 2219c1e) has been built twice with identical results (Rui Song, and an independent rebuild, both in Issue #1): 249,762 rows, 199,838 train / 49,924 test. On that sample only 36.4% of listings are labeled and 79.8% lack a price, so the pilot shares below are not representative; the models are rerun on it in Week 2 ([Issue #2](https://github.com/johnha912/amazon-clothing-material/issues/2)). Raw data is never committed — the `data/` folder is gitignored.

Labels are not given; we extract them. A label is the highest-percentage fiber in a composition string such as "90% Polyester, 10% Spandex" (ties go to the first fiber listed). Only listings that carry such a string can be labeled:

| Pilot EDA (10,000 listings) | Share |
|---|---|
| Listings with a percentage composition in their bullets | 46.0% |
| Listings with material in a structured details field | 3.6% |
| Listings that name a known material anywhere | 79.0% |
| Listings with a missing price | 45.7% |

Among labeled listings, polyester (~34%) and cotton (~31%) dominate, followed by leather (~13%). Classes with fewer than 20 labeled rows are dropped for the pilot, leaving 9 classes (3,641 train / 903 test). Task B uses all 10,000 listings (8,046 train / 1,954 test). On the 250k sample the 90,979 labeled listings fall into 10 classes (cotton 33.0%, polyester 31.3%, leather 17.3%, then nylon, rayon, other, spandex, wool, rubber, and metal), so pilot and 250k scores are not directly comparable. Shoes stay in scope: dropping the 113 shoe rows from the pilot test set changes macro-F1 by -0.017 to -0.019 for the three models, and every bootstrap 95% confidence interval of that change includes 0, but 105 of the 115 leather test rows are shoes, so the leather class mostly measures footwear.

## Approach

**Leakage controls** (the part we are most careful about):

- The composition strings the label is parsed from are **removed from the model input text**, so Task A is not a lookup of the answer key.
- Task A never uses `average_rating` or `rating_number` as features.
- In Task B, `rating_number` is never a feature; it is only used as a sample weight (`log1p`), as recorded in [AGENTS.md](AGENTS.md).
- One fixed split (`split` column, seed 42) for every run; the test split is never tuned on.

**Models**

| Task | Model | Status |
|---|---|---|
| A | TF-IDF + logistic regression | ✅ Pilot results below |
| A | TF-IDF + linear SVM | ✅ Pilot results below |
| A | DistilBERT fine-tune (GPU) | ✅ Pilot results below |
| B | Mean baseline | ✅ Pilot results below |
| B | Random forest | ✅ Pilot results below |
| B | XGBoost | ✅ Pilot results below |

Each model family with results (TF-IDF material models, DistilBERT, rating models) has a note in [`docs/model-notes/`](docs/model-notes/) explaining how it works, why we chose it for this dataset, and an error analysis with real examples.

## Pilot results (10,000 listings)

### Task A — Material classification

| Model | Macro-F1 | Accuracy |
|---|---|---|
| TF-IDF + logistic regression | **0.5572** | 0.6766 |
| TF-IDF + linear SVM | 0.5239 | **0.7364** |
| DistilBERT (fine-tuned, Colab T4) | 0.3563 | 0.6899 |

Per-class F1 (logistic regression): leather is nearly solved (0.94) and cotton is solid (0.74), while spandex (0.20), wool (0.40), and nylon (0.48) are hard — they are rare and often blended. The most common confusion is polyester ↔ cotton. DistilBERT matches or beats logistic regression on cotton, polyester, leather, and nylon but gets none of the three rarest fibers right (polyurethane, spandex, and wool, F1 0), which accounts for about 60% of its macro-F1 gap; rayon (F1 0.07) and acrylic account for the rest. One untested explanation is that DistilBERT was fine-tuned without class weights while the TF-IDF models use balanced weights. All three models were scored on the same 903 test rows (matching `test_ids_sha256` in [`results/material_comparison.csv`](results/material_comparison.csv)); these are single-seed pilot numbers without confidence intervals. The DistilBERT row is provisional until its independent audit. Full tables, the confusion matrix, and error examples are in [`results/`](results/).

### Task B — Rating prediction

| Model | RMSE | MAE |
|---|---|---|
| Mean baseline | 0.5863 | 0.3834 |
| Random forest | 0.5854 | 0.3851 |
| XGBoost | **0.5780** | **0.3736** |

An honest reading: listing-side features barely beat predicting the mean. Ratings seem to be driven more by fit, quality, and expectations than by what a listing says — we treat that as a finding, not a failure, and the report's analysis section digs into where the remaining error concentrates (hardest group: listings whose material could not be parsed, RMSE 0.677).

## Getting started

Requirements: Python 3.12 or later.

```bash
git clone https://github.com/johnha912/amazon-clothing-material.git
cd amazon-clothing-material
pip install -r requirements.txt

# Pilot data: the first 10,000 clothing metadata records
# (the full category file is several GB)
mkdir -p data
curl -sL https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/meta_categories/meta_Clothing_Shoes_and_Jewelry.jsonl | head -n 10000 > data/meta_head.jsonl

python explore_sample.py data/meta_head.jsonl   # label-coverage statistics
python analysis/eda.py                          # EDA charts -> results/eda/
python analysis/material_tfidf.py               # Task A baselines -> results/material_tfidf.csv
python analysis/rating.py                       # Task B models -> results/rating.csv

# Optional: the ~250k sample (downloads the full 18 GB category file;
# building the sample then took about 3 minutes on an Intel Core 7 150U laptop)
curl -L -C - -o data/meta_Clothing_Shoes_and_Jewelry.jsonl https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/meta_categories/meta_Clothing_Shoes_and_Jewelry.jsonl
python -m src.make_sample data/meta_Clothing_Shoes_and_Jewelry.jsonl --n 250000   # -> data/sample.parquet
```

The analysis scripts share one loader ([`analysis/load_data.py`](analysis/load_data.py)). When `data/sample.parquet` exists (built by `src/make_sample.py`) they use it automatically; until then they fall back to the 10k pilot above, with the same schema, labeling rule, and split.

## Repository layout

```
.
├── AGENTS.md                # working rules: ownership, labels, leakage, evaluation, agent roles
├── ai-prompts.md            # log of AI-assistant prompts (course requirement, see below)
├── explore_sample.py        # label-coverage statistics; home of the shared labeling rule
├── analysis/
│   ├── load_data.py         # shared loader (sample.parquet preferred, pilot fallback)
│   ├── eda.py               # exploratory analysis + Plotly charts
│   ├── material_tfidf.py    # Task A: TF-IDF + logistic regression / linear SVM
│   ├── material_bert.py     # Task A: DistilBERT fine-tune (Colab GPU)
│   └── rating.py            # Task B: mean baseline, random forest, XGBoost
├── docs/
│   ├── code-style.md
│   └── model-notes/         # one note per model family: method, rationale, error analysis
├── app/                     # Streamlit demo
├── report/                  # report sections, meeting notes, pipeline figure
├── results/                 # metric CSVs, EDA charts, confusion matrix, error examples
├── src/                     # sample builder, label module and tests, shared evaluation
└── data/, models/           # local only — never committed
```

The DistilBERT weights and Colab notebook are attached to the [`yanglin-distilbert-2026-11-02` release](https://github.com/johnha912/amazon-clothing-material/releases/tag/yanglin-distilbert-2026-11-02), not committed.

## Status and roadmap

- ✅ Proposal submitted; pilot EDA and both baseline families run on 10,000 listings
- ✅ Sample builder: ~250k-listing `data/sample.parquet` (249,762 rows, 36.4% labeled)
- ✅ DistilBERT pilot fine-tune for Task A; first Streamlit demo in `app/`
- ✅ Week 1 closed ([Issue #1](https://github.com/johnha912/amazon-clothing-material/issues/1)): same test rows confirmed for all three Task A models, shoe scope decided, Related Work and Related Implementations drafted
- 🚧 Week 2 ([Issue #2](https://github.com/johnha912/amazon-clothing-material/issues/2)): rerun all models on the 250k sample through `src/evaluate.py`, with 95% confidence intervals
- 📝 Report (PDF, max 8 pages) and a 5 ± 1 minute recorded presentation with a live demo, both due **December 6, 2026, 11:55 PM EST**

## Team

| Name | Program | GitHub | Main responsibilities |
|---|---|---|---|
| Nguyen Ha | M.S. in Computer Science | [@johnha912](https://github.com/johnha912) | TF-IDF and rating models, EDA, report integration |
| Yanglin Hu | M.S. in Data Science | [@Elliothu-oss](https://github.com/Elliothu-oss) | DistilBERT modeling (lead) |
| Rui Song | M.S. in Computer Science | [@ruisnow-e](https://github.com/ruisnow-e) | Data pipeline (`src/`), demo app, related work |

## Related work and implementations

[`report/sections/related_work.md`](report/sections/related_work.md) places the project among six published papers: the Amazon Reviews 2023 dataset paper, earlier Amazon rating-prediction work (HFT), product attribute extraction (OpenTag, MAVE), DistilBERT, and a comparison of TF-IDF linear SVMs with fine-tuned transformers.

We also compare our approach with three public Kaggle notebooks on the same dataset family (star-rating prediction, metadata-based recommendation, and EDA) in [`report/sections/related_implementations.md`](report/sections/related_implementations.md), including what each one does and where we try to do better.

Five more papers that support the Method and Analysis sections (TF-IDF vs. DistilBERT, NB vs. SVM baselines, class imbalance, the Rakuten product-title challenge, and fastText) are listed with verified citations in [`docs/method-analysis-sources.md`](docs/method-analysis-sources.md).

## A note on AI assistance

This project is developed with AI assistants under the rules in [AGENTS.md](AGENTS.md) — file ownership per member, cross-tool audits for feature work, and no metric entering the report unless it was produced on the untouched test split. Every assistant prompt is logged in [`ai-prompts.md`](ai-prompts.md), which feeds the "AI Prompts Used" section of the final report.

## References

- Hou, Yupeng, et al. "Bridging Language and Items for Retrieval and Recommendation." *arXiv*, 2024. [arxiv.org/abs/2403.03952](https://arxiv.org/abs/2403.03952) — the Amazon Reviews 2023 dataset paper.
- Chen, Tianqi, and Carlos Guestrin. "XGBoost: A Scalable Tree Boosting System." *Proceedings of the 22nd ACM SIGKDD*, ACM, 2016, pp. 785–94.
- Sanh, Victor, et al. "DistilBERT, a Distilled Version of BERT: Smaller, Faster, Cheaper and Lighter." *arXiv*, 2019. [arxiv.org/abs/1910.01108](https://arxiv.org/abs/1910.01108)

---

Dataset courtesy of the McAuley Lab (UC San Diego). This repository is a course project; code and results are shared for learning and review.

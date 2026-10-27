# What Is It Made Of? Predicting Clothing Material and Buyer Ratings from Amazon Product Listings

Final project for CS6140 Machine Learning (Fall 2026, Northeastern University, Professor Mohammad Toutiaee).

## Authors

| Name | Program | GitHub |
|---|---|---|
| Nguyen Ha | M.S. in Computer Science | [@johnha912](https://github.com/johnha912) |
| Yanglin Hu | M.S. in Data Science | [@Elliothu-oss](https://github.com/Elliothu-oss) |
| Rui Song | M.S. in Computer Science | [@ruisnow-e](https://github.com/ruisnow-e) |

## Problem

Clothing material drives comfort, durability, and satisfaction, yet Amazon listings usually bury it in free-text bullets rather than a searchable field. We ask two questions:

1. Can a model infer a product's dominant material from its listing text?
2. Does material, together with price and department, help predict the product's average rating?

## Data

The Clothing, Shoes and Jewelry category of [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) (Hou et al.), with metadata for about 7.2 million products. Each product has a title, feature bullets, description, price, department, and average rating.

Labels come from parsing composition strings such as "90% Polyester, 10% Spandex": the highest-percentage fiber is the dominant material, and rare materials are grouped into about ten classes.

## Methods

| Task | Models | Baseline | Metrics |
|---|---|---|---|
| Material classification | TF-IDF + logistic regression, TF-IDF + linear SVM, fine-tuned DistilBERT | — | Macro-F1 |
| Rating prediction | Random forest, XGBoost | Mean rating | RMSE, MAE |

- The composition strings used to create the labels are removed from the model input, so the task is not a simple lookup.
- Hyperparameters are tuned with five-fold cross-validation on a fixed-seed sample of the category.
- Known challenges: noisy shoe labels (sole vs. upper material), missing prices, and the size of the full category.

## Preliminary results (first 10,000 metadata records)

| Finding | Value |
|---|---|
| Products with material in a structured details field | 3.6% |
| Products with a percentage composition in their bullets | 46.0% |
| Products that name a known material | 79.0% |
| Products with a missing price | 45.7% |

Top dominant materials: polyester (24.3%), cotton (22.0%), leather (14.3%), and rubber (13.5%, mostly shoe soles).

## Setup

Requirements: Python 3.12 or later. Install the project libraries with `pip install -r requirements.txt`.

```bash
git clone https://github.com/johnha912/amazon-clothing-material.git
cd amazon-clothing-material

# Download the first 10,000 clothing metadata records (the full file is several GB)
mkdir -p data
curl -sL https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/meta_categories/meta_Clothing_Shoes_and_Jewelry.jsonl | head -n 10000 > data/meta_head.jsonl

# Reproduce the preliminary results
python explore_sample.py data/meta_head.jsonl
```

The `data/` folder is ignored by Git, so raw data never gets committed.

## Repository layout

```
.
├── AGENTS.md                # rules for AI assistants: ownership, labels, leakage, eval, agent roles
├── analysis/                # load_data.py (shared loader), eda.py, material_tfidf.py, rating.py
├── docs/                    # code-style.md, model-notes/ (how and why each model works)
├── explore_sample.py        # material coverage statistics and the shared labeling rule
├── report/                  # report sections, meeting notes, figures
├── results/                 # metrics CSVs, EDA and error-analysis charts
└── data/                    # local data, not committed
```

## References

- Chen, Tianqi, and Carlos Guestrin. "XGBoost: A Scalable Tree Boosting System." *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, ACM, 2016, pp. 785–94.
- Hou, Yupeng, et al. "Bridging Language and Items for Retrieval and Recommendation." *arXiv*, 2024, [arxiv.org/abs/2403.03952](https://arxiv.org/abs/2403.03952).
- Sanh, Victor, et al. "DistilBERT, a Distilled Version of BERT: Smaller, Faster, Cheaper and Lighter." *arXiv*, 2019, [arxiv.org/abs/1910.01108](https://arxiv.org/abs/1910.01108).

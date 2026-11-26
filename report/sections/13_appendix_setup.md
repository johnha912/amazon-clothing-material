# Appendix A: Setup and reproduction

Everything below runs from a clean clone of the public repository, https://github.com/johnha912/amazon-clothing-material. Python 3.12 or later is required.

Install the dependencies:

```bash
git clone https://github.com/johnha912/amazon-clothing-material.git
cd amazon-clothing-material
pip install -r requirements.txt
```

Download the Clothing, Shoes and Jewelry files of Amazon Reviews 2023 (McAuley Lab, https://amazon-reviews-2023.github.io/) into `data/`: the product metadata file, saved uncompressed as `data/meta_Clothing_Shoes_and_Jewelry.jsonl` (about 18 GB), and the review file, saved as `data/reviews_Clothing_Shoes_and_Jewelry.jsonl.gz` (about 7.1 GB, read only by Task C). The `data/` folder is not committed, so a fresh clone has to fetch both files before anything runs. Build the sample every result in this report uses (249,762 rows, seed 42; the script prints its row count and the commit it was built from):

```bash
mkdir -p data
curl -L -C - -o data/meta_Clothing_Shoes_and_Jewelry.jsonl https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/meta_categories/meta_Clothing_Shoes_and_Jewelry.jsonl
curl -L -C - -o data/reviews_Clothing_Shoes_and_Jewelry.jsonl.gz https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Clothing_Shoes_and_Jewelry.jsonl.gz
python -m src.make_sample data/meta_Clothing_Shoes_and_Jewelry.jsonl --n 250000
```

Run the analyses from the repository root, so that the shared evaluation in `src/evaluate.py` is importable. Each script writes its metrics, the command, the seed, and the data version into `results/`:

```bash
python analysis/eda.py               # exploratory charts into results/eda/
python analysis/material_tfidf.py    # Task A: TF-IDF + logistic regression / SVM + calibrated demo SVM (CPU, ~6 to 26 min)
python analysis/rating.py            # Task B: mean baseline, random forest, XGBoost (CPU, ~21 min)
python analysis/review_sentiment.py  # Task C: lexicon sentiment (first run streams the
                                     # reviews file, ~13 min by console
                                     # timing, and caches
                                     # data/reviews_sample.parquet; a rerun takes ~8 min)
```

Confidence intervals are computed from the saved prediction files, without retraining, with the shared bootstrap helper in `src/evaluate.py` (1,000 resamples, seed 42):

```bash
python analysis/ci_from_predictions.py
```

Two runs need a GPU and are not part of the CPU reproduction above. The DistilBERT fine-tune (`python analysis/material_bert.py --overwrite`, seed 42, the same 249,762-row sample) runs on a GPU or in Colab, never on a CPU laptop; we ran it on one NVIDIA GeForce RTX 5090 Laptop GPU (24 GB) with torch 2.14.1 for CUDA 13.0, where tokenizing, training, and predicting both settings took 1,719.8 s (about 28.7 min, timed after the data were loaded). The confidence intervals and the four-model comparison come from `python analysis/material_bert.py --compare-only`, and the run note is `docs/model-notes/material_bert.md`. The Task C version-2 check (DistilBERT fine-tuned on SST-2 over the 209,830 material sentences) (`analysis/taskc_sst2_agreement.py`) runs as a Kaggle notebook on a GPU, reading `data/reviews_sample.parquet` uploaded as a private Kaggle dataset; it reported 85.3% agreement with the lexicon (Cohen's kappa 0.617) on the 146,981 polar sentences, and its agreement tables are saved as `results/taskc_sst2_agreement.csv` and `results/taskc_sst2_by_rating.csv`.

Run the demo from the repository root:

```bash
streamlit run app/app.py
```

The demo loads the saved Task A pipeline from `models/` (created by `analysis/material_tfidf.py`; the folder is not committed). On a fresh clone, work through the steps above in order before starting the demo: download the data, build the sample with `make_sample`, then run `analysis/material_tfidf.py` to train and save the pipeline the demo loads.

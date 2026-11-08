# Meeting notes

Kept by Nguyen Ha. One entry per meeting, newest first. Decisions and owners only — discussion detail lives in the issue tracker.

## Sun Nov 8, 2026 — integration check (Issue #1)

- Attendees: none (async review by Nguyen, not a meeting)
- Test rows for the three Task A models are consistent (Nguyen): identical per-class test counts, and all 292 TF-IDF logistic-regression errors are in the DistilBERT test predictions with the same labels. Exact check pending: the TF-IDF run must save its test IDs so they can be hashed against `test_ids_sha256`.
- DistilBERT macro-F1 recomputed from `results/material_bert_predictions.csv` with `src.evaluate.material_scores`: 0.3563, same as the run. Still needs the independent Auditor verdict before it is final.
- Sample delivered (Rui, Nov 4, Issue #1): `python -m src.make_sample data/meta_Clothing_Shoes_and_Jewelry.jsonl --n 250000` gave `data/sample.parquet` 249,762 rows, `src/make_sample.py` commit 2219c1e, seed 42; 36.4% labeled and 79.8% missing price. The pilot's 46% / 45.7% came from the first 10k lines and are not representative.
- Results merged (pilot, `data/meta_head.jsonl` 10,000 rows, seed 42, 903 Task A test rows): Task A macro-F1 (TF-IDF LR / SVM / DistilBERT) = 0.5572 / 0.5239 / 0.3563 (`results/material_tfidf.csv`, `results/material_bert.csv`, merged in `results/material_comparison.csv`); Task B RMSE–MAE in `results/rating.csv` = mean baseline 0.5863–0.3834, random forest 0.5854–0.3851, XGBoost 0.5780–0.3736.
- DistilBERT handoff (Yanglin, Nov 5): code, model note, and results are in the repo; weights and the Colab notebook are in the `yanglin-distilbert-2026-11-02` release.
- Next week's owners: rerun TF-IDF and rating models on the 250k sample (Nguyen); rerun DistilBERT on the 250k sample with class weights, through `src/evaluate.py` (Yanglin); papers for `report/sections/related_work.md` (Rui); shoe scope decision (Yanglin + Nguyen) before any rerun.

## Sun Nov 1, 2026 — week review

- Attendees:
- Results merged: see the Nov 8 entry (DistilBERT results arrived Nov 5).
- Canvas Report and Presentation pages re-checked (Nguyen): changes = ___
- Next week's owners: see the Nov 8 entry.

## Wed Oct 28, 2026 — data check

- Attendees:
- Sample status: `data/sample.parquet` rows = ___ (target 50k); commit of `src/make_sample.py` = ___
- Decisions needed:
  - Label rule: percentage strings only (AGENTS.md) vs the first-mentioned-fiber fallback in `explore_sample.dominant()` — settle before labels are frozen.
  - Task B: confirm `rating_number` may weight/filter rows but is never a feature.
  - Shoes in scope? (Nguyen + Yanglin)
- Blockers:

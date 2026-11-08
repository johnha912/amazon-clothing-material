# Meeting notes

Kept by Nguyen Ha. One entry per meeting, newest first. Decisions and owners only — discussion detail lives in the issue tracker.

## Wed Oct 28, 2026 — data check

- Attendees:
- Sample status: `data/sample.parquet` rows = ___ (target 50k); commit of `src/make_sample.py` = ___
- Decisions needed:
  - Label rule: percentage strings only (AGENTS.md) vs the first-mentioned-fiber fallback in `explore_sample.dominant()` — settle before labels are frozen.
  - Task B: confirm `rating_number` may weight/filter rows but is never a feature.
  - Shoes in scope? (Nguyen + Yanglin)
- Blockers:

## Sun Nov 1, 2026 — week review

- Attendees:
- Results merged (pilot, `data/meta_head.jsonl` 10,000 rows, seed 42, 903 Task A test rows): Task A macro-F1 (TF-IDF LR / SVM / DistilBERT) = 0.5572 / 0.5239 / 0.3563 (`results/material_tfidf.csv`, `results/material_bert.csv`, merged in `results/material_comparison.csv`); Task B RMSE–MAE in `results/rating.csv` = mean baseline 0.5863–0.3834, random forest 0.5854–0.3851, XGBoost 0.5780–0.3736.
- Same test rows confirmed for all three Task A models (Nguyen, Nov 8): identical per-class test counts, and all 292 TF-IDF logistic-regression errors are in the DistilBERT test manifest with the same labels. An exact `parent_asin` hash check against `test_ids_sha256` still needs a rerun with the data present.
- Sample delivered (Rui, Nov 4): `data/sample.parquet` 249,762 rows, `src/make_sample.py` commit 2219c1e, seed 42; 36.4% labeled and 79.8% missing price. The pilot's 46% / 45.7% came from the first 10k lines and are not representative.
- DistilBERT handoff (Yanglin, Nov 5): code, model note, and results are in the repo; weights and the Colab notebook are in the `yanglin-distilbert-2026-11-02` release.
- Canvas Report and Presentation pages re-checked (Nguyen): changes = ___
- Next week's owners: rerun TF-IDF and rating models on the 250k sample (Nguyen); rerun DistilBERT on the 250k sample with class weights, through `src/evaluate.py` (Yanglin); papers for `report/sections/related_work.md` (Rui); shoe scope decision (Yanglin + Nguyen) before any rerun.

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
- Results merged: Task A macro-F1 (TF-IDF LR / SVM / DistilBERT) = ___ / ___ / ___; Task B RMSE–MAE in `results/rating.csv` = ___
- Canvas Report and Presentation pages re-checked (Nguyen): changes = ___
- Next week's owners:

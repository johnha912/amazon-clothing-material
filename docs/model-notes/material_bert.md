# DistilBERT material classification

Owner: Yanglin Hu. This is a completed pilot run, not a final-report acceptance.

## Method and rationale

The model fine-tunes pretrained DistilBERT on the shared composition-free listing text. Its contextual token representations can capture phrases and word relationships beyond TF-IDF word counts. A classification head transforms the first-token representation into logits z, with p = softmax(z). The unweighted training loss is cross-entropy, L = -mean(log(p_correct)). Macro-F1 is the mean of the per-class F1 scores, giving rare and common materials equal importance.

Labels and cleaned text come from the shared loader, without reimplementing the composition parser. Only labeled classes with the same minimum support as the TF-IDF baseline are retained. Neither average_rating nor rating_number is supplied to the tokenizer. Percentage composition segments are removed because they generated the labels.

## Experimental setup and provenance

The data version is data/meta_head.jsonl pilot (10,000 rows). The seed is 42; the frozen population contains 3641 training rows, 903 test rows and 9 classes. Epoch selection used 2912 fit rows and 729 validation rows drawn only from the training split. Validation selected 2 epoch(s). A fresh pretrained model was refit on every training row for that many epochs before one final test prediction.

The exact command was `python analysis/material_bert.py`. The GPU was Tesla T4; recorded elapsed time was 110.43 seconds. Pretrained revision: `12040accade4e8a0f71eabdb258fecc2e7e948be`. Task fingerprint: `2fbda68db0a8be27fda0a7fa31a7c48273f93cdc910613e75c156167d46b9d73`. Full hyperparameters, package versions, source hash, test-ID hash and runtime provenance are in `results/material_bert_run.json`; the exact split manifest and validation history are in `models/material_bert/`.

## Pilot results

| model | macro_f1 | accuracy |
| --- | --- | --- |
| tfidf_logreg | 0.5572 | 0.6766 |
| tfidf_svm | 0.5239 | 0.7364 |
| distilbert | 0.3563 | 0.6899 |

Comparison verification is `legacy_metadata_match_only; Nguyen must confirm frozen rows`. The legacy TF-IDF CSV does not prove identical listing identities/text/labels through a fingerprint. Nguyen must confirm that all three runs used the same frozen population before the table is treated as final. The evaluator for this run was `sklearn.metrics (shared src/evaluate.py not yet available)`. Required shared-evaluator verification and independent audit remain pending before report acceptance.

## Error analysis

The per-class table shows which materials contribute to the macro-F1 result. Accuracy gives more influence to frequent labels, whereas macro-F1 gives every class equal weight; therefore accuracy alone is insufficient for this imbalanced task.

| material | precision | recall | f1-score | support |
| --- | --- | --- | --- | --- |
| acrylic | 0.375 | 0.1667 | 0.2308 | 18.0 |
| cotton | 0.7422 | 0.802 | 0.771 | 298.0 |
| leather | 0.9322 | 0.9565 | 0.9442 | 115.0 |
| nylon | 0.56 | 0.4746 | 0.5138 | 59.0 |
| polyester | 0.6061 | 0.7571 | 0.6732 | 317.0 |
| polyurethane | 0.0 | 0.0 | 0.0 | 3.0 |
| rayon | 0.3333 | 0.0417 | 0.0741 | 72.0 |
| spandex | 0.0 | 0.0 | 0.0 | 15.0 |
| wool | 0.0 | 0.0 | 0.0 | 6.0 |

The most frequent true-to-predicted error pairs in this saved test prediction are:

| material | predicted | count |
| --- | --- | --- |
| rayon | polyester | 59 |
| polyester | cotton | 56 |
| cotton | polyester | 53 |
| nylon | polyester | 24 |
| polyester | nylon | 14 |

Five actual error examples, or all errors if fewer than five exist, are:

| parent_asin | material | predicted | text |
| --- | --- | --- | --- |
| B07H9N8HLJ | rayon | polyester | Air Curvey 2&4 PCS Basic Camisole for Women Cami Tanks Adjustable Spaghetti Strap Tank Tops. , . Imported. Pull On closure. Machine Wash. . Feature: Our womens tank tops are adjustable shoulder straps for a customized fit. Occasion: This Camisoles & Tanks is well with shirts, jeans,leggings; Fashion |
| B07X8P882H | cotton | nylon | Rosme Womens Balconette Bra with Padded Straps. , , . Hook and Eye closure. Underwired. Lightly Padded Cups. Adjustable Straps. Made in Latvia (EU) |
| B09BQ26MTX | cotton | polyester | Gioberti Men's Long Sleeve Solid Dress Shirt. , . Imported. Button closure. Machine Wash. Regular Fit, Solid Color. Long Sleeve, Double-button cuffs. Chest Pocket, Round Bottom. Classic Dress Shirt For All Occasions: Business, Work, Interviews, Wedding, Party and Casual LookS.. ***PLEASE refer to si |
| B09TRD59DG | rayon | polyester | Ferlema Womens Casual Striped Short Sleeve Summer Dress Round Neck Empire Waist Pocket Sundresses with Belt. , . Tie closure. Machine Wash. FEATURES：Striped print,short sleeve,round/boat neck,knee length,elastic waist,wrap dress with pockets,t shirt dress,midi dress,womens elegant sundress,2021 summ |
| B08627YMFY | cotton | polyester | MorwenVeo Men's Casual Hooded T-Shirts - Fashion Short Sleeve Solid Color Pullover Top Summer Blouse. . Pull On closure. Machine Wash. . ▲Machine Wash. ▲Design: Pullover hoodie, soft fabric, strong and durable. ▲Occasion：Fitness, cycling, Ideal for daily or outdoor or street wear. ▲Note: We offer de |

The full confusion matrix is `results/material_bert_confusion.csv` and its interactive view is `results/material_bert_confusion.html`. The complete saved test predictions and short error excerpts are available in the corresponding prediction/error CSVs. Review these examples before assigning a specific cause to an individual mistake.

Possible contributors include imbalanced class support, listings mentioning multiple fibers, and truncation of late evidence. This run uses unweighted cross-entropy, while the TF-IDF classifiers use balanced class weights. These are explanations to investigate using training/validation data, not causes established by this single experiment. This pilot does not establish that a contextual model will outperform the simpler baselines. Preserve the test result; future configuration selection must remain on training/validation data.

## Scope and transfer

The shared scope still includes shoes; Yanglin and Nguyen have not jointly resolved the upper-versus-sole label ambiguity. Inspect training examples to decide that scope with the pipeline owner, freeze any changed sample, and rerun every model consistently. Do not change scope in response to test errors.

The reusable parts are the shared label rule, composition-leakage guard, validation-only epoch selection, frozen test evaluation and saved provenance. Applying this workflow to another category requires checking its material vocabulary, label coverage and component ambiguity again.

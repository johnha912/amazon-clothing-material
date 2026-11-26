# Introduction

Material decides how a garment feels, how long it lasts, and whether the buyer is satisfied, yet Amazon rarely stores it as a searchable field. It sits in free-text bullets as a composition string such as "90% Polyester, 10% Spandex", so a buyer with a wool allergy has no reliable filter.

We study three linked tasks on the Clothing, Shoes and Jewelry category of Amazon Reviews 2023 (Hou et al.), about 7.2 million products. Task A predicts a product's dominant material, the fiber with the highest percentage in its composition string, from listing text with the composition strings removed, so the task is inference rather than lookup (macro-F1). Task B predicts the average star rating of products with at least 5 ratings from price, department, and material (RMSE and MAE). Task C, added after instructor feedback, starts from the observation that a star average is noisy (a 1-star review can be about late shipping rather than the fabric): it builds a target from what buyers write about the material and asks whether listing fields can predict it (RMSE and MAE).

The dataset poses three challenges. Only 36.4% of listings in our ~250,000-product sample (90,979 of 249,762) carry a composition string, so the labeled ones may not be representative. The labels are imbalanced: cotton and polyester make up 64.3% of labeled listings, while metal has 152 and rubber 300; and a shoe's composition string often describes the sole.

## Proposed method in brief

For Task A we classify TF-IDF vectors of the cleaned listing text with a class-weighted logistic regression or linear SVM and compare both with a fine-tuned DistilBERT. For Task B we compare a weighted mean baseline with a random forest and XGBoost on listing fields; for Task C, whose target averages lexicon scores of material-related review sentences per product, we compare the baseline with XGBoost.

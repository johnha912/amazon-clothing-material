# Introduction

Material decides how a garment feels, how long it lasts, and whether the buyer is satisfied, yet Amazon rarely stores it as a searchable field. It sits in free-text feature bullets as a composition string such as "90% Polyester, 10% Spandex", so a shopper who wants only cotton shirts has no reliable filter.

We ask two linked questions on the Clothing, Shoes and Jewelry category of Amazon Reviews 2023 (Hou et al.). Task A asks whether the missing field can be recovered from the listing text. Task B asks whether the recovered field is worth anything, that is, whether material helps predict how buyers rate a product.

| | Task A: material | Task B: rating |
|---|---|---|
| Target | Dominant fiber (highest percentage in the composition string) | Average rating, 1 to 5 |
| Input | Title, bullets, and description, with every composition string removed | Listing fields (price, department, text length) plus the material label |
| Models | TF-IDF + logistic regression, TF-IDF + linear SVM, fine-tuned DistilBERT | Mean baseline, random forest, XGBoost |
| Metric | Macro-F1 | RMSE and MAE |

The dataset poses three challenges. Only 46% of listings carry a composition string, so most products have no label. The labels are imbalanced, with polyester and cotton making up 65% of labeled listings. Footwear composition strings often describe the sole rather than the upper, so the same label can mean different things.

All experiments share one train/test split and one seed (42), and every reported number records the command, seed, and data version that produced it.

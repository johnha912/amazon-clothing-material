# Introduction

Material decides how a garment feels, how long it lasts, and whether the buyer is satisfied, yet Amazon rarely stores it as a searchable field. It sits in free-text feature bullets as a composition string such as "90% Polyester, 10% Spandex", so a shopper who wants only cotton shirts, or a buyer with a wool allergy, has no reliable filter. Recovering this field automatically would make a large catalog searchable by a property shoppers care about, and it would tell sellers and marketplaces whether that property is worth stating clearly.

We study two linked questions on the Clothing, Shoes and Jewelry category of Amazon Reviews 2023 (Hou et al.), about 7.2 million products. Task A asks whether a product's dominant material can be recovered from its listing text. Its target is the fiber with the highest percentage in the composition string, and the composition strings themselves are removed from the input so that the task is genuine inference rather than lookup. Task B asks whether material is worth anything once recovered, that is, whether it helps predict a product's average star rating (1 to 5) together with price and department.

| | Task A: material | Task B: rating |
|---|---|---|
| Target | Dominant fiber in the composition string | Average rating, 1 to 5 |
| Input | Title, bullets, and description, composition clauses removed | Listing fields plus the material label |
| Models | TF-IDF + logistic regression, TF-IDF + linear SVM, DistilBERT | Mean baseline, random forest, XGBoost |
| Metric | Macro-F1 | RMSE and MAE |

The dataset poses three challenges. First, only 46% of listings carry a composition string, so most products have no label and the labeled ones may not be representative. Second, the labels are heavily imbalanced: polyester and cotton make up 65% of labeled listings, while several fibers have only a handful of examples. Third, the label can mean different things across products, because a shoe's composition string often describes the sole rather than the upper.

## Proposed method in brief

For Task A we represent each cleaned listing as a TF-IDF vector of words and word pairs and classify it with class-weighted logistic regression. On the pilot it reached a macro-F1 of 0.557 against 0.524 for a linear SVM on the same features, but that lead rests mostly on nine test listings of two tiny classes and the SVM is stronger on most materials, so the two are effectively tied; we keep logistic regression for the demo because it outputs class probabilities that let the app show its confidence, and we will choose between them, and against fine-tuned DistilBERT, on the 50,000-product sample. For Task B we use XGBoost on listing fields with log-count sample weights, the only model that beat a weighted mean baseline on both error metrics, although by a small margin that is itself our main finding about ratings.

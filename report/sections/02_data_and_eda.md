# Data and exploratory analysis

![Project pipeline: from raw metadata to the sample, the three model tracks, and the merged outputs.](../assets/project-pipeline.png)

We use the product metadata of the Clothing, Shoes and Jewelry category of Amazon Reviews 2023 (Hou et al.), about 7.2 million products. Each record has a title, feature bullets, a description, a price, a department, the average rating, and the number of ratings. The pilot results below use the first 10,000 records; the same scripts rerun unchanged on the 50,000- and 250,000-product samples drawn by hashing the product identifier.

## Labels

A listing's label is the known fiber with the highest percentage in its composition string, taken from the structured details or the feature bullets:

$$m^* = \arg\max_{m \in \mathcal{M}} p_m,$$

where $p_m$ is the stated percentage of fiber $m$ and ties go to the fiber listed first. A listing with no composition string gets no label and is left out of Task A; we never guess one. In the pilot, 4,602 of 10,000 listings (46.0%) receive a label.

| Material | Labeled listings | Share |
|---|---|---|
| Polyester | 1,583 | 34.4% |
| Cotton | 1,416 | 30.8% |
| Leather | 618 | 13.4% |
| Nylon | 341 | 7.4% |
| Rayon | 336 | 7.3% |
| Other (spandex, acrylic, wool, ...) | 308 | 6.7% |

This long tail is why Task A reports macro-F1, which gives every class the same weight regardless of its size:

$$\text{F1}_{\text{macro}} = \frac{1}{K}\sum_{k=1}^{K} \frac{2\,P_k R_k}{P_k + R_k},$$

where $P_k$ and $R_k$ are the precision and recall of class $k$. Accuracy alone would reward a model that only learns polyester and cotton.

## What the exploration shows

The exploratory analysis (`analysis/eda.py`, charts in `results/eda/`) found three patterns, each of which changed a modeling decision.

| Finding | Evidence (pilot) | Decision |
|---|---|---|
| Prices are often missing, and not at random | 45.7% of listings have no price; among departments with at least 100 listings the share ranges from 34% (Sport Specific Clothing) to 77% (Westlake). Unpriced products average 4.10 stars against 4.27 for priced ones. | Keep every row and add a has-price indicator, since missingness itself carries signal |
| Ratings are compressed near the top | Mean 4.20, standard deviation 0.57; 92% of products sit between 3.5 and 5. Mean rating by material spans only 4.16 (spandex) to 4.35 (acrylic). | Expect a weak Task B signal and judge every model against a mean baseline |
| Price is heavily right-skewed | Median $26, mean $39, skewness 18.0 | Use $\log(1 + \text{price})$, which cuts the skewness to 0.76 |

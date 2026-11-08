# Sources for the Method and Analysis sections

Five papers Rui Song found on 2026-11-08 that fit the Method and Analysis sections better than Related Work. Each entry below was opened at its URL on 2026-11-08; the "What it says" lines come from the paper's abstract or first page, not from memory. None of them is cited in the report yet.

| # | Paper | What it says | Where it can help us |
|---|---|---|---|
| 1 | Galke and Scherp (2022) | A wide MLP on bag-of-words input outperforms the graph-based TextGCN and HeteGCN; fine-tuned BERT and DistilBERT outperform all of them. | Analysis: a counterpoint to Wahba et al. (2022), since here DistilBERT wins. Our pilot result (TF-IDF LR macro-F1 0.557 vs DistilBERT 0.356) should be read against both. |
| 2 | Wang and Manning (2012) | NB vs SVM rankings depend on the variant, features and dataset: bigrams give consistent gains on sentiment tasks, NB beats SVM on short snippets while SVM wins on longer documents, and NBSVM performs well across tasks. | Method: supports unigram+bigram TF-IDF. Analysis: text length can flip which linear model wins. Note: the paper compares NB and SVM, not logistic regression. |
| 3 | Henning et al. (2023) | NLP models "tend to perform poorly on less frequent classes"; the survey groups fixes into sampling, data augmentation, loss function choice, staged learning, and model design. | Method: background for balanced class weights. Analysis: a loss-function fix is one way to test whether DistilBERT's unweighted loss explains its low recall on rare fibers. The abstract does not discuss macro-F1; check the full text before citing it for that. |
| 4 | Lin, Das and Datta (2018) | Rakuten released 1M product titles (0.8M train, 0.2M test) for taxonomy classification; 26 teams submitted 28 systems, scored by weighted precision, recall and F1, with a top weighted F1 of 0.8513. | Analysis: product-title classification at scale, a task similar to ours ("similar datasets" paragraph). Their metric is weighted F1, ours macro-F1, so the numbers are not comparable. |
| 5 | Joulin et al. (2017) | fastText "is often on par with deep learning classifiers in terms of accuracy" and "many orders of magnitude faster" to train and evaluate. | Method: supports a fast linear baseline over a transformer on a catalog of millions of products. |

## References

Galke, Lukas, and Ansgar Scherp. "Bag-of-Words vs. Graph vs. Sequence in Text Classification: Questioning the Necessity of Text-Graphs and the Surprising Strength of a Wide MLP." *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, ACL, 2022, pp. 4038–4051. https://doi.org/10.18653/v1/2022.acl-long.279 (https://aclanthology.org/2022.acl-long.279/)

Henning, Sophie, William Beluch, Alexander Fraser, and Annemarie Friedrich. "A Survey of Methods for Addressing Class Imbalance in Deep-Learning Based Natural Language Processing." *Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics*, ACL, 2023, pp. 523–540. https://doi.org/10.18653/v1/2023.eacl-main.38 (https://aclanthology.org/2023.eacl-main.38/)

Joulin, Armand, Edouard Grave, Piotr Bojanowski, and Tomas Mikolov. "Bag of Tricks for Efficient Text Classification." *Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics: Volume 2, Short Papers*, ACL, 2017, pp. 427–431. https://aclanthology.org/E17-2068/

Lin, Yiu-Chang, Pradipto Das, and Ankur Datta. "Overview of the SIGIR 2018 eCom Rakuten Data Challenge." *The SIGIR 2018 Workshop on eCommerce (eCom 2018)*, CEUR Workshop Proceedings, vol. 2319, 2018. https://ceur-ws.org/Vol-2319/ecom18DC_paper_13.pdf

Wang, Sida, and Christopher D. Manning. "Baselines and Bigrams: Simple, Good Sentiment and Topic Classification." *Proceedings of the 50th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)*, ACL, 2012, pp. 90–94. https://aclanthology.org/P12-2018/

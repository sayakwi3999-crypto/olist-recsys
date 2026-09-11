# Resume Entry (English): E-commerce Recommendation Optimization

**Project**: E-commerce Recommendation Algorithm Optimization (Olist, Brazilian e-commerce)

**Data**: Public Olist dataset — ~100K orders, 96K customers, 33K products (8 relational tables)

**What I did**:

- Cleaned eight relational tables in MySQL before modeling: removed duplicates, handled missing and anomalous records, and logged each cleaning step for traceability. Also fixed a user-mapping issue that had dropped repeat-purchase history after account-level deduplication.
- Engineered features from a people–product–context framework (RFM-style user profiles, product category/price/quality attributes, order time, region, logistics) and built a sparse 95K-user × 5K-item interaction matrix. Set up two leakage-free evaluation schemes: a time-based split and a last-order holdout for repeat buyers.
- Implemented and compared eight recommenders — popularity and random baselines, ItemCF, UserCF, Funk-SVD, ALS, content-based, and a weighted hybrid — using Precision@K, Recall@K, NDCG@K, coverage, and diversity. For repeat buyers, ItemCF/UserCF roughly doubled Recall@10 versus the popularity baseline (0.129 vs. 0.063).
- Ran a grid search over 34 hyperparameter configurations. The tuned hybrid model raised NDCG@10 from 0.090 to 0.096 (+6.2%) with no regression on cold-start users, and the results were turned into segment-level recommendation strategies (new users, high-value users, regional supply).

**Tools**: MySQL, Python (pandas, NumPy, SciPy, scikit-learn), Matplotlib

---

**One-line version**:

Built an e-commerce recommendation project on the public Olist dataset (~100K orders): MySQL data cleaning, feature engineering, eight model implementations, and offline evaluation; collaborative filtering doubled recall for repeat buyers, and a tuned hybrid reached NDCG@10 of 0.096.


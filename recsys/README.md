# Recommender Optimisation (Olist, Brazilian e-commerce)

## Files

```
recsys/
├── config.py          # database connection, paths, experiment settings
├── export_data.py     # export the cleaned tables from MySQL(olist_clean) to data/raw/
├── features.py        # interactions, user/item features, time split, candidate pool
├── models.py          # 8 recommenders: popular / random / itemcf / usercf / svd / als / content / hybrid
├── evaluate.py        # Top-K metrics, coverage/diversity/novelty, segment evaluation, figures
├── tune.py            # grid search over the main hyperparameters
├── run_all.py         # run the whole pipeline
├── data/              # exported and intermediate data (git-ignored)
└── output/            # evaluation results and figures
```

## Environment

Python 3.9+ with pandas / numpy / scipy / scikit-learn / matplotlib / pymysql:

```bash
pip install pandas numpy scipy scikit-learn matplotlib pymysql
```

**Configure the database password first**: copy `.env.example` to `.env` and fill in `DB_PASSWORD`
(`.env` is git-ignored and never committed), or set the environment variables
`DB_HOST / DB_PORT / DB_USER / DB_PASSWORD / DB_NAME`.

Run everything:

```bash
python run_all.py
```

Or step by step:

```bash
python export_data.py   # MySQL -> data/raw/
python features.py      # features and both splits
python tune.py          # optional: grid search
```

## Experiment design

- **Interactions**: the cleaned `olist_clean` database, as user x item (review score 1-5, quantity, order value).
  User identity is `customer_unique_id`: every historical account of the same person is merged, so
  repeat purchases survive the account-level de-duplication applied during cleaning.
- **Scenario A (time split)**: the last 20% of interactions form the test set, which avoids leaking future
  information. Only ~3% of customers ever repurchase, so almost every test-period user is cold-start; this
  scenario therefore measures the cold-start fallback (popularity / content features).
- **Scenario B (repeat-buyer holdout)**: keep users with >=2 orders and hold out their last order, which
  measures what CF and matrix factorisation can really achieve for users with history.
- **Candidate pool**: the 5,000 most popular training items, shared by every model. Items already bought
  during training are filtered out, so recommendations are always "new" items.
- **Models (8 in total)**:
  - `popular`: most popular training items (baseline)
  - `random`: random recommendations (lower bound)
  - `itemcf`: item-based CF (cosine similarity, Top-50 neighbours, popularity prior)
  - `usercf`: user-based CF (similarity matrix built for active users only)
  - `svd`: Funk-SVD (SGD, rating prediction)
  - `als`: implicit-feedback matrix factorisation (simplified alternating least squares)
  - `content`: content-based recommendation on item features (cold-start friendly)
  - `hybrid`: weighted blend (itemcf 0.55 / content 0.15 / als 0.10 / svd 0.10 / popular 0.10 after tuning)
- **Metrics**: Precision@K, Recall@K, NDCG@K (K=5/10/20), coverage, diversity, novelty, plus segment
  comparisons across new users, high-value users, top-five states and other regions.

## Output files

- `output/time_split_metrics_summary.csv` / `output/repeat_metrics_summary.csv`: per-model metrics for both scenarios
- `output/time_split_segment_metrics.csv` / `output/repeat_segment_metrics.csv`: segment-level metrics
- `output/time_split_report.txt` / `output/repeat_report.txt`: text reports
- `output/*_model_comparison.png`, `output/*_coverage_diversity_novelty.png`: figures
- `output/tuning_results.csv`, `output/tuning_report.txt`, `output/tuning_best_comparison.png`: tuning results

## Notes

- The database password is never hard-coded: it is read from environment variables first, then from
  `recsys/.env`; `export_data.py` fails with a clear message when it is missing.
- Reviews are order-level (all items in one order share a single score). This is a structural limitation of
  the dataset and is discussed in the data-reflection section of the report.

# Code walkthrough (data cleaning -> recommender modelling)

> Companion project: Olist recommender optimisation.
> How to read it: follow the actual files in `recsys/` and the SQL scripts; line numbers refer to the current files.

---

## 0. The whole data flow

```
MySQL raw database olist_analysis
  ├── 01_data_profiling.sql      (read-only: quantify dirty and duplicate data first)
  ├── 02_cleaning_and_dedup.sql  (build the clean database olist_clean + cleaning log)
  ├── Python export_data.py      (export the clean tables to CSV)
  ├── Python features.py         (build user x item interactions, features, train/test splits)
  ├── Python models.py           (train the 8 recommenders)
  └── Python evaluate.py         (offline evaluation + figures + report)
```

---

## 1. 01_data_profiling.sql (read-only checks)

The first two lines:

- `USE olist_analysis;` – switch the session to the raw database.
- `SET NAMES utf8mb4;` – make the client use utf8mb4 so non-ASCII characters are not mangled.

### Section 1: row counts and primary-key uniqueness

```sql
SELECT 'orders' AS tbl, COUNT(*) AS rows_total, COUNT(DISTINCT order_id) AS distinct_pk FROM orders
UNION ALL SELECT 'customers', ...
```

- `COUNT(*)`: how many rows the table has;
- `COUNT(DISTINCT order_id)`: how many remain after de-duplicating the key;
- if the two are equal the key has no duplicates; `UNION ALL` stacks the seven tables into one result.

### Section 2: duplicate detection

The pattern is always "group first, then count the duplicated groups":

```sql
SELECT COUNT(*) AS dup_orders_rows FROM (
  SELECT order_id FROM orders GROUP BY order_id HAVING COUNT(*) > 1
) t;
```

- `GROUP BY order_id HAVING COUNT(*) > 1` finds keys that appear more than once;
- the outer `COUNT(*)` counts how many such groups exist;
- the same pattern covers repeated `customer_unique_id` (one person, several accounts), repeated
  `(order_id, product_id, seller_id)` (several rows for one item), `review_id` reused across orders, and
  duplicate primary keys in payments, products and sellers.

### Section 3: missing values

```sql
SUM(customer_id IS NULL OR customer_id = '') AS null_customer_id
```

- In MySQL, `SUM(boolean)` counts TRUE as 1 and FALSE as 0, so this line is literally "how many rows lack a
  customer_id";
- each table checks its own key fields: customer/time/status for orders, product id/price/freight for items,
  score/text for reviews, category/weight/description for products, type/value for payments.

### Section 4: dirty data

- `4.1` order-status distribution via `GROUP BY order_status`, to spot illegal values by eye;
- `4.2` impossible timestamps: three `OR` conditions covering "approved before purchase", "delivered before
  shipped" and "delivered before approved", none of which can happen in reality;
- `4.3` purchase timestamps outside 2016-09 to 2018-10;
- `4.4` negative prices; `4.5` review scores outside 1-5;
- `4.6` deliveries longer than 30 days (a business concern rather than dirty data, counted separately).

### Section 5: referential integrity (orphans)

```sql
SELECT COUNT(DISTINCT o.order_id) AS orders_without_items
FROM orders o LEFT JOIN order_items oi ON o.order_id = oi.order_id
WHERE oi.order_id IS NULL;
```

- `LEFT JOIN` keeps every row of the left table (orders) and fills NULL when the right table does not match;
- `WHERE oi.order_id IS NULL` isolates exactly the orders with no items;
- the reverse query finds item rows whose order does not exist (orphan items).

### Section 6: interaction health check (preparation for modelling)

- user / item / interaction counts: `COUNT(DISTINCT ...)` plus a three-table join;
- **repeat-purchase rate** (the key figure of this project):
```sql
SELECT COUNT(*) AS total_users,
       SUM(order_cnt >= 2) AS repeat_buyers,
       ROUND(SUM(order_cnt >= 2) * 100.0 / COUNT(*), 2) AS repeat_rate_pct
FROM (
  SELECT c.customer_unique_id, COUNT(DISTINCT o.order_id) AS order_cnt
  FROM orders o JOIN customers c ON o.customer_id = c.customer_id
  GROUP BY c.customer_unique_id
) t;
```
  - the subquery groups by unique customer id and counts orders per person;
  - `SUM(order_cnt >= 2)` counts repeat buyers; dividing by the total gives 3.12%;
- score distribution (58% are fives, so the rating signal is skewed);
- items-per-order distribution (the vast majority of orders contain a single item, a clear long tail).

---

## 2. 02_cleaning_and_dedup.sql

### Opening: create the database and the log table

```sql
CREATE DATABASE IF NOT EXISTS olist_clean DEFAULT CHARACTER SET utf8mb4;
USE olist_clean;
```

- a separate database `olist_clean` is created; the raw `olist_analysis` is never touched;

```sql
CREATE TABLE cleaning_log (
  id INT AUTO_INCREMENT PRIMARY KEY,
  tbl VARCHAR(50) NOT NULL,
  reason VARCHAR(200) NOT NULL,
  rows_affected INT NOT NULL DEFAULT 0,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

- this table records how many rows each cleaning action removed and why, so the process is traceable.

### The same three-step pattern per table

Taking orders as the example:

**Step 1 – count the problem rows into `@variables`**

```sql
SET @dup_orders := 0;              -- initialise the variable
SELECT COUNT(*) INTO @dup_orders FROM (
  SELECT order_id FROM olist_analysis.orders
  GROUP BY order_id HAVING COUNT(*) > 1
) t;                               -- count duplicate orders
SELECT COUNT(*) INTO @bad_time FROM olist_analysis.orders
WHERE order_approved_at < order_purchase_timestamp OR ...;  -- count impossible timestamps
```

- `@name` is a MySQL session variable and `SELECT ... INTO @x` stores a result for later INSERTs.

**Step 2 – write the numbers into the log**

```sql
INSERT INTO cleaning_log(tbl, reason, rows_affected) VALUES
 ('orders', 'duplicate order_id (earliest kept)', @dup_orders),
 ('orders', 'impossible timestamp order (approval/shipping/delivery out of sequence)', @bad_time);
```

**Step 3 – actually build the clean table**

```sql
DROP TABLE IF EXISTS clean_orders;                 -- makes the script re-runnable
CREATE TABLE clean_orders AS                       -- create a table straight from a query
SELECT ... FROM (
  SELECT o.*,
         ROW_NUMBER() OVER (PARTITION BY order_id
                            ORDER BY order_purchase_timestamp ASC, order_id ASC) AS rn
  FROM olist_analysis.orders o
  WHERE customer_id IS NOT NULL
    AND order_purchase_timestamp BETWEEN '2016-09-01' AND '2018-10-31'
    AND <timestamp logic is valid>
) o
WHERE rn = 1;                                      -- keep the first row of each group
ALTER TABLE clean_orders ADD PRIMARY KEY (order_id);
```

- `ROW_NUMBER() OVER (PARTITION BY order_id ...)` is a window function numbering rows inside each key group;
- `WHERE rn = 1` keeps the earliest order, implementing "duplicate order_id, earliest kept";
- `CREATE TABLE ... AS SELECT` materialises the result and `ALTER TABLE ... ADD PRIMARY KEY` restores the key.

### Cleaning rules per table

| Table | De-duplication / cleaning rule |
|---|---|
| clean_orders | duplicate order_id keeps the earliest; orders with no customer, impossible timestamps or out-of-range dates are dropped |
| clean_customers | one unique id with several accounts keeps the most recent account (**note: this is a trap for modelling, see below**) |
| clean_sellers | primary key de-duplicated |
| clean_products | primary key de-duplicated; empty categories become `'uncategorised'` (`CASE WHEN ... THEN 'uncategorised'`) |
| clean_order_items | same order/item/seller merged into `quantity` (`GROUP BY` + `COUNT(*)`), values summed |
| clean_payments | only payments of clean orders; a separate order-level summary `order_payment_summary` is built (`GROUP_CONCAT` merges payment types) |
| clean_reviews | only clean orders with scores 1-5; `review_id` reused across orders is kept but flagged |

### The `clean_customers` trap (important)

Keeping only the most recent account per multi-account user is correct for **counting customers**.
A recommender, however, needs *all* historical orders of the same user, while the old accounts' orders still
sit in `clean_orders`. Joining `clean_orders` to `clean_customers` on `customer_id` therefore drops those
orders and the repeat-buyer count collapses to 0.

**Fix: do not re-run the cleaning.** Instead, `export_data.py` adds a `user_map` table that reads the full
mapping straight from the raw `olist_analysis.customers` table (99,197 rows), and the modelling code
aggregates all orders of a user through that mapping.

### Closing: the before/after comparison report

```sql
SELECT 'orders' AS tbl,
       (SELECT COUNT(*) FROM olist_analysis.orders) AS raw_rows,        -- before cleaning
       (SELECT COUNT(*) FROM olist_clean.clean_orders) AS clean_rows,   -- after cleaning
       (SELECT COUNT(*) FROM olist_analysis.orders) - (SELECT COUNT(*) FROM olist_clean.clean_orders) AS removed_rows
UNION ALL SELECT 'customers', ...
```

- one row per table, with subqueries fetching the before count, after count and difference;
- finally `SELECT ... FROM cleaning_log ORDER BY id` lists how many rows each cleaning reason removed.

---

## 3. export_data.py (Python export)

- `lines 3-9`: import `time` (timing), `Path` (paths), pandas (tables), pymysql (MySQL client) and the
  connection settings plus directories from `config`.
- `lines 11-36`, the `TABLES` dictionary: maps each table name to its query. The important entry is
  `user_map` (lines 15-19):
  ```python
  "SELECT DISTINCT c.customer_id, c.customer_unique_id, c.customer_state "
  "FROM olist_analysis.customers c "
  "WHERE c.customer_id IN (SELECT customer_id FROM olist_clean.clean_orders)"
  ```
  it reads the complete `customer_id → customer_unique_id` mapping from the **raw** database, restricted to
  accounts that appear in the clean orders — the line that fixes the lost repeat buyers.
- `main()`:
  - `RAW_DIR.mkdir(...)` creates `data/raw` if needed;
  - a guard raises a clear error when no database password is configured;
  - `pymysql.connect(**DB_CONFIG)` connects using the settings from `config` (`**` unpacks the dictionary);
  - the loop over tables runs `pd.read_sql(sql, conn)` into a DataFrame and writes it with
    `to_csv(..., encoding="utf-8-sig")` (the BOM keeps Excel happy);
  - `finally: conn.close()` closes the connection whether the export succeeded or not.

---

## 4. features.py (feature engineering)

### `_load_raw()`

- reads every `data/raw/*.csv`;
- `dtype={"customer_id": str}` forces ids to be read as strings so 32-character hex ids are not turned into
  numbers;
- `parse_dates=[...]` converts timestamp columns to real datetimes so they can be subtracted later;
- returns `user_map, orders, items, reviews, products`.

### `build_interactions()`

Goal: turn order-item rows into one **user x item interaction table**.

- `reviews.groupby("order_id")["review_score"].first()` takes the first score per order (one review per order
  in Olist) as the order's score;
- `items.merge(orders[...], on="order_id", how="left")` joins items to orders to obtain the customer and the
  timestamp (left join so no item row is lost);
- a further join to `user_map` replaces `customer_id` with `customer_unique_id` (**the user identity**) and
  the state;
- the review score is joined, then `dropna(...)` removes rows missing key fields;
- user and item id tables are built (`drop_duplicates()` then `reset_index()`), with `user_id` / `item_id`
  starting at 0 and merged back into the detail rows;
- three derived columns are added: `timestamp`, `state` (missing filled with `unknown`) and `delivery_days`
  (delivered minus purchased);
- `groupby(["user_id", "item_id"]).agg(...)` accumulates quantity and value per user-item pair;
- after sorting, `drop_duplicates(["user_id", "item_id"], keep="last")` keeps the most recent interaction per
  pair, and the accumulated quantity/value are merged back.

### `_build_from_split()` (shared by both scenarios)

- the number of users is `max(user_id) + 1`, since ids start at 0;
- the cut-off date is the latest training timestamp, and the train/test interaction counts are printed;
- **candidate pool**: `train.groupby("item_id")["user_id"].nunique()` counts distinct buyers per item, then
  `sort_values(ascending=False)` and `head(5000)` keep the 5,000 most popular items; `item_pos` maps each item
  id to a position 0-4999;
- **user features**: `groupby("user_id").agg(...)` produces orders, units bought, total spend, average score,
  most recent purchase and state; `recency_days` is the distance to the cut-off; `reindex(range(n_users))`
  makes sure every user has a row, with cold-start users filled with 0/unknown;
- **item features**:
  - `pd.get_dummies(category)` one-hot encodes categories;
  - numeric features (buyer counts, score, price, weight, photo count, description length) are zero-filled,
    `log1p`-transformed for the long-tailed columns and standardised with `StandardScaler`;
  - numeric and one-hot blocks are concatenated into a feature matrix and row-normalised for cosine
    similarity later;
  - `item_pop_norm` is the popularity normalised to 0-1 and serves as the popularity prior at prediction time;
- **sparse interaction matrices** use `scipy.sparse.csr_matrix` to store only non-zeros; three views are kept:
  purchased or not (0/1), review score (1-5) and log quantity;
- **SVD training triples** are `(user, item, score)` arrays, with missing scores filled as 3;
- **per-state popularity** builds a `state_pop` dictionary used as cold-start fallback (see below);
- **user profiles**: `X_spend @ item_features` is the spend-weighted sum of the features of purchased items,
  normalised — the "taste vector" used by the content model;
- **test construction**:
  - `train_items`: pool items each user bought during training (filtered out of the recommendations);
  - `test_pos`: pool items each user bought in the test period (the ground truth), with training items removed
    via `np.setdiff1d`;
  - `X_eval`: **the evaluation input uses training interactions only**; cold-start users get an all-zero row —
    the line that prevents leakage;
- everything is packaged into a `data` dictionary and returned.

### `build()` – Scenario A

- `quantile(1 - TEST_RATIO)` takes the 80th percentile timestamp as the cut-off;
- training data is everything before it, test data everything after;
- the result is cached to `features.pkl`, so a later run with `force=False` reads the cache.

### `build_repeat()` – Scenario B

- `groupby("user_id")["order_id"].nunique()` counts orders per user; `>=2` identifies repeat buyers;
- each repeat buyer's last order (sorted by time, `groupby.last()`) becomes the test set and everything before
  it the training set;
- it reuses `_build_from_split`, so the only difference between the two scenarios is how the split is made.

---

## 5. models.py (the eight recommenders)

All models share one interface: `fit(data)` trains and `predict(user_ids, data)` returns a score for every user
against the 5,000 candidate items.

### 5.1 popular – popularity baseline

**What it is**: no personalisation at all; recommend the globally most popular items.

```python
def predict(self, user_ids, data):
    return np.tile(self.pop, (len(user_ids), 1))
```

- `self.pop` is the 0-1 popularity vector over the 5,000 items;
- `np.tile(..., (n, 1))` copies it n times so every user receives an identical ranking.

### 5.2 random – lower bound

**What it is**: random scores, used to confirm the metrics behave — a non-zero score for `random` would mean a
bug in the evaluation code.

```python
return self.rng.random((len(user_ids), data["pool_size"]))
```

- `np.random.default_rng(42)` fixes the seed so results are reproducible.

### 5.3 itemcf – item-based collaborative filtering

**What it is**: "people who bought A also bought things similar to A". Item-item cosine similarity is computed
first, then items similar to what the user bought are recommended.

```python
col_sum = np.asarray(X.multiply(X).sum(axis=0)).ravel()  # buyers per item
col_norm = np.sqrt(col_sum) + 1e-8                       # vector length (+1e-8 avoids division by zero)
Xn = X.multiply(1.0 / col_norm)                          # normalise each column
S = (Xn.T @ Xn).toarray()                                # similarity = inner product of normalised columns
np.fill_diagonal(S, 0.0)                                 # an item is not similar to itself
top_idx = np.argpartition(-S, k, axis=1)[:, :k]          # keep only the Top-50 neighbours per item
self.S = sparse.csr_matrix((vals, (rows, cols)), shape=S.shape)
```

- cosine similarity is the cosine of the angle between two vectors: closer to 1 means more alike;
- `argpartition` finds the top-k values efficiently without fully sorting;
- keeping only the Top-50 neighbours turns a dense 5,000 x 5,000 matrix into a sparse one, saving memory and
  removing noise;
- prediction multiplies the user's purchase vector by the similarity matrix (`X @ S`) to score each candidate;
- `+ 0.05 * self.pop` adds a small popularity prior so cold-start users do not end up with all-zero scores.

### 5.4 usercf – user-based collaborative filtering

**What it is**: "people who bought what I bought also bought these other items".

```python
row_cnt = np.asarray(X.sum(axis=1)).ravel()
self.repeat_idx = np.where(row_cnt >= 2)[0]   # only active users (>=2 training interactions)
Xr = X[self.repeat_idx]
self.S = (Xrn @ Xrn.T).toarray()              # user-user cosine similarity
```

- a full similarity matrix over ~95K users would not fit in memory, so it is built only for active users;
- prediction computes `sim = Xn @ Xrn.T` (similarity between the target user and each active user) and then
  `scores = sim @ Xr`, a similarity-weighted sum of what those users bought.

### 5.5 svd – Funk-SVD matrix factorisation

**What it is**: factorising the user x item rating matrix into two small matrices — one latent vector per user
and one per item — with the dot product of the two giving the predicted score. The latent vectors capture
"what kind of thing this user likes" and "what kind of thing this item is".

```python
pred = self.mu + self.bu[u] + self.bi[i] + float(self.P[u] @ self.Q[i])
err = r - pred                                        # prediction error
self.bu[u] += SVD_LR * (err - SVD_REG * self.bu[u])   # update the user bias
self.bi[i] += SVD_LR * (err - SVD_REG * self.bi[i])   # update the item bias
self.P[u] = pu + SVD_LR * (err * self.Q[i] - SVD_REG * pu)  # update the user latent vector
self.Q[i] = qi + SVD_LR * (err * pu - SVD_REG * qi)        # update the item latent vector
```

- `mu` is the global mean, `bu` / `bi` are user and item biases (some users always score high, some items are
  well liked);
- one observed rating is drawn at a time and the parameters are nudged against the error (SGD);
- `SVD_REG` is regularisation that limits overfitting;
- prediction is `P @ Q.T + mu + bi`, and RMSE converges to about 1.05.

### 5.6 als – implicit-feedback matrix factorisation

**What it is**: like SVD, but instead of predicting a score it uses behaviour (bought or not, how often) as
the feedback signal. Suited to data that has orders but no ratings.

```python
for it in range(ALS_ITERS):
    P = (A @ Q) @ np.linalg.inv(Q.T @ Q + eye)   # fix Q, solve the optimal P (least squares)
    Q = (A.T @ P) @ np.linalg.inv(P.T @ P + eye) # fix P, solve the optimal Q
```

- alternating between fixing one side and solving the other is the "alternating least squares" loop;
- `eye = ALS_REG * np.eye(k)` is the regularisation term;
- consequence: a cold-start user with no behaviour has a zero latent vector and near-zero scores, so ALS helps
  returning users only.

### 5.7 content – content-based recommendation

**What it is**: ignore what other people bought and look at item attributes (category, price band,
popularity). The user profile is the average of the features of purchased items, and items whose features
match it best are recommended.

```python
prof = self.profiles[user_ids]                  # user profile computed during feature engineering
scores = prof @ self.item_feat.T                # profile x item features = cosine similarity
cold = norms < 1e-6                             # new users with no history
if cold.any():
    scores[j] = self.state_pop.get(st, self.pop)  # fall back to popularity within their state
```

- cold-start friendly: a new user without a profile degrades to "what people in their state buy".

### 5.8 hybrid – weighted blend

**What it is**: rescale the scores of itemcf/content/als/svd/popular to 0-1 and add them up with weights.

```python
norm = (s - lo) / (hi - lo)                     # scale each user's scores to 0-1
scores += self.weights[m.name] * norm           # accumulate with weights
```

- weights: itemcf 0.40, content 0.25, als 0.15, svd 0.10, popular 0.10 (tuning later favours itemcf 0.55);
- rationale: the components complement each other — collaborative filtering looks at other people, content
  looks at attributes, popularity covers cold start.

---

## 6. evaluate.py (offline evaluation)

### What the metrics mean

- **Precision@K**: of the top K recommendations, how many the user actually bought (is the list accurate?);
- **Recall@K**: of everything the user bought, how much was recommended (is the list complete?);
- **NDCG@K**: not only whether a hit occurred but where it ranked, with a logarithmic position discount;
- **Coverage@10**: how many distinct items appear across all lists (higher means less monotonous);
- **Diversity@10**: category variety inside the recommendation list (not ten near-identical items);
- **Novelty@10**: how obscure the recommended items are (1 minus popularity).

### Key functions

- `_ndcg_at_k`: per-user DCG/IDCG with `rel / log2(position + 1)`, so hits near the top count more;
- `rank_items`:
  - calls `predict` in batches of 2,000 users so no oversized matrix is built;
  - `scores[j, bought] = -np.inf` **pushes items bought during training to negative infinity, banning
    already-purchased items**;
  - `np.argpartition(-scores, kth=19)` takes the top 20 scores, which are then sorted;
- `evaluate_one`: computes precision/recall/ndcg per K, with coverage, diversity and novelty over the Top-10;
- `segment_metrics`: evaluates by segment — cold-start users (no training purchase), high-value users (top 20%
  by spend), users in the top five states, and everyone else;
- `run`: iterates over all models, assembles the DataFrame, writes CSVs and the text report, and draws the bar
  charts (`matplotlib` with the `Agg` backend, so figures are written to disk instead of popping up).

---

## 7. run_all.py (one command)

```python
export_data.main()                          # 1. export CSVs
data = features.build(force=True)           # 2. Scenario A features
model_dict = models_mod.build_models(data)  # 3. train the 8 models
evaluate.run(data, model_dict, tag="time_split", ...)  # 4. Scenario A evaluation
data2 = features.build_repeat(force=True)   # 5. Scenario B features
model_dict2 = models_mod.build_models(data2)# 6. retrain (different data, so new models)
evaluate.run(data2, model_dict2, tag="repeat", ...)    # 7. Scenario B evaluation
```

- `if __name__ == "__main__":` means the block runs only when the file is executed directly, not when it is
  imported.

---

## 8. Why each step exists, in one line

1. **Profiling**: quantify the problems before cleaning, so you know what to fix and how much is involved;
2. **Cleaning**: remove duplicated, missing and anomalous rows while logging every removal;
3. **Export**: freeze MySQL results into CSV so Python and the database stay decoupled;
4. **Feature engineering**: turn business data into a user x item interaction matrix plus feature vectors,
   with leakage protection built in;
5. **Modelling**: compare eight different strategies for guessing what a user wants to buy next;
6. **Evaluation**: answer, with one consistent offline metric set, which algorithm is better and for which
   kind of user.

# Project outline: e-commerce recommender optimisation from a people–product–context view

> The pipeline skeleton (preprocessing -> analysis -> people/product/context -> strategy) follows the
> original reference project, but the research question is different. That project asked "what does the
> business look like and how should we operate it"; this one asks "how do we put the right product in front
> of the right person, in which context, and how do we evaluate the result". The emphasis shifts from
> post-hoc reporting to algorithm modelling, offline evaluation and deployable strategy.

## 1. Positioning and goals

- Dataset: Olist Brazilian e-commerce (about 100K orders, 96K customers, 33K products, 8 relational
  tables, 2016-09 to 2018-10)
- Stack: MySQL (cleaning and feature extraction) + Python (pandas / scikit-learn for modelling and evaluation)
- Goal: build a complete recommendation pipeline — **data cleaning -> user/item/context profiles ->
  algorithm comparison and tuning -> offline evaluation -> operational strategy**
- Distinctive output: an honest reflection on what this dataset can and cannot support for recommendation

## 2. Stage 1: data preprocessing and cleaning (removing dirty rows, de-duplication first)

**2.1 Load the tables and check the data dictionary**

Import all 8 tables into MySQL and verify fields, types, null rates and uniqueness constraints.

**2.2 Missing values**

- orders: missing `customer_id`, `order_approved_at`, `order_delivered_*` timestamps
- reviews: missing review text (and missing scores)
- order_items: missing `freight_value`, `price`
- Rule: missing key identifiers are dropped; missing non-key fields are filled or nulled depending on the analysis

**2.3 Duplicate detection and removal (the core of this stage)**

| Table | Duplicate rule | Handling |
|---|---|---|
| orders | repeated `order_id` | keep the first occurrence, log the removed count |
| order_items | multiple rows for the same order + item (i.e. several units) | merge into a quantity field so the interaction matrix counts once |
| customers | one `customer_unique_id` mapped to several `customer_id` | keep one user entity, using the most recent purchase |
| reviews | repeated `review_id`; several reviews per order | keep the first; flag orders where items share one review |
| payments | several instalment rows per order | merge into the order total; not treated as duplicates |
| sellers / products | duplicate primary keys | de-duplicate and verify whether they are the same entity |

**2.4 Anomalous and dirty rows**

- invalid order statuses, empty orders (no items or no payment)
- negative or implausible prices
- impossible timestamps (payment before purchase, delivery before shipment)
- reviews that do not match any order (orphans)
- rows outside the 2016-09 to 2018-10 window

**2.5 Build the recommendation-specific datasets**

- interactions: `user x item x rating x purchase count x timestamp x value`
- user profiles: region, spending power, activity level, RFM segments
- item profiles: category, price band, rating, sales, ABC tier
- context: order month/weekday, region, delivery lead time, payment method
- **before/after cleaning report**: raw rows -> removed rows -> kept rows per table, with reason categories

## 3. Stage 2: people–product–context analysis (feature engineering, not description)

Every conclusion in this stage feeds the models rather than restating the business:

- **People (user features)**: purchase frequency, average order value, RFM tier, price-band preference,
  regional preference -> user feature vectors
- **Product (item features)**: category, price band, rating, popularity, ABC tier -> item feature vectors
- **Context (situation features)**: time window, region, delivery lead time, payment method -> context features
- **Interaction health check**: matrix sparsity, long-tail distribution, score distribution, repeat-purchase
  rate — these decide the algorithm choice (a low repeat rate means sparse CF data, which calls for content
  features and a hybrid strategy)

## 4. Stage 3: recommender modelling and optimisation (core chapter)

**4.1 Problem definition and data split**

- split interactions by **time** into train and test to avoid future leakage
- document the negative-sampling strategy (how "not purchased" is constructed without impression data)

**4.2 Baselines**

- most-popular and random recommendation as the lower bound

**4.3 Collaborative filtering**

- ItemCF (item similarity with a popularity penalty to reduce head bias)
- UserCF (user similarity, to observe behaviour on sparse data)

**4.4 Matrix factorisation**

- Funk-SVD (explicit ratings)
- ALS (implicit feedback, weighted by purchase count and value)
- tuning: number of latent factors, regularisation, learning rate, iterations

**4.5 Feature-based recommendation (cold start)**

- category preference, price-band preference, delivery preference matching
- new users -> regional and category popularity; new items -> items similar to previously bought categories

**4.6 Hybrid recommendation**

- weighted fusion / cascading
- people–product–context weighting: new releases and high-margin items for high-value users; fast-delivery
  items for remote regions; best-sellers for dormant users

## 5. Stage 4: offline evaluation and comparison

- metrics: Precision@K, Recall@K, F1, NDCG@K, coverage, diversity, novelty
- comparison table: baseline vs ItemCF vs UserCF vs SVD vs hybrid
- segment evaluation: high-value / new / low-frequency users and different regions
- tuning takeaways: which parameter matters most, and why

## 6. Stage 5: operational strategy (putting people–product–context to work)

- People: design copy and candidate pools by RFM tier (new releases plus membership perks for high-value
  users; best-sellers plus win-back coupons for dormant users)
- Product: exposure strategy by ABC tier (A drives traffic, B supports cross-sell, C clears stock or is
  given away)
- Context: situational recommendation (promotion windows, logistics optimisation in remote areas,
  payment-method preferences)

## 7. Stage 6: data reflection and improvement ideas (the distinctive chapter)

**7.1 What this dataset cannot do**

- no browsing, click, cart or search logs, so "purchase or not" is only a proxy for the recommendation target
- no impression data, so position bias cannot be corrected and offline evaluation carries selection bias
- scores cluster at 5 and comment coverage is low, which weakens the explicit signal
- the repeat-purchase rate is very low and the interaction matrix is sparse, which limits collaborative filtering
- only two years of data with an observable cut-off, so recency bias is present
- users concentrate in south-eastern Brazil, so the sample is unbalanced and generalisation is risky

**7.2 What it can do**

- supports offline simulation and teaching-level validation of recommender algorithms
- delimits how far order-based preference modelling can go in a sparse setting

**7.3 Improvement ideas**

- add behavioural tracking (impressions, clicks, carts), plus time decay and implicit-feedback weighting
- use real impression logs for position-bias correction and confirm with online A/B tests
- add item attributes, images and text to enrich the content signal

## 8. Deliverables

1. Cleaning comparison report (duplicates and dirty rows, item by item)
2. People–product–context feature engineering code and profile tables
3. Recommender implementations with tuning logs
4. Offline evaluation comparison report
5. Operational strategy document
6. Data reflection and improvement chapter

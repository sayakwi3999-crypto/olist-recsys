# Comparing and Optimising Recommender Algorithms on Order Data: the Case of Olist

**Abstract.** This paper studies how well classical recommender algorithms work on sparse order data,
using the public Brazilian e-commerce dataset from Olist (about 100K orders, 96K customers, 33K products).
On the data side, profiling and cleaning were carried out in MySQL to build a clean dataset covering
customers, products, orders, payments, reviews and sellers, including a fix for the loss of historical
orders caused by account-level de-duplication. On the modelling side, features were organised around a
people–product–context framework, a user x item interaction matrix was built, and eight algorithms were
implemented: popularity and random baselines, ItemCF, UserCF, Funk-SVD, ALS, content-based and hybrid
recommendation. Both cold-start and repeat-buyer scenarios were evaluated offline, followed by a
hyperparameter grid search. The repeat-purchase rate is only 3.12%, and after a time-based split 98.6% of
test users are cold-start, where popularity and content fallback define the performance ceiling. For repeat
buyers, collaborative filtering roughly doubles the recall of the popularity baseline. The explicit-rating
model (SVD) ranks worst, while the implicit-feedback model (ALS) is clearly better. The recommended hybrid
reaches NDCG@10 = 0.096 after tuning (+6.2% over defaults) with no regression on cold-start users.

**Keywords:** recommender systems; collaborative filtering; matrix factorisation; cold start; offline evaluation

## 1 Introduction

An e-commerce recommender exists to put the right product in front of the right user, and how well it does so
depends heavily on the structure of the behavioural data available. The public Olist dataset — eight
relational tables covering orders, customers, products, payments, reviews and sellers — is a good test case
for the question "can order and review data alone support recommender modelling?". Its defining property is
an extremely low repeat-purchase rate (3.12%): almost every customer buys once, which makes cold start the
central problem.

This paper aims to: compare eight classical recommenders under one consistent data and evaluation protocol;
test them separately on cold-start and returning users; produce a deployable hybrid configuration through
hyperparameter tuning; and discuss, based on the results, the boundaries of this dataset for recommender
modelling.

## 2 Data and preprocessing

### 2.1 Data at a glance

The raw data covers orders (99,441 rows), customers (99,441), order items (112,650), payments (103,886),
reviews (99,224), products (32,951) and sellers (3,095), spanning September 2016 to October 2018.

### 2.2 Profiling

Before cleaning, data quality was quantified. Main findings: 2,997 customers hold several accounts (one
unique identifier mapped to multiple orders); 7,088 groups of repeated rows share the same order, item and
seller; 789 review ids are reused across orders; 84 orders have impossible timestamps (for example delivery
before shipment); 610 products have no category; 58,247 reviews have no text. Review scores are strongly
right-skewed, with about 58% of them equal to 5.

### 2.3 Cleaning and user identity

Cleaning was implemented in MySQL: orders were de-duplicated on the primary key and records with missing
customers, impossible timestamps or out-of-range dates were dropped; repeated item rows inside one order
were merged into a quantity; orphan items, payments and reviews were removed; missing categories were
flagged as "uncategorised". Every removal was written to a cleaning log, producing a traceable before/after
comparison. The cleaned data has 99,197 orders, 95,918 customers, 102,304 order items, 103,620 payments and
98,986 reviews.

Cleaning exposed one issue that directly affected downstream modelling: after de-duplicating customers to
one row per unique identifier, orders placed under older accounts no longer matched any customer record, so
the repeat-buyer count briefly collapsed to zero. The root cause is that customer reporting and recommender
modelling need different definitions of a user — the former de-duplicates entities, the latter must merge
every historical order of the same identifier. The fix was to rebuild a complete user mapping from the raw
customer table during the modelling stage, merging all orders of each unique identifier; the repeat-buyer
count returned to 2,943 (against 2,997 under the raw definition, the difference being invalid orders removed
during cleaning).

## 3 Feature engineering and experiment design

### 3.1 Features and interaction matrix

Following the people–product–context framework, user features were built on the people side (number of
orders, units bought, total spend, average review score, recency, state), item features on the product side
(one-hot category, price, average score, popularity, weight and listing specification) and time, region and
delivery lead time were kept as context. Interactions form a sparse 95,316 x 5,000 matrix (the 5,000 most
popular training items), stored in three value views: purchased or not, review score, and purchase quantity.

### 3.2 Evaluation scenarios

To avoid leaking future information, model input uses training-period interactions only, and items already
purchased during training are filtered out of the recommendation list. Two scenarios are used: Scenario A
splits by time and takes the last 20% of interactions as the test set; Scenario B applies a holdout to repeat
buyers, holding each user's last order out as the test set to measure what collaborative filtering can really
achieve for users with history. Metrics include Precision@K, Recall@K and NDCG@K (K = 5/10/20), plus
platform-side coverage, diversity and novelty.

## 4 Models and results

### 4.1 Models

Eight models were implemented and compared: popularity (Popular) and random (Random) as baseline and lower
bound; item-based (ItemCF) and user-based (UserCF) collaborative filtering as memory-based methods;
Funk-SVD and ALS as matrix factorisation methods, the former minimising explicit rating error and the latter
reconstructing the interaction matrix from implicit feedback (purchase or not, quantity); content-based
recommendation matching item attributes against the user profile; and a hybrid that combines the weighted
scores of the previous five components.

### 4.2 Scenario A: cold start dominates

In Scenario A, 98.6% of the 7,408 test users have no purchase at all in the training period. Differences
between models are tiny: popularity, content and hybrid are essentially level (Recall@10 around 0.031) and
the random lower bound is close to zero. With cold start dominating, collaborative methods have no history
to work with, and popularity plus content fallback defines the ceiling.

### 4.3 Scenario B: users with history

Across the 1,257 evaluable repeat buyers the gaps widen sharply: ItemCF and UserCF both reach Recall@10 =
0.129, roughly twice the popularity baseline (0.063); the hybrid reaches Recall@20 = 0.167, the best overall;
content-based reaches Recall@10 = 0.078 and provides useful fallback for users with thin behaviour. Funk-SVD
converges to RMSE 1.05 on the training set — decent rating accuracy — yet ranks last on every ranking metric
(NDCG@10 = 0.003), while ALS is clearly better (NDCG@10 = 0.036). On sparse purchase data, implicit feedback
therefore suits ranking better than explicit ratings. Segment results show collaborative filtering reaching
Recall@10 = 0.171 for returning users in the top five states and the hybrid reaching Recall@10 = 0.127 for
high-value users: the gains concentrate where users have history and data is dense enough.

![Figure 1 Model comparison in Scenario B (repeat-buyer holdout)](recsys/output/repeat_model_comparison.png)

## 5 Hyperparameter tuning

A grid search over 34 configurations was run on Scenario B, optimising NDCG@10. Results: for ItemCF, 20
neighbours gives exactly the same result as 50 or 200, showing that on sparse data the similarity signal
concentrates in a few neighbours and saturates quickly; increasing ALS latent dimensions from 20 to 40 helps
clearly (NDCG@10 from 0.036 to 0.042); SVD does not improve materially under any configuration, which
reinforces the limits of explicit-rating models on this data. The best hybrid weights are ItemCF 0.55,
content 0.15, ALS 0.10, SVD 0.10 and popularity 0.10, lifting NDCG@10 from 0.090 to 0.096 (+6.2%) with no
regression in Scenario A.

![Figure 2 Best configuration per model after tuning](recsys/output/tuning_best_comparison.png)

## 6 Operational strategy

Based on these results, the strategy is layered by user status: new users are served by regional popularity,
global popularity and the content profile, while returning users get personalised ranking from collaborative
filtering and the hybrid. On the product side, an ABC tier determines exposure: tier A guarantees hits, tier B
gains traffic through cross-sell, tier C has its exposure capped. The experiments show that list coverage has
to be balanced between "popular items only" and "spray everything", with the hybrid the best trade-off between
hit rate and coverage. On the context side, well-covered regions can lean on collaborative filtering while
remote regions are served items with better delivery lead times.

## 7 Conclusions and reflection

On order data with a very low repeat-purchase rate, cold start is the dominant problem, and popularity plus
content fallback is the sensible answer there. For returning users, collaborative filtering remains the most
cost-effective method, and a hybrid achieves the best overall balance by fusing several views. Explicit
rating models are not suitable for ranking on this data; implicit feedback modelling is the more reliable
direction.

There are clear boundaries to what this dataset supports. Without browsing, click or cart logs, purchase is
only a proxy for the recommendation target; without impression data, position bias cannot be corrected and
offline evaluation remains an approximation; users are geographically concentrated, which puts generalisation
at risk. Future work should add behavioural tracking, introduce time decay and position-bias correction, and
validate the recommendations online with a small-traffic A/B test measured by click-through rate, conversion
rate and GMV.

**References**

[1] Olist. Brazilian E-Commerce Public Dataset by Olist. Kaggle. https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

[2] Xiang Liang. Recommender Systems in Practice. Beijing: People's Posts and Telecommunications Press, 2012.

# -*- coding: utf-8 -*-
"""Feature engineering: interactions, user/item features, time split, candidate pool."""
import pickle

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import StandardScaler

from config import PROCESSED_DIR, RAW_DIR, RANDOM_SEED, TEST_RATIO, TOP_ITEMS_POOL


def _load_raw():
    customers = pd.read_csv(RAW_DIR / "clean_customers.csv", dtype={"customer_id": str})
    user_map = pd.read_csv(RAW_DIR / "user_map.csv", dtype={"customer_id": str, "customer_unique_id": str})
    orders = pd.read_csv(
        RAW_DIR / "clean_orders.csv",
        dtype={"order_id": str, "customer_id": str},
        parse_dates=["order_purchase_timestamp", "order_delivered_customer_date", "order_estimated_delivery_date"],
    )
    items = pd.read_csv(
        RAW_DIR / "clean_order_items.csv",
        dtype={"order_id": str, "product_id": str, "seller_id": str},
    )
    reviews = pd.read_csv(RAW_DIR / "clean_reviews.csv", dtype={"order_id": str, "review_id": str})
    products = pd.read_csv(RAW_DIR / "clean_products.csv", dtype={"product_id": str})
    return user_map, orders, items, reviews, products


def build_interactions(user_map, orders, items, reviews):
    """Order-level reviews -> user x item interaction table (latest row per pair; qty and value summed)."""
    rating_map = reviews.groupby("order_id")["review_score"].first().rename("rating")
    df = items.merge(
        orders[["order_id", "customer_id", "order_purchase_timestamp", "order_delivered_customer_date"]],
        on="order_id",
        how="left",
    )
    df = df.merge(
        user_map[["customer_id", "customer_unique_id", "customer_state"]],
        on="customer_id",
        how="left",
    )
    df = df.merge(rating_map, on="order_id", how="left")
    df = df.dropna(subset=["customer_unique_id", "product_id", "order_purchase_timestamp"])

    users = df[["customer_unique_id"]].drop_duplicates().reset_index(drop=True)
    users["user_id"] = users.index
    prod = df[["product_id"]].drop_duplicates().reset_index(drop=True)
    prod["item_id"] = prod.index
    df = df.merge(users, on="customer_unique_id").merge(prod, on="product_id")

    df["timestamp"] = df["order_purchase_timestamp"]
    df["state"] = df["customer_state"].fillna("unknown")
    df["delivery_days"] = (
        (df["order_delivered_customer_date"] - df["order_purchase_timestamp"]).dt.days
    )

    # Same user-item pair: keep the most recent review/status, sum quantity and value
    agg = df.groupby(["user_id", "item_id"]).agg(
        purchase_count=("quantity", "sum"),
        total_spend=("price_total", "sum"),
    ).reset_index()
    inter = (
        df.sort_values("timestamp")
        .drop_duplicates(["user_id", "item_id"], keep="last")
        .reset_index(drop=True)
    )
    inter = inter.merge(agg, on=["user_id", "item_id"])
    inter = inter[
        ["user_id", "item_id", "order_id", "product_id", "rating", "purchase_count", "total_spend",
         "timestamp", "state", "delivery_days"]
    ]
    return inter, users, prod


def _build_from_split(inter, train, test, products, tag):
    """Given train/test interactions, build features, matrices and evaluation data (shared by both scenarios)."""
    n_users = int(inter["user_id"].max()) + 1

    cut_date = train["timestamp"].max()
    print(f"[features/{tag}] train interactions {len(train):,} | test interactions {len(test):,} | cut-off {cut_date}")

    # Candidate pool: the TOP_ITEMS_POOL most popular items in the training set
    pop = train.groupby("item_id")["user_id"].nunique().sort_values(ascending=False)
    pool = pop.head(TOP_ITEMS_POOL).index.tolist()
    pool_set = set(pool)
    item_pos = {iid: p for p, iid in enumerate(pool)}
    pool_size = len(pool)
    print(f"[features/{tag}] candidate pool: {pool_size} items, covering {train['item_id'].isin(pool_set).mean()*100:.1f}% of train interactions")

    # User features (from the training set)
    train_pool = train[train["item_id"].isin(pool_set)].copy()
    train_pool["pool_pos"] = train_pool["item_id"].map(item_pos)
    uf = train_pool.groupby("user_id").agg(
        n_orders=("timestamp", "count"),
        n_items=("purchase_count", "sum"),
        total_spend=("total_spend", "sum"),
        avg_rating=("rating", "mean"),
        last_ts=("timestamp", "max"),
        state=("state", lambda s: s.mode().iloc[0] if len(s) else "unknown"),
    )
    uf["recency_days"] = (cut_date - uf["last_ts"]).dt.days
    uf = uf.drop(columns=["last_ts"])
    uf = uf.reindex(range(n_users)).fillna(
        {"n_orders": 0, "n_items": 0, "total_spend": 0, "avg_rating": 3, "recency_days": np.nan, "state": "unknown"}
    )

    # Item features (candidate pool)
    all_items = inter[["item_id", "product_id"]].drop_duplicates()
    prod_feat = all_items.merge(products, on="product_id", how="left")
    prod_feat["product_category_name"] = prod_feat["product_category_name"].fillna("uncategorised")
    pstats = train_pool.groupby("item_id").agg(
        n_buyers=("user_id", "nunique"),
        avg_rating=("rating", "mean"),
        median_price=("total_spend", "median"),
    )
    item_feat = prod_feat[prod_feat["item_id"].isin(pool_set)].set_index("item_id").loc[pool]
    item_feat = item_feat.merge(pstats, left_index=True, right_index=True, how="left")
    item_feat["n_buyers"] = item_feat["n_buyers"].fillna(0)
    item_feat["avg_rating"] = item_feat["avg_rating"].fillna(3.0)
    item_feat["median_price"] = item_feat["median_price"].fillna(0)

    cat_dummies = pd.get_dummies(item_feat["product_category_name"], prefix="cat")
    numeric = item_feat[
        ["n_buyers", "avg_rating", "median_price", "product_weight_g",
         "product_photos_qty", "product_description_length", "product_name_length"]
    ].copy()
    numeric = numeric.fillna(0)
    for c in ["median_price", "product_weight_g", "product_description_length", "product_name_length"]:
        numeric[c] = np.log1p(numeric[c].clip(lower=0))
    num_scaled = StandardScaler().fit_transform(numeric.values)
    item_features = np.hstack([num_scaled, cat_dummies.values]).astype(np.float32)
    item_features /= (np.linalg.norm(item_features, axis=1, keepdims=True) + 1e-8)
    item_cats = pd.factorize(item_feat["product_category_name"])[0]
    item_pop_norm = np.log1p(pop.loc[pool].values.astype(float))
    item_pop_norm = item_pop_norm / (item_pop_norm.max() + 1e-9)

    # Sparse interaction matrix (users x candidate pool)
    def make_matrix(values):
        u = train_pool["user_id"].values
        i = train_pool["pool_pos"].values
        return sparse.csr_matrix((values, (u, i)), shape=(n_users, pool_size))

    X_train_bin = make_matrix(np.ones(len(train_pool), dtype=np.float32))
    X_train_rating = make_matrix(train_pool["rating"].fillna(3.0).values.astype(np.float32))
    X_train_count = make_matrix(np.log1p(train_pool["purchase_count"].values).astype(np.float32))

    # Training triples (for SVD)
    train_pairs = np.column_stack(
        [train_pool["user_id"].values, train_pool["pool_pos"].values,
         train_pool["rating"].fillna(3.0).values]
    ).astype(np.float32)

    # User status + per-state popularity (for cold-start fallback)
    user_states = uf["state"].values
    state_item_count = train_pool.groupby(["state", "pool_pos"])["user_id"].nunique()
    state_pop = {}
    for st, sub in state_item_count.groupby(level=0):
        vec = np.zeros(pool_size, dtype=np.float32)
        vec[sub.index.get_level_values(1).values] = np.log1p(sub.values.astype(float))
        state_pop[st] = vec / (vec.max() + 1e-9)

    # User profile (for content-based): spending-weighted average of purchased item features
    X_spend = make_matrix(np.log1p(train_pool["total_spend"].clip(lower=0).values).astype(np.float32))
    user_profiles = (X_spend @ item_features).astype(np.float32)
    norms = np.linalg.norm(user_profiles, axis=1, keepdims=True) + 1e-8
    user_profiles /= norms

    # Test set (restricted to the candidate pool)
    test_pool = test[test["item_id"].isin(pool_set)].copy()
    test_pool["pool_pos"] = test_pool["item_id"].map(item_pos)
    test_users_all = np.sort(test_pool["user_id"].unique())
    test_pos = {}
    train_items = {}
    for u, g in train_pool.groupby("user_id"):
        train_items[int(u)] = g["pool_pos"].values.astype(np.int32)
    kept_users = []
    for u, g in test_pool.groupby("user_id"):
        pos = np.setdiff1d(g["pool_pos"].values.astype(np.int32),
                            train_items.get(int(u), np.array([], dtype=np.int32)))
        if len(pos) > 0:
            test_pos[int(u)] = pos
            kept_users.append(int(u))
    test_users = np.array(sorted(kept_users), dtype=np.int32)

    # Test interaction matrix (evaluated users only)
    tmask = test_pool["user_id"].isin(test_users)
    tt = test_pool[tmask]
    u_idx = np.searchsorted(test_users, tt["user_id"].values)
    i_idx = tt["pool_pos"].values
    X_test_bin = sparse.csr_matrix(
        (np.ones(len(tt), dtype=np.float32), (u_idx, i_idx)), shape=(len(test_users), pool_size)
    )
    # Model input matrix: training interactions only (using test-period purchases as input would leak)
    # Cold-start users get an all-zero row and fall back to popularity/content features
    X_eval = X_train_bin[test_users]

    data = dict(
        train=train, test=test, pool=np.array(pool), item_pos=item_pos,
        n_users=n_users, pool_size=pool_size,
        X_train_bin=X_train_bin, X_train_rating=X_train_rating, X_train_count=X_train_count,
        train_pairs=train_pairs,
        user_features=uf, item_features=item_features, item_cats=item_cats,
        item_pop_norm=item_pop_norm, user_states=user_states, state_pop=state_pop,
        user_profiles=user_profiles,
        test_users=test_users, test_pos=test_pos, train_items=train_items,
        X_test_bin=X_test_bin, X_eval=X_eval, cut=cut_date,
    )
    print(f"[features/{tag}] features ready: {n_users:,} users, {pool_size} candidate items, {len(test_users):,} evaluated users")
    return data


def build(force=False):
    """Scenario A: time-based split (last 20% of interactions as test set)."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    pkl = PROCESSED_DIR / "features.pkl"
    if pkl.exists() and not force:
        print("[features] loading cached features:", pkl)
        with open(pkl, "rb") as f:
            return pickle.load(f)

    user_map, orders, items, reviews, products = _load_raw()
    inter, users, prod = build_interactions(user_map, orders, items, reviews)
    cut = inter["timestamp"].quantile(1 - TEST_RATIO)
    train = inter[inter["timestamp"] <= cut].copy()
    test = inter[inter["timestamp"] > cut].copy()
    data = _build_from_split(inter, train, test, products, "time_split")
    with open(pkl, "wb") as f:
        pickle.dump(data, f)
    return data


def build_repeat(force=False):
    """Scenario B: leave-one-out for repeat buyers.
    Keep users with >=2 orders, hold out each user's last order as the test set and
    train on everything before it. This measures CF/MF on users who do have history.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    pkl = PROCESSED_DIR / "repeat_features.pkl"
    if pkl.exists() and not force:
        print("[features] loading cached features:", pkl)
        with open(pkl, "rb") as f:
            return pickle.load(f)

    user_map, orders, items, reviews, products = _load_raw()
    inter, users, prod = build_interactions(user_map, orders, items, reviews)
    order_cnt = inter.groupby("user_id")["order_id"].nunique()
    repeat_users = set(order_cnt[order_cnt >= 2].index.tolist())
    print(f"[features/repeat] repeat buyers: {len(repeat_users):,}")

    # The last order of each repeat buyer becomes the test item
    last_order = inter[inter["user_id"].isin(repeat_users)].sort_values(
        "timestamp"
    ).groupby("user_id")["order_id"].last()
    lo_map = inter["user_id"].map(last_order)
    mask = inter["user_id"].isin(repeat_users) & (inter["order_id"] == lo_map)
    test = inter[mask].copy()
    train = inter[~mask].copy()
    data = _build_from_split(inter, train, test, products, "repeat")
    with open(pkl, "wb") as f:
        pickle.dump(data, f)
    return data


if __name__ == "__main__":
    build(force=True)
    build_repeat(force=True)

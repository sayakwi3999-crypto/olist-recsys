# -*- coding: utf-8 -*-
"""Recommenders: baselines / collaborative filtering / matrix factorisation / content / hybrid."""
import numpy as np
from scipy import sparse

from config import ALS_ITERS, ALS_K, ALS_REG, ITEMCF_TOP_K, RANDOM_SEED, SVD_EPOCHS, SVD_K, SVD_LR, SVD_REG


def _row_slice(X, user_ids, test_users):
    """Map global user ids to row positions of X_test_bin."""
    pos = np.searchsorted(test_users, user_ids)
    return X[pos]


class BaseModel:
    name = "base"

    def fit(self, data):
        pass

    def predict(self, user_ids, data):
        raise NotImplementedError


class PopularModel(BaseModel):
    name = "popular"

    def fit(self, data):
        self.pop = data["item_pop_norm"].astype(np.float32)

    def predict(self, user_ids, data):
        return np.tile(self.pop, (len(user_ids), 1))


class RandomModel(BaseModel):
    name = "random"

    def fit(self, data):
        self.rng = np.random.default_rng(RANDOM_SEED)

    def predict(self, user_ids, data):
        return self.rng.random((len(user_ids), data["pool_size"])).astype(np.float32)


class ItemCFModel(BaseModel):
    """Item-based CF: cosine similarity + Top-K neighbour truncation + popularity prior."""
    name = "itemcf"

    def __init__(self, top_k=ITEMCF_TOP_K, pop_prior=0.05):
        self.top_k = top_k
        self.pop_prior = pop_prior

    def fit(self, data):
        X = data["X_train_bin"].astype(np.float32)
        col_sum = np.asarray(X.multiply(X).sum(axis=0)).ravel()
        col_norm = np.sqrt(col_sum) + 1e-8
        Xn = X.multiply(1.0 / col_norm)
        S = (Xn.T @ Xn).toarray().astype(np.float32)
        np.fill_diagonal(S, 0.0)
        k = min(self.top_k, S.shape[0])
        top_idx = np.argpartition(-S, k, axis=1)[:, :k]
        rows = np.repeat(np.arange(S.shape[0]), k)
        cols = top_idx.ravel()
        vals = S[rows, cols].ravel()
        self.S = sparse.csr_matrix((vals, (rows, cols)), shape=S.shape)
        self.pop = data["item_pop_norm"].astype(np.float32)

    def predict(self, user_ids, data):
        X = _row_slice(data["X_eval"], user_ids, data["test_users"])
        scores = (X @ self.S).toarray().astype(np.float32)
        return scores + self.pop_prior * self.pop


class UserCFModel(BaseModel):
    """User-based CF: the similarity matrix is built only for users with >=2 training interactions."""
    name = "usercf"

    def __init__(self, pop_prior=0.05):
        self.pop_prior = pop_prior

    def fit(self, data):
        X = data["X_train_bin"].astype(np.float32)
        row_cnt = np.asarray(X.sum(axis=1)).ravel()
        self.repeat_idx = np.where(row_cnt >= 2)[0]
        Xr = X[self.repeat_idx]
        rn = np.sqrt(np.asarray(Xr.multiply(Xr).sum(axis=1)).ravel()) + 1e-8
        Xrn = Xr.multiply(1.0 / rn[:, None])
        self.S = (Xrn @ Xrn.T).toarray().astype(np.float32)
        np.fill_diagonal(self.S, 0.0)
        self.Xr = Xr.astype(np.float32)
        self.Xrn = Xrn
        self.pop = data["item_pop_norm"].astype(np.float32)

    def predict(self, user_ids, data):
        X = _row_slice(data["X_eval"], user_ids, data["test_users"])
        rn = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel()) + 1e-8
        Xn = X.multiply(1.0 / rn[:, None])
        sim = (Xn @ self.Xrn.T).toarray().astype(np.float32)
        scores = sim @ self.Xr
        scores = scores.toarray().astype(np.float32) if sparse.issparse(scores) else scores
        return scores + self.pop_prior * self.pop


class FunkSVDModel(BaseModel):
    """Funk-SVD: rating prediction (1-5) trained with SGD."""
    name = "svd"

    def __init__(self, k=SVD_K, epochs=SVD_EPOCHS, lr=SVD_LR, reg=SVD_REG):
        self.k = k
        self.epochs = epochs
        self.lr = lr
        self.reg = reg

    def fit(self, data):
        pairs = data["train_pairs"]
        rng = np.random.default_rng(RANDOM_SEED)
        n_users = data["n_users"]
        n_items = data["pool_size"]
        self.mu = float(pairs[:, 2].mean())
        self.P = rng.normal(0, 0.1, (n_users, self.k)).astype(np.float32)
        self.Q = rng.normal(0, 0.1, (n_items, self.k)).astype(np.float32)
        self.bu = np.zeros(n_users, dtype=np.float32)
        self.bi = np.zeros(n_items, dtype=np.float32)
        n = len(pairs)
        for epoch in range(self.epochs):
            order = rng.permutation(n)
            rmse_sum = 0.0
            for t in order:
                u, i, r = int(pairs[t, 0]), int(pairs[t, 1]), pairs[t, 2]
                pred = self.mu + self.bu[u] + self.bi[i] + float(self.P[u] @ self.Q[i])
                err = r - pred
                rmse_sum += err * err
                self.bu[u] += self.lr * (err - self.reg * self.bu[u])
                self.bi[i] += self.lr * (err - self.reg * self.bi[i])
                pu = self.P[u]
                self.P[u] = pu + self.lr * (err * self.Q[i] - self.reg * pu)
                qi = self.Q[i]
                self.Q[i] = qi + self.lr * (err * pu - self.reg * qi)
            print(f"  [svd] epoch {epoch+1}/{self.epochs} rmse={np.sqrt(rmse_sum/n):.4f}")

    def predict(self, user_ids, data):
        P = self.P[user_ids]
        scores = P @ self.Q.T + self.mu + self.bi[None, :]
        return scores.astype(np.float32)

    def predict_ratings(self, users, items):
        preds = (self.P[users] * self.Q[items]).sum(axis=1) + self.mu + self.bu[users] + self.bi[items]
        return preds


class ALSModel(BaseModel):
    """Implicit-feedback ALS (simplified): alternating least squares, unobserved entries weighted 0."""
    name = "als"

    def __init__(self, k=ALS_K, iters=ALS_ITERS, reg=ALS_REG):
        self.k = k
        self.iters = iters
        self.reg = reg

    def fit(self, data):
        A = data["X_train_count"].astype(np.float32)
        n_users, n_items = A.shape
        rng = np.random.default_rng(RANDOM_SEED)
        Q = rng.normal(0, 0.1, (n_items, self.k)).astype(np.float32)
        eye = self.reg * np.eye(self.k)
        for it in range(self.iters):
            P = (A @ Q) @ np.linalg.inv(Q.T @ Q + eye)
            Q = (A.T @ P) @ np.linalg.inv(P.T @ P + eye)
            print(f"  [als] iter {it+1}/{self.iters}")
        self.P = P.astype(np.float32)
        self.Q = Q.astype(np.float32)

    def predict(self, user_ids, data):
        return (self.P[user_ids] @ self.Q.T).astype(np.float32)


class ContentModel(BaseModel):
    """Content-based: the user profile is the weighted average of purchased item features."""
    name = "content"

    def fit(self, data):
        self.profiles = data["user_profiles"].astype(np.float32)
        self.item_feat = data["item_features"].astype(np.float32)
        self.state_pop = data["state_pop"]
        self.user_states = data["user_states"]
        self.pop = data["item_pop_norm"].astype(np.float32)

    def predict(self, user_ids, data):
        prof = self.profiles[user_ids]
        scores = prof @ self.item_feat.T
        norms = np.linalg.norm(prof, axis=1)
        cold = norms < 1e-6
        if cold.any():
            for j, uid in enumerate(np.where(cold)[0]):
                st = self.user_states[user_ids[j]]
                scores[j] = self.state_pop.get(st, self.pop)
        return scores.astype(np.float32)


class HybridModel(BaseModel):
    name = "hybrid"

    def __init__(self, models, weights):
        self.models = models
        self.weights = weights

    def fit(self, data):
        for m in self.models:
            m.fit(data)

    def predict(self, user_ids, data):
        scores = np.zeros((len(user_ids), data["pool_size"]), dtype=np.float32)
        for m in self.models:
            s = m.predict(user_ids, data).astype(np.float32)
            lo = s.min(axis=1, keepdims=True)
            hi = s.max(axis=1, keepdims=True)
            norm = np.where(hi - lo > 1e-9, (s - lo) / (hi - lo), np.zeros_like(s))
            scores += self.weights[m.name] * norm
        return scores


def build_models(data):
    """Build and train every model; returns {name: model}."""
    models = {
        "popular": PopularModel(),
        "random": RandomModel(),
        "itemcf": ItemCFModel(),
        "usercf": UserCFModel(),
        "svd": FunkSVDModel(),
        "als": ALSModel(),
        "content": ContentModel(),
    }
    for name, m in models.items():
        print(f"[model] training {name} ...")
        m.fit(data)
    hybrid = HybridModel(
        [models[k] for k in ["itemcf", "content", "als", "svd", "popular"]],
        {k: v for k, v in {"itemcf": 0.40, "content": 0.25, "als": 0.15, "svd": 0.10, "popular": 0.10}.items()},
    )
    models["hybrid"] = hybrid
    return models

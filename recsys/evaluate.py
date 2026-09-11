# -*- coding: utf-8 -*-
"""Offline evaluation: Top-K metrics, coverage/diversity/novelty, segments, plots and reports."""
import numpy as np
import pandas as pd

from config import K_VALUES, OUTPUT_DIR


def _ndcg_at_k(recs, test_pos, k):
    n = len(recs)
    dcg = np.zeros(n)
    idcg = np.zeros(n)
    for i in range(n):
        rel = np.isin(recs[i, :k], test_pos[i])
        dcg[i] = np.sum(rel / np.log2(np.arange(2, k + 2)))
        m = min(k, len(test_pos[i]))
        idcg[i] = np.sum(1.0 / np.log2(np.arange(2, m + 2)))
    return dcg / np.maximum(idcg, 1e-9)


def rank_items(model, data, chunk=2000):
    """Return (recs_top20, scores); items already bought during training are filtered out."""
    test_users = data["test_users"]
    recs_all, scores_all = [], []
    for s in range(0, len(test_users), chunk):
        uids = test_users[s : s + chunk]
        scores = model.predict(uids, data).astype(np.float64)
        for j, uid in enumerate(uids):
            bought = data["train_items"].get(int(uid))
            if bought is not None and len(bought):
                scores[j, bought] = -np.inf
        k = 20
        idx = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
        for j in range(len(uids)):
            order = np.argsort(-scores[j, idx[j]])
            idx[j] = idx[j, order]
        recs_all.append(idx)
        scores_all.append(scores)
    return np.vstack(recs_all), np.vstack(scores_all)


def evaluate_one(model, data):
    test_users = data["test_users"]
    pos = [data["test_pos"][int(u)] for u in test_users]
    recs, _ = rank_items(model, data)
    rows = {}
    for k in K_VALUES:
        hits = np.array([np.intersect1d(recs[i, :k], pos[i]).size for i in range(len(test_users))])
        prec = hits / k
        rec = hits / np.maximum(np.array([len(p) for p in pos]), 1)
        ndcg = _ndcg_at_k(recs, pos, k)
        rows[f"precision@{k}"] = float(prec.mean())
        rows[f"recall@{k}"] = float(rec.mean())
        rows[f"ndcg@{k}"] = float(ndcg.mean())
    # coverage / diversity / novelty over the Top-10 list
    top10 = recs[:, :10]
    cov = len(np.unique(top10)) / data["pool_size"]
    cats = data["item_cats"]
    same = 0.0
    nov = 0.0
    for i in range(len(test_users)):
        c = cats[top10[i]]
        cnt = np.bincount(c, minlength=cats.max() + 1)
        same += (cnt * (cnt - 1) / 2).sum()
        nov += (1 - data["item_pop_norm"][top10[i]]).mean()
    div = 1 - same / (len(test_users) * 45)
    rows["coverage@10"] = float(cov)
    rows["diversity@10"] = float(div)
    rows["novelty@10"] = float(nov / len(test_users))
    return rows


def segment_metrics(model, data, k=10):
    test_users = data["test_users"]
    pos = [data["test_pos"][int(u)] for u in test_users]
    recs, _ = rank_items(model, data)
    uf = data["user_features"]
    spend = uf["total_spend"].values
    thr = np.quantile(spend[spend > 0], 0.8) if (spend > 0).any() else 0
    states_top = ["SP", "RJ", "MG", "RS", "PR"]

    segs = {}
    for i, uid in enumerate(test_users):
        st = data["user_states"][uid]
        if len(data["train_items"].get(int(uid), [])) == 0:
            seg = "cold_users"
        elif spend[uid] >= thr:
            seg = "high_value"
        elif st in states_top:
            seg = "top5_states"
        else:
            seg = "other_users"
        segs.setdefault(seg, []).append(i)

    out = {}
    for seg, idx in segs.items():
        hits = np.array([np.intersect1d(recs[i, :k], pos[i]).size for i in idx])
        out[seg] = {
            "users": len(idx),
            f"precision@{k}": float((hits / k).mean()),
            f"recall@{k}": float((hits / np.maximum(np.array([len(pos[i]) for i in idx]), 1)).mean()),
        }
    return out


def run(data, models, tag="time_split",
        title="Time-based split (last 20% of interactions as test set)", plot=True):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}
    seg_all = {}
    for name, model in models.items():
        print(f"[eval] evaluating {name} ...")
        summary[name] = evaluate_one(model, data)
        seg_all[name] = segment_metrics(model, data)

    df = pd.DataFrame(summary).T
    df.index.name = "model"
    df.to_csv(OUTPUT_DIR / f"{tag}_metrics_summary.csv", encoding="utf-8-sig")

    seg_rows = []
    for m, segs in seg_all.items():
        for seg, v in segs.items():
            seg_rows.append({"model": m, "segment": seg, **v})
    sdf = pd.DataFrame(seg_rows)
    sdf.to_csv(OUTPUT_DIR / f"{tag}_segment_metrics.csv", index=False, encoding="utf-8-sig")

    lines = []
    lines.append("=" * 70)
    lines.append(f"Offline evaluation results ({title})")
    lines.append("=" * 70)
    lines.append(df.to_string())
    lines.append("")
    lines.append("Segment evaluation (precision@10 / recall@10)")
    lines.append(sdf.to_string(index=False))
    lines.append("")
    lines.append("Notes:")
    lines.append("- Candidate pool = the 5,000 most popular items in the training set; items already bought are filtered out.")
    n_cold = sum(1 for u in data["test_users"] if len(data["train_items"].get(int(u), [])) == 0)
    lines.append(f"- Of the {len(data['test_users']):,} users in the test period, {n_cold:,} ({n_cold/len(data['test_users'])*100:.1f}%) "
                 "have no purchase in the training period (cold-start users), a direct consequence of the ~3% repeat-purchase rate.")
    lines.append("- Only training interactions are used as model input; cold-start users fall back to popularity/content features, so no leakage.")
    lines.append("- cold_users = new users in the test period (no training purchase); high_value = top 20% by training spend;")
    lines.append("  top5_states = returning users in SP/RJ/MG/RS/PR; other_users = returning users elsewhere.")
    text = "\n".join(lines)
    (OUTPUT_DIR / f"{tag}_report.txt").write_text(text, encoding="utf-8")
    print(text)

    if plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            metrics = ["precision@10", "recall@10", "ndcg@10"]
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            x = np.arange(len(df))
            for ax, m in zip(axes, metrics):
                ax.bar(x, df[m].values)
                ax.set_xticks(x)
                ax.set_xticklabels(df.index, rotation=45)
                ax.set_title(m)
                ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()
            fig.savefig(OUTPUT_DIR / f"{tag}_model_comparison.png", dpi=150)
            plt.close(fig)

            fig2, ax2 = plt.subplots(1, 3, figsize=(15, 4))
            for ax, m in zip(ax2, ["coverage@10", "diversity@10", "novelty@10"]):
                ax.bar(x, df[m].values)
                ax.set_xticks(x)
                ax.set_xticklabels(df.index, rotation=45)
                ax.set_title(m)
                ax.grid(axis="y", alpha=0.3)
            fig2.tight_layout()
            fig2.savefig(OUTPUT_DIR / f"{tag}_coverage_diversity_novelty.png", dpi=150)
            plt.close(fig2)
            print("[eval] figures saved to output/")
        except Exception as e:
            print("[eval] figure generation skipped:", e)
    return df, sdf

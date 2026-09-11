# -*- coding: utf-8 -*-
"""Hyperparameter tuning: grid search over ItemCF / UserCF / SVD / ALS and hybrid weights
on Scenario B (repeat buyers). Writes tuning_results.csv, tuning_report.txt and a figure."""
import json
import time

import numpy as np
import pandas as pd

import evaluate
import features
import models as M
from config import OUTPUT_DIR

METRIC = "ndcg@10"  # metric being optimised


def main():
    t0 = time.time()
    data = features.build_repeat(force=False)
    rows = []

    def record(family, params, model):
        r = evaluate.evaluate_one(model, data)
        rows.append({"family": family,
                     "params": json.dumps(params, ensure_ascii=False),
                     **{k: round(v, 5) for k, v in r.items()}})
        print(f"[tune] {family:8s} {params} -> ndcg@10={r['ndcg@10']:.5f} "
              f"recall@10={r['recall@10']:.5f} precision@10={r['precision@10']:.5f}")
        return r

    # ---------- base models (trained once with default params, reused by the hybrid search) ----------
    print("[tune] training base models ...")
    popular = M.PopularModel(); popular.fit(data)
    random_m = M.RandomModel(); random_m.fit(data)
    content = M.ContentModel(); content.fit(data)
    record("popular", {}, popular)
    record("random", {}, random_m)
    record("content", {}, content)

    # ---------- ItemCF: neighbourhood size x popularity prior ----------
    best = None
    for top_k in [20, 50, 100, 200]:
        for prior in [0.0, 0.05, 0.10]:
            m = M.ItemCFModel(top_k=top_k, pop_prior=prior)
            m.fit(data)
            r = record("itemcf", {"top_k": top_k, "pop_prior": prior}, m)
            if best is None or r[METRIC] > best[0][METRIC]:
                best = (r, m, {"top_k": top_k, "pop_prior": prior})
    best_itemcf, best_itemcf_params = best[1], best[2]

    # ---------- UserCF: popularity prior ----------
    best = None
    for prior in [0.0, 0.05, 0.10]:
        m = M.UserCFModel(pop_prior=prior)
        m.fit(data)
        r = record("usercf", {"pop_prior": prior}, m)
        if best is None or r[METRIC] > best[0][METRIC]:
            best = (r, m, {"pop_prior": prior})
    best_usercf, best_usercf_params = best[1], best[2]

    # ---------- SVD: latent factors x regularisation (lr=0.01, epochs=12 fixed), plus one extra lr ----------
    best = None
    for k in [10, 20, 40]:
        for reg in [0.02, 0.05]:
            m = M.FunkSVDModel(k=k, reg=reg, lr=0.01, epochs=12)
            m.fit(data)
            r = record("svd", {"k": k, "reg": reg, "lr": 0.01}, m)
            if best is None or r[METRIC] > best[0][METRIC]:
                best = (r, m, {"k": k, "reg": reg, "lr": 0.01})
    m = M.FunkSVDModel(k=20, reg=0.05, lr=0.005, epochs=12)
    m.fit(data)
    r = record("svd", {"k": 20, "reg": 0.05, "lr": 0.005}, m)
    if r[METRIC] > best[0][METRIC]:
        best = (r, m, {"k": 20, "reg": 0.05, "lr": 0.005})
    best_svd, best_svd_params = best[1], best[2]

    # ---------- ALS: latent factors x regularisation ----------
    best = None
    for k in [10, 20, 40]:
        for reg in [5, 10, 20]:
            m = M.ALSModel(k=k, reg=reg, iters=10)
            m.fit(data)
            r = record("als", {"k": k, "reg": reg}, m)
            if best is None or r[METRIC] > best[0][METRIC]:
                best = (r, m, {"k": k, "reg": reg})
    best_als, best_als_params = best[1], best[2]

    # ---------- Hybrid: best components, weights only (no retraining) ----------
    base = {"itemcf": best_itemcf, "content": content,
            "als": best_als, "svd": best_svd, "popular": popular}
    combos = [
        {"itemcf": 0.40, "content": 0.25, "als": 0.15, "svd": 0.10, "popular": 0.10},
        {"itemcf": 0.55, "content": 0.15, "als": 0.10, "svd": 0.10, "popular": 0.10},
        {"itemcf": 0.35, "content": 0.30, "als": 0.15, "svd": 0.10, "popular": 0.10},
        {"itemcf": 0.30, "content": 0.20, "als": 0.25, "svd": 0.10, "popular": 0.15},
        {"itemcf": 0.30, "content": 0.15, "als": 0.10, "svd": 0.05, "popular": 0.40},
        {"itemcf": 0.50, "content": 0.25, "als": 0.10, "svd": 0.05, "popular": 0.10},
    ]
    best = None
    for w in combos:
        hy = M.HybridModel([base[k] for k in ["itemcf", "content", "als", "svd", "popular"]], w)
        r = record("hybrid", w, hy)
        if best is None or r[METRIC] > best[0][METRIC]:
            best = (r, w)
    best_hybrid_weights = best[1]

    # ---------- summary and output ----------
    df = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_DIR / "tuning_results.csv", index=False, encoding="utf-8-sig")

    best_rows = {}
    for fam in ["popular", "random", "content", "itemcf", "usercf", "svd", "als", "hybrid"]:
        sub = df[df["family"] == fam]
        if len(sub):
            best_rows[fam] = sub.loc[sub[METRIC].idxmax()]

    # Scenario A sanity check: re-evaluate the best hybrid setup on the time-based split
    print("[tune] validating the best configuration on Scenario A ...")
    dataA = features.build(force=False)
    popA = M.PopularModel(); popA.fit(dataA)
    conA = M.ContentModel(); conA.fit(dataA)
    icA = M.ItemCFModel(**{k: best_itemcf_params[k] for k in ("top_k", "pop_prior")}); icA.fit(dataA)
    svdA = M.FunkSVDModel(**best_svd_params); svdA.fit(dataA)
    alsA = M.ALSModel(**best_als_params); alsA.fit(dataA)
    hyA = M.HybridModel([icA, conA, alsA, svdA, popA], best_hybrid_weights)
    rA = evaluate.evaluate_one(hyA, dataA)

    lines = []
    lines.append("=" * 70)
    lines.append("Hyperparameter tuning report (Scenario B: repeat buyers, optimising ndcg@10)")
    lines.append("=" * 70)
    lines.append("")
    for fam in ["popular", "random", "content", "itemcf", "usercf", "svd", "als", "hybrid"]:
        if fam in best_rows:
            b = best_rows[fam]
            lines.append(f"[{fam}] best params: {b['params']}")
            lines.append(f"    precision@10={b['precision@10']}  recall@10={b['recall@10']}  "
                         f"ndcg@10={b['ndcg@10']}  coverage@10={b['coverage@10']}")
    lines.append("")
    lines.append(f"Overall best: {best_rows['hybrid']['params']} (ndcg@10={best_rows['hybrid']['ndcg@10']})")
    lines.append(f"Best hybrid on Scenario A (time split): "
                 f"precision@10={rA['precision@10']} recall@10={rA['recall@10']} ndcg@10={rA['ndcg@10']}")
    lines.append("")
    lines.append("To enable these results, write them into config.py and re-run run_all.py:")
    lines.append(f"  ITEMCF_TOP_K = {best_itemcf_params['top_k']}")
    lines.append(f"  ItemCF/UserCF popularity prior = {best_itemcf_params['pop_prior']}")
    lines.append(f"  SVD_K={best_svd_params['k']}  SVD_LR={best_svd_params['lr']}  SVD_REG={best_svd_params['reg']}")
    lines.append(f"  ALS_K={best_als_params['k']}  ALS_REG={best_als_params['reg']}")
    lines.append(f"  HYBRID_WEIGHTS = {best_hybrid_weights}")
    text = "\n".join(lines)
    (OUTPUT_DIR / "tuning_report.txt").write_text(text, encoding="utf-8")
    print(text)

    # ---------- comparison figure ----------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fams = ["itemcf", "usercf", "svd", "als", "hybrid"]
        metric_pairs = [("ndcg@10", "recall@10"), ("precision@10", "coverage@10")]
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        for ax, (m1, m2) in zip(axes, metric_pairs):
            vals = []
            for fam in fams:
                b = best_rows[fam]
                vals.append((fam, b[m1], b[m2]))
            x = np.arange(len(fams))
            ax.bar(x - 0.18, [v[1] for v in vals], 0.36, label=m1)
            ax.bar(x + 0.18, [v[2] for v in vals], 0.36, label=m2)
            ax.set_xticks(x)
            ax.set_xticklabels([v[0] for v in vals])
            ax.legend()
            ax.grid(axis="y", alpha=0.3)
        fig.suptitle("Tuned best config per model family (repeat scenario)")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "tuning_best_comparison.png", dpi=150)
        plt.close(fig)
        print("[tune] figure saved to output/tuning_best_comparison.png")
    except Exception as e:
        print("[tune] figure generation skipped:", e)

    print(f"\n[tune] finished in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()

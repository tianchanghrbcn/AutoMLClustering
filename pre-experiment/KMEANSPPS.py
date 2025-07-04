#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
K-MC² + Vanilla K-means (tracking)
目标函数: α·Sil + β·DB^{-1} + γ·(1−SSE/SSE_max) (α+β+γ=1)
权重通过 --alpha --beta [--gamma] 传入；省略 γ 时自动 γ = 1−α−β
"""

import argparse, json, math, os, time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score)
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------
# 0. CLI：解析权重参数
# --------------------------------------------------
cli = argparse.ArgumentParser()
cli.add_argument("--alpha", type=float, default=0.4)
cli.add_argument("--beta",  type=float, default=0.4)
cli.add_argument("--gamma", type=float, default=None,
                 help="若省略则 γ = 1-α-β")
cli.add_argument("--trials", type=int, default=30,
                 help="Optuna trial 数 (默认 30)")
args, _ = cli.parse_known_args()

alpha = args.alpha
beta  = args.beta
gamma = args.gamma if args.gamma is not None else 1.0 - alpha - beta
if not math.isclose(alpha + beta + gamma, 1.0, abs_tol=1e-6):
    raise ValueError(f"α+β+γ 必须等于 1，当前 = {alpha + beta + gamma}")

# --------------------------------------------------
# 1. 环境变量与数据读取
# --------------------------------------------------
csv_file_path  = os.getenv("CSV_FILE_PATH")
dataset_id     = os.getenv("DATASET_ID")
algorithm_name = os.getenv("ALGO")
if not csv_file_path:
    raise SystemExit("CSV_FILE_PATH env 未设置")
csv_file_path = os.path.normpath(csv_file_path)

df = pd.read_csv(csv_file_path)
excluded = [c for c in df.columns if 'id' in c.lower()]
X = df[df.columns.difference(excluded)].copy()
for c in X.select_dtypes(["object", "category"]).columns:
    X[c] = X[c].map(X[c].value_counts(normalize=True))
X = X.dropna()
X_scaled = StandardScaler().fit_transform(X)

# 预计算 SSE_max
SSE_max = float(((X_scaled - X_scaled.mean(0)) ** 2).sum())
start_time = time.time()

# --------------------------------------------------
# 2. K-MC² 初始化
# --------------------------------------------------
def k_mc2(X, k, m, rng):
    n = X.shape[0]
    centers = [X[rng.integers(n)]]
    for _ in range(1, k):
        x = X[rng.integers(n)]
        dx = np.min(np.sum((x - centers) ** 2, axis=1))
        for _ in range(m):
            y = X[rng.integers(n)]
            dy = np.min(np.sum((y - centers) ** 2, axis=1))
            if rng.random() < dy / dx:
                x, dx = y, dy
        centers.append(x)
    return np.vstack(centers)

# --------------------------------------------------
# 3. 手写 K-means（带追踪）
# --------------------------------------------------
def kmeans_track(X, init_centers, max_iter=300, tol=1e-4):
    rng = np.random.default_rng(0)
    centers = init_centers.copy()
    k = centers.shape[0]
    history = []

    for it in range(1, max_iter + 1):
        dmat = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        labels = dmat.argmin(axis=1)

        new_centers = np.vstack([
            X[labels == j].mean(0) if (labels == j).any()
            else X[rng.integers(X.shape[0])]
            for j in range(k)
        ])

        delta = float(np.linalg.norm(new_centers - centers))
        rel_d = delta / (np.linalg.norm(centers) + 1e-12)
        sse   = float(((X - new_centers[labels]) ** 2).sum())
        history.append({"iter": it, "delta": delta,
                        "relative_delta": rel_d, "sse": sse})
        if delta < tol:
            centers = new_centers
            break
        centers = new_centers

    return labels, centers, history

# --------------------------------------------------
# 4. Optuna 搜索
# --------------------------------------------------
records = []

def combined(db, sil, sse):
    db = max(db, 1e-6)
    return alpha*sil + beta*(1.0/db) + gamma*(1.0 - sse/SSE_max)

def add_conv(rec):
    deltas = [h["delta"] for h in rec["history"] if h["delta"] > 1e-12]
    if len(deltas) > 1:
        rec["auc_delta"] = float(sum(deltas))
        rec["geo_decay"] = float((deltas[-1]/deltas[0])**(1/(len(deltas)-1)))
    else:
        rec["auc_delta"] = rec["geo_decay"] = 0.0

def objective(trial):
    k = trial.suggest_int("k", 5, max(5, int(math.sqrt(X_scaled.shape[0]))))
    m = trial.suggest_int("m", 100, 500)
    init = k_mc2(X_scaled, k, m, np.random.default_rng(trial.number))
    labels, _, hist = kmeans_track(X_scaled, init)

    db  = davies_bouldin_score(X_scaled, labels)
    sil = silhouette_score(X_scaled, labels)
    ch  = calinski_harabasz_score(X_scaled, labels)
    sse = hist[-1]["sse"]
    combo = combined(db, sil, sse)

    rec = {"trial": trial.number, "k": k, "m": m,
           "iterations": len(hist), "history": hist,
           "silhouette": sil, "davies_bouldin": db,
           "calinski_harabasz": ch, "sse": sse,
           "combined": combo}
    add_conv(rec)
    records.append(rec)
    return combo

optuna.create_study(direction="maximize").optimize(objective, n_trials=args.trials)
best = max(records, key=lambda r: r["combined"])
k_best, m_best = best["k"], best["m"]

# --------------------------------------------------
# 5. 最终模型
# --------------------------------------------------
final_init = k_mc2(X_scaled, k_best, m_best, np.random.default_rng(42))
labels_fin, _, hist_fin = kmeans_track(X_scaled, final_init)

fin_sse = hist_fin[-1]["sse"]
fin_db  = davies_bouldin_score(X_scaled, labels_fin)
fin_sil = silhouette_score(X_scaled, labels_fin)
fin_ch  = calinski_harabasz_score(X_scaled, labels_fin)
fin_comb = combined(fin_db, fin_sil, fin_sse)
fin_iters = len(hist_fin)

best_final = best.copy()
best_final.update({"history": hist_fin,
                   "iterations": fin_iters,
                   "sse": fin_sse,
                   "silhouette": fin_sil,
                   "davies_bouldin": fin_db,
                   "calinski_harabasz": fin_ch,
                   "combined": fin_comb})
records.append(best_final)

# --------------------------------------------------
# 6. 输出
# --------------------------------------------------
root = (Path.cwd() / "results" / "clustered_data" / "KMEANSPPS" / f"clustered_{dataset_id}")
root.mkdir(parents=True, exist_ok=True)
base = Path(csv_file_path).stem

(root / f"{base}.txt").write_text(
    "\n".join([
        f"Best parameters: k={k_best}",
        f"Number of clusters: {k_best}",
        f"Final Combined Score: {fin_comb}",
        f"Final Silhouette Score: {fin_sil}",
        f"Final Davies-Bouldin Score: {fin_db}",
        f"Calinski-Harabasz: {fin_ch}",
        f"Iterations to converge: {fin_iters}",
        f"Final SSE: {fin_sse}"
    ]),
    encoding="utf-8")

(root / f"{base}_centroid_history.json").write_text(
    json.dumps(records, indent=4), encoding="utf-8")

summary = {
    "n_trials": len(records),
    "avg_iterations": float(np.mean([r["iterations"] for r in records])),
    "median_iterations": float(np.median([r["iterations"] for r in records])),
    "avg_auc_delta": float(np.mean([r["auc_delta"] for r in records])),
    "avg_geo_decay": float(np.mean([r["geo_decay"] for r in records])),
    "best_k": k_best,
    "best_combined_score": fin_comb,
    "best_sse": fin_sse,
    "weights": {"alpha": alpha, "beta": beta, "gamma": gamma},
    "total_runtime_sec": time.time() - start_time
}
(root / f"{base}_summary.json").write_text(
    json.dumps(summary, indent=4), encoding="utf-8")

print("History and summary files saved.")
print(f"Total runtime: {summary['total_runtime_sec']:.2f} s")

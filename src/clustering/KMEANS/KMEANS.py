#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
K-means with full tracking
* 目标函数改为: α·Sil + β·DB^{-1} + γ·(1 − SSE/SSE_max) ，α+β+γ=1
* 其它输入/输出行为保持不变
"""
import os
import time
import math
import json
import numpy as np
import pandas as pd
import optuna
from kneed import KneeLocator
from sklearn.cluster import kmeans_plusplus
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (silhouette_score,
                             davies_bouldin_score,
                             calinski_harabasz_score)

# --------------------------------------------------
# 0. 环境与数据读取
# --------------------------------------------------
csv_file_path  = os.getenv("CSV_FILE_PATH")
dataset_id     = os.getenv("DATASET_ID")
algorithm_name = os.getenv("ALGO")

if not csv_file_path:
    raise SystemExit("Error: CSV_FILE_PATH env not set.")

csv_file_path = os.path.normpath(csv_file_path)
try:
    df = pd.read_csv(csv_file_path)
    print("Data loaded successfully.")
except FileNotFoundError:
    raise SystemExit(f"File '{csv_file_path}' not found.")

start_time = time.time()

# ---------------------- 权重设置 (α+β+γ=1) ----------------------
alpha = 0.35
beta  = 0.65
gamma = 1.0 - alpha - beta   # 自动补足

# --------------------------------------------------
# 1. 预处理
# --------------------------------------------------
excluded_cols = [c for c in df.columns if 'id' in c.lower()]
X = df[df.columns.difference(excluded_cols)].copy()

for col in X.columns:
    if X[col].dtype in ("object", "category"):
        X[col] = X[col].map(X[col].value_counts(normalize=True))
X = X.dropna()
X_scaled = StandardScaler().fit_transform(X)

# 预计算 SSE_max（所有样本视为单簇）
global_centroid = X_scaled.mean(axis=0)
SSE_max = float(np.sum((X_scaled - global_centroid) ** 2))

# --------------------------------------------------
# 2. K-means with full tracking
# --------------------------------------------------
def run_kmeans_tracking(X, k, max_iter=300, tol=1e-4, seed=0):
    rng = np.random.default_rng(seed)
    centers, _ = kmeans_plusplus(X, n_clusters=k, random_state=seed)

    history = []
    for t in range(1, max_iter + 1):
        dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        labels = dists.argmin(axis=1)
        new_centers = np.vstack([
            X[labels == j].mean(axis=0) if np.any(labels == j)
            else X[rng.integers(0, X.shape[0])]
            for j in range(k)
        ])

        delta = float(np.linalg.norm(new_centers - centers))
        rel_delta = delta / (np.linalg.norm(centers) + 1e-12)
        sse = float(np.sum((X - new_centers[labels]) ** 2))
        history.append({
            "iter": t,
            "delta": delta,
            "relative_delta": rel_delta,
            "sse": sse
        })

        if delta < tol:
            centers = new_centers
            break
        centers = new_centers

    return labels, history, t, sse

# --------------------------------------------------
# 3. Optuna search + stats
# --------------------------------------------------
optuna_trials = []

def add_extra_stats(rec):
    deltas = [h["delta"] for h in rec["history"] if h["delta"] > 1e-12]
    if len(deltas) > 1:
        auc_delta = float(np.sum(deltas))
        geo_decay = float((deltas[-1] / deltas[0]) ** (1 / (len(deltas) - 1)))
    else:
        auc_delta, geo_decay = 0.0, 0.0
    rec.update({"auc_delta": auc_delta, "geo_decay": geo_decay})

def combined_score(db, sil, sse):
    db = max(db, 1e-6)
    return alpha * sil + beta * (1.0 / db) + gamma * (1.0 - sse / SSE_max)

def make_record(trial_number, k, labels, hist, iters, sse):
    db  = davies_bouldin_score(X_scaled, labels)
    sil = silhouette_score(X_scaled, labels)
    ch  = calinski_harabasz_score(X_scaled, labels)
    combo = combined_score(db, sil, sse)
    rec = {
        "trial_number": trial_number,
        "n_clusters": k,
        "sse": sse,
        "iterations": iters,
        "combined_score": combo,
        "silhouette": sil,
        "davies_bouldin": db,
        "calinski_harabasz": ch,
        "history": hist
    }
    add_extra_stats(rec)
    return rec

def objective(trial):
    k = trial.suggest_int("n_clusters", 5, max(2, math.isqrt(X.shape[0])))
    labels, hist, iters, sse = run_kmeans_tracking(X_scaled, k)
    rec = make_record(trial.number, k, labels, hist, iters, sse)
    optuna_trials.append(rec)
    return rec["combined_score"]          # 最大化综合得分

optuna.create_study(direction="maximize").optimize(objective, n_trials=20)

best_combo_trial = max(optuna_trials, key=lambda d: d["combined_score"])
k_opt = best_combo_trial["n_clusters"]
print(f"Optimal k (Optuna): {k_opt}")

# --------------------------------------------------
# 4. Kneedle refinement (可选)
# --------------------------------------------------
cluster_range = range(2, math.isqrt(X.shape[0]) + 1)
sse_curve = [
    run_kmeans_tracking(X_scaled, k, max_iter=30, tol=1e-3)[3]
    for k in cluster_range
]
try:
    kneedle = KneeLocator(cluster_range, sse_curve,
                          curve="convex", direction="decreasing")
    k_kneedle = kneedle.elbow
except ValueError:
    k_kneedle = None

if k_kneedle and k_kneedle != k_opt:
    k_low, k_high = sorted([k_opt, k_kneedle])
    def refined_obj(trial):
        k = trial.suggest_int("n_clusters", k_low, k_high)
        labels, hist, iters, sse = run_kmeans_tracking(X_scaled, k)
        rec = make_record(trial.number, k, labels, hist, iters, sse)
        optuna_trials.append(rec)
        return rec["combined_score"]
    optuna.create_study(direction="maximize").optimize(refined_obj, n_trials=10)
    best_combo_trial = max(optuna_trials, key=lambda d: d["combined_score"])

k_final = best_combo_trial["n_clusters"]
print(f"Final k: {k_final}")

# --------------------------------------------------
# 5. Final model & files
# --------------------------------------------------
labels_final, hist_final, final_iters, final_sse = run_kmeans_tracking(X_scaled, k_final)
final_db  = davies_bouldin_score(X_scaled, labels_final)
final_sil = silhouette_score(X_scaled, labels_final)
final_ch  = calinski_harabasz_score(X_scaled, labels_final)
final_combo = combined_score(final_db, final_sil, final_sse)

base = os.path.splitext(os.path.basename(csv_file_path))[0]
root_out = os.path.join(os.getcwd(), "..", "..", "..",
                        "results", "clustered_data", "KMEANS",
                        algorithm_name, f"clustered_{dataset_id}")
os.makedirs(root_out, exist_ok=True)

with open(os.path.join(root_out, f"{base}.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join([
        f"Best parameters: k={k_final}",
        f"Number of clusters: {k_final}",
        f"Final Combined Score: {final_combo}",
        f"Final Silhouette Score: {final_sil}",
        f"Final Davies-Bouldin Score: {final_db}",
        f"Calinski-Harabasz: {final_ch}",
        f"Iterations to converge: {final_iters}",
        f"Final SSE: {final_sse}"
    ]))

summary = {
    "n_trials": len(optuna_trials),
    "avg_iterations": float(np.mean([t["iterations"] for t in optuna_trials])),
    "median_iterations": float(np.median([t["iterations"] for t in optuna_trials])),
    "avg_auc_delta": float(np.mean([t["auc_delta"] for t in optuna_trials])),
    "avg_geo_decay": float(np.mean([t["geo_decay"] for t in optuna_trials])),
    "best_k": k_final,
    "best_combined_score": final_combo,
    "best_sse": best_combo_trial["sse"],
    "weights": {"alpha": alpha, "beta": beta, "gamma": gamma},
    "total_runtime_sec": time.time() - start_time
}

with open(os.path.join(root_out, f"{base}_centroid_history.json"), "w", encoding="utf-8") as fp:
    json.dump(optuna_trials, fp, indent=4)
with open(os.path.join(root_out, f"{base}_summary.json"), "w", encoding="utf-8") as fp:
    json.dump(summary, fp, indent=4)

print("History and summary files saved.")
print(f"Total runtime: {summary['total_runtime_sec']:.2f} s")

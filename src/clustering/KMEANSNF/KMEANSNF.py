#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improved K-Means (new formulation) — 目标函数升级版
★ 与基准脚本保持完全一致的输入、日志与文件输出格式 ★
"""

import os, time, math, json
import numpy as np, pandas as pd, optuna
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score)

# --------------------------------------------------
# 0. 环境参数与数据读取
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

# ---------------------- 目标函数权重 (α+β+γ=1) ----------------------
alpha = 0.35
beta  = 0.65
gamma = 1.0 - alpha - beta     # 自动补足

# --------------------------------------------------
# 1. 预处理（与基准一致）
# --------------------------------------------------
excluded_cols = [c for c in df.columns if 'id' in c.lower()]
X = df[df.columns.difference(excluded_cols)].copy()
for col in X.select_dtypes(['object', 'category']):
    X[col] = X[col].map(X[col].value_counts(normalize=True))
X = X.dropna()
X_scaled = StandardScaler().fit_transform(X)     # (n_samples, n_features)

# 预计算 SSE_max（把全部样本当作一个簇）
global_centroid = X_scaled.mean(axis=0)
SSE_max = float(np.sum((X_scaled - global_centroid) ** 2))

# --------------------------------------------------
# 2. 改进 K-means (new formulation) with history
# --------------------------------------------------
def _init_labels(n, k, rng):
    return rng.integers(0, k, size=n)

def _indicator(labels, k, n):
    F = np.zeros((n, k), dtype=np.float64)
    F[np.arange(n), labels] = 1.0
    return F

def kmeans_nf_history(Xt, k, max_iter=1000, inner_max_iter=100,
                      tol=1e-4, seed=0):
    """
    Xt: (d, n) 形式的矩阵（特征×样本）
    返回: labels, history(list), iters, sse
    """
    rng = np.random.default_rng(seed)
    n = Xt.shape[1]
    labels = _init_labels(n, k, rng)
    F = _indicator(labels, k, n)

    A = Xt.T @ Xt                              # n×n Gram
    s = np.ones(k, dtype=np.float64)
    M = np.zeros((n, k), dtype=np.float64)

    history, prev_centers = [], None
    for t in range(1, max_iter + 1):
        # —— 更新 s_i —— #
        for i in range(k):
            f = F[:, i]
            s[i] = np.sqrt(f.T @ A @ f) / (f.T @ f + 1e-10)

        # —— 内层标签更新 —— #
        for _ in range(inner_max_iter):
            for j in range(k):
                f = F[:, j]
                temp4 = A @ f
                temp3 = np.sqrt(f.T @ temp4)
                M[:, j] = temp4 / (temp3 + 1e-10)
            S = np.tile(s, (n, 1))
            labels_new = np.argmin(S**2 - 2*S*M, axis=1)
            if np.array_equal(labels, labels_new):
                break
            labels = labels_new
            F = _indicator(labels, k, n)

        # —— 计算中心及位移 —— #
        centers = (Xt @ F) / (np.sum(F, axis=0, keepdims=True) + 1e-10)  # d×k
        if prev_centers is not None:
            delta = float(np.linalg.norm(centers - prev_centers))
            rel_d = delta / (np.linalg.norm(prev_centers) + 1e-12)
            history.append({"iter": t, "delta": delta,
                            "relative_delta": rel_d})
            if delta < tol:
                break
        prev_centers = centers.copy()

        # 标签静止也可提前结束
        if t > 1 and history[-1]["delta"] == 0.0:
            break

    # SSE 目标
    sse = float(np.linalg.norm(Xt - centers @ F.T, ord="fro")**2)
    return labels, history, len(history) + 1, sse

# --------------------------------------------------
# 3. Optuna 搜索 + 记录 trial 历史
# --------------------------------------------------
optuna_trials = []

def _add_conv_stats(rec):
    deltas = [h["delta"] for h in rec["history"] if h["delta"] > 1e-12]
    if len(deltas) > 1:
        rec["auc_delta"] = float(sum(deltas))
        rec["geo_decay"] = float((deltas[-1] / deltas[0]) ** (1/(len(deltas)-1)))
    else:
        rec["auc_delta"] = 0.0
        rec["geo_decay"] = 0.0

def combined_score(db, sil, sse):
    db = max(db, 1e-6)
    return alpha * sil + beta * (1.0 / db) + gamma * (1.0 - sse / SSE_max)

def objective(trial):
    k = trial.suggest_int("n_clusters", 5,
                          max(2, int(math.isqrt(X_scaled.shape[0]))))
    labels, hist, iters, sse = kmeans_nf_history(X_scaled.T, k)

    db  = davies_bouldin_score(X_scaled, labels)
    sil = silhouette_score(X_scaled, labels)
    ch  = calinski_harabasz_score(X_scaled, labels)
    combo = combined_score(db, sil, sse)

    rec = {"trial_number": trial.number,
           "n_clusters":   k,
           "sse":          sse,
           "iterations":   iters,
           "combined_score": combo,
           "silhouette":   sil,
           "davies_bouldin": db,
           "calinski_harabasz": ch,
           "history":      hist}
    _add_conv_stats(rec)
    optuna_trials.append(rec)
    return combo                        # 最大化综合得分

optuna.create_study(direction="maximize").optimize(objective, n_trials=30)

best_trial = max(optuna_trials, key=lambda d: d["combined_score"])
k_opt      = best_trial["n_clusters"]
print(f"Optimal k (Optuna): {k_opt}")

# --------------------------------------------------
# 4. 以最佳 k 重新训练最终模型
# --------------------------------------------------
labels_final, hist_final, final_iters, final_sse = kmeans_nf_history(X_scaled.T, k_opt)
final_db  = davies_bouldin_score(X_scaled, labels_final)
final_sil = silhouette_score(X_scaled, labels_final)
final_ch  = calinski_harabasz_score(X_scaled, labels_final)
final_combo = combined_score(final_db, final_sil, final_sse)

# —— 把最终一次运行也追加到历史，以保持和基准“centroid_history”一致 —— #
best_trial_final = best_trial.copy()
best_trial_final.update({"history": hist_final,
                         "iterations": final_iters,
                         "sse": final_sse,
                         "silhouette": final_sil,
                         "davies_bouldin": final_db,
                         "calinski_harabasz": final_ch,
                         "combined_score": final_combo})
optuna_trials.append(best_trial_final)

# --------------------------------------------------
# 5. 目录结构 & 文件输出（完全对齐基准）
# --------------------------------------------------
root_out = os.path.join(os.getcwd(), "..", "..", "..", "results",
                        "clustered_data", "KMEANSNF",
                        algorithm_name,
                        f"clustered_{dataset_id}")
os.makedirs(root_out, exist_ok=True)

base = os.path.splitext(os.path.basename(csv_file_path))[0]

# ---- TXT ----
with open(os.path.join(root_out, f"{base}.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join([
        f"Best parameters: k={k_opt}",
        f"Number of clusters: {k_opt}",
        f"Final Combined Score: {final_combo}",
        f"Final Silhouette Score: {final_sil}",
        f"Final Davies-Bouldin Score: {final_db}",
        f"Calinski-Harabasz: {final_ch}",
        f"Iterations to converge: {final_iters}",
        f"Final SSE: {final_sse}"
    ]))

# ---- JSON ----
with open(os.path.join(root_out, f"{base}_centroid_history.json"),
          "w", encoding="utf-8") as fp:
    json.dump(optuna_trials, fp, indent=4)

summary = {
    "n_trials":          len(optuna_trials),
    "avg_iterations":    float(np.mean([t["iterations"] for t in optuna_trials])),
    "median_iterations": float(np.median([t["iterations"] for t in optuna_trials])),
    "avg_auc_delta":     float(np.mean([t["auc_delta"] for t in optuna_trials])),
    "avg_geo_decay":     float(np.mean([t["geo_decay"] for t in optuna_trials])),
    "best_k":            k_opt,
    "best_combined_score": final_combo,
    "best_sse":          final_sse,
    "weights":           {"alpha": alpha, "beta": beta, "gamma": gamma},
    "total_runtime_sec": time.time() - start_time
}

with open(os.path.join(root_out, f"{base}_summary.json"),
          "w", encoding="utf-8") as fp:
    json.dump(summary, fp, indent=4)

print("History and summary files saved.")
print(f"Total runtime: {summary['total_runtime_sec']:.2f} s")

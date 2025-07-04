#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GMM clustering with full EM-iteration tracking
* 保持原有输入/输出格式
* 目标函数改为: α·Sil + β·DB^{-1} + γ·(1-SSE/SSE_max) ，α+β+γ=1
* 额外记录: n_iter, lower_bound 曲线, log-likelihood AUC, 迭代衰减率、参数偏移
"""

import numpy as np
import pandas as pd
import optuna
import os
import time
import math
import json
from kneed import KneeLocator
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score

# --------------------------------------------------
# 0. 环境读取
# --------------------------------------------------
csv_file_path  = os.getenv("CSV_FILE_PATH")
dataset_id     = os.getenv("DATASET_ID")
algorithm_name = os.getenv("ALGO")
clean_state    = os.getenv("CLEAN_STATE", "raw")    # raw / cleaned (用于后续Δk 计算)

if not csv_file_path:
    raise SystemExit("Error: CSV_FILE_PATH env not set.")
csv_file_path = os.path.normpath(csv_file_path)

try:
    df = pd.read_csv(csv_file_path)
except FileNotFoundError:
    raise SystemExit(f"File '{csv_file_path}' not found.")

start_time = time.time()

# --------------------------------------------------
# 1. 预处理
# --------------------------------------------------
excluded_cols = [c for c in df.columns if 'id' in c.lower()]
X = df[df.columns.difference(excluded_cols)].copy()

for col in X.columns:
    if X[col].dtype in ("object", "category"):
        X.loc[:, col] = X[col].map(X[col].value_counts(normalize=True))
X = X.dropna()
X_scaled = StandardScaler().fit_transform(X)

# --------------------------------------------------
# 2. 目标函数权重 (α+β+γ=1)
# --------------------------------------------------
alpha = 0.35
beta  = 0.65
gamma = 1.0 - alpha - beta   # 自动补足和为 1

# 预计算全局 SSE_max（所有样本当作一个簇时的 SSE）
global_centroid = X_scaled.mean(axis=0)
SSE_max = float(np.sum((X_scaled - global_centroid) ** 2))

# --------------------------------------------------
# 3. 辅助函数
# --------------------------------------------------
def _sse(labels: np.ndarray) -> float:
    """计算聚类方案的总 SSE（忽略噪声标签 -1, 虽然 GMM 不会产生 -1）。"""
    sse = 0.0
    for lbl in np.unique(labels):
        pts = X_scaled[labels == lbl]
        if pts.size == 0:
            continue
        centroid = pts.mean(axis=0)
        sse += np.sum((pts - centroid) ** 2)
    return float(sse)

def combined_score(db, sil, sse):
    """综合得分: α·Sil + β·DB^{-1} + γ·(1 - SSE/SSE_max)."""
    db = max(db, 1e-6)                       # 防止除零
    return alpha * sil + beta * (1.0 / db) + gamma * (1.0 - sse / SSE_max)

def gmm_with_tracking(n_components, cov_type):
    """返回 (labels, n_iter, lower_bounds list, gmm_model)"""
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=cov_type,
        max_iter=1,            # 每次只跑 1 iter
        warm_start=True,
        random_state=0
    )
    lower_bounds = []
    for _ in range(300):      # 与默认 max_iter 保持一致
        gmm.fit(X_scaled)
        lower_bounds.append(gmm.lower_bound_)
        if gmm.converged_:
            break
    labels = gmm.predict(X_scaled)
    return labels, gmm.n_iter_, lower_bounds, gmm

def make_trial_record(trial_no, k, cov_type, labels, n_iter, lb_curve):
    db  = davies_bouldin_score(X_scaled, labels)
    sil = silhouette_score(X_scaled, labels)
    sse = _sse(labels)
    combo = combined_score(db, sil, sse)
    # 计算 EM 收敛 “面积”(AUC) 与衰减率
    if len(lb_curve) > 1:
        auc_ll = float(np.trapz(lb_curve))
        geo_decay = float(abs(lb_curve[-1] - lb_curve[0]) /
                          max(abs(lb_curve[0]), 1e-12))
    else:
        auc_ll, geo_decay = 0.0, 0.0
    return {
        "trial_number": trial_no,
        "n_components": k,
        "covariance_type": cov_type,
        "combined_score": combo,
        "silhouette": sil,
        "davies_bouldin": db,
        "sse": sse,
        "n_iter": n_iter,
        "ll_start": lb_curve[0],
        "ll_end": lb_curve[-1],
        "auc_ll": auc_ll,
        "ll_geo_decay": geo_decay,
        "ll_curve": lb_curve
    }

# --------------------------------------------------
# 4. 第一轮 Optuna 搜索
# --------------------------------------------------
optuna_trials = []

def objective(trial):
    max_k = max(5, math.isqrt(X.shape[0]))
    k = trial.suggest_int("n_components", 5, max_k)
    cov_type = trial.suggest_categorical("covariance_type",
                                         ['full', 'tied', 'diag', 'spherical'])
    labels, n_iter, lb_curve, _ = gmm_with_tracking(k, cov_type)
    rec = make_trial_record(trial.number, k, cov_type, labels, n_iter, lb_curve)
    optuna_trials.append(rec)
    return rec["combined_score"]

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)

best_rec = max(optuna_trials, key=lambda d: d["combined_score"])
k_optuna, best_cov_type = best_rec["n_components"], best_rec["covariance_type"]

# --------------------------------------------------
# 5. Kneedle (基于负对数似然 ≈ SSE)
# --------------------------------------------------
cluster_range = range(2, max(3, math.isqrt(X.shape[0])) + 1)
sse_curve = []
for k in cluster_range:
    labels, _, _, gmm_tmp = gmm_with_tracking(k, best_cov_type)
    nll = -gmm_tmp.score(X_scaled) * len(X_scaled)
    sse_curve.append(nll)

def moving_average(x, w=3):
    return np.convolve(x, np.ones(w) / w, mode="valid")
sse_smoothed = moving_average(sse_curve, 3)

try:
    kneedle = KneeLocator(cluster_range[:len(sse_smoothed)],
                          sse_smoothed,
                          curve="convex",
                          direction="decreasing")
    k_kneedle = kneedle.elbow
except ValueError:
    k_kneedle = None

# 第二轮局部搜索
if k_kneedle and k_kneedle != k_optuna:
    k_low, k_high = sorted([k_optuna, k_kneedle])
else:
    k_low, k_high = k_optuna, k_optuna + 2

def refined_objective(trial):
    k = trial.suggest_int("n_components", k_low, k_high)
    labels, n_iter, lb_curve, _ = gmm_with_tracking(k, best_cov_type)
    rec = make_trial_record(trial.number, k, best_cov_type, labels, n_iter, lb_curve)
    optuna_trials.append(rec)
    return rec["combined_score"]

optuna.create_study(direction="maximize").optimize(refined_objective, n_trials=10)
best_rec = max(optuna_trials, key=lambda d: d["combined_score"])
final_best_k = best_rec["n_components"]

# --------------------------------------------------
# 6. 最终模型 + 输出
# --------------------------------------------------
labels_final, n_iter_final, lb_curve_final, gmm_final = gmm_with_tracking(
    final_best_k, best_cov_type)

final_db   = davies_bouldin_score(X_scaled, labels_final)
final_sil  = silhouette_score(X_scaled, labels_final)
final_sse  = _sse(labels_final)
final_comb = combined_score(final_db, final_sil, final_sse)

# 文本输出（保持原有 4 行）
base   = os.path.splitext(os.path.basename(csv_file_path))[0]
root   = os.path.join(os.getcwd(), "..", "..", "..", "results",
                      "clustered_data", "GMM", algorithm_name,
                      f"clustered_{dataset_id}")
os.makedirs(root, exist_ok=True)
txt_fp = os.path.join(root, f"{base}.txt")
with open(txt_fp, "w", encoding="utf-8") as fh:
    fh.write("\n".join([
        f"Best parameters: n_components={final_best_k}, covariance type={best_cov_type}",
        f"Final Combined Score: {final_comb}",
        f"Final Silhouette Score: {final_sil}",
        f"Final Davies-Bouldin Score: {final_db}"
    ]))

# JSON:  trial 级历史 + 版本 summary
hist_path   = os.path.join(root, f"{base}_{clean_state}_gmm_history.json")
summary_path = os.path.join(root, f"{base}_{clean_state}_summary.json")
with open(hist_path, "w", encoding="utf-8") as fp:
    json.dump(optuna_trials, fp, indent=4)

summary = {
    "clean_state": clean_state,
    "best_k": final_best_k,
    "best_cov_type": best_cov_type,
    "best_combined": final_comb,
    "best_silhouette": final_sil,
    "best_db": final_db,
    "best_sse": final_sse,
    "weights": {"alpha": alpha, "beta": beta, "gamma": gamma},
    "n_iter_final": n_iter_final,
    "ll_curve_final": lb_curve_final,
    "total_runtime_sec": time.time() - start_time
}
with open(summary_path, "w", encoding="utf-8") as fp:
    json.dump(summary, fp, indent=4)

# 若存在另一版本 summary，则写出参数偏移
other_state = "cleaned" if clean_state == "raw" else "raw"
other_path  = os.path.join(root, f"{base}_{other_state}_summary.json")
if os.path.exists(other_path):
    with open(other_path) as fp:
        other = json.load(fp)
    shift = {
        "dataset_id": dataset_id,
        "delta_k": summary["best_k"] - other["best_k"],
        "delta_combined": summary["best_combined"] - other["best_combined"],
        "rel_shift": abs(summary["best_k"] - other["best_k"]) / max(other["best_k"], 1)
    }
    shift_path = os.path.join(root, f"{base}_param_shift.json")
    with open(shift_path, "w", encoding="utf-8") as fp:
        json.dump(shift, fp, indent=4)

print(f"History, summary and (if applicable) shift files saved in {root}")
print(f"Program completed in: {summary['total_runtime_sec']:.2f} seconds")

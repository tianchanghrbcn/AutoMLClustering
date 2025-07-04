#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DBSCAN with core / border / noise statistics + param-shift tracking
保持 4 行固定文本输出字段
"""
import os, time, json
import numpy as np
import pandas as pd
import optuna
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, pairwise_distances

# ----------------------------- 环境变量 -----------------------------
csv_file_path  = os.getenv("CSV_FILE_PATH")
dataset_id     = os.getenv("DATASET_ID")
algorithm_name = os.getenv("ALGO")
clean_state    = os.getenv("CLEAN_STATE", "raw")          # raw / cleaned

if not csv_file_path:
    raise SystemExit("CSV_FILE_PATH not set")
csv_file_path = os.path.normpath(csv_file_path)

# ------------------------------ 读取数据 -----------------------------
df = pd.read_csv(csv_file_path)
excluded = [c for c in df.columns if 'id' in c.lower()]
X = df[df.columns.difference(excluded)].copy()
for c in X.columns:
    if X[c].dtype in ("object", "category"):
        # 用出现频率映射类别
        X[c] = X[c].map(X[c].value_counts(normalize=True))
X = X.dropna()
X_scaled = StandardScaler().fit_transform(X)

# ------------------------ 指标权重 (α+β+γ=1) ------------------------
alpha = 0.35
beta  = 0.65
gamma = 1.0 - alpha - beta      # 自动保证和为 1

# -------------------------- 预计算 SSE_max --------------------------
global_centroid = X_scaled.mean(axis=0)
SSE_max = float(np.sum((X_scaled - global_centroid) ** 2))

start_time = time.time()

# -------------------------- 评价函数 -------------------------------
def _sse(labels: np.ndarray) -> float:
    """计算聚类方案的总 SSE（忽略噪声点 −1）。"""
    sse = 0.0
    for lbl in np.unique(labels):
        if lbl == -1:                         # 跳过噪声
            continue
        pts = X_scaled[labels == lbl]
        if pts.size == 0:
            continue
        centroid = pts.mean(axis=0)
        sse += np.sum((pts - centroid) ** 2)
    return float(sse)

def evaluate(labels: np.ndarray):
    """返回 (综合得分, silhouette, DB, noise_ratio, SSE)"""
    n_clusters = len(np.unique(labels)) - (1 if -1 in labels else 0)
    noise_ratio = float((labels == -1).mean())

    # 若不足两个簇，silhouette/DB 无法定义，直接给极差分数
    if n_clusters < 2:
        return -np.inf, np.nan, np.nan, noise_ratio, np.nan

    sil = silhouette_score(X_scaled, labels)
    db  = davies_bouldin_score(X_scaled, labels)
    db  = max(db, 1e-6)                       # 防止除零

    sse = _sse(labels)
    sse_term = 1.0 - sse / SSE_max            # 归一化后越大越好

    combined = alpha * sil + beta * (1.0 / db) + gamma * sse_term
    return combined, sil, db, noise_ratio, sse

# -------------------------- Optuna 搜索 -----------------------------
def objective(trial):
    eps         = trial.suggest_float("eps", 0.1, 2.0, step=0.05)
    min_samples = trial.suggest_int("min_samples", 5, 50)

    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X_scaled)
    score, sil, db, noise, _ = evaluate(labels)

    # 噪声惩罚
    return score * (1.0 - noise)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=150)

best_params = study.best_params

# -------------------------- 最终模型 & 统计 --------------------------
dbscan = DBSCAN(eps=best_params["eps"],
                min_samples=best_params["min_samples"])
labels = dbscan.fit_predict(X_scaled)
combined_score, sil_score, db_score, noise_ratio, sse_score = evaluate(labels)

# —— 统计核心 / 边界 / 噪声 —— #
dmat = pairwise_distances(X_scaled, metric="euclidean")
core_mask    = np.sum(dmat <= best_params["eps"], axis=1) >= best_params["min_samples"]
noise_mask   = labels == -1
border_mask  = ~core_mask & ~noise_mask
core_count   = int(core_mask.sum())
border_count = int(border_mask.sum())
noise_count  = int(noise_mask.sum())

neighbor_hist = np.bincount(np.clip(np.sum(dmat <= best_params["eps"], axis=1), 0, 49))

core_stats = {
    "core_count": core_count,
    "border_count": border_count,
    "noise_count": noise_count,
    "noise_ratio": noise_ratio,
    "neighbor_hist": neighbor_hist.tolist(),      # 0..49+
    "combined_score": combined_score,            # 方便后续 param-shift
    "silhouette": sil_score,
    "davies_bouldin": db_score,
    "sse": sse_score,
    "weights": {"alpha": alpha, "beta": beta, "gamma": gamma}
}

# ------------------------------ 输出 -------------------------------
base = os.path.splitext(os.path.basename(csv_file_path))[0]
root = os.path.join(os.getcwd(), "..", "..", "..", "results",
                    "clustered_data", "DBSCAN", algorithm_name,
                    f"clustered_{dataset_id}")
os.makedirs(root, exist_ok=True)

# 4 行固定格式文本
txt_path = os.path.join(root, f"{base}.txt")
with open(txt_path, "w", encoding="utf-8") as fh:
    fh.write("\n".join([
        f"Best parameters: n_components={best_params['min_samples']}, covariance type={best_params['eps']}",
        f"Final Combined Score: {combined_score}",
        f"Final Silhouette Score: {sil_score}",
        f"Final Davies-Bouldin Score: {db_score}"
    ]))

# 保存核心统计
core_path = os.path.join(root, f"{base}_{clean_state}_core_stats.json")
with open(core_path, "w", encoding="utf-8") as fp:
    json.dump(core_stats, fp, indent=4)

# -------------------------- param-shift -----------------------------
other_state = "cleaned" if clean_state == "raw" else "raw"
other_path  = os.path.join(root, f"{base}_{other_state}_core_stats.json")
if os.path.exists(other_path):
    with open(other_path) as fp:
        other = json.load(fp)

    shift = {
        "dataset_id": dataset_id,
        "delta_eps": best_params["eps"] - float(other.get("covariance type", 0)),
        "delta_min_samples": best_params["min_samples"] - int(other.get("n_components", 0)),
        "delta_combined": combined_score - float(other.get("combined_score", 0))
    }
    shift_path = os.path.join(root, f"{base}_param_shift.json")
    with open(shift_path, "w", encoding="utf-8") as fp:
        json.dump(shift, fp, indent=4)

print(f"All files saved in: {root}")
print(f"Program completed in {time.time() - start_time:.2f} sec")

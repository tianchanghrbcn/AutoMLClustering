#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Agglomerative Clustering (HC) with full merge-tree tracking
· 新目标函数: α·Sil + β·DB^{-1} + γ·(1-SSE/SSE_max) ， α+β+γ=1
· 额外记录 h_max —— merge-tree 的最终合并高度
· 其余输入 / 输出行为与旧版完全一致
"""
import os, time, math, json
import numpy as np
import pandas as pd
import optuna
from pathlib import Path
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (silhouette_score,
                             davies_bouldin_score,
                             pairwise_distances)

# --------------------------------------------------
# 0. 环境变量
# --------------------------------------------------
csv_file_path  = os.getenv("CSV_FILE_PATH")
dataset_id     = os.getenv("DATASET_ID")
algorithm_name = os.getenv("ALGO")
clean_state    = os.getenv("CLEAN_STATE", "raw")   # raw / cleaned

if not csv_file_path:
    raise SystemExit("CSV_FILE_PATH not provided")
csv_file_path = os.path.normpath(csv_file_path)

# --------------------------------------------------
# 1. 读取与预处理
# --------------------------------------------------
df = pd.read_csv(csv_file_path)
excluded = [c for c in df.columns if 'id' in c.lower()]
X = df[df.columns.difference(excluded)].copy()

for col in X.columns:
    if X[col].dtype in ("object", "category"):
        X[col] = X[col].map(X[col].value_counts(normalize=True))
X = X.dropna()
X_scaled = StandardScaler().fit_transform(X)

# ------------------ 目标函数权重 (α+β+γ=1) ------------------
alpha = 0.35
beta  = 0.65
gamma = 1.0 - alpha - beta     # 自动补足到 1

# 预计算全局 SSE_max（所有样本视为一个簇）
global_centroid = X_scaled.mean(axis=0)
SSE_max = float(np.sum((X_scaled - global_centroid) ** 2))

start_time = time.time()

# --------------------------------------------------
# 2. HC + merge-tree 追踪函数
# --------------------------------------------------
def _sse(labels: np.ndarray) -> float:
    """聚类方案的 SSE（层次聚类没有噪声标签）"""
    sse = 0.0
    for lbl in np.unique(labels):
        pts = X_scaled[labels == lbl]
        centroid = pts.mean(axis=0)
        sse += np.sum((pts - centroid) ** 2)
    return float(sse)

def combined(db, sil, sse):
    """α·Sil + β·DB^{-1} + γ·(1 - SSE/SSE_max)"""
    db = max(db, 1e-6)
    return alpha * sil + beta * (1.0 / db) + gamma * (1.0 - sse / SSE_max)

def run_hc_tracking(k: int, linkage: str, metric: str):
    """返回 labels, merge_history(list[dict]), core_stats"""
    hc = AgglomerativeClustering(
        n_clusters=k,
        linkage=linkage,
        affinity=metric,
        compute_distances=True          # sklearn ≥1.2
    )
    labels = hc.fit_predict(X_scaled)

    merges = []
    if hasattr(hc, "children_") and hasattr(hc, "distances_"):
        for step, (i, j, d) in enumerate(
                zip(hc.children_[:, 0], hc.children_[:, 1], hc.distances_)):
            merges.append({"step": int(step + 1),
                           "cluster_i": int(i),
                           "cluster_j": int(j),
                           "dist": float(d)})

    # --- core / border 稳定度（近似） ---
    dist_mat = pairwise_distances(X_scaled, metric="euclidean")
    intra, inter, cnt_intra, cnt_inter = 0.0, 0.0, 0, 0
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if labels[i] == labels[j]:
                intra += dist_mat[i, j]; cnt_intra += 1
            else:
                inter += dist_mat[i, j]; cnt_inter += 1
    intra_mean = intra / max(cnt_intra, 1)
    inter_mean = inter / max(cnt_inter, 1)
    core_stats = {"intra_dist_mean": intra_mean,
                  "inter_dist_mean": inter_mean,
                  "ratio_intra_inter": intra_mean / (inter_mean + 1e-12)}
    return labels, merges, core_stats

# --------------------------------------------------
# 3. Optuna 超参数搜索
# --------------------------------------------------
optuna_trials = []

def objective(trial):
    k = trial.suggest_int("n_clusters", 5, max(5, math.isqrt(X.shape[0])))
    linkage = trial.suggest_categorical("linkage",
                                        ["ward", "complete", "average", "single"])
    metric = trial.suggest_categorical("metric",
                                       ["euclidean", "manhattan", "cosine"])
    if linkage == "ward" and metric != "euclidean":
        raise optuna.exceptions.TrialPruned()

    labels, merges, core_stats = run_hc_tracking(k, linkage, metric)
    sil = silhouette_score(X_scaled, labels)
    db  = davies_bouldin_score(X_scaled, labels)
    sse = _sse(labels)
    comb = combined(db, sil, sse)
    h_max = merges[-1]["dist"] if merges else 0.0

    optuna_trials.append({
        "trial_number": trial.number,
        "n_clusters": k,
        "linkage": linkage,
        "metric": metric,
        "combined_score": comb,
        "silhouette": sil,
        "davies_bouldin": db,
        "sse": sse,
        "n_merge_steps": len(merges),
        "h_max": h_max,                  # ★ 记录高度
        **core_stats
    })
    return comb

optuna.create_study(direction="maximize").optimize(objective, n_trials=100)
best = max(optuna_trials, key=lambda d: d["combined_score"])

# --------------------------------------------------
# 4. 最优模型 + merge 记录
# --------------------------------------------------
final_best_k   = best["n_clusters"]
linkage_optuna = best["linkage"]
metric_optuna  = best["metric"]
best_cov_type  = f"{linkage_optuna}-{metric_optuna}"   # 占位，与文本输出保持一致

labels_final, merge_history, core_stats_final = run_hc_tracking(
    final_best_k, linkage_optuna, metric_optuna)

final_db   = davies_bouldin_score(X_scaled, labels_final)
final_sil  = silhouette_score(X_scaled, labels_final)
final_sse  = _sse(labels_final)
final_comb = combined(final_db, final_sil, final_sse)
final_hmax = merge_history[-1]["dist"] if merge_history else 0.0    # ★

# --------------------------------------------------
# 5. 输出
# --------------------------------------------------
base = Path(csv_file_path).stem
root = Path.cwd() / ".." / ".." / ".." / "results" / "clustered_data" / "HC" / algorithm_name / f"clustered_{dataset_id}"
root.mkdir(parents=True, exist_ok=True)

# 5-1 TXT（4 行固定，不新增字段）
(root / f"{base}.txt").write_text(
    "\n".join([
        f"Best parameters: n_components={final_best_k}, covariance type={best_cov_type}",
        f"Final Combined Score: {final_comb}",
        f"Final Silhouette Score: {final_sil}",
        f"Final Davies-Bouldin Score: {final_db}"
    ]),
    encoding="utf-8"
)

# 5-2 JSON: merge history & summary
with open(root / f"{base}_{clean_state}_merge_history.json", "w", encoding="utf-8") as fp:
    json.dump(merge_history, fp, indent=4)

summary = {
    "clean_state": clean_state,
    "best_k": final_best_k,
    "linkage": linkage_optuna,
    "metric": metric_optuna,
    "combined": final_comb,
    "silhouette": final_sil,
    "davies_bouldin": final_db,
    "sse": final_sse,
    "weights": {"alpha": alpha, "beta": beta, "gamma": gamma},
    **core_stats_final,
    "n_merge_steps": len(merge_history),
    "h_max": final_hmax,                 # ★ 新增
    "total_runtime_sec": time.time() - start_time
}
with open(root / f"{base}_{clean_state}_summary.json", "w", encoding="utf-8") as fp:
    json.dump(summary, fp, indent=4)

# 5-3 Δk / Δcombined 偏移（若另一版本存在）
other_state = "cleaned" if clean_state == "raw" else "raw"
other_path  = root / f"{base}_{other_state}_summary.json"
if other_path.exists():
    other = json.loads(other_path.read_text(encoding="utf-8"))
    shift = {
        "dataset_id": dataset_id,
        "delta_k": summary["best_k"] - other["best_k"],
        "delta_combined": summary["combined"] - other["combined"],
        "rel_shift": abs(summary["best_k"] - other["best_k"]) / max(other["best_k"], 1)
    }
    with open(root / f"{base}_param_shift.json", "w", encoding="utf-8") as fp:
        json.dump(shift, fp, indent=4)

print(f"All files saved in: {root}")
print(f"Program completed in {summary['total_runtime_sec']:.2f} sec")

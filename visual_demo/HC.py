#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Agglomerative Clustering (HC) with full merge-tree tracking
· 新目标函数: α·Sil + β·DB^{-1} + γ·(1-SSE/SSE_max) ， α+β+γ=1
· 额外记录 h_max —— merge-tree 的最终合并高度
· 其余输入 / 输出行为与旧版完全一致
· NEW: 额外落盘 —— 最终簇标签、索引映射、试验全记录、全树谱系与SSE收敛曲线、X_scaled
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
# NEW: 记录被用于聚类的数据在原始df中的行索引（用于可视化/对齐）
used_index = X.index.copy()

X_scaled = StandardScaler().fit_transform(X)

# ------------------ 目标函数权重 (α+β=1) ------------------
alpha = 0.47
beta  = 1 - alpha
gamma = 0.00

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
    S = (sil + 1.0) / 2.0  # [-1,1] → [0,1]
    D = 1.0 / (1.0 + db)   # [0,∞) → (0,1]
    eps = 1e-12
    return 1.0 / (alpha / max(S, eps) + beta / max(D, eps))

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
    t0 = time.time()  # NEW: trial计时
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
        "trial_number": int(trial.number),
        "n_clusters": int(k),
        "linkage": str(linkage),
        "metric": str(metric),
        "combined_score": float(comb),
        "silhouette": float(sil),
        "davies_bouldin": float(db),
        "sse": float(sse),
        "n_merge_steps": int(len(merges)),
        "h_max": float(h_max),          # ★ 记录高度
        **{k: (float(v) if isinstance(v, (int, float, np.floating)) else float(v))
           for k, v in {
               "intra_dist_mean": core_stats["intra_dist_mean"],
               "inter_dist_mean": core_stats["inter_dist_mean"],
               "ratio_intra_inter": core_stats["ratio_intra_inter"]
           }.items()},
        "trial_duration_sec": float(time.time() - t0)  # NEW
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

# NEW: 计算每簇规模与中心（标准化空间）
unique_labels = np.unique(labels_final)
cluster_sizes = [int(np.sum(labels_final == c)) for c in unique_labels]
cluster_centers = []
for c in unique_labels:
    pts = X_scaled[labels_final == c]
    centroid = pts.mean(axis=0)
    cluster_centers.append([float(x) for x in centroid])

# --------------------------------------------------
# 5. 输出（保持原有I/O不变 + 追加新文件）
# --------------------------------------------------
base = Path(csv_file_path).stem
root = Path.cwd() / ".." / ".." / ".." / "results" / "clustered_data" / "HC" / algorithm_name / f"clustered_{dataset_id}"
root.mkdir(parents=True, exist_ok=True)

# 5-1 TXT（4 行固定，不新增字段） —— 原样保持
(Path(root) / f"{base}.txt").write_text(
    "\n".join([
        f"Best parameters: n_components={final_best_k}, covariance type={best_cov_type}",
        f"Final Combined Score: {final_comb}",
        f"Final Silhouette Score: {final_sil}",
        f"Final Davies-Bouldin Score: {final_db}"
    ]),
    encoding="utf-8"
)

# 5-2 JSON: merge history & summary —— 原样保持（但允许在 summary 中增添字段）
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
    "total_runtime_sec": time.time() - start_time,
    # NEW: 便于后续统计分析
    "label_order": [int(x) for x in unique_labels.tolist()],
    "cluster_sizes": cluster_sizes,
    "cluster_centers": cluster_centers
}
with open(root / f"{base}_{clean_state}_summary.json", "w", encoding="utf-8") as fp:
    json.dump(summary, fp, indent=4)

# 5-3 Δk / Δcombined 偏移（若另一版本存在） —— 原样保持
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

# -------------------- 以下为新增输出，不影响既有I/O --------------------

# NEW-1: 最终簇标签（点级） —— 用于散点/Alluvial/对齐
# 1) 完整行+簇标签
df_out = df.loc[used_index].copy()
df_out.insert(0, "orig_index", df_out.index)  # 保留原始索引
df_out["cluster"] = labels_final.astype(int)
df_out.to_csv(root / f"{base}_{clean_state}_clusters.csv", index=False, encoding="utf-8")

# 2) 精简映射（簇 -> 原始索引列表）
labels_map = {
    int(c): [ (int(ix) if isinstance(ix, (int, np.integer)) else str(ix))
              for ix in df_out.loc[df_out["cluster"] == c, "orig_index"].tolist() ]
    for c in unique_labels
}
with open(root / f"{base}_{clean_state}_labels.json", "w", encoding="utf-8") as fp:
    json.dump(labels_map, fp, indent=2)

# 3) 行位置 <-> 原索引 映射（绘图/对齐用）
index_map = pd.DataFrame({
    "row_pos": np.arange(len(used_index), dtype=int),
    "orig_index": used_index
})
index_map.to_csv(root / f"{base}_{clean_state}_index_map.csv", index=False, encoding="utf-8")

# NEW-2: Optuna 全部试验记录（便于绘 trial 曲线）
pd.DataFrame(optuna_trials).to_csv(root / f"{base}_{clean_state}_trials.csv",
                                   index=False, encoding="utf-8")
with open(root / f"{base}_{clean_state}_trials.json", "w", encoding="utf-8") as fp:
    json.dump(optuna_trials, fp, indent=2)

# NEW-3: 全量聚合树谱系 + SSE收敛曲线（不改变主流程；单独再拟合一次 full tree）
def _full_tree_profiles(linkage: str, metric: str):
    """
    使用 distance_threshold=0 拟合全树（n_clusters=None），
    计算每次合并后的 SSE_total 与对应的簇数 k。
    """
    hc_full = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=0.0,
        linkage=linkage,
        affinity=metric,
        compute_distances=True
    )
    hc_full.fit(X_scaled)
    children = hc_full.children_
    distances = getattr(hc_full, "distances_", None)
    n_samples, n_features = X_scaled.shape
    n_merges = children.shape[0]

    # 初始化各节点统计量（叶节点为样本）
    n_nodes = n_samples + n_merges
    counts = np.zeros(n_nodes, dtype=np.int64)
    sums   = np.zeros((n_nodes, n_features), dtype=float)
    sumsq  = np.zeros(n_nodes, dtype=float)
    sse_node = np.zeros(n_nodes, dtype=float)

    counts[:n_samples] = 1
    sums[:n_samples]   = X_scaled
    sumsq[:n_samples]  = np.einsum('ij,ij->i', X_scaled, X_scaled)

    sse_total_list, k_list, dist_list = [], [], []
    sse_total = 0.0

    for step in range(n_merges):
        a, b = int(children[step, 0]), int(children[step, 1])
        new_id = n_samples + step

        # 当前两个被合并簇的 SSE
        sse_a, sse_b = sse_node[a], sse_node[b]

        # 合并后统计量
        counts[new_id] = counts[a] + counts[b]
        sums[new_id]   = sums[a] + sums[b]
        sumsq[new_id]  = sumsq[a] + sumsq[b]

        # 新簇 SSE = sum||x||^2 - ||sum(x)||^2 / n
        sse_new = float(sumsq[new_id] - (sums[new_id] @ sums[new_id]) / max(counts[new_id], 1))
        sse_node[new_id] = sse_new

        # 更新全局 SSE_total
        sse_total = sse_total - sse_a - sse_b + sse_new

        # 记录（合并一步后，簇数 = n_samples - (step + 1)）
        sse_total_list.append(sse_total)
        k_list.append(n_samples - (step + 1))
        dist_list.append(float(distances[step]) if distances is not None else float("nan"))

    return {
        "children": children,
        "distances": distances if distances is not None else np.full(len(k_list), np.nan),
        "counts": counts,
        "k_by_step": np.array(k_list, dtype=int),
        "sse_by_step": np.array(sse_total_list, dtype=float),
        "merge_dist_by_step": np.array(dist_list, dtype=float)
    }

profiles = _full_tree_profiles(linkage_optuna, metric_optuna)

# 3-1: 存储谱系原始数组（npz）
np.savez(root / f"{base}_{clean_state}_tree.npz",
         children=profiles["children"],
         distances=profiles["distances"],
         counts=profiles["counts"])

# 3-2: 存储SSE收敛曲线（CSV）
profile_df = pd.DataFrame({
    "step": np.arange(1, len(profiles["k_by_step"]) + 1, dtype=int),
    "n_clusters": profiles["k_by_step"],
    "merge_distance": profiles["merge_dist_by_step"],
    "sse_total": profiles["sse_by_step"],
    "sse_ratio": profiles["sse_by_step"] / max(SSE_max, 1e-12)
})
profile_df.to_csv(root / f"{base}_{clean_state}_tree_profile.csv",
                  index=False, encoding="utf-8")

# NEW-4: 标准化后的特征矩阵（便于快速重绘与复现实验）
np.save(root / f"{base}_{clean_state}_X_scaled.npy", X_scaled)
with open(root / f"{base}_{clean_state}_columns.json", "w", encoding="utf-8") as fp:
    json.dump({"columns": X.columns.tolist()}, fp, indent=2)

print(f"All files saved in: {root}")
print(f"Program completed in {summary['total_runtime_sec']:.2f} sec")

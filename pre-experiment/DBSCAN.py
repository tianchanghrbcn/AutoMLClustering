#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DBSCAN with core/border/noise statistics (+ param-shift tracking)
α、β、γ 通过 CLI 传入 (--alpha --beta --gamma)，省略 γ 时自动补 1-α-β
★ 结果统一写入 <base>_summary.json（raw / cleaned 两个 section）
"""
import os, time, json, argparse, numpy as np
from pathlib import Path
import pandas as pd, optuna
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, pairwise_distances

# --------------------------- CLI：权重 --------------------------- #
cli = argparse.ArgumentParser()
cli.add_argument("--alpha", type=float, default=0.4)
cli.add_argument("--beta",  type=float, default=0.4)
cli.add_argument("--gamma", type=float, default=None)
cli.add_argument("--trials", type=int, default=150)
args, _ = cli.parse_known_args()

α, β = args.alpha, args.beta
γ = args.gamma if args.gamma is not None else 1.0 - α - β
if not np.isclose(α + β + γ, 1.0, atol=1e-6):
    raise ValueError(f"α+β+γ 必须等于 1 (得到 {α+β+γ})")

# --------------------------- 环境变量 --------------------------- #
csv_path  = os.getenv("CSV_FILE_PATH")
ds_id     = os.getenv("DATASET_ID")
algo_name = os.getenv("ALGO") or "DBSCAN"
clean_tag = os.getenv("CLEAN_STATE", "raw")   # raw / cleaned

if not csv_path:
    raise SystemExit("CSV_FILE_PATH env 未设置")
csv_path = os.path.normpath(csv_path)

# --------------------------- 读数据 ----------------------------- #
df = pd.read_csv(csv_path)
X  = df[df.columns.difference([c for c in df.columns if 'id' in c.lower()])].copy()
for c in X.columns:
    if X[c].dtype in ("object", "category"):
        X[c] = X[c].map(X[c].value_counts(normalize=True))
X = X.dropna()
X_std = StandardScaler().fit_transform(X)

centroid_all = X_std.mean(axis=0, keepdims=True)
SSE_max = float(np.sum((X_std - centroid_all) ** 2))
start_t = time.time()

# ------------------------- 评价函数 ----------------------------- #
def _sse(labels: np.ndarray) -> float:
    s = 0.0
    for lb in np.unique(labels):
        if lb == -1:       # noise
            continue
        pts = X_std[labels == lb]
        if pts.size:
            cen = pts.mean(axis=0, keepdims=True)
            s += np.sum((pts - cen) ** 2)
    return float(s)

def _evaluate(labels: np.ndarray):
    n_clusters = len(np.unique(labels)) - (1 if -1 in labels else 0)
    noise_r    = float((labels == -1).mean())
    if n_clusters < 2:
        return -np.inf, np.nan, np.nan, noise_r, np.nan
    sil = silhouette_score(X_std, labels)
    db  = max(davies_bouldin_score(X_std, labels), 1e-6)
    sse = _sse(labels)
    comb = α*sil + β*(1/db) + γ*(1 - sse/SSE_max)
    return comb, sil, db, noise_r, sse

# ---------------------- Optuna 搜索 ----------------------------- #
def _objective(tr):
    eps = tr.suggest_float("eps", 0.1, 2.0, step=0.05)
    ms  = tr.suggest_int("min_samples", 5, 50)
    lb  = DBSCAN(eps=eps, min_samples=ms).fit_predict(X_std)
    score, _, _, noise, _ = _evaluate(lb)
    return score * (1-noise)

study = optuna.create_study(direction="maximize")
study.optimize(_objective, n_trials=args.trials)
best = study.best_params

# ---------------------- 最终模型 ------------------------------- #
model  = DBSCAN(**best).fit(X_std)
labels = model.labels_
comb, sil, db, noise_r, sse = _evaluate(labels)

# 邻域统计
dmat = pairwise_distances(X_std)
core_mask = np.sum(dmat <= best["eps"], axis=1) >= best["min_samples"]
stats = {
    "core_count":   int(core_mask.sum()),
    "border_count": int((~core_mask & (labels!=-1)).sum()),
    "noise_count":  int((labels == -1).sum()),
    "noise_ratio":  noise_r,
    "neighbor_hist": np.bincount(np.clip(np.sum(dmat <= best["eps"], axis=1),0,49)).tolist(),
    "combined": comb, "silhouette": sil, "davies_bouldin": db, "sse": sse,
    "best_eps": best["eps"], "best_min_samples": best["min_samples"],
    "weights": {"alpha": α, "beta": β, "gamma": γ},
    "runtime_sec": time.time()-start_t
}

# ------------------------ 文件输出 ------------------------------ #
base = Path(csv_path).stem
out_dir = Path.cwd()/ "results" / "clustered_data" / "DBSCAN" / f"clustered_{ds_id}"
out_dir.mkdir(parents=True, exist_ok=True)

# ① TXT（4 行固定）
(out_dir/f"{base}.txt").write_text(
    "\n".join([
        f"Best parameters: n_components={best['min_samples']}, covariance type={best['eps']}",
        f"Final Combined Score: {comb}",
        f"Final Silhouette Score: {sil}",
        f"Final Davies-Bouldin Score: {db}"
    ]), encoding="utf-8")

# ② 统一 summary.json
sum_fp = out_dir/f"{base}_summary.json"
if sum_fp.exists():
    whole = json.loads(sum_fp.read_text())
else:
    whole = {}
whole[clean_tag] = stats
sum_fp.write_text(json.dumps(whole, indent=4), encoding="utf-8")

# ③ param-shift（若两个版本都有时写）
if {"raw","cleaned"}<=set(whole):
    a,b = whole["raw"], whole["cleaned"]
    shift = {
        "dataset_id": ds_id,
        "delta_eps":        a["best_eps"]        - b["best_eps"],
        "delta_min_samples":a["best_min_samples"]- b["best_min_samples"],
        "delta_combined":   a["combined"]        - b["combined"],
        "relative_shift":   abs(a["combined"]-b["combined"])/(abs(b["combined"])+1e-9)
    }
    (out_dir/f"{base}_param_shift.json").write_text(json.dumps(shift,indent=4),encoding="utf-8")

print(f"All files saved in: {out_dir}")
print(f"Program completed in {stats['runtime_sec']:.2f} sec")

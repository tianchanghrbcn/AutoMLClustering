#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improved K-Means (new formulation 统一版)

目标函数: α·Sil + β·DB^{-1} + γ·(1−SSE/SSE_max) (α+β+γ=1)
CLI: --alpha --beta [--gamma]  (γ 缺省 = 1-α-β)

文件输出：
  <base>.txt               # 8 行，人读用
  <base>_history.json      # Optuna + 历史
  <base>_summary.json      # {"raw": {...}, "cleaned": {...}}
  <base>_param_shift.json  # raw vs cleaned (若两者皆在 summary 中)
"""

import argparse, json, math, os, time
from pathlib import Path

import numpy as np, pandas as pd, optuna
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# ---------- CLI ---------- #
cli = argparse.ArgumentParser()
cli.add_argument("--alpha", type=float, default=0.4)
cli.add_argument("--beta",  type=float, default=0.4)
cli.add_argument("--gamma", type=float, default=None,
                 help="若省略则 γ = 1-α-β")
cli.add_argument("--trials", type=int, default=30)
args, _ = cli.parse_known_args()

α, β = args.alpha, args.beta
γ = args.gamma if args.gamma is not None else 1.0 - α - β
if not math.isclose(α + β + γ, 1.0, abs_tol=1e-6):
    raise ValueError(f"α+β+γ 必须 = 1 (当前 {α+β+γ})")

# ---------- 环境 ---------- #
csv_path  = os.getenv("CSV_FILE_PATH")
ds_id     = os.getenv("DATASET_ID")
algo_name = os.getenv("ALGO") or "KMEANSNF"
state_tag = os.getenv("CLEAN_STATE", "raw")          # raw | cleaned

if not csv_path:
    raise SystemExit("CSV_FILE_PATH env 未设置")
csv_path = os.path.normpath(csv_path)

# ---------- 读数据 ---------- #
df = pd.read_csv(csv_path)
X = df[df.columns.difference([c for c in df.columns if 'id' in c.lower()])].copy()
for c in X.columns:
    if X[c].dtype in ("object", "category"):
        X[c] = X[c].map(X[c].value_counts(normalize=True))
X = X.dropna()
X_std = StandardScaler().fit_transform(X)
Xt = X_std.T                                     # (d, n)  – 本算法按列样本

SSE_max = float(((X_std - X_std.mean(0)) ** 2).sum())
t0 = time.time()

# ---------- NF-K-Means ---------- #
def _indicator(lbl, k, n):
    F = np.zeros((n, k))
    F[np.arange(n), lbl] = 1.0
    return F

def kmeans_nf(Xt, k, max_iter=1000, inner_iter=100, tol=1e-4, seed=0):
    rng = np.random.default_rng(seed)
    n = Xt.shape[1]
    lbl = rng.integers(0, k, size=n)
    F = _indicator(lbl, k, n)
    A = Xt.T @ Xt
    s = np.ones(k)
    hist, prev = [], None

    for t in range(1, max_iter+1):
        # 更新 s_i
        for i in range(k):
            f = F[:, i]
            s[i] = np.sqrt(f.T @ A @ f) / (f.T @ f + 1e-10)

        # 内层标签更新
        for _ in range(inner_iter):
            M = np.empty((n, k))
            for j in range(k):
                f = F[:, j]
                tmp = A @ f
                M[:, j] = tmp / (math.sqrt(f.T @ tmp) + 1e-10)
            lbl_new = np.argmin((s**2) - 2*s*M, axis=1)
            if np.array_equal(lbl, lbl_new):
                break
            lbl = lbl_new
            F = _indicator(lbl, k, n)

        cent = Xt @ F / (np.sum(F, axis=0, keepdims=True) + 1e-10)
        if prev is not None:
            delta = float(np.linalg.norm(cent - prev))
            hist.append({"iter": t, "delta": delta,
                         "relative_delta": delta/(np.linalg.norm(prev)+1e-12)})
            if delta < tol:
                break
        prev = cent.copy()
        if t > 1 and hist[-1]["delta"] == 0.0:
            break

    sse = float(np.linalg.norm(Xt - cent @ F.T, 'fro')**2)
    return lbl, hist, len(hist)+1, sse

# ---------- 评价函数 ---------- #
def _combined(db, sil, sse):
    db = max(db, 1e-6)
    return α*sil + β*(1/db) + γ*(1 - sse/SSE_max)

def _make_rec(tno,k,lbl,hist,iters,sse):
    db  = davies_bouldin_score(X_std,lbl)
    sil = silhouette_score(X_std,lbl)
    ch  = calinski_harabasz_score(X_std,lbl)
    combo = _combined(db,sil,sse)
    auc  = float(sum(h["delta"] for h in hist))
    geo  = float(hist[-1]["delta"]/hist[0]["delta"]) if len(hist)>1 else 0.0
    return {"trial":tno,"k":k,"combined":combo,"silhouette":sil,
            "davies_bouldin":db,"calinski_harabasz":ch,"sse":sse,
            "iterations":iters,"history":hist,"auc_delta":auc,"geo_decay":geo}

# ---------- Optuna 搜索 ---------- #
records=[]
def _objective(tr):
    k = tr.suggest_int("k",5,max(5,int(math.sqrt(X_std.shape[0]))))
    lbl,hist,iters,sse = kmeans_nf(Xt,k)
    rec = _make_rec(tr.number,k,lbl,hist,iters,sse)
    records.append(rec)
    return rec["combined"]

optuna.create_study(direction="maximize").optimize(_objective,n_trials=args.trials)
best = max(records,key=lambda r:r["combined"])
k_final = best["k"]

# ---------- 最终模型 ---------- #
lbl_fin,hist_fin,it_fin,sse_fin = kmeans_nf(Xt,k_final)
fin_db  = davies_bouldin_score(X_std,lbl_fin)
fin_sil = silhouette_score(X_std,lbl_fin)
fin_ch  = calinski_harabasz_score(X_std,lbl_fin)
fin_comb = _combined(fin_db,fin_sil,sse_fin)

best_fin = best.copy()
best_fin.update({"history":hist_fin,"iterations":it_fin,
                 "sse":sse_fin,"silhouette":fin_sil,
                 "davies_bouldin":fin_db,"calinski_harabasz":fin_ch,
                 "combined":fin_comb})
records.append(best_fin)

# ---------- 输出 ---------- #
base = Path(csv_path).stem
out  = Path.cwd()/"results"/"clustered_data"/"KMEANSNF"/f"clustered_{ds_id}"
out.mkdir(parents=True,exist_ok=True)

# ① txt
(out/f"{base}.txt").write_text(
    "\n".join([
        f"Best parameters: k={k_final}",
        f"Number of clusters: {k_final}",
        f"Final Combined Score: {fin_comb}",
        f"Final Silhouette Score: {fin_sil}",
        f"Final Davies-Bouldin Score: {fin_db}",
        f"Calinski-Harabasz: {fin_ch}",
        f"Iterations to converge: {it_fin}",
        f"Final SSE: {sse_fin}"
    ]), encoding="utf-8")

# ② history
(out/f"{base}_history.json").write_text(json.dumps(records,indent=4),encoding="utf-8")

# ③ summary (统一结构)
sum_fp = out/f"{base}_summary.json"
sec = {
    "best_k":k_final,"combined":fin_comb,"silhouette":fin_sil,
    "davies_bouldin":fin_db,"calinski_harabasz":fin_ch,"sse":sse_fin,
    "iterations":it_fin,"weights":{"alpha":α,"beta":β,"gamma":γ},
    "total_runtime_sec":time.time()-t0,
    "n_trials":len(records),
    "avg_iterations":float(np.mean([r["iterations"] for r in records])),
    "median_iterations":float(np.median([r["iterations"] for r in records])),
    "avg_auc_delta":float(np.mean([r["auc_delta"] for r in records])),
    "avg_geo_decay":float(np.mean([r["geo_decay"] for r in records]))
}
if sum_fp.exists():
    whole=json.loads(sum_fp.read_text())
else:
    whole={}
whole[state_tag]=sec
sum_fp.write_text(json.dumps(whole,indent=4),encoding="utf-8")

# ④ param_shift
if {"raw","cleaned"}<=set(whole):
    r,c = whole["raw"], whole["cleaned"]
    shift = {
        "dataset_id": ds_id,
        "delta_k": r["best_k"] - c["best_k"],
        "delta_combined": r["combined"] - c["combined"],
        "rel_shift": abs(r["combined"]-c["combined"]) / (abs(c["combined"])+1e-9)
    }
    (out/f"{base}_param_shift.json").write_text(json.dumps(shift,indent=4),encoding="utf-8")

print(f"All files saved in: {out}")
print(f"Program completed in {time.time()-t0:.2f} sec")

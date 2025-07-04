#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
K-means (tracking) – 统一输出版
目标函数: α·Sil + β·DB^{-1} + γ·(1−SSE/SSE_max)，α+β+γ=1
CLI: --alpha --beta [--gamma]  (γ 缺省 = 1-α-β)
"""

import argparse, json, math, os, time
from pathlib import Path

import numpy as np, pandas as pd, optuna
from kneed import KneeLocator
from sklearn.cluster import kmeans_plusplus
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# ---------- CLI ---------- #
cli = argparse.ArgumentParser()
cli.add_argument("--alpha", type=float, default=0.4)
cli.add_argument("--beta",  type=float, default=0.4)
cli.add_argument("--gamma", type=float, default=None,
                 help="若省略则 γ = 1-α-β")
cli.add_argument("--trials", type=int, default=20)
args, _ = cli.parse_known_args()

α, β = args.alpha, args.beta
γ = args.gamma if args.gamma is not None else 1.0 - α - β
if not math.isclose(α + β + γ, 1.0, abs_tol=1e-6):
    raise ValueError(f"α+β+γ 必须为 1 (当前 {α+β+γ})")

# ---------- 环境 & 数据 ---------- #
csv_path  = os.getenv("CSV_FILE_PATH")
ds_id     = os.getenv("DATASET_ID")
algo_name = os.getenv("ALGO") or "KMEANS"
state_tag = os.getenv("CLEAN_STATE", "raw")          # raw | cleaned

if not csv_path:
    raise SystemExit("CSV_FILE_PATH env 未设置")
csv_path = os.path.normpath(csv_path)

df = pd.read_csv(csv_path)
X = df[df.columns.difference([c for c in df.columns if 'id' in c.lower()])].copy()
for c in X.columns:
    if X[c].dtype in ("object", "category"):
        X[c] = X[c].map(X[c].value_counts(normalize=True))
X = X.dropna()
X_std = StandardScaler().fit_transform(X)

SSE_max = float(((X_std - X_std.mean(axis=0, keepdims=True)) ** 2).sum())
t0 = time.time()

# ---------- K-means 实现 ---------- #
def kmeans_track(data, k, max_iter=300, tol=1e-4, seed=0):
    rng = np.random.default_rng(seed)
    centers, _ = kmeans_plusplus(data, n_clusters=k, random_state=seed)
    hist=[]
    for it in range(1, max_iter+1):
        dmat = np.linalg.norm(data[:,None,:]-centers[None,:,:], axis=2)
        lbl  = dmat.argmin(axis=1)
        new_centers = np.vstack([
            data[lbl==j].mean(axis=0) if (lbl==j).any()
            else data[rng.integers(0, data.shape[0])]
            for j in range(k)
        ])
        delta=float(np.linalg.norm(new_centers-centers))
        rel  = delta/(np.linalg.norm(centers)+1e-12)
        sse  = float(((data-new_centers[lbl])**2).sum())
        hist.append({"iter":it,"delta":delta,"relative_delta":rel,"sse":sse})
        if delta<tol:
            centers=new_centers; break
        centers=new_centers
    return lbl, hist, hist[-1]["iter"], hist[-1]["sse"]

def _score(db,sil,sse):
    db=max(db,1e-6)
    return α*sil + β*(1/db) + γ*(1 - sse/SSE_max)

records=[]
def _make_rec(tno,k,lbl,hist,iters,sse):
    db  = davies_bouldin_score(X_std,lbl)
    sil = silhouette_score(X_std,lbl)
    ch  = calinski_harabasz_score(X_std,lbl)
    combo=_score(db,sil,sse)
    auc = float(sum(h["delta"] for h in hist))
    decay=float(hist[-1]["delta"]/hist[0]["delta"]) if len(hist)>1 else 0.0
    return {"trial":tno,"k":k,"combined":combo,"silhouette":sil,
            "davies_bouldin":db,"calinski_harabasz":ch,"sse":sse,
            "iterations":iters,"history":hist,"auc_delta":auc,"geo_decay":decay}

# ---------- Optuna 初轮 ---------- #
def _objective(tr):
    k=tr.suggest_int("k",5,max(5,int(math.sqrt(X.shape[0]))))
    lbl,hist,iters,sse=kmeans_track(X_std,k)
    rec=_make_rec(tr.number,k,lbl,hist,iters,sse)
    records.append(rec)
    return rec["combined"]

optuna.create_study(direction="maximize").optimize(_objective,n_trials=args.trials)
best=max(records,key=lambda r:r["combined"])
k_opt=best["k"]

# ---------- Kneedle 微调 ---------- #
ks = range(2,int(math.sqrt(X.shape[0]))+1)
sse_curve=[kmeans_track(X_std,k,30,1e-3)[3] for k in ks]
try:
    knee=KneeLocator(ks,sse_curve,curve="convex",direction="decreasing").elbow
except ValueError:
    knee=None
if knee and knee!=k_opt:
    lo,hi=sorted([k_opt,knee])
    def _obj2(tr):
        k=tr.suggest_int("k",lo,hi)
        lbl,hist,iters,sse=kmeans_track(X_std,k)
        rec=_make_rec(tr.number,k,lbl,hist,iters,sse)
        records.append(rec)
        return rec["combined"]
    optuna.create_study(direction="maximize").optimize(_obj2,n_trials=10)
    best=max(records,key=lambda r:r["combined"])

# ---------- 最终模型 ---------- #
k_final=best["k"]
lbl_fin,hist_fin,it_fin,sse_fin=kmeans_track(X_std,k_final)
fin_db  = davies_bouldin_score(X_std,lbl_fin)
fin_sil = silhouette_score(X_std,lbl_fin)
fin_ch  = calinski_harabasz_score(X_std,lbl_fin)
fin_comb=_score(fin_db,fin_sil,sse_fin)

# ---------- 输出 ---------- #
base = Path(csv_path).stem
out  = Path.cwd()/"results"/"clustered_data"/"KMEANS"/f"clustered_{ds_id}"
out.mkdir(parents=True,exist_ok=True)

# ① txt（保持原 8 行格式）
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
    ]),encoding="utf-8")

# ② history
(out/f"{base}_history.json").write_text(json.dumps(records,indent=4),encoding="utf-8")

# ③ unified summary
sum_fp = out/f"{base}_summary.json"
sec = {
    "best_k":k_final,"combined":fin_comb,"silhouette":fin_sil,
    "davies_bouldin":fin_db,"calinski_harabasz":fin_ch,"sse":sse_fin,
    "iterations":it_fin,"weights":{"alpha":α,"beta":β,"gamma":γ},
    "total_runtime_sec":time.time()-t0,
    "n_trials":len(records),"avg_iterations":float(np.mean([r["iterations"] for r in records])),
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
    r,c=whole["raw"],whole["cleaned"]
    shift={
        "dataset_id":ds_id,
        "delta_k":r["best_k"]-c["best_k"],
        "delta_combined":r["combined"]-c["combined"],
        "rel_shift":abs(r["combined"]-c["combined"])/(abs(c["combined"])+1e-9)
    }
    (out/f"{base}_param_shift.json").write_text(json.dumps(shift,indent=4),encoding="utf-8")

print(f"All files saved in: {out}")
print(f"Program completed in {time.time()-t0:.2f} sec")

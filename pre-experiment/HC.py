#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Agglomerative Clustering (HC) with merge-tree tracking
目标函数: α·Sil + β·DB^{-1} + γ·(1-SSE/SSE_max)，α+β+γ=1
CLI: --alpha --beta [--gamma]  (γ 缺省 = 1-α-β)
所有结果统一写入 <base>_summary.json （raw / cleaned 两 section）
"""

import argparse, json, math, os, time
from pathlib import Path

import numpy as np, pandas as pd, optuna
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, pairwise_distances
from kneed import KneeLocator

# ---------- CLI ---------- #
cli = argparse.ArgumentParser()
cli.add_argument("--alpha", type=float, default=0.4)
cli.add_argument("--beta",  type=float, default=0.4)
cli.add_argument("--gamma", type=float, default=None,
                 help="缺省则 γ = 1-α-β")
cli.add_argument("--trials", type=int, default=100)
args, _ = cli.parse_known_args()

α, β = args.alpha, args.beta
γ = args.gamma if args.gamma is not None else 1.0 - α - β
if not math.isclose(α + β + γ, 1.0, abs_tol=1e-6):
    raise ValueError(f"α+β+γ 必须等于 1 (当前 {α+β+γ})")

# ---------- 环境变量 & 数据 ---------- #
csv_path  = os.getenv("CSV_FILE_PATH")
ds_id     = os.getenv("DATASET_ID")
algo_name = os.getenv("ALGO") or "HC"
state_tag = os.getenv("CLEAN_STATE", "raw")          # raw | cleaned

if not csv_path:
    raise SystemExit("CSV_FILE_PATH env 未设置")
csv_path = os.path.normpath(csv_path)

df = pd.read_csv(csv_path)
X = df[df.columns.difference([c for c in df.columns if 'id' in c.lower()])].copy()
for col in X.columns:
    if X[col].dtype in ("object", "category"):
        X[col] = X[col].map(X[col].value_counts(normalize=True))
X = X.dropna()
X_std = StandardScaler().fit_transform(X)

# ---------- 预计算 ---------- #
SSE_max = float(((X_std - X_std.mean(axis=0, keepdims=True)) ** 2).sum())
t0 = time.time()

# ---------- 工具函数 ---------- #
def _sse(lbl):
    s = 0.0
    for k in np.unique(lbl):
        pts = X_std[lbl == k]
        cen = pts.mean(axis=0, keepdims=True)
        s += ((pts - cen) ** 2).sum()
    return float(s)

def _score(db, sil, sse):
    db = max(db, 1e-6)
    return α*sil + β*(1/db) + γ*(1 - sse/SSE_max)

def _run_hc(k, lk, mt):
    hc = AgglomerativeClustering(
        n_clusters=k,
        linkage=lk,
        metric=mt,  # ← 用 metric，别再写 affinity
        compute_distances=True
    )

    lbl = hc.fit_predict(X_std)
    merges=[]
    if hasattr(hc,"children_"):
        for i,(a,b,d) in enumerate(zip(hc.children_[:,0],
                                       hc.children_[:,1],
                                       hc.distances_)):
            merges.append({"step":i+1,"i":int(a),"j":int(b),"dist":float(d)})
    # 简易 intra / inter
    dmat = pairwise_distances(X_std)
    tri  = np.triu_indices(len(lbl),1)
    intra = dmat[tri][(lbl[:,None]==lbl)[tri]]
    inter = dmat[tri][(lbl[:,None]!=lbl)[tri]]
    stats = {"intra_mean": float(intra.mean()) if intra.size else 0.0,
             "inter_mean": float(inter.mean()) if inter.size else 0.0}
    return lbl, merges, stats

# ---------- Optuna 搜索 ---------- #
records=[]
def _objective(tr):
    k = tr.suggest_int("k",5,max(5,int(math.sqrt(X.shape[0]))))
    lk= tr.suggest_categorical("linkage",["ward","complete","average","single"])
    mt= tr.suggest_categorical("metric",["euclidean","manhattan","cosine"])
    if lk=="ward" and mt!="euclidean":
        raise optuna.TrialPruned()
    lbl,merges,stats=_run_hc(k,lk,mt)
    sil,db,sse = silhouette_score(X_std,lbl), davies_bouldin_score(X_std,lbl), _sse(lbl)
    comb=_score(db,sil,sse)
    records.append({"trial":tr.number,"k":k,"linkage":lk,"metric":mt,
                    "combined":comb,"silhouette":sil,"davies_bouldin":db,
                    "sse":sse,"h_max":merges[-1]["dist"] if merges else 0.0,
                    "n_merge":len(merges),**stats})
    return comb

optuna.create_study(direction="maximize").optimize(_objective, n_trials=args.trials)
best = max(records,key=lambda r:r["combined"])

# ---------- Kneedle 微调 (可选) ---------- #
ks   = range(2,max(3,int(math.sqrt(X.shape[0])))+1)
sse_curve=[]
for k in ks:
    lbl,_,_=_run_hc(k,best["linkage"],best["metric"])
    sse_curve.append(_sse(lbl))
try:
    knee = KneeLocator(ks,sse_curve,curve="convex",direction="decreasing").elbow
except ValueError:
    knee=None
if knee and knee!=best["k"]:
    def _local(tr):
        k=tr.suggest_int("k",min(best["k"],knee),max(best["k"],knee))
        lbl,merges,stats=_run_hc(k,best["linkage"],best["metric"])
        sil,db,sse = silhouette_score(X_std,lbl), davies_bouldin_score(X_std,lbl), _sse(lbl)
        comb=_score(db,sil,sse)
        records.append({"trial":tr.number,"k":k,"linkage":best["linkage"],
                        "metric":best["metric"],"combined":comb,
                        "silhouette":sil,"davies_bouldin":db,"sse":sse,
                        "h_max":merges[-1]["dist"] if merges else 0.0,
                        "n_merge":len(merges),**stats})
        return comb
    optuna.create_study(direction="maximize").optimize(_local,n_trials=30)
    best=max(records,key=lambda r:r["combined"])

# ---------- 最终模型 ---------- #
lbl_fin, merges_fin, stats_fin = _run_hc(best["k"],best["linkage"],best["metric"])
fin_sil  = silhouette_score(X_std,lbl_fin)
fin_db   = davies_bouldin_score(X_std,lbl_fin)
fin_sse  = _sse(lbl_fin)
fin_comb = _score(fin_db,fin_sil,fin_sse)
fin_hmax = merges_fin[-1]["dist"] if merges_fin else 0.0

# ---------- 输出 ---------- #
base = Path(csv_path).stem
out  = Path.cwd()/"results"/"clustered_data"/"HC"/f"clustered_{ds_id}"
out.mkdir(parents=True,exist_ok=True)

# ① txt (4 行)
(out/f"{base}.txt").write_text(
    "\n".join([
        f"Best parameters: n_components={best['k']}, covariance type={best['linkage']}-{best['metric']}",
        f"Final Combined Score: {fin_comb}",
        f"Final Silhouette Score: {fin_sil}",
        f"Final Davies-Bouldin Score: {fin_db}"
    ]),encoding="utf-8")

# ② merge history
(out/f"{base}_{state_tag}_merge_history.json").write_text(
    json.dumps(merges_fin,indent=4),encoding="utf-8")

# ③ unified summary.json
sum_fp = out/f"{base}_summary.json"
sec = {
    "best_k":best["k"],"linkage":best["linkage"],"metric":best["metric"],
    "combined":fin_comb,"silhouette":fin_sil,"davies_bouldin":fin_db,
    "sse":fin_sse,"h_max":fin_hmax,"weights":{"alpha":α,"beta":β,"gamma":γ},
    **stats_fin,"runtime_sec":time.time()-t0
}
if sum_fp.exists():
    whole=json.loads(sum_fp.read_text())
else:
    whole={}
whole[state_tag]=sec
sum_fp.write_text(json.dumps(whole,indent=4),encoding="utf-8")

# ④ param_shift （若 raw & cleaned 均存在）
if {"raw","cleaned"}<=set(whole):
    r,c = whole["raw"],whole["cleaned"]
    shift = {
        "dataset_id":ds_id,
        "delta_k":r["best_k"]-c["best_k"],
        "delta_combined":r["combined"]-c["combined"],
        "rel_shift":abs(r["combined"]-c["combined"])/(abs(c["combined"])+1e-9)
    }
    (out/f"{base}_param_shift.json").write_text(json.dumps(shift,indent=4),encoding="utf-8")

print(f"All files saved in: {out}")
print(f"Program completed in {(time.time()-t0):.2f} sec")

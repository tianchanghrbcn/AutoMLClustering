#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
K-means – 调和平均综合指标版
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
cli.add_argument("--alpha", type=float, default=0.5)
cli.add_argument("--beta",  type=float, default=None)
cli.add_argument("--gamma", type=float, default=None)
cli.add_argument("--trials", type=int, default=20)
args, _ = cli.parse_known_args()
α = float(args.alpha)
if not (0.0 <= α <= 1.0):
    raise ValueError("--alpha 必须在 [0,1]")

# ---------- 环境 & 数据 ---------- #
csv_path = os.getenv("CSV_FILE_PATH")
ds_id    = os.getenv("DATASET_ID")
state_tag= os.getenv("CLEAN_STATE", "raw")
if not csv_path:
    raise SystemExit("CSV_FILE_PATH env 未设置")
csv_path = os.path.normpath(csv_path)

df = pd.read_csv(csv_path)
X  = df[df.columns.difference([c for c in df.columns if 'id' in c.lower()])].copy()
for c in X.columns:
    if X[c].dtype in ("object", "category"):
        X[c] = X[c].map(X[c].value_counts(normalize=True))
X = X.dropna()
X_std = StandardScaler().fit_transform(X)
t0 = time.time()

# ---------- 工具 ---------- #
EPS = 1e-12
def _ci_harm(sil, db):
    S = (sil + 1.0) / 2.0
    D = 1.0 / (1.0 + db)
    return 1.0 / (α / max(S, EPS) + (1.0 - α) / max(D, EPS))

def kmeans_track(data, k, max_iter=300, tol=1e-4, seed=0):
    rng = np.random.default_rng(seed)
    centers, _ = kmeans_plusplus(data, n_clusters=k, random_state=seed)
    hist=[]
    for it in range(1, max_iter+1):
        dmat = np.linalg.norm(data[:, None, :] - centers[None, :, :], axis=2)
        lbl  = dmat.argmin(axis=1)
        new_centers = np.vstack([
            data[lbl == j].mean(axis=0) if (lbl == j).any()
            else data[rng.integers(0, data.shape[0])]
            for j in range(k)
        ])
        delta = float(np.linalg.norm(new_centers - centers))
        rel   = delta / (np.linalg.norm(centers) + 1e-12)
        hist.append({"iter": it, "delta": delta, "relative_delta": rel})
        if delta < tol:
            centers = new_centers
            break
        centers = new_centers
    return lbl, hist, hist[-1]["iter"]

def _make_rec(tno, k, lbl, hist, iters):
    db  = davies_bouldin_score(X_std, lbl)
    sil = silhouette_score(X_std, lbl)
    ch  = calinski_harabasz_score(X_std, lbl)
    combo = _ci_harm(sil, db)
    auc   = float(sum(h["delta"] for h in hist))
    decay = float(hist[-1]["delta"]/hist[0]["delta"]) if len(hist) > 1 else 0.0
    return {"trial": tno, "k": k, "combined": combo, "silhouette": sil,
            "davies_bouldin": db, "calinski_harabasz": ch,
            "iterations": iters, "history": hist,
            "auc_delta": auc, "geo_decay": decay}

# ---------- Optuna ---------- #
records=[]
def _objective(tr):
    k = tr.suggest_int("k", 5, max(5, int(math.sqrt(X.shape[0]))))
    lbl, hist, iters = kmeans_track(X_std, k)
    rec = _make_rec(tr.number, k, lbl, hist, iters)
    records.append(rec)
    return rec["combined"]

optuna.create_study(direction="maximize").optimize(_objective, n_trials=args.trials)
best = max(records, key=lambda r: r["combined"])
k_opt = best["k"]

# ---------- Kneedle 微调 ---------- #
ks = range(2, int(math.sqrt(X.shape[0])) + 1)
wcss = []
for k in ks:
    lbl, _, _ = kmeans_track(X_std, k, max_iter=30, tol=1e-3)
    wcss.append(((X_std - X_std[lbl][:, None]).var(axis=0)).sum())
try:
    knee = KneeLocator(ks, wcss, curve="convex",
                       direction="decreasing").elbow
except ValueError:
    knee = None
if knee and knee != k_opt:
    lo, hi = sorted([k_opt, knee])
    def _obj2(tr):
        k = tr.suggest_int("k", lo, hi)
        lbl, hist, iters = kmeans_track(X_std, k)
        rec = _make_rec(tr.number, k, lbl, hist, iters)
        records.append(rec)
        return rec["combined"]
    optuna.create_study(direction="maximize").optimize(_obj2, n_trials=10)
    best = max(records, key=lambda r: r["combined"])

# ---------- 最终模型 ---------- #
k_final = best["k"]
lbl_fin, hist_fin, it_fin = kmeans_track(X_std, k_final)
fin_db  = davies_bouldin_score(X_std, lbl_fin)
fin_sil = silhouette_score(X_std, lbl_fin)
fin_ch  = calinski_harabasz_score(X_std, lbl_fin)
fin_comb = _ci_harm(fin_sil, fin_db)

# ---------- 输出 ---------- #
base = Path(csv_path).stem
out  = Path.cwd() / "results" / "clustered_data" / "KMEANS" / f"clustered_{ds_id}"
out.mkdir(parents=True, exist_ok=True)

(out / f"{base}.txt").write_text(
    "\n".join([
        f"Best parameters: k={k_final}",
        f"Number of clusters: {k_final}",
        f"Final Combined Score: {fin_comb}",
        f"Final Silhouette Score: {fin_sil}",
        f"Final Davies-Bouldin Score: {fin_db}",
        f"Calinski-Harabasz: {fin_ch}",
        f"Iterations to converge: {it_fin}"
    ]), encoding="utf-8")

(out / f"{base}_history.json").write_text(
    json.dumps(records, indent=4), encoding="utf-8")

sum_fp = out / f"{base}_summary.json"
sec = {"best_k": k_final, "combined": fin_comb,
       "silhouette": fin_sil, "davies_bouldin": fin_db,
       "calinski_harabasz": fin_ch, "iterations": it_fin,
       "weights": {"alpha": α}, "runtime_sec": time.time() - t0}
if sum_fp.exists():
    whole = json.loads(sum_fp.read_text())
else:
    whole = {}
whole["combined"] = fin_comb
whole[state_tag] = sec
sum_fp.write_text(json.dumps(whole, indent=4), encoding="utf-8")

print(f"All files saved in: {out}")
print(f"Program completed in {(time.time()-t0):.2f} sec")

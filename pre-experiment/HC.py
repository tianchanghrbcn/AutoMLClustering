#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Agglomerative Clustering – 调和平均综合指标版
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
cli.add_argument("--alpha", type=float, default=0.5)
cli.add_argument("--beta",  type=float, default=None)   # 占位
cli.add_argument("--gamma", type=float, default=None)   # 占位
cli.add_argument("--trials", type=int, default=100)
args, _ = cli.parse_known_args()
α = float(args.alpha)
if not (0.0 <= α <= 1.0):
    raise ValueError("--alpha 必须在 [0,1]")

# ---------- 环境 & 数据 ---------- #
csv_path  = os.getenv("CSV_FILE_PATH")
ds_id     = os.getenv("DATASET_ID")
state_tag = os.getenv("CLEAN_STATE", "raw")
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
t0 = time.time()

# ---------- 工具 ---------- #
EPS = 1e-12
def _ci_harm(sil, db):
    S = (sil + 1.0) / 2.0
    D = 1.0 / (1.0 + db)
    return 1.0 / (α / max(S, EPS) + (1.0 - α) / max(D, EPS))

def _run_hc(k, lk, mt):
    hc = AgglomerativeClustering(n_clusters=k, linkage=lk,
                                 metric=mt, compute_distances=True)
    lbl = hc.fit_predict(X_std)
    merges = [{"step": i + 1, "i": int(a), "j": int(b), "dist": float(d)}
              for i, (a, b, d) in enumerate(
                  zip(hc.children_[:, 0], hc.children_[:, 1], hc.distances_))] \
             if hasattr(hc, "children_") else []
    dmat = pairwise_distances(X_std)
    tri  = np.triu_indices(len(lbl), 1)
    intra = dmat[tri][(lbl[:, None] == lbl)[tri]]
    inter = dmat[tri][(lbl[:, None] != lbl)[tri]]
    stats = {"intra_mean": float(intra.mean()) if intra.size else 0.0,
             "inter_mean": float(inter.mean()) if inter.size else 0.0}
    return lbl, merges, stats

# ---------- Optuna 搜索 ---------- #
records = []
def _objective(tr):
    k  = tr.suggest_int("k", 5, max(5, int(math.sqrt(X.shape[0]))))
    lk = tr.suggest_categorical("linkage",
                                ["ward", "complete", "average", "single"])
    mt = tr.suggest_categorical("metric",
                                ["euclidean", "manhattan", "cosine"])
    if lk == "ward" and mt != "euclidean":
        raise optuna.TrialPruned()
    lbl, merges, stats = _run_hc(k, lk, mt)
    sil = silhouette_score(X_std, lbl)
    db  = davies_bouldin_score(X_std, lbl)
    comb = _ci_harm(sil, db)
    records.append({"trial": tr.number, "k": k, "linkage": lk, "metric": mt,
                    "combined": comb, "silhouette": sil, "davies_bouldin": db,
                    "h_max": merges[-1]["dist"] if merges else 0.0,
                    "n_merge": len(merges), **stats})
    return comb

optuna.create_study(direction="maximize").optimize(_objective, n_trials=args.trials)
best = max(records, key=lambda r: r["combined"])

# ---------- Kneedle 微调 ---------- #
ks, sse_curve = [], []
for k in range(2, max(3, int(math.sqrt(X.shape[0]))) + 1):
    lbl, _, _ = _run_hc(k, best["linkage"], best["metric"])
    sse_curve.append(((X_std - X_std[lbl][:, None]).var(axis=0)).sum())
    ks.append(k)
try:
    knee = KneeLocator(ks, sse_curve, curve="convex",
                       direction="decreasing").elbow
except ValueError:
    knee = None
if knee and knee != best["k"]:
    lo, hi = sorted([best["k"], knee])
    def _local(tr):
        k = tr.suggest_int("k", lo, hi)
        lbl, merges, stats = _run_hc(k, best["linkage"], best["metric"])
        sil = silhouette_score(X_std, lbl)
        db  = davies_bouldin_score(X_std, lbl)
        comb = _ci_harm(sil, db)
        records.append({"trial": tr.number, "k": k,
                        "linkage": best["linkage"], "metric": best["metric"],
                        "combined": comb, "silhouette": sil,
                        "davies_bouldin": db, **stats})
        return comb
    optuna.create_study(direction="maximize").optimize(_local, n_trials=30)
    best = max(records, key=lambda r: r["combined"])

# ---------- 最终模型 ---------- #
lbl_fin, merges_fin, stats_fin = _run_hc(best["k"], best["linkage"], best["metric"])
fin_sil = silhouette_score(X_std, lbl_fin)
fin_db  = davies_bouldin_score(X_std, lbl_fin)
fin_comb = _ci_harm(fin_sil, fin_db)
fin_hmax = merges_fin[-1]["dist"] if merges_fin else 0.0

# ---------- 输出 ---------- #
base = Path(csv_path).stem
out  = Path.cwd() / "results" / "clustered_data" / "HC" / f"clustered_{ds_id}"
out.mkdir(parents=True, exist_ok=True)

(out / f"{base}.txt").write_text(
    "\n".join([
        f"Best parameters: k={best['k']}, linkage-metric={best['linkage']}-{best['metric']}",
        f"Final Combined Score: {fin_comb}",
        f"Final Silhouette Score: {fin_sil}",
        f"Final Davies-Bouldin Score: {fin_db}"
    ]), encoding="utf-8")

(out / f"{base}_{state_tag}_merge_history.json").write_text(
    json.dumps(merges_fin, indent=4), encoding="utf-8")

sum_fp = out / f"{base}_summary.json"
sec = {"best_k": best["k"], "linkage": best["linkage"], "metric": best["metric"],
       "combined": fin_comb, "silhouette": fin_sil, "davies_bouldin": fin_db,
       "h_max": fin_hmax, **stats_fin, "weights": {"alpha": α},
       "runtime_sec": time.time() - t0}
if sum_fp.exists():
    whole = json.loads(sum_fp.read_text())
else:
    whole = {}
whole["combined"] = fin_comb
whole[state_tag] = sec
sum_fp.write_text(json.dumps(whole, indent=4), encoding="utf-8")

print(f"All files saved in: {out}")
print(f"Program completed in {(time.time()-t0):.2f} sec")

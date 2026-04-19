#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NF‑K‑Means – 调和平均综合指标版
"""

import argparse, json, math, os, time
from pathlib import Path
import numpy as np, pandas as pd, optuna
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# ---------- CLI ---------- #
cli = argparse.ArgumentParser()
cli.add_argument("--alpha", type=float, default=0.5)
cli.add_argument("--beta",  type=float, default=None)
cli.add_argument("--gamma", type=float, default=None)
cli.add_argument("--trials", type=int, default=30)
args, _ = cli.parse_known_args()
α = float(args.alpha)
if not (0.0 <= α <= 1.0):
    raise ValueError("--alpha 必须在 [0,1]")

# ---------- 环境 ---------- #
csv_path = os.getenv("CSV_FILE_PATH")
ds_id    = os.getenv("DATASET_ID")
state_tag= os.getenv("CLEAN_STATE", "raw")
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
Xt = X_std.T
t0 = time.time()

# ---------- 工具 ---------- #
EPS = 1e-12
def _ci_harm(sil, db):
    S = (sil + 1.0) / 2.0
    D = 1.0 / (1.0 + db)
    return 1.0 / (α / max(S, EPS) + (1.0 - α) / max(D, EPS))

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
    history, prev = [], None

    for t in range(1, max_iter + 1):
        for i in range(k):
            f = F[:, i]
            s[i] = np.sqrt(f.T @ A @ f) / (f.T @ f + 1e-10)

        for _ in range(inner_iter):
            M = np.empty((n, k))
            for j in range(k):
                f = F[:, j]
                tmp = A @ f
                M[:, j] = tmp / (math.sqrt(f.T @ tmp) + 1e-10)
            lbl_new = np.argmin((s**2) - 2 * s * M, axis=1)
            if np.array_equal(lbl, lbl_new):
                break
            lbl = lbl_new
            F = _indicator(lbl, k, n)

        cent = Xt @ F / (np.sum(F, axis=0, keepdims=True) + 1e-10)
        if prev is not None:
            delta = float(np.linalg.norm(cent - prev))
            history.append({"iter": t, "delta": delta})
            if delta < tol:
                break
        prev = cent.copy()
        if t > 1 and history[-1]["delta"] == 0.0:
            break

    return lbl, history, len(history) + 1

# ---------- Optuna ---------- #
records=[]
def _objective(tr):
    k = tr.suggest_int("k", 5, max(5, int(math.sqrt(X_std.shape[0]))))
    lbl, hist, iters = kmeans_nf(Xt, k)
    db  = davies_bouldin_score(X_std, lbl)
    sil = silhouette_score(X_std, lbl)
    ch  = calinski_harabasz_score(X_std, lbl)
    combo = _ci_harm(sil, db)
    rec = {"trial": tr.number, "k": k, "combined": combo, "silhouette": sil,
           "davies_bouldin": db, "calinski_harabasz": ch,
           "iterations": iters, "history": hist}
    records.append(rec)
    return combo

optuna.create_study(direction="maximize").optimize(_objective, n_trials=args.trials)
best = max(records, key=lambda r: r["combined"])
k_final = best["k"]

# ---------- 最终模型 ---------- #
lbl_fin, hist_fin, it_fin = kmeans_nf(Xt, k_final)
fin_db  = davies_bouldin_score(X_std, lbl_fin)
fin_sil = silhouette_score(X_std, lbl_fin)
fin_ch  = calinski_harabasz_score(X_std, lbl_fin)
fin_comb = _ci_harm(fin_sil, fin_db)

# ---------- 输出 ---------- #
base = Path(csv_path).stem
out  = Path.cwd() / "results" / "clustered_data" / "KMEANSNF" / f"clustered_{ds_id}"
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

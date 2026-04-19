#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
K‑MC² + Vanilla K‑means – 调和平均综合指标版
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

# ---------- 环境 & 数据 ---------- #
csv_path  = os.getenv("CSV_FILE_PATH")
ds_id     = os.getenv("DATASET_ID")
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

def k_mc2(X, k, m, rng):
    n = X.shape[0]
    centers = [X[rng.integers(n)]]
    for _ in range(1, k):
        x = X[rng.integers(n)]
        dx = np.min(np.sum((x - centers) ** 2, axis=1))
        for _ in range(m):
            y = X[rng.integers(n)]
            dy = np.min(np.sum((y - centers) ** 2, axis=1))
            if rng.random() < dy / dx:
                x, dx = y, dy
        centers.append(x)
    return np.vstack(centers)

def kmeans_track(X, init_centers, max_iter=300, tol=1e-4):
    rng = np.random.default_rng(0)
    centers = init_centers.copy()
    k = centers.shape[0]
    history = []

    for it in range(1, max_iter + 1):
        dmat = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        labels = dmat.argmin(axis=1)

        new_centers = np.vstack([
            X[labels == j].mean(0) if (labels == j).any()
            else X[rng.integers(X.shape[0])]
            for j in range(k)
        ])

        delta = float(np.linalg.norm(new_centers - centers))
        history.append({"iter": it, "delta": delta})
        if delta < tol:
            centers = new_centers
            break
        centers = new_centers

    return labels, centers, history

# ---------- Optuna ---------- #
records=[]
def objective(trial):
    k = trial.suggest_int("k", 5, max(5, int(math.sqrt(X_std.shape[0]))))
    m = trial.suggest_int("m", 100, 500)
    init = k_mc2(X_std, k, m, np.random.default_rng(trial.number))
    labels, _, hist = kmeans_track(X_std, init)

    db  = davies_bouldin_score(X_std, labels)
    sil = silhouette_score(X_std, labels)
    ch  = calinski_harabasz_score(X_std, labels)
    combo = _ci_harm(sil, db)

    rec = {"trial": trial.number, "k": k, "m": m,
           "iterations": len(hist), "history": hist,
           "silhouette": sil, "davies_bouldin": db,
           "calinski_harabasz": ch, "combined": combo}
    records.append(rec)
    return combo

optuna.create_study(direction="maximize").optimize(objective, n_trials=args.trials)
best = max(records, key=lambda r: r["combined"])
k_best, m_best = best["k"], best["m"]

# ---------- 最终模型 ---------- #
final_init = k_mc2(X_std, k_best, m_best, np.random.default_rng(42))
labels_fin, _, hist_fin = kmeans_track(X_std, final_init)

fin_db  = davies_bouldin_score(X_std, labels_fin)
fin_sil = silhouette_score(X_std, labels_fin)
fin_ch  = calinski_harabasz_score(X_std, labels_fin)
fin_comb = _ci_harm(fin_sil, fin_db)
fin_iters = len(hist_fin)

# ---------- 输出 ---------- #
root = (Path.cwd() / "results" / "clustered_data" /
        "KMEANSPPS" / f"clustered_{ds_id}")
root.mkdir(parents=True, exist_ok=True)
base = Path(csv_path).stem

(root / f"{base}.txt").write_text(
    "\n".join([
        f"Best parameters: k={k_best}, m={m_best}",
        f"Number of clusters: {k_best}",
        f"Final Combined Score: {fin_comb}",
        f"Final Silhouette Score: {fin_sil}",
        f"Final Davies-Bouldin Score: {fin_db}",
        f"Calinski-Harabasz: {fin_ch}",
        f"Iterations to converge: {fin_iters}"
    ]), encoding="utf-8")

(root / f"{base}_centroid_history.json").write_text(
    json.dumps(records + [{
        "trial": "final", "k": k_best, "m": m_best,
        "iterations": fin_iters, "history": hist_fin,
        "silhouette": fin_sil, "davies_bouldin": fin_db,
        "calinski_harabasz": fin_ch, "combined": fin_comb
    }], indent=4), encoding="utf-8")

summary = {
    "best_k": k_best, "combined": fin_comb,
    "silhouette": fin_sil, "davies_bouldin": fin_db,
    "calinski_harabasz": fin_ch, "iterations": fin_iters,
    "weights": {"alpha": α}, "total_runtime_sec": time.time() - t0
}
(root / f"{base}_summary.json").write_text(
    json.dumps({"combined": fin_comb, "raw": summary}, indent=4), encoding="utf-8")

print(f"All files saved in: {root}")
print(f"Program completed in {(time.time()-t0):.2f} sec")

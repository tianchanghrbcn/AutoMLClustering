#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GMM clustering  –  调和平均综合指标版
CI_harm = 1 / (α / S + (1-α) / D)
          S = (Sil+1)/2,  D = 1/(1+DB)
"""

import argparse, json, math, os, time
from pathlib import Path
import numpy as np, pandas as pd, optuna
from kneed import KneeLocator
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score

# -------- CLI -------- #
cli = argparse.ArgumentParser()
cli.add_argument("--alpha", type=float, default=0.5)
cli.add_argument("--beta",  type=float, default=None)   # 占位
cli.add_argument("--gamma", type=float, default=None)   # 占位
cli.add_argument("--trials", type=int, default=20)
args, _ = cli.parse_known_args()
α = float(args.alpha)
if not (0.0 <= α <= 1.0):
    raise ValueError("--alpha 必须在 [0,1]")

# -------- 环境变量 / 读数据 -------- #
csv_path  = os.getenv("CSV_FILE_PATH")
ds_id     = os.getenv("DATASET_ID")
clean_tag = os.getenv("CLEAN_STATE", "raw")
if not csv_path:
    raise SystemExit("CSV_FILE_PATH env 未设置")
csv_path = os.path.normpath(csv_path)

df = pd.read_csv(csv_path)
X  = df[df.columns.difference([c for c in df.columns if 'id' in c.lower()])].copy()
for col in X.columns:
    if X[col].dtype in ("object", "category"):
        X[col] = X[col].map(X[col].value_counts(normalize=True))
X = X.dropna()
X_std = StandardScaler().fit_transform(X)
start_t = time.time()

# -------- 工具函数 -------- #
EPS = 1e-12
def _ci_harm(sil: float, db: float) -> float:
    S = (sil + 1.0) / 2.0
    D = 1.0 / (1.0 + db)
    return 1.0 / (α / max(S, EPS) + (1.0 - α) / max(D, EPS))

def _sse(lbl):
    s = 0.0
    for k in np.unique(lbl):
        pts = X_std[lbl == k]
        if pts.size:
            cen = pts.mean(axis=0, keepdims=True)
            s += ((pts - cen) ** 2).sum()
    return float(s)

def _gmm_track(k, cov):
    gmm = GaussianMixture(n_components=k, covariance_type=cov,
                          max_iter=1, warm_start=True, random_state=0)
    lbs = []
    for _ in range(300):
        gmm.fit(X_std)
        lbs.append(gmm.lower_bound_)
        if gmm.converged_:
            break
    lbl = gmm.predict(X_std)
    return lbl, gmm.n_iter_, lbs, gmm

def _record(tno, k, cov, lbl, n_iter, lbs):
    db  = davies_bouldin_score(X_std, lbl)
    sil = silhouette_score(X_std, lbl)
    sse = _sse(lbl)
    combo = _ci_harm(sil, db)
    auc   = float(np.trapz(lbs)) if len(lbs) > 1 else 0.0
    decay = float(abs(lbs[-1] - lbs[0]) /
                  max(abs(lbs[0]), 1e-12)) if len(lbs) > 1 else 0.0
    return {"trial": tno, "k": k, "cov": cov, "combined": combo,
            "silhouette": sil, "db": db, "sse": sse, "n_iter": n_iter,
            "ll_curve": lbs, "auc_ll": auc, "ll_decay": decay}

# -------- Optuna 粗搜索 -------- #
records = []
def _objective(tr):
    k   = tr.suggest_int("n_components", 5, max(5, int(math.sqrt(X.shape[0]))))
    cov = tr.suggest_categorical("cov_type",
                                 ["full", "tied", "diag", "spherical"])
    lbl, it, lbs, _ = _gmm_track(k, cov)
    rec = _record(tr.number, k, cov, lbl, it, lbs)
    records.append(rec)
    return rec["combined"]

optuna.create_study(direction="maximize").optimize(_objective, n_trials=args.trials)
best = max(records, key=lambda r: r["combined"])
k_best, cov_best = best["k"], best["cov"]

# -------- Kneedle 微调 -------- #
ks, nll = [], []
for k in range(2, max(3, int(math.sqrt(X.shape[0]))) + 1):
    _, _, _, g = _gmm_track(k, cov_best)
    ks.append(k); nll.append(-g.score(X_std) * len(X_std))
try:
    knee = KneeLocator(ks, nll, curve="convex",
                       direction="decreasing").elbow
except ValueError:
    knee = None
k_low, k_high = sorted([k_best, knee]) if knee else (k_best, k_best + 2)

def _local(tr):
    k   = tr.suggest_int("n_components", k_low, k_high)
    lbl, it, lbs, _ = _gmm_track(k, cov_best)
    rec = _record(tr.number, k, cov_best, lbl, it, lbs)
    records.append(rec)
    return rec["combined"]

optuna.create_study(direction="maximize").optimize(_local, n_trials=10)
best = max(records, key=lambda r: r["combined"])

# -------- 最终模型 -------- #
lbl_fin, it_fin, lbs_fin, _ = _gmm_track(best["k"], cov_best)
fin_db  = davies_bouldin_score(X_std, lbl_fin)
fin_sil = silhouette_score(X_std, lbl_fin)
fin_sse = _sse(lbl_fin)
fin_comb = _ci_harm(fin_sil, fin_db)

# -------- 输出 -------- #
base = Path(csv_path).stem
out  = Path.cwd() / "results" / "clustered_data" / "GMM" / f"clustered_{ds_id}"
out.mkdir(parents=True, exist_ok=True)

(out / f"{base}.txt").write_text(
    "\n".join([
        f"Best parameters: n_components={best['k']}, covariance type={cov_best}",
        f"Final Combined Score: {fin_comb}",
        f"Final Silhouette Score: {fin_sil}",
        f"Final Davies-Bouldin Score: {fin_db}"
    ]), encoding="utf-8")

(out / f"{base}_{clean_tag}_gmm_history.json").write_text(
    json.dumps(records, indent=4), encoding="utf-8")

summary_fp = out / f"{base}_summary.json"
summary_sec = {
    "best_k": best["k"], "cov_type": cov_best, "combined": fin_comb,
    "silhouette": fin_sil, "davies_bouldin": fin_db, "sse": fin_sse,
    "weights": {"alpha": α}, "n_iter_final": it_fin,
    "ll_curve_final": lbs_fin, "runtime_sec": time.time() - start_t
}
if summary_fp.exists():
    whole = json.loads(summary_fp.read_text())
else:
    whole = {}
whole["combined"] = fin_comb      # 顶层冗余
whole[clean_tag] = summary_sec
summary_fp.write_text(json.dumps(whole, indent=4), encoding="utf-8")

print(f"All files saved in: {out}")
print(f"Program completed in {(time.time() - start_t):.2f} sec")

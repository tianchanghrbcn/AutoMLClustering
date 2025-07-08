#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
search.py · Hierarchical UCB search  (batch mode)

自动遍历:
    ../../../datasets/train/{ds}/{ds}_{num}.csv
        ds   ∈ {beers, flights, hospital, rayyan}
        num  ∈ 1 … 15
其余逻辑（特征构造、UCB-Top-k-ε 剪枝等）与论文算法 1/2 完全一致。
"""

import json, joblib, math, os, sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import heapq

# ------------------ 全局常量（原默认值） ------------------
UCB_COEF = 1.96   # --ucb
TOP_K    = 5      # --topk
EPS_PRUNE= 0.005  # --eps

# ---------- 1. util: data quality ----------
def dataset_quality(df: pd.DataFrame):
    miss = df.isna().values.mean() * 100.0
    num = df.select_dtypes(include=["number"])
    if num.empty:
        anom = 0.0
    else:
        z = (num - num.mean()) / num.std(ddof=0)
        anom = (np.abs(z) > 3).values.mean() * 100.0
    return miss, anom, *df.shape  # (miss, anomaly, m, n)

# ---------- 2. helpers ----------
CLUSTER_MAP = {
    "KMEANS": "kmeans", "KMEANSNF": "kmeans", "KMEANSPPS": "kmeans",
    "GMM": "kmeans", "HC": "hierarchical", "DBSCAN": "dbscan",
}
def safe_float(x, default=0.0):
    return default if (x is None or (isinstance(x, float) and not math.isfinite(x))) else float(x)

def build_row(cleaner, clus_key, k, eps, minpts, miss, anom, m, n, cleaners):
    clus = CLUSTER_MAP[clus_key]
    row = {
        "missing": miss, "anomaly": anom,
        "log_m": math.log10(m + 1), "log_d": math.log10(n + 1),
        "k": safe_float(k), "eps": safe_float(eps), "minPts": safe_float(minpts),
        "miss_kmeans": miss if clus == "kmeans" else 0.0,
        "miss_dbscan": miss if clus == "dbscan" else 0.0,
        "miss_hierarchical": miss if clus == "hierarchical" else 0.0,
        "out_kmeans": anom if clus == "kmeans" else 0.0,
        "out_dbscan": anom if clus == "dbscan" else 0.0,
        "out_hierarchical": anom if clus == "hierarchical" else 0.0,
        "cluster_method": clus,
    }
    row.update({c: int(c == cleaner) for c in cleaners})
    return row

# ---------- 3. search node ----------
@dataclass(order=True)
class Node:
    sort_index: float = field(init=False, repr=False)
    ucb: float
    cleaner: str
    cluster: str
    k: float
    eps: float
    minpts: float
    level: int
    mu: float
    sigma: float
    def __post_init__(self):
        object.__setattr__(self, "sort_index", -self.ucb)

# ---------- 4. prediction helpers ----------
def make_feature_df(row_dict: dict, base_cols: List[str]) -> pd.DataFrame:
    df = pd.DataFrame([row_dict])
    for c in base_cols:
        if c not in df:
            df[c] = 0
    return df[base_cols]

def predict_stats(row_dict, base_cols, preproc, folds, cache):
    key = tuple(row_dict.values())
    if key not in cache:
        df = make_feature_df(row_dict, base_cols)
        X23 = df
        X37 = preproc.transform(df)
        vals = []
        for mdl in folds:
            try:
                vals.append(float(mdl.predict(X23)))
            except Exception:
                vals.append(float(mdl.predict(X37)))
        vals = np.array(vals)
        cache[key] = (vals.mean(), vals.std(ddof=1))
    return cache[key]

# ---------- 5. single-file search ----------
def run_search(csv_path: Path, CLEANERS, CLUSTERS, K_LIST, EPS_LIST, MIN_LIST,
               base_cols, preproc, folds, f_min, f_max):
    print(f"\n=== Processing {csv_path} ===")
    try:
        df_raw = pd.read_csv(csv_path)
    except Exception:
        try:
            df_raw = pd.read_excel(csv_path)
        except Exception as e:
            print(f"[WARN] Skip (cannot read): {e}")
            return
    miss, anom, m, n = dataset_quality(df_raw)
    print(f"  Data: m={m:,}, n={n}, missing={miss:.2f}%, anomaly={anom:.2f}%")

    cache, heap, leaf_nodes = {}, [], []
    best_mu = -float("inf")

    # L0
    for cln in CLEANERS:
        for clu in CLUSTERS:
            row = build_row(cln, clu, np.nan, np.nan, np.nan, miss, anom, m, n, CLEANERS)
            mu, sigma = predict_stats(row, base_cols, preproc, folds, cache)
            ucb = mu + UCB_COEF * sigma
            heapq.heappush(heap, Node(ucb, cln, clu, np.nan, np.nan, np.nan, 0, mu, sigma))

    # search
    while heap:
        node = heapq.heappop(heap)
        if node.ucb < best_mu - EPS_PRUNE:
            continue
        if node.level == 1:
            leaf_nodes.append(node)
            if node.mu > best_mu:
                best_mu = node.mu
            continue
        children = []
        if node.cluster == "DBSCAN":
            for eps in EPS_LIST:
                for mp in MIN_LIST:
                    row = build_row(node.cleaner, node.cluster, np.nan, eps, mp, miss, anom, m, n, CLEANERS)
                    mu, sigma = predict_stats(row, base_cols, preproc, folds, cache)
                    ucb = mu + UCB_COEF * sigma
                    children.append(Node(ucb, node.cleaner, node.cluster, np.nan, eps, mp, 1, mu, sigma))
        else:
            for k in K_LIST:
                row = build_row(node.cleaner, node.cluster, k, np.nan, np.nan, miss, anom, m, n, CLEANERS)
                mu, sigma = predict_stats(row, base_cols, preproc, folds, cache)
                ucb = mu + UCB_COEF * sigma
                children.append(Node(ucb, node.cleaner, node.cluster, k, np.nan, np.nan, 1, mu, sigma))
        children.sort(key=lambda n: n.ucb, reverse=True)
        for ch in children[:TOP_K]:
            if ch.ucb >= best_mu - EPS_PRUNE:
                heapq.heappush(heap, ch)

    if not leaf_nodes:
        print("  [WARN] No candidates kept after pruning.")
        return
    leaf_nodes.sort(key=lambda n: n.ucb, reverse=True)
    top = leaf_nodes[:10]
    print("  Top-10 by UCB:")
    print(f"  {'rk':<3}{'cleaner':<12}{'cluster':<12}{'k':>4}{'eps':>6}{'minPts':>7}{'μ̂':>8}{'σ':>6}{'UCB':>8}")
    for i, n in enumerate(top, 1):
        print(f"  {i:<3}{n.cleaner:<12}{n.cluster:<12}"
              f"{'' if math.isnan(n.k) else int(n.k):>4}"
              f"{'' if math.isnan(n.eps) else f'{n.eps:>6.2f}'}"
              f"{'' if math.isnan(n.minpts) else int(n.minpts):>7}"
              f"{n.mu:>8.3f}{n.sigma:>6.3f}{n.ucb:>8.3f}")

    best = max(leaf_nodes, key=lambda n: n.mu)
    real_score = best.mu * (f_max - f_min) + f_min
    print("  >>> Best pipeline:",
          best.cleaner, best.cluster,
          f"k={int(best.k)}"   if not math.isnan(best.k)   else "",
          f"eps={best.eps:.3f}" if not math.isnan(best.eps) else "",
          f"minPts={int(best.minpts)}" if not math.isnan(best.minpts) else "",
          f"score={real_score:.4f}")

# ---------- 6. bootstrap & batch loop ----------
def main():
    here = Path(__file__).resolve().parent
    models_dir = here / "models"
    try:
        meta      = joblib.load(models_dir / "meta.pkl")
        preproc   = joblib.load(models_dir / "preproc.pkl")
        folds     = [joblib.load(p) for p in sorted((models_dir / "folds").glob("f*.joblib"))]
    except Exception as e:
        sys.exit(f"[ERROR] Cannot load models: {e}")

    space = json.loads((here.parent / "train/models/search_meta.json").read_text())
    CLEANERS = space["search_space"]["cleaner"]
    CLUSTERS = space["search_space"]["cluster"]
    K_LIST   = [k for k in space["search_space"]["k"]      if k > 0]
    EPS_LIST = [e for e in space["search_space"]["eps"]    if e > 0]
    MIN_LIST = [p for p in space["search_space"]["minPts"] if p > 0]

    base_cols = list(preproc.feature_names_in_)
    f_min, f_max = meta["f_min"], meta["f_max"]

    # 遍历所有数据集文件
    root = Path(__file__).resolve().parent / "../../../datasets/train"
    datasets = ["beers", "flights", "hospital", "rayyan"]
    for ds in datasets:
        for num in range(1, 16):
            csv_path = root / ds / f"{ds}_{num}.csv"
            if csv_path.exists():
                run_search(csv_path, CLEANERS, CLUSTERS, K_LIST, EPS_LIST, MIN_LIST,
                           base_cols, preproc, folds, f_min, f_max)

if __name__ == "__main__":
    main()

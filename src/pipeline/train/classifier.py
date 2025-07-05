#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LightGBMRegressor · 0-1 label · stratified 10-fold
目标 δ ≈ 0.05（无 early-stopping）
"""

import re, pickle, warnings, numpy as np, pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMRegressor
from joblib import dump

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMRegressor was fitted",
)

# ---------- files ----------
FILES = [
    "../../../results/analysis_results/beers_summary.xlsx",
    "../../../results/analysis_results/flights_summary.xlsx",
    "../../../results/analysis_results/hospital_summary.xlsx",
    "../../../results/analysis_results/rayyan_summary.xlsx",
]

# ---------- columns ----------
MISS_COL, OUT_COL = "missing", "anomaly"
CLN_METHOD_COL    = "cleaning_method"
CLUSTER_TYPE_COL  = "cluster_method"
PARAM_COL         = "parameters"
M_COL, D_COL      = "m", "n"
TARGET_COL        = "Combined Score"

# ---------- 1. load ----------
df = pd.concat([pd.read_excel(p) for p in FILES], ignore_index=True)
df = df[np.isfinite(df[TARGET_COL])].reset_index(drop=True)
print(f"[INFO] Loaded {len(df):,} samples.")

# ---------- 2. label 0-1 ----------
f_min, f_max = df[TARGET_COL].min(), df[TARGET_COL].max()
df["f_norm"] = (df[TARGET_COL] - f_min) / (f_max - f_min)

# ---------- 3. one-hot ----------
clean_ops = sorted(df[CLN_METHOD_COL].dropna().unique())
for op in clean_ops:
    df[op] = (df[CLN_METHOD_COL] == op).astype(int)

# ---------- 4. parse params ----------
def parse_param(s):
    kv = dict(re.findall(r"(\w+)=([0-9.]+)", str(s)))
    return float(kv.get("k",0)), float(kv.get("eps",0)), float(kv.get("minPts",0))
df[["k","eps","minPts"]] = df[PARAM_COL].apply(lambda x: pd.Series(parse_param(x)))

# ---------- 5. scale & interactions ----------
df["log_m"] = np.log10(df[M_COL]+1)
df["log_d"] = np.log10(df[D_COL]+1)
for a in ["kmeans","dbscan","hierarchical"]:
    m = (df[CLUSTER_TYPE_COL]==a).astype(int)
    df[f"miss_{a}"] = df[MISS_COL]*m
    df[f"out_{a}"]  = df[OUT_COL]*m

num_cols = ([MISS_COL, OUT_COL, "log_m","log_d","k","eps","minPts"] +
            [f"{p}_{a}" for p in ["miss","out"] for a in ["kmeans","dbscan","hierarchical"]])
cat_cols = clean_ops + [CLUSTER_TYPE_COL]

preproc = ColumnTransformer(
    [("num", MinMaxScaler(), num_cols),
     ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)],
    remainder="drop"
)

X = df[num_cols + cat_cols]
y = df["f_norm"].astype(float)

# stratified 10-fold
bins = pd.qcut(y, 10, duplicates="drop", labels=False)
kf   = StratifiedKFold(n_splits=10, shuffle=True, random_state=2025)

params = dict(
    objective="regression_l1",
    n_estimators=6000,          # ↑ 迭代数以弥补无早停
    learning_rate=0.02,
    num_leaves=511,
    min_child_samples=3,
    subsample=0.75,
    colsample_bytree=0.7,
    lambda_l1=1.0,
    lambda_l2=1.0,
    force_row_wise=True,
    verbosity=-1,
)

Path("models").mkdir(exist_ok=True)
oof = np.zeros(len(X))

for fold,(tr,val) in enumerate(kf.split(X,bins),1):
    model = Pipeline([
        ("prep", preproc),
        ("lgb",  LGBMRegressor(**params, random_state=2025+fold))
    ])
    model.fit(X.iloc[tr], y.iloc[tr])
    oof[val] = model.predict(X.iloc[val])
    dump(model, f"models/reg_fold{fold}.joblib")
    print(f"[INFO] Fold {fold} done")

delta = np.percentile(np.abs(oof - y), 95)
print(f"[INFO] δ (95-th abs error, norm) = {delta:.4f}")

meta = {"delta": delta, "f_min": f_min, "f_max": f_max}
dump(meta, "models/regressor.pkl")
dump(preproc, "models/scaler.pkl")

# ---------- predictor ----------
class Predictor:
    def __init__(self, m):
        self.delta, self.f_min, self.f_max = m["delta"], m["f_min"], m["f_max"]
        self.children = [pickle.load(open(f"models/reg_fold{i}.joblib","rb")) for i in range(1,11)]
    def _denorm(self,x): return x*(self.f_max-self.f_min)+self.f_min
    def _avg(self,X):    return np.mean([m.predict(X) for m in self.children], axis=0)
    def predict(self, df): return self._denorm(self._avg(df))
    def ucb(self, df):    return self._denorm(np.clip(self._avg(df)+self.delta,0,1))

with open("models/predictor.pkl","wb") as fp:
    pickle.dump(Predictor(meta), fp)

print("[INFO] Training complete – artefacts in ./models/")

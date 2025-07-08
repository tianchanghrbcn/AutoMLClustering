#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
classifier_v2.py  · 训练 + 生成 search_meta.json

用法
----
# 默认：先训练再写 artefacts ＋ search_meta.json
python classifier_v2.py

# 只根据训练数据快速重新写 search_meta.json，不重新跑 Optuna
python classifier_v2.py --gen-json-only
"""

import re, json, time, warnings, argparse
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
from lightgbm import LGBMRegressor, early_stopping
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold
from joblib import dump

warnings.filterwarnings("ignore",
    message="X does not have valid feature names, but LGBMRegressor was fitted")

# ---------- CLI ----------
ap = argparse.ArgumentParser()
ap.add_argument("--gen-json-only", action="store_true",
                help="仅根据训练数据写 models/search_meta.json，不重新训练")
args = ap.parse_args()

# ---------- 0. files ----------
FILES = [
    "../../../results/analysis_results/beers_summary.xlsx",
    "../../../results/analysis_results/flights_summary.xlsx",
    "../../../results/analysis_results/hospital_summary.xlsx",
    "../../../results/analysis_results/rayyan_summary.xlsx",
]

# ---------- 1. columns ----------
MISS_COL, OUT_COL   = "missing", "anomaly"
CLN_METHOD_COL      = "cleaning_method"
CLUSTER_TYPE_COL    = "cluster_method"
PARAM_COL           = "parameters"
M_COL, D_COL        = "m", "n"
TARGET_COL          = "Combined Score"

# ---------- 2. load ----------
df = pd.concat([pd.read_excel(p) for p in FILES], ignore_index=True)
df = df[np.isfinite(df[TARGET_COL])].reset_index(drop=True)
print(f"[INFO] Loaded {len(df):,} samples.")

# ---------- 3. 生成 search_meta.json (无论是否训练) ----------
def write_search_meta(df: pd.DataFrame):
    Path("models").mkdir(parents=True, exist_ok=True)

    cleaner_list = sorted(df[CLN_METHOD_COL].dropna().unique())
    cluster_list = sorted(df[CLUSTER_TYPE_COL].dropna().unique())

    # 取出现过的、合法的超参取值
    k_vals   = sorted({int(k) for k in df["k"] if k > 0})
    eps_vals = sorted({round(float(e), 3) for e in df["eps"] if e > 0})
    min_vals = sorted({int(p) for p in df["minPts"] if p > 0})

    meta = {
        "search_space": {
            "cleaner": cleaner_list,
            "cluster": cluster_list,
            "k":       k_vals,
            "eps":     eps_vals,
            "minPts":  min_vals
        }
    }
    with open("models/search_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("[INFO] 📝 search_meta.json 写入 models/ 完成")

# ---------- 4. 若仅写 JSON 则提前退出 ----------
if args.gen_json_only:
    # 先解析参数列，避免 KeyError
    def parse_param(s):
        kv = dict(re.findall(r"(\w+)=([0-9.]+)", str(s)))
        return float(kv.get("k", 0)), float(kv.get("eps", 0)), float(kv.get("minPts", 0))
    df[["k", "eps", "minPts"]] = df[PARAM_COL].apply(lambda x: pd.Series(parse_param(x)))
    write_search_meta(df)
    exit(0)

# ---------- 5. 全量训练流程 ----------
# 5.1 label 0-1
f_min, f_max = df[TARGET_COL].min(), df[TARGET_COL].max()
df["f_norm"] = (df[TARGET_COL] - f_min) / (f_max - f_min)

# 5.2 cleaning-method One-Hot
clean_ops = sorted(df[CLN_METHOD_COL].dropna().unique())
for op in clean_ops:
    df[op] = (df[CLN_METHOD_COL] == op).astype(int)

# 5.3 parse params
def parse_param(s):
    kv = dict(re.findall(r"(\w+)=([0-9.]+)", str(s)))
    return float(kv.get("k", 0)), float(kv.get("eps", 0)), float(kv.get("minPts", 0))
df[["k", "eps", "minPts"]] = df[PARAM_COL].apply(lambda x: pd.Series(parse_param(x)))

# 5.4 feature engineering
df["log_m"] = np.log10(df[M_COL] + 1)
df["log_d"] = np.log10(df[D_COL] + 1)
cluster_variants = sorted(df[CLUSTER_TYPE_COL].dropna().unique())
for a in cluster_variants:
    mask = (df[CLUSTER_TYPE_COL] == a).astype(int)
    df[f"miss_{a.lower()}"] = df[MISS_COL] * mask
    df[f"out_{a.lower()}"]  = df[OUT_COL] * mask

num_cols = (
    [MISS_COL, OUT_COL, "log_m", "log_d", "k", "eps", "minPts"] +
    [c for c in df.columns if re.match(r"^(miss|out)_", c)]
)
cat_cols = clean_ops + [CLUSTER_TYPE_COL]

preproc = ColumnTransformer(
    [("num", MinMaxScaler(), num_cols),
     ("cat", OneHotEncoder(handle_unknown='ignore'), cat_cols)],
    remainder="drop"
)

X_full = preproc.fit_transform(df[num_cols + cat_cols]).astype(np.float32)
y_full = df["f_norm"].values.astype(np.float32)
bins   = pd.qcut(y_full, 5, duplicates="drop", labels=False)

# 5.5 GPU auto detect
def has_gpu():
    try:
        import pynvml; pynvml.nvmlInit(); return pynvml.nvmlDeviceGetCount() > 0
    except Exception:
        return False
USE_GPU = has_gpu()
print("[INFO] GPU detected – using GPU" if USE_GPU else "[INFO] CPU hist mode")
gpu_params = (dict(device_type="gpu", gpu_platform_id=0, gpu_device_id=0,
                   max_bin=63, force_row_wise=False) if USE_GPU else
              dict(device_type="cpu", force_row_wise=True))

# 5.6 pruning callback
class DedupPruningCallback:
    def __init__(self, trial, metric="l1"):
        self.trial, self.metric, self.seen = trial, metric, set()
    def __call__(self, env):
        step = env.iteration
        if step in self.seen:
            return
        for name, m, val, _ in env.evaluation_result_list:
            if m == self.metric and name.endswith("valid"):
                self.trial.report(val, step)
                self.seen.add(step)
                if self.trial.should_prune():
                    raise optuna.TrialPruned(); break

# 5.7 Optuna objective
best_delta, best_models = np.inf, None
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2025)

def objective(trial):
    global best_delta, best_models
    params = {
        **gpu_params, "objective": "l1", "n_jobs": -1, "verbosity": -1,
        "learning_rate":   trial.suggest_float("lr", 1e-3, 0.2, log=True),
        "n_estimators":    trial.suggest_int("n_estimators", 500, 5000, step=500),
        "num_leaves":      trial.suggest_int("num_leaves", 31, 255, step=8),
        "min_child_samples": trial.suggest_int("min_child_samples", 2, 50),
        "subsample":       trial.suggest_float("subsample", 0.6, 0.9),
        "colsample_bytree":trial.suggest_float("colsample_bytree", 0.5, 0.9),
        "reg_alpha":       trial.suggest_float("reg_alpha", 0.0, 5.0),
        "reg_lambda":      trial.suggest_float("reg_lambda", 0.0, 5.0),
    }
    oof = np.zeros_like(y_full)
    fold_models = []
    for f, (tr, val) in enumerate(skf.split(X_full, bins)):
        mdl = LGBMRegressor(**params, random_state=2025 + f)
        mdl.fit(X_full[tr], y_full[tr],
                eval_set=[(X_full[val], y_full[val])],
                eval_metric="l1",
                callbacks=[early_stopping(200, verbose=False),
                           DedupPruningCallback(trial)])
        oof[val] = mdl.predict(X_full[val])
        fold_models.append(mdl)
    delta = np.percentile(np.abs(oof - y_full), 95)
    if delta < best_delta:
        best_delta, best_models = delta, fold_models
    return delta

print("[INFO] ⏳ Optuna search running …")
study = optuna.create_study(direction="minimize",
                            sampler=optuna.samplers.TPESampler(seed=2025),
                            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))
t0 = time.time()
study.optimize(objective, n_trials=100, n_jobs=4, show_progress_bar=True)
print(f"[TIME] Optuna finished in {(time.time()-t0)/60:.1f} min")
print(f"[INFO] Best δ = {best_delta:.4f}")
print(json.dumps(study.best_params, indent=2, ensure_ascii=False))

# 5.8 保存 artefacts
Path("models/folds").mkdir(parents=True, exist_ok=True)
for i, m in enumerate(best_models, 1):
    dump(m, f"models/folds/f{i}.joblib")
dump({"delta": best_delta, "f_min": f_min, "f_max": f_max}, "models/meta.pkl")
dump(preproc, "models/preproc.pkl")
print("[INFO] ✅ Training artefacts 写入 ./models/")

# 5.9 写 search_meta.json
write_search_meta(df)

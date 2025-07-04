#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import re
import numpy as np
import pandas as pd
from scipy import stats

# IsolationForest
try:
    from sklearn.ensemble import IsolationForest
    has_isolation_forest = True
except ImportError:
    has_isolation_forest = False

# POT for multi-dim Wasserstein
try:
    import ot
    has_pot = True
except ImportError:
    has_pot = False


###############################################################################
# 1) load_csv_drop_id
###############################################################################
def load_csv_drop_id(path):
    """读取CSV, 去掉 'ID' 列. 文件不存在/异常时返回 None."""
    if not os.path.isfile(path):
        print(f"[WARN] CSV not found: {path}")
        return None
    try:
        df = pd.read_csv(path)
        df.drop(columns=["ID"], errors="ignore", inplace=True)
        return df
    except Exception as e:
        print(f"[WARN] fail read {path}: {e}")
        return None


###############################################################################
# 2) parse_details => anomaly, missing, format, knowledge
###############################################################################
def parse_details(details_str):
    """从 e.g. "anomaly=2.50%, missing=0.00%, format=2.50%, knowledge=0.00%" 中解析比例."""
    result = {"anomaly": 0.0, "missing": 0.0}
    if not details_str:
        return result
    pattern = r'(\w+)\s*=\s*([\d\.]+)\%'
    matches = re.findall(pattern, details_str)
    for (k, v) in matches:
        if k in result:
            result[k] = float(v)
    return result


###############################################################################
# 3) 一些通用函数: numeric_cols, cat_cols
###############################################################################
def numeric_cols(df):
    if df is None or df.empty:
        return []
    return df.select_dtypes(include=[np.number]).columns.tolist()

def cat_cols(df):
    """object/category columns."""
    if df is None or df.empty:
        return []
    return df.select_dtypes(include=["object","category"]).columns.tolist()


###############################################################################
# 4) IsolationForest => outlier_ratio
###############################################################################
def detect_outliers_iforest(df, sample_size=1000):
    """返回离群点比例. 若没安装sklearn则返回None."""
    if not has_isolation_forest:
        return None
    if df is None or df.empty:
        return None
    num_cols = numeric_cols(df)
    if not num_cols:
        return None

    tmp = df[num_cols].dropna()
    if tmp.shape[0] > sample_size:
        tmp = tmp.sample(sample_size, random_state=42)

    if tmp.empty:
        return None

    iso = IsolationForest(random_state=42)
    iso.fit(tmp)
    preds = iso.predict(tmp)
    out_count = (preds == -1).sum()
    return float(out_count / len(preds))


###############################################################################
# 5) 多维Wasserstein
###############################################################################
def multi_dim_wasserstein(df1, df2):
    if not has_pot:
        return None
    if df1 is None or df2 is None:
        return None
    shared = list(set(numeric_cols(df1)) & set(numeric_cols(df2)))
    if not shared:
        return None

    A = df1[shared].dropna()
    B = df2[shared].dropna()
    if A.empty or B.empty:
        return None

    import ot
    a = ot.unif(len(A))
    b = ot.unif(len(B))
    M = ot.dist(A.values, B.values, metric='euclidean')
    cost = ot.emd2(a, b, M)
    return cost**0.5


###############################################################################
# 6) single_col_metrics => (mean, var, kl(dirty||repaired))
###############################################################################
def single_col_metrics(dcol, rcol, bins=10):

    out = {
        "mean_dirty": np.nan,
        "var_dirty":  np.nan,
        "mean_rep":   np.nan,
        "var_rep":    np.nan,
        "kl":         np.nan
    }

    if dcol is not None and not dcol.dropna().empty:
        sd = dcol.dropna()
        out["mean_dirty"] = sd.mean()
        out["var_dirty"]  = sd.var()

    # 再算 mean/var for repaired
    if rcol is not None and not rcol.dropna().empty:
        sr = rcol.dropna()
        out["mean_rep"] = sr.mean()
        out["var_rep"]  = sr.var()

    # kl(dirty||repaired)
    if dcol is not None and rcol is not None and not dcol.dropna().empty and not rcol.dropna().empty:
        sd = dcol.dropna()
        sr = rcol.dropna()
        minv = min(sd.min(), sr.min())
        maxv = max(sd.max(), sr.max())
        if minv < maxv:
            be = np.linspace(minv, maxv, bins+1)
            hd, _ = np.histogram(sd, bins=be)
            hr, _ = np.histogram(sr, bins=be)
            if hd.sum() > 0 and hr.sum() > 0:
                pd_ = hd / hd.sum()
                pr_ = hr / hr.sum()
                kl_ = 0.0
                for i in range(len(pd_)):
                    if pd_[i] > 0 and pr_[i] > 0:
                        kl_ += pd_[i] * np.log(pd_[i] / pr_[i])
                out["kl"] = kl_

    return out


def average_corr(df):
    if df is None or df.empty:
        return (np.nan, np.nan)
    cols = numeric_cols(df)
    if len(cols) < 2:
        return (np.nan, np.nan)
    cmat = df[cols].corr().abs()
    mask = np.triu(np.ones_like(cmat, dtype=bool))
    vals = cmat.where(~mask).stack()
    if vals.empty:
        return (np.nan, np.nan)
    return (vals.mean(), vals.max())


###############################################################################
# 8) cat_kl => just one average
###############################################################################
def cat_kl(dser, rser):
    """类别列 KL(d||r)."""
    if dser is None or dser.dropna().empty or rser is None or rser.dropna().empty:
        return np.nan
    f_d = dser.value_counts(dropna=False)
    f_r = rser.value_counts(dropna=False)
    allk = set(f_d.index) | set(f_r.index)
    sd_ = float(f_d.sum())
    sr_ = float(f_r.sum())
    kl_ = 0.0
    for val in allk:
        pd_ = f_d.get(val, 0) / sd_
        pr_ = f_r.get(val, 0) / sr_
        if pd_ > 0 and pr_ > 0:
            kl_ += pd_ * np.log(pd_ / pr_)
    return kl_


def cat_analysis(dirty_df, rep_df):
    out = {}
    if rep_df is None or rep_df.empty:
        out["cat_kl_avg"] = np.nan
        return out
    c1 = cat_cols(dirty_df)
    c2 = cat_cols(rep_df)
    shared = sorted(list(set(c1) & set(c2)))
    kl_list = []
    for col in shared:
        klv = cat_kl(dirty_df[col], rep_df[col])
        if not np.isnan(klv):
            kl_list.append(klv)
    if kl_list:
        out["cat_kl_avg"] = float(np.mean(kl_list))
    else:
        out["cat_kl_avg"] = np.nan
    return out


###############################################################################
# 帮助函数：用来计算变化率
###############################################################################
def calc_change_rate(before, after):
    """
    计算 (after - before)/before.
    若 before=0 或 NaN，则返回 NaN;
    若 after=NaN，也返回 NaN.
    """
    if before is None or np.isnan(before) or before == 0:
        return np.nan
    if after is None or np.isnan(after):
        return np.nan
    return (after - before) / before


###############################################################################
# 9) compute_all_metrics => 包括 "清洗前后对比" for outlier_ratio, mean, var, corr, etc.
###############################################################################
def compute_all_metrics(dirty_df, rep_df):

    res = {}

    # multi-wasserstein
    if has_pot and dirty_df is not None and rep_df is not None:
        wd = multi_dim_wasserstein(dirty_df, rep_df)
        res["multi_wasserstein"] = wd
    else:
        res["multi_wasserstein"] = np.nan

    # outlier ratio => dirty
    odir = detect_outliers_iforest(dirty_df) if dirty_df is not None else None
    # outlier ratio => rep
    orep = detect_outliers_iforest(rep_df) if rep_df is not None else None
    res["outlier_ratio_dirty"] = odir
    res["outlier_ratio_rep"]   = orep
    # 变化率(以百分数存储)
    if (odir is not None) and (orep is not None):
        change_val = calc_change_rate(odir, orep)
        if not np.isnan(change_val):
            res["outlier_ratio_diff"] = change_val * 100.0
        else:
            res["outlier_ratio_diff"] = np.nan
    else:
        res["outlier_ratio_diff"] = np.nan

    # numeric columns
    if dirty_df is not None and rep_df is not None:
        shared_num = sorted(list(set(numeric_cols(dirty_df)) & set(numeric_cols(rep_df))))
    else:
        shared_num = []

    kl_list = []
    idx = 1
    for col in shared_num:
        mets = single_col_metrics(dirty_df[col], rep_df[col])
        # store dirty & rep stats
        res[f"mean_col{idx}_dirty"] = mets["mean_dirty"]
        res[f"mean_col{idx}_rep"]   = mets["mean_rep"]

        # 变化率(以百分数存储)
        mc_rate = calc_change_rate(mets["mean_dirty"], mets["mean_rep"])
        res[f"mean_col{idx}_diff"] = mc_rate * 100.0 if not np.isnan(mc_rate) else np.nan

        res[f"var_col{idx}_dirty"] = mets["var_dirty"]
        res[f"var_col{idx}_rep"]   = mets["var_rep"]

        # 变化率(以百分数存储)
        vc_rate = calc_change_rate(mets["var_dirty"], mets["var_rep"])
        res[f"var_col{idx}_diff"]  = vc_rate * 100.0 if not np.isnan(vc_rate) else np.nan

        # kl
        if not np.isnan(mets["kl"]):
            kl_list.append(mets["kl"])

        idx += 1

    # kl_numeric_avg => all shared num col
    if kl_list:
        res["kl_numeric_avg"] = float(np.mean(kl_list))
    else:
        res["kl_numeric_avg"] = np.nan

    # corr => dirty & rep
    cavg_d, cmax_d = average_corr(dirty_df) if dirty_df is not None else (np.nan, np.nan)
    cavg_r, cmax_r = average_corr(rep_df)   if rep_df is not None else (np.nan, np.nan)
    res["corr_avg_dirty"] = cavg_d
    res["corr_avg_rep"]   = cavg_r

    # 变化率(以百分数存储)
    avg_rate = calc_change_rate(cavg_d, cavg_r)
    res["corr_avg_diff"] = avg_rate * 100.0 if not np.isnan(avg_rate) else np.nan

    res["corr_max_dirty"] = cmax_d
    res["corr_max_rep"]   = cmax_r

    # 变化率(以百分数存储)
    max_rate = calc_change_rate(cmax_d, cmax_r)
    res["corr_max_diff"] = max_rate * 100.0 if not np.isnan(max_rate) else np.nan

    # cat => cat_kl_avg
    catd = cat_analysis(dirty_df, rep_df)
    res.update(catd)

    return res


###############################################################################
def main():
    comparison_file = "comparison.json"
    out_dir = "../../../results/analysis_results"
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isfile(comparison_file):
        print(f"[ERROR] {comparison_file} not found.")
        return

    with open(comparison_file, 'r', encoding='utf-8') as f:
        comp_list = json.load(f)

    task_map = {}

    for entry in comp_list:
        task_name  = entry.get("task_name","")
        num        = entry.get("num","")
        dataset_id = entry.get("dataset_id","")
        error_rate = entry.get("error_rate",None)
        m_         = entry.get("m",None)
        n_         = entry.get("n",None)

        d_dict = parse_details(entry.get("details",""))
        anomaly_val   = d_dict["anomaly"]
        missing_val   = d_dict["missing"]

        paths     = entry.get("paths", {})
        dirty_csv = paths.get("dirty_csv","")
        df_dirty  = load_csv_drop_id(dirty_csv)

        rep_map   = paths.get("repaired_paths",{})
        for method, rep_path in rep_map.items():
            df_rep = load_csv_drop_id(rep_path)

            mets = compute_all_metrics(df_dirty, df_rep)

            row = {
                "task_name": task_name,
                "num": num,
                "dataset_id": dataset_id,
                "error_rate": error_rate,
                "m": m_,
                "n": n_,
                "anomaly": anomaly_val,
                "missing": missing_val,
                "cleaning_method": method
            }
            row.update(mets)

            if task_name not in task_map:
                task_map[task_name] = []
            task_map[task_name].append(row)

    # 输出
    for tname, rows in task_map.items():
        df_out = pd.DataFrame(rows)
        out_csv = os.path.join(out_dir, f"{tname}_analysis.csv")
        df_out.to_csv(out_csv, index=False, encoding='utf-8')
        print(f"[INFO] Wrote => {out_csv}, rows={len(rows)}")

    print("[INFO] All done. Check CSV in", out_dir)


if __name__=="__main__":
    main()

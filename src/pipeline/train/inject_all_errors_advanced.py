#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inject_errors_v2.py  · 仅向非主键列注入“异常 + 缺失”错误
CLI、输入输出格式与旧版完全一致。
"""
import argparse, os, random, string
import numpy as np
import pandas as pd

# ---------- CLI ----------
def parse_arguments():
    p = argparse.ArgumentParser(
        description="Inject 'anomaly' & 'missing' errors; first column treated as primary key."
    )
    p.add_argument("--input", required=True,  help="Clean CSV path")
    p.add_argument("--output", required=True, help="Output directory")
    p.add_argument("--task_name", required=True,
                   help="Files: {task_name}_{n}.csv & {task_name}_explanation.txt")
    p.add_argument("--seed", type=int, default=None, help="Random seed (optional)")
    return p.parse_args()

# ---------- 注入核心 ----------
def inject_anomaly_and_missing(df, cols, a_rate, m_rate, rng):
    """
    精确抽取   n_anom = round(N_valid * a_rate)
            n_miss = round(N_valid * m_rate)
    保证：
        - 仅对原本非空单元格采样
        - A、M 互不重叠
    返回 (n_anom, n_miss)
    """
    # ------- 1) 收集所有候选单元格 (原本非 NaN) -------
    candidates = [(r, c)
                  for c in cols
                  for r in df.index
                  if not pd.isna(df.at[r, c])]

    N_valid = len(candidates)
    if N_valid == 0:
        return 0, 0

    n_anom = round(N_valid * a_rate)
    n_miss = round(N_valid * m_rate)

    # ------- 2) 随机抽取异常、缺失 -------
    idx_all = rng.choice(N_valid, size=n_anom + n_miss, replace=False)
    anom_cells = [candidates[i] for i in idx_all[:n_anom]]
    miss_cells = [candidates[i] for i in idx_all[n_anom:]]

    # ------- 3) 写入异常 -------
    for (r, c) in anom_cells:
        df.at[r, c] = generate_anomaly_value(df.at[r, c], rng)

    # ------- 4) 写入缺失 -------
    for (r, c) in miss_cells:
        df.at[r, c] = np.nan

    return n_anom, n_miss

def generate_anomaly_value(val, rng):
    """数值列放大 3–6 倍；文本列插怪符号"""
    if pd.isnull(val):
        return val
    try:
        f = float(val)
        return f * rng.uniform(3, 6)
    except Exception:
        specs = ["#", "$", "%", "??", "@@"]
        times = rng.integers(1, 3)
        s = str(val)
        for _ in range(times):
            pos = rng.integers(0, len(s) + 1)
            s = s[:pos] + rng.choice(specs) + s[pos:]
        return s

# ---------- 主流程 ----------
def main():
    args = parse_arguments()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    df_clean = pd.read_csv(args.input).reset_index(drop=True)
    if df_clean.shape[1] < 2:
        raise SystemExit("Dataset 应 ≥2 列。")

    pk, other_cols = df_clean.columns[0], df_clean.columns[1:]
    # 把整数列显式转成 float，避免放大后 dtype 报错
    for c in other_cols:
        if pd.api.types.is_integer_dtype(df_clean[c]):
            df_clean[c] = df_clean[c].astype(float)

    os.makedirs(args.output, exist_ok=True)
    expl_path = os.path.join(args.output, f"{args.task_name}_explanation.txt")
    rates = [0.00, 0.05, 0.10, 0.15]
    rng = np.random.default_rng()

    explain_lines = []
    combo_idx = 0
    for a_r in rates:
        for m_r in rates:
            if a_r == 0 and m_r == 0:
                continue
            combo_idx += 1
            df_cor = df_clean.copy()
            n_a, n_m = inject_anomaly_and_missing(df_cor, other_cols, a_r, m_r, rng)

            # 计算观测误差率
            n_valid = (~df_clean[other_cols].isna()).values.sum()
            r_a = n_a / n_valid
            r_m = n_m / n_valid
            r_tot = r_a + r_m

            fname = f"{args.task_name}_{combo_idx}.csv"
            df_cor.to_csv(os.path.join(args.output, fname), index=False)

            explain_lines.append(
                f"{combo_idx:02d} | Anom={a_r:.0%}, Miss={m_r:.0%}  "
                f"→  r_anom={r_a:.2%}, r_miss={r_m:.2%}, r_tot={r_tot:.2%}"
            )
            print(f"[{combo_idx:02d}] {fname} done.")

    with open(expl_path, "w", encoding="utf-8") as f:
        f.write("Error‑injection summary\n\n")
        f.write("\n".join(explain_lines))

    print("\n✅  All corrupted files & explanation saved to:", args.output)

if __name__ == "__main__":
    main()

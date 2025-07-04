#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import pandas as pd
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Mode-based data correction script.")
    parser.add_argument("--clean_path", required=True, help="Path to the clean CSV (unused here).")
    parser.add_argument("--dirty_path", required=True, help="Path to the dirty CSV to be repaired.")
    parser.add_argument("--task_name", required=True, help="Task name, used to name output file.")
    return parser.parse_args()

def main():
    args = parse_args()
    clean_csv_path = args.clean_path   # 这里不使用
    dirty_csv_path = args.dirty_path
    task_name = args.task_name

    # 1. 读取脏数据
    if not os.path.isfile(dirty_csv_path):
        print(f"[ERROR] Dirty file not found: {dirty_csv_path}")
        return

    df_dirty = pd.read_csv(dirty_csv_path)

    # 2. 区分数值列和非数值列
    numeric_cols = df_dirty.select_dtypes(include=[np.number]).columns
    cat_cols = df_dirty.select_dtypes(exclude=[np.number]).columns

    # 3. 对数值列的缺失值：中位数填充
    for col in numeric_cols:
        if df_dirty[col].isnull().any():
            median_val = df_dirty[col].median(skipna=True)
            df_dirty[col].fillna(median_val, inplace=True)

    # 4. 对非数值列的缺失值：众数填充 (mode)
    for col in cat_cols:
        if df_dirty[col].isnull().any():
            mode_val = df_dirty[col].mode(dropna=True)
            if not mode_val.empty:
                df_dirty[col].fillna(mode_val.iloc[0], inplace=True)
            else:
                # 如果整列都是 NaN，则用“Unknown”占位
                df_dirty[col].fillna("Unknown", inplace=True)

    # 5. 与“else: ... pattern = re.compile(r'^repaired_.*\.csv$')” 匹配对齐
    #    所以要把修复结果放到:
    #      /root/AutoMLClustering/src/cleaning/Repaired_res/mode/<task_name>/repaired_<task_name>.csv

    out_dir = f"/root/AutoMLClustering/src/cleaning/Repaired_res/mode/{task_name}"
    os.makedirs(out_dir, exist_ok=True)
    out_filename = os.path.join(out_dir, f"repaired_{task_name}.csv")

    df_dirty.to_csv(out_filename, index=False)

    # 6. 打印提示 (必须包含 "Repaired data saved to ...")
    print(f"Repaired data saved to {out_filename}")

if __name__ == "__main__":
    main()

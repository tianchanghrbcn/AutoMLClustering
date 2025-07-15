#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pre-processing.py · 生成各脏文件的特征向量 (missing_rate, noise_rate, error_rate 等)
运行流程:
  1) 先调用 4 条注入脚本, 生成 15 份含错 CSV
  2) 遍历 datasets/train 下各数据集文件夹, 为每个脏 CSV 计算特征
  3) 输出 results/eigenvectors.json
"""

import os
import json
import pandas as pd
import numpy as np

# ========== 全局配置 ==========

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "datasets", "train")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "..", "results", "eigenvectors.json")
K_VALUE = 5  # 固定 K=5

# ========== 运行 4 条注入命令的函数 ==========

def run_inject_scripts():
    """
    依次执行用户给出的 4 行注入命令。
    如需更灵活控制，可改用 subprocess.run()。
    """
    commands = [
        "python inject_all_errors_advanced.py --input ../../../datasets/train/beers/clean.csv "
        "--output ../../../datasets/train/beers --task_name beers --seed 42",
        "python inject_all_errors_advanced.py --input ../../../datasets/train/flights/clean.csv "
        "--output ../../../datasets/train/flights --task_name flights --seed 42",
        "python inject_all_errors_advanced.py --input ../../../datasets/train/hospital/clean.csv "
        "--output ../../../datasets/train/hospital --task_name hospital --seed 42",
        "python inject_all_errors_advanced.py --input ../../../datasets/train/rayyan/clean.csv "
        "--output ../../../datasets/train/rayyan --task_name rayyan --seed 42"
    ]

    print("===== 开始执行错误注入命令 =====")
    for i, cmd in enumerate(commands, 1):
        print(f"[{i}] 执行命令: {cmd}")
        ret = os.system(cmd)  # 0 表示成功
        if ret != 0:
            print(f"警告: 命令执行出错 (返回码 {ret}): {cmd}")
        else:
            print(f"完成: {cmd}")
    print("===== 四条错误注入命令执行完毕 =====\n")

# ========== 计算相关函数（新定义） ==========

def _strip_pk(df: pd.DataFrame) -> pd.DataFrame:
    """返回去掉主键列 (第 1 列) 的副本。"""
    return df.iloc[:, 1:].copy() if df.shape[1] > 1 else df.copy()

def compute_missing_rate(df: pd.DataFrame, df_clean: pd.DataFrame) -> float:
    """
    r_miss = |M| / N_non‑NaN
    只统计『原本非空 ➜ 现在 NaN』的比例。
    """
    df_no_pk        = _strip_pk(df)
    clean_no_pk     = _strip_pk(df_clean)

    mask_non_nan_clean = ~clean_no_pk.isna()      # 原本非空
    N_non_nan = mask_non_nan_clean.sum().sum()
    if N_non_nan == 0:
        return 0.0

    mask_miss = mask_non_nan_clean & df_no_pk.isna()
    n_miss    = mask_miss.sum().sum()
    return n_miss / N_non_nan


def compute_noise_rate(df: pd.DataFrame, df_clean: pd.DataFrame) -> float:
    """
    r_anom = |A| / N_non‑NaN
    只统计『原本非空且现在 ≠ 原值且非 NaN』的比例。
    """
    df_no_pk    = _strip_pk(df)
    clean_no_pk = _strip_pk(df_clean)

    mask_non_nan_clean = ~clean_no_pk.isna()
    N_non_nan = mask_non_nan_clean.sum().sum()
    if N_non_nan == 0:
        return 0.0

    mask_anom = mask_non_nan_clean & (~df_no_pk.isna()) & (df_no_pk != clean_no_pk)
    n_anom    = mask_anom.sum().sum()
    return n_anom / N_non_nan

# ========== 单文件处理 ==========

def process_single_file(csv_path: str,
                        dataset_name: str,
                        dataset_id: int,
                        df_clean: pd.DataFrame) -> dict:
    """
    读取 CSV，计算特征向量并返回字典。
    - missing_rate, noise_rate ∈ [0,1]
    - error_rate = (missing_rate + noise_rate) × 100  (百分比，与注入脚本 r_tot 对齐)
    """
    file_name = os.path.basename(csv_path)
    if file_name == "clean.csv":            # clean.csv 不写 JSON
        return {}

    df = pd.read_csv(csv_path)

    missing_rate = compute_missing_rate(df, df_clean)
    noise_rate   = compute_noise_rate(df, df_clean)
    error_rate   = (missing_rate + noise_rate) * 100  # 百分数

    feature_vector = {
        "dataset_id":   dataset_id,
        "dataset_name": dataset_name,
        "csv_file":     file_name,
        "error_rate":   error_rate,
        "K":            K_VALUE,
        "missing_rate": missing_rate,
        "noise_rate":   noise_rate,
        "m":            df.shape[1],   # 列数
        "n":            df.shape[0],   # 行数
    }
    return feature_vector

# ========== 主逻辑 ==========

def main():
    """
    每次运行:
      1) 先执行 4 条错误注入命令
      2) 再扫描 DATA_DIR 下子文件夹, 读取 CSV, 生成 eigenvectors.json
    """
    # 1) 执行注入脚本
    run_inject_scripts()

    # 2) 遍历数据集
    if not os.path.isdir(DATA_DIR):
        print(f"错误: DATA_DIR {DATA_DIR} 不存在或不是文件夹。")
        return

    all_vectors = []
    dataset_id_counter = 0

    for dataset_name in sorted(os.listdir(DATA_DIR)):
        sub_folder = os.path.join(DATA_DIR, dataset_name)
        if not os.path.isdir(sub_folder):
            continue

        csv_files = [f for f in os.listdir(sub_folder) if f.endswith(".csv")]
        if not csv_files:
            print(f"警告: 数据集 {dataset_name} 无 CSV，跳过。")
            continue

        clean_path = os.path.join(sub_folder, "clean.csv")
        if not os.path.isfile(clean_path):
            print(f"警告: {dataset_name} 缺少 clean.csv，跳过。")
            continue

        df_clean = pd.read_csv(clean_path)

        for csv_file in sorted(csv_files):
            if csv_file == "clean.csv":     # clean.csv 不写入 JSON
                continue

            csv_path = os.path.join(sub_folder, csv_file)
            vector = process_single_file(csv_path, dataset_name,
                                         dataset_id_counter, df_clean)
            if not vector:                  # 安全检查
                continue

            all_vectors.append(vector)
            print(f"[{dataset_id_counter}] 处理完成: {dataset_name}/{csv_file} "
                  f"=> error_rate={vector['error_rate']:.2f}%")
            dataset_id_counter += 1

    # 3) 写入输出文件
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_vectors, f, indent=4, ensure_ascii=False)

    print(f"\n✅  全部处理完成，共 {len(all_vectors)} 条记录写入: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
    依次执行您给出的 4 行注入命令。
    根据环境可改用 subprocess.run() 等更灵活方式。
    """
    commands = [
        "python inject_all_errors_advanced.py --input ../../../datasets/train/beers/clean.csv --output ../../../datasets/train/beers --task_name beers --seed 42",
        "python inject_all_errors_advanced.py --input ../../../datasets/train/flights/clean.csv --output ../../../datasets/train/flights --task_name flights --seed 42",
        "python inject_all_errors_advanced.py --input ../../../datasets/train/hospital/clean.csv --output ../../../datasets/train/hospital --task_name hospital --seed 42",
        "python inject_all_errors_advanced.py --input ../../../datasets/train/rayyan/clean.csv --output ../../../datasets/train/rayyan --task_name rayyan --seed 42"
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


# ========== 计算相关函数 ==========

def compute_missing_rate(df: pd.DataFrame) -> float:
    """
    计算缺失值占比。
    跳过第1列（主键列）。
    """
    if df.shape[1] <= 1:
        return 0.0

    df_no_pk = df.iloc[:, 1:]
    total_cells = df_no_pk.size
    missing_cells = df_no_pk.isnull().sum().sum()
    return missing_cells / total_cells if total_cells > 0 else 0.0


def compute_noise_rate(df: pd.DataFrame) -> float:
    """
    基于IQR检测数值型离群点，返回占比。
    同样跳过第1列（主键列）。
    """
    if df.shape[1] <= 1:
        return 0.0

    df_no_pk = df.iloc[:, 1:]
    numeric_df = df_no_pk.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return 0.0

    Q1 = numeric_df.quantile(0.25)
    Q3 = numeric_df.quantile(0.75)
    IQR = Q3 - Q1

    outliers = ((numeric_df < (Q1 - 1.5 * IQR)) | (numeric_df > (Q3 + 1.5 * IQR))).sum().sum()
    total_elements = numeric_df.size
    noise_rate = outliers / total_elements if total_elements > 0 else 0.0
    return noise_rate


def compute_error_rate_by_comparison(df: pd.DataFrame, df_clean: pd.DataFrame) -> float:
    """
    与同目录下的 clean.csv 做逐元素对比，统计不一致单元格占比。
    同样跳过第一列（主键列）不参与对比。
    """
    if df.shape[1] <= 1 or df_clean.shape[1] <= 1:
        return 0.0

    df_no_pk = df.iloc[:, 1:].copy()
    df_clean_no_pk = df_clean.iloc[:, 1:].copy()

    total_cells = df_no_pk.size
    diff_count = (df_no_pk != df_clean_no_pk).sum().sum()

    error_rate = diff_count / total_cells if total_cells > 0 else 0.0
    return error_rate


def process_single_file(csv_path: str,
                        dataset_name: str,
                        dataset_id: int,
                        df_clean: pd.DataFrame) -> dict:
    """
    读取 CSV 文件，计算特征向量并返回字典。
    """
    file_name = os.path.basename(csv_path)
    df = pd.read_csv(csv_path)

    num_samples = df.shape[0]
    num_features = df.shape[1]

    missing_rate = compute_missing_rate(df)
    noise_rate = compute_noise_rate(df)

    # 这里我们只对非 clean.csv 计算 error_rate
    # 若是别的带有注入错误的文件，与 clean.csv 对比
    if file_name == "clean.csv":
        error_rate = 0.0
    else:
        error_rate = compute_error_rate_by_comparison(df, df_clean) * 100

    feature_vector = {
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "csv_file": file_name,
        "error_rate": error_rate,
        "K": K_VALUE,
        "missing_rate": missing_rate,
        "noise_rate": noise_rate,
        "m": num_features,         # 列数
        "n": num_samples,          # 行数
    }
    return feature_vector


# ========== 主逻辑 ==========

def main():
    """
    每次运行:
      1) 先执行 4 条错误注入命令
      2) 再扫描 DATA_DIR 下子文件夹, 读取 CSV, 生成 eigenvectors.json
    """
    # 先执行注入脚本
    run_inject_scripts()

    existing_data = []

    if not os.path.isdir(DATA_DIR):
        print(f"错误: DATA_DIR {DATA_DIR} 不存在或不是文件夹。")
        return

    dataset_id_counter = 0

    for dataset_name in os.listdir(DATA_DIR):
        sub_folder = os.path.join(DATA_DIR, dataset_name)
        if not os.path.isdir(sub_folder):
            continue

        # 找出子文件夹下所有 CSV
        csv_files = [f for f in os.listdir(sub_folder) if f.endswith(".csv")]
        if not csv_files:
            print(f"警告: 数据集 {dataset_name} 文件夹下无 CSV 文件，跳过。")
            continue

        # 确认是否存在 clean.csv，用于对比 error_rate
        clean_csv_path = os.path.join(sub_folder, "clean.csv")
        if not os.path.isfile(clean_csv_path):
            print(f"警告: {dataset_name} 数据集中缺少 clean.csv，跳过该数据集。")
            continue

        df_clean = pd.read_csv(clean_csv_path)

        for csv_file in csv_files:
            # 跳过 clean.csv，不将其写入 JSON
            if csv_file == "clean.csv":
                continue

            csv_path = os.path.join(sub_folder, csv_file)

            dataset_id = dataset_id_counter
            dataset_id_counter += 1

            feature_vector = process_single_file(csv_path, dataset_name, dataset_id, df_clean)
            existing_data.append(feature_vector)

            print(f"[{dataset_id}] 完成处理: {dataset_name}/{csv_file} => error_rate={feature_vector['error_rate']:.4f}")

    # 生成输出文件
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, indent=4, ensure_ascii=False)

    print(f"\n所有数据处理完成，共 {len(existing_data)} 条记录已写入: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

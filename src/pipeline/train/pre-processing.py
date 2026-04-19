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
from typing import Tuple, Union
import pandas as pd
import numpy as np

# ========== 版本&全局配置 ==========
SCRIPT_VERSION = "eigenvectors/align-v2.2"

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "datasets", "train")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "..", "results", "eigenvectors.json")
K_VALUE = 5  # 固定 K=5

# 主键列：可为 int(列序号，从0起) 或 str(列名)。默认=第1列，与原逻辑一致。
PK_COL: Union[int, str] = 0

# 对齐与退化策略
ALIGN_VERBOSE = True                # 打印对齐时的提示
ALLOW_POSITIONAL_FALLBACK = True    # 无法按主键对齐时，允许“公共列 + 按位置对齐”

# ========== 运行 4 条注入命令的函数 ==========

def run_inject_scripts():
    """
    依次执行 4 行注入命令。
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

# ========== 对齐与清理工具函数 ==========

def _drop_unnamed_cols(df: pd.DataFrame) -> pd.DataFrame:
    """去掉由保存索引带来的 'Unnamed:*' 临时列。"""
    if df.empty:
        return df
    keep = ~df.columns.astype(str).str.startswith("Unnamed")
    return df.loc[:, keep].copy()

def _strip_pk(df: pd.DataFrame) -> pd.DataFrame:
    """返回去掉主键列(第1列)的副本（仅用于旧代码兼容，不在最终比较中直接使用）。"""
    return df.iloc[:, 1:].copy() if df.shape[1] > 1 else df.copy()

def _resolve_pk_name(df: pd.DataFrame, df_clean: pd.DataFrame, pk_col: Union[int, str]) -> Union[str, None]:
    """
    解析主键列名：
      - pk_col 为 int：两侧该位置列名必须一致，否则返回 None
      - pk_col 为 str：两侧都包含此列名才返回，否则 None
    """
    if isinstance(pk_col, int):
        if pk_col < df.shape[1] and pk_col < df_clean.shape[1]:
            a, b = df.columns[pk_col], df_clean.columns[pk_col]
            return a if a == b else None
        return None
    else:
        return pk_col if (str(pk_col) in df.columns and str(pk_col) in df_clean.columns) else None

def _align_for_compare(
    df_dirty: pd.DataFrame,
    df_clean: pd.DataFrame,
    pk_col: Union[int, str] = PK_COL,
    dedup_keep: str = "first",
    verbose: bool = ALIGN_VERBOSE,
    allow_positional_fallback: bool = ALLOW_POSITIONAL_FALLBACK,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    返回 (dirty_aligned, clean_aligned)，二者：
      - 行：已对齐（优先主键交集；否则位置对齐）
      - 列：仅保留公共列，顺序以 clean 为准
      - 均不包含主键列（若按主键对齐）

    额外：在最终比较前再做一次 pandas 的 align + 排序，确保标签一致。
    """
    A = _drop_unnamed_cols(df_dirty).copy()
    B = _drop_unnamed_cols(df_clean).copy()

    # —— 1) 优先尝试按主键对齐 ——
    pk_name = _resolve_pk_name(A, B, pk_col)
    if pk_name is not None:
        # 去重（若存在重复主键）
        if not A[pk_name].is_unique:
            if verbose:
                print(f"警告: 脏数据主键 `{pk_name}` 存在重复，将保留 {dedup_keep} 条记录。")
            A = A.drop_duplicates(subset=[pk_name], keep=dedup_keep)
        if not B[pk_name].is_unique:
            if verbose:
                print(f"警告: clean.csv 主键 `{pk_name}` 存在重复，将保留 {dedup_keep} 条记录。")
            B = B.drop_duplicates(subset=[pk_name], keep=dedup_keep)

        # set_index 后主键列不再在 columns 中，相当于“去掉主键列”
        A = A.set_index(pk_name, drop=True)
        B = B.set_index(pk_name, drop=True)

        # 列交集（以 clean 顺序为准）
        common_cols = [c for c in B.columns if c in A.columns]
        extra_in_A = [c for c in A.columns if c not in B.columns]
        extra_in_B = [c for c in B.columns if c not in A.columns]
        if verbose and (extra_in_A or extra_in_B):
            if extra_in_A:
                print(f"提示: 脏数据额外列被忽略: {extra_in_A}")
            if extra_in_B:
                print(f"提示: clean 额外列被忽略: {extra_in_B}")

        A = A[common_cols]
        B = B[common_cols]

        # 主键交集（行对齐 & 排序）
        common_idx = A.index.intersection(B.index)
        if len(common_idx) > 0:
            A = A.loc[common_idx].sort_index()
            B = B.loc[common_idx].sort_index()

            # 再做一次严格对齐（含索引与列）
            A, B = A.align(B, join="inner", axis=None, copy=False)
            # 显式排序，防止某些版本对齐后残留顺序差异
            A = A.sort_index(axis=0).sort_index(axis=1)
            B = B.sort_index(axis=0).sort_index(axis=1)
            return A, B
        else:
            if verbose:
                print("提示: 主键存在，但两侧主键无交集，将尝试位置兜底对齐。")

    # —— 2) 退化：公共列 + 位置对齐（保底不抛异常） ——
    if allow_positional_fallback:
        common_cols = A.columns.intersection(B.columns)
        if verbose:
            if len(common_cols) == 0:
                print("提示: 无公共列，位置兜底无法进行；将返回空对齐结果。")
        A2 = A[common_cols].reset_index(drop=True)
        B2 = B[common_cols].reset_index(drop=True)
        n = min(len(A2), len(B2))
        A2 = A2.iloc[:n].copy()
        B2 = B2.iloc[:n].copy()

        # 再做一次严格对齐，确保标签一致（此时两边都是 RangeIndex + 同列）
        A2, B2 = A2.align(B2, join="inner", axis=None, copy=False)
        A2 = A2.sort_index(axis=0).sort_index(axis=1)
        B2 = B2.sort_index(axis=0).sort_index(axis=1)
        return A2, B2

    # —— 3) 不允许退化：报错 ——
    raise ValueError("无法按主键对齐，且未允许位置对齐（allow_positional_fallback=False）。")

# ========== 计算函数（使用对齐后的 DataFrame，并用 NumPy 比较） ==========

def compute_missing_rate(df: pd.DataFrame, df_clean: pd.DataFrame) -> float:
    """
    r_miss = |M| / N_non‑NaN
    统计『原本非空(以 clean 为准) ➜ 现在 dirty 为 NaN』的比例。
    """
    A, B = _align_for_compare(df, df_clean, pk_col=PK_COL)

    # 以 clean 的“原本非空”作为分母
    mask_non_nan_clean = ~B.isna()
    N_non_nan = int(mask_non_nan_clean.values.sum())
    if N_non_nan == 0:
        return 0.0

    # 用 NumPy 进行元素级比较，避免 pandas 再次基于标签对齐
    miss_mask = mask_non_nan_clean.to_numpy() & A.isna().to_numpy()
    n_miss = int(miss_mask.sum())
    return n_miss / N_non_nan

def compute_noise_rate(df: pd.DataFrame, df_clean: pd.DataFrame) -> float:
    """
    r_anom = |A| / N_non‑NaN
    统计『原本非空(以 clean 为准) 且现在 dirty 非 NaN 且 ≠ 原值』的比例。
    """
    A, B = _align_for_compare(df, df_clean, pk_col=PK_COL)

    mask_non_nan_clean = ~B.isna()
    N_non_nan = int(mask_non_nan_clean.values.sum())
    if N_non_nan == 0:
        return 0.0

    # 三个条件：原本非空 & 脏值非空 & 值不相等（用 NumPy 比较）
    cond_clean_non_nan = mask_non_nan_clean.to_numpy()
    cond_dirty_non_nan = ~A.isna().to_numpy()
    # 注意：值比较也转 NumPy，避免标签对齐
    cond_value_diff   = (A.to_numpy() != B.to_numpy())

    anom_mask = cond_clean_non_nan & cond_dirty_non_nan & cond_value_diff
    n_anom = int(anom_mask.sum())
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
        "m":            df.shape[1],   # 列数（含主键）
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
    print(f"== Running {os.path.basename(__file__)} · {SCRIPT_VERSION} ==")

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

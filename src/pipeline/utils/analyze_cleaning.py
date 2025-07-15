#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cleaning_metrics.py
-------------------
基于 comparison.json，计算各清洗算法的 Precision / Recall / F1 / EDR
（符合 LaTeX 表 ~\ref{tab:q1-metrics-definition} 的最新公式），
并按 task_name 输出 <task_name>_cleaning.csv。
"""
import csv
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

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

def main() -> None:
    # 1. 读取 comparison.json（相对于本脚本）
    here = Path(__file__).resolve().parent
    comparison_path = here / "../train/comparison.json"
    with open(comparison_path, "r", encoding="utf-8") as f:
        comparison_data = json.load(f)

    # 2. 保存所有结果行
    results = []

    # 3. 遍历每条配置
    for item in comparison_data:
        task_name = item["task_name"]
        num_ = item["num"]
        dataset_id = item["dataset_id"]
        error_rate = item["error_rate"]
        m_ = item["m"]
        n_ = item["n"]

        # anomaly / missing
        details_str = item.get("details", "")
        d_dict = parse_details(details_str)
        anomaly = d_dict["anomaly"]
        missing = d_dict["missing"]

        # 构造路径
        clean_csv_path = (comparison_path.parent / item["paths"]["clean_csv"]).resolve()
        dirty_csv_path = (comparison_path.parent / item["paths"]["dirty_csv"]).resolve()

        # 3.1 读取 clean / dirty
        df_clean = pd.read_csv(clean_csv_path, keep_default_na=False)
        df_dirty = pd.read_csv(dirty_csv_path, keep_default_na=False)

        if df_clean.shape != df_dirty.shape:
            print(
                f"[WARNING] {task_name}-{num_}: clean.csv 与 dirty.csv 形状不匹配，跳过。"
            )
            continue

        n_rows, n_cols = df_clean.shape

        # 3.2 标记原始错误单元格 (n_w)
        is_wrong_cell = (df_dirty.astype(str) != df_clean.astype(str)).to_numpy()
        n_w = int(is_wrong_cell.sum())  # 原始错误总数

        # 3.3 遍历每个算法的 repaired.csv
        for cleaning_method, rep_rel_path in item["paths"]["repaired_paths"].items():
            repaired_csv_path = (comparison_path.parent / rep_rel_path).resolve()
            if not repaired_csv_path.exists():
                print(f"[WARNING] 缺少文件: {repaired_csv_path}，跳过 {cleaning_method}")
                continue

            df_rep = pd.read_csv(repaired_csv_path, keep_default_na=False)
            if df_rep.shape != df_clean.shape:
                print(
                    f"[WARNING] {task_name}-{num_}: repaired 形状与 clean 不匹配，跳过 {cleaning_method}"
                )
                continue

            # ---------- 统计 n_w2r, n_w2w, n_r2r, n_r2w ----------
            n_w2r = n_w2w = n_r2r = n_r2w = 0

            # 向量化比较以加速
            rep_eq_clean = (df_rep.astype(str) == df_clean.astype(str)).to_numpy()

            # originally wrong
            n_w2r = int((is_wrong_cell & rep_eq_clean).sum())
            n_w2w = int((is_wrong_cell & ~rep_eq_clean).sum())

            # originally right
            n_r2r = int((~is_wrong_cell & rep_eq_clean).sum())
            n_r2w = int((~is_wrong_cell & ~rep_eq_clean).sum())

            # ---------- (1) Precision / Recall / F1 ----------
            # Precision = n_w2r / (n_w2r + n_r2w)
            precision_den = n_w2r + n_r2w
            precision = n_w2r / precision_den if precision_den else 0.0

            # Recall = n_w2r / n_w
            recall = n_w2r / n_w if n_w else 0.0

            # F1 = 2PR/(P+R)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

            # ---------- (2) EDR ----------
            # EDR = (n_w2r - n_r2w) / n_w
            edr = (n_w2r - n_r2w) / n_w if n_w else 0.0

            # 收集
            results.append(
                {
                    "task_name": task_name,
                    "num": num_,
                    "dataset_id": dataset_id,
                    "error_rate": error_rate,
                    "m": m_,
                    "n": n_,
                    "anomaly": anomaly,
                    "missing": missing,
                    "cleaning_method": cleaning_method,
                    "precision": precision,
                    "recall": recall,
                    "F1": f1,
                    "EDR": edr,
                }
            )

    # 4. 按 task_name 分组写出 CSV
    grouped = defaultdict(list)
    for row in results:
        grouped[row["task_name"]].append(row)

    output_base = here / "../../../results/analysis_results"
    output_base.mkdir(parents=True, exist_ok=True)

    out_columns = [
        "task_name",
        "num",
        "dataset_id",
        "error_rate",
        "m",
        "n",
        "anomaly",
        "missing",
        "cleaning_method",
        "precision",
        "recall",
        "F1",
        "EDR",
    ]

    for tname, rows in grouped.items():
        out_path = output_base / f"{tname}_cleaning.csv"
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=out_columns)
            writer.writeheader()
            writer.writerows(rows)
        print(f"[INFO] Cleaning analysis results saved to {out_path}")


if __name__ == "__main__":
    main()

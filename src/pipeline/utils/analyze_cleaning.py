import json
import os
import pandas as pd
import numpy as np
import csv

from src.pipeline.train.distribution_analysis import parse_details


def main():
    # 1. 读取配置文件（相对于脚本位置）
    comparison_path = os.path.normpath(os.path.join(os.path.dirname(__file__),
                                                    "../train/comparison.json"))
    with open(comparison_path, "r", encoding="utf-8") as f:
        comparison_data = json.load(f)

    # 2. 存放所有算法评价行的列表
    results = []

    # 3. 遍历每条配置
    for item in comparison_data:
        task_name = item["task_name"]
        num_ = item["num"]
        dataset_id = item["dataset_id"]
        error_rate = item["error_rate"]
        m_ = item["m"]
        n_ = item["n"]

        # 从 details 中解析 anomaly, missing
        details_str = item.get("details", "")
        d_dict = parse_details(details_str)
        anomaly = d_dict["anomaly"]
        missing = d_dict["missing"]


        # 读取相对于 comparison.json 的路径 => 需要拼接
        clean_csv_path = os.path.normpath(os.path.join(os.path.dirname(comparison_path),
                                                       item["paths"]["clean_csv"]))
        dirty_csv_path = os.path.normpath(os.path.join(os.path.dirname(comparison_path),
                                                       item["paths"]["dirty_csv"]))

        # --- 3.1 读取clean.csv, dirty.csv
        df_clean = pd.read_csv(clean_csv_path, keep_default_na=False)
        df_dirty = pd.read_csv(dirty_csv_path, keep_default_na=False)

        # 确保行数、列数匹配
        if df_clean.shape != df_dirty.shape:
            print(f"[WARNING] {task_name}-{num_} clean.csv and dirty.csv shape mismatch. Skipped.")
            continue

        n_rows, n_cols = df_clean.shape

        # 3.2 确定最初哪些单元格是错误(#dw) 或 正确(#dr)
        is_wrong_cell = np.zeros((n_rows, n_cols), dtype=bool)
        for r in range(n_rows):
            for c in range(n_cols):
                val_dirty = str(df_dirty.iat[r,c])
                val_clean = str(df_clean.iat[r,c])
                if val_dirty == val_clean:
                    is_wrong_cell[r,c] = False  # initially correct
                else:
                    is_wrong_cell[r,c] = True   # initially wrong

        dw = is_wrong_cell.sum()  # #dw

        # 3.3 遍历各算法repaired_csv，计算Precision/Recall/F1, EDR
        for cleaning_method, rep_path in item["paths"]["repaired_paths"].items():
            # 拼接绝对路径
            repaired_csv_path = os.path.normpath(os.path.join(os.path.dirname(comparison_path),
                                                              rep_path))
            if not os.path.exists(repaired_csv_path):
                print(f"[WARNING] Repaired file not found: {repaired_csv_path}. Skipped.")
                continue

            df_repaired = pd.read_csv(repaired_csv_path, keep_default_na=False)
            # 形状检查
            if df_repaired.shape != df_clean.shape:
                print(f"[WARNING] {task_name}-{num_} repaired shape mismatch with clean. Skipped {cleaning_method}.")
                continue

            # ---------- 统计 dw2r, dw2w, dr2r, dr2w ----------
            dw2r, dw2w, dr2r, dr2w = 0, 0, 0, 0
            for r in range(n_rows):
                for c in range(n_cols):
                    val_clean = str(df_clean.iat[r,c])
                    val_rep   = str(df_repaired.iat[r,c])

                    if is_wrong_cell[r,c]:
                        # originally wrong
                        if val_rep == val_clean:
                            dw2r += 1
                        else:
                            dw2w += 1
                    else:
                        # originally right
                        if val_rep == val_clean:
                            dr2r += 1
                        else:
                            dr2w += 1

            # (1) Precision, Recall, F1
            total_repaired = dw2r + dw2w + dr2r + dr2w
            precision = 0.0
            recall = 0.0
            f1 = 0.0

            if total_repaired > 0:
                precision = (dw2r + dr2r) / total_repaired
            total_dirty_repairs = dw2r + dw2w
            if total_dirty_repairs > 0:
                recall = dw2r / total_dirty_repairs
            if (precision + recall) > 1e-12:
                f1 = 2 * precision * recall / (precision + recall)

            # (2) EDR = (#dw2r - #dr2w) / #dw
            edr = 0.0
            if dw > 0:
                edr = (dw2r - dr2w) / dw

            # 收集结果
            row_dict = {
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
                "EDR": edr
            }
            results.append(row_dict)

    # 4. 输出CSV: ../../../results/analysis_results/{task_name}_cleaning.csv
    #    将所有记录汇总在一起再根据 task_name 分文件
    from collections import defaultdict
    grouped = defaultdict(list)
    for row in results:
        grouped[row["task_name"]].append(row)

    output_base = os.path.normpath(os.path.join(os.path.dirname(__file__),
        "../../../results/analysis_results"))
    os.makedirs(output_base, exist_ok=True)

    # 输出列名顺序
    out_columns = ["task_name", "num", "dataset_id", "error_rate",
                   "m", "n", "anomaly", "missing", "cleaning_method",
                   "precision", "recall", "F1", "EDR"]

    for tname, rows in grouped.items():
        out_path = os.path.join(output_base, f"{tname}_cleaning.csv")
        with open(out_path, mode="w", newline='', encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=out_columns)
            writer.writeheader()
            for rdict in rows:
                writer.writerow(rdict)

        print(f"[INFO] Cleaning analysis results saved to {out_path}")

if __name__ == "__main__":
    main()

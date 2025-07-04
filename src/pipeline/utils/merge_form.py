"""
merge_to_summary.py
-------------------
把 <dataset>_cleaning.csv 与 <dataset>_cluster.csv 合并为 <dataset>_summary.xlsx。

依赖:
    pandas>=1.3
    openpyxl   # 由 pandas.to_excel 自动调用

用法:
    python merge_to_summary.py      # 一次性处理四个数据集
    python merge_to_summary.py beers  # 只处理指定数据集
"""
from pathlib import Path
import sys
import pandas as pd

BASE_DIR   = Path("../../../results/analysis_results")
DATASETS   = ["beers", "hospital", "flights", "rayyan"]

# 用于左右表连接的公共字段
KEY_COLS = [
    "task_name", "num", "dataset_id", "error_rate",
    "m", "n", "anomaly", "missing", "cleaning_method"
]

# summary 中严格的列顺序
FINAL_COL_ORDER = [
    "task_name", "num", "dataset_id", "error_rate", "m", "n",
    "anomaly", "missing", "cleaning_method",
    "precision", "recall", "F1", "EDR",
    "cluster_method", "parameters",
    "Silhouette Score", "Davies-Bouldin Score", "Combined Score",
    "Sil_relative", "DB_relative", "Comb_relative",
]

def build_summary(ds: str) -> pd.DataFrame:
    """生成单个数据集的 summary DataFrame 并保存为 xlsx。"""
    cleaning_path = BASE_DIR / f"{ds}_cleaning.csv"
    cluster_path  = BASE_DIR / f"{ds}_cluster.csv"
    out_path      = BASE_DIR / f"{ds}_summary.xlsx"

    # 读取两张表
    clean_df   = pd.read_csv(cleaning_path)
    cluster_df = pd.read_csv(cluster_path)

    # 1️⃣ 内连接得到主表
    merged = pd.merge(clean_df, cluster_df, on=KEY_COLS, how="inner")

    # 2️⃣ 提取 GroundTruth 作为基线
    gt = (
        merged[merged["cleaning_method"] == "GroundTruth"]
        .loc[:, ["task_name", "num", "dataset_id", "cluster_method",
                 "Silhouette Score", "Davies-Bouldin Score", "Combined Score"]]
        .rename(columns={
            "Silhouette Score":      "Sil_gt",
            "Davies-Bouldin Score":  "DB_gt",
            "Combined Score":        "Comb_gt",
        })
    )

    # 3️⃣ 把基线指标并回主表
    merged = merged.merge(
        gt,
        on=["task_name", "num", "dataset_id", "cluster_method"],
        how="left",
    )

    # 4️⃣ 计算相对指标（GroundTruth 本身会自然得到 1）
    merged["Sil_relative"]  = merged["Silhouette Score"]       / merged["Sil_gt"]
    merged["DB_relative"]   = merged["DB_gt"]                  / merged["Davies-Bouldin Score"]
    merged["Comb_relative"] = merged["Combined Score"]         / merged["Comb_gt"]

    # 5️⃣ 只保留指定列并保持顺序
    summary = merged[FINAL_COL_ORDER].copy()

    # 6️⃣ 保存为 xlsx（Sheet 名与默认一致）
    summary.to_excel(out_path, index=False)
    print(f"[✓] {ds}: 生成 {out_path.relative_to(BASE_DIR.parent.parent)}")

    return summary

def main():
    targets = DATASETS if len(sys.argv) == 1 else [sys.argv[1]]
    for ds in targets:
        if ds not in DATASETS:
            print(f"[!] 未识别的数据集名称: {ds}")
            continue
        build_summary(ds)

if __name__ == "__main__":
    main()

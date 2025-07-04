import os
import json
import pandas as pd
import numpy as np

# ========== 全局配置 ==========
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "datasets", "test")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "..", "results", "test_eigenvectors.json")

def compute_missing_rate(df: pd.DataFrame) -> float:
    """
    计算缺失值占比。
    """
    total_cells = df.size
    missing_cells = df.isnull().sum().sum()
    return missing_cells / total_cells if total_cells > 0 else 0.0

def compute_noise_rate(df: pd.DataFrame) -> float:
    """
    基于IQR检测数值型离群点，返回占比。
    """
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return 0.0

    Q1 = numeric_df.quantile(0.25)
    Q3 = numeric_df.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((numeric_df < (Q1 - 1.5 * IQR)) | (numeric_df > (Q3 + 1.5 * IQR))).sum().sum()
    total_elements = numeric_df.size
    noise_rate = outliers / total_elements if total_elements > 0 else 0.0
    return noise_rate

def process_single_file_test(csv_path: str, dataset_id: int, dataset_name: str) -> dict:
    """
    读取测试数据集的 CSV 文件，计算特征向量并返回字典。
    """
    df = pd.read_csv(csv_path)

    # 基础信息
    num_samples = df.shape[0]
    num_features = df.shape[1]

    # 计算
    missing_rate = compute_missing_rate(df)
    noise_rate = compute_noise_rate(df)

    file_name = os.path.basename(csv_path)

    # 生成 x 特征向量
    x = [
        float(file_name.replace('%', '').replace('.csv', '')),  # 从文件名提取 error_rate
        missing_rate,
        noise_rate,
        num_features,
        num_samples
    ]

    return {
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "csv_file": file_name,
        "x": x
    }

def process_test_datasets():
    """
    处理测试数据集文件夹，生成特征向量并保存为 JSON 文件。
    """
    existing_data = []

    if not os.path.isdir(TEST_DATA_DIR):
        print(f"错误: TEST_DATA_DIR {TEST_DATA_DIR} 不存在或不是文件夹。")
        return None

    dataset_id_counter = 0

    for dataset_name in os.listdir(TEST_DATA_DIR):
        sub_folder = os.path.join(TEST_DATA_DIR, dataset_name)
        if not os.path.isdir(sub_folder):
            continue

        csv_files = [f for f in os.listdir(sub_folder) if f.endswith(".csv")]
        if not csv_files:
            print(f"警告: 测试数据集 {dataset_name} 文件夹下无 CSV 文件，跳过。")
            continue

        for csv_file in csv_files:
            csv_path = os.path.join(sub_folder, csv_file)

            # 从文件名中提取 error_rate，跳过 0.01% 的数据集
            if csv_file == "clean.csv":
                print(f"跳过 clean 的数据集: {dataset_name}/{csv_file}")
                continue

            # 将 dataset_id 设置为递增的整数值
            dataset_id = dataset_id_counter
            dataset_id_counter += 1

            feature_vector = process_single_file_test(csv_path, dataset_id, dataset_name)
            existing_data.append(feature_vector)

            print(f"[{dataset_id}] 完成处理: {dataset_name}/{csv_file}")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, indent=4, ensure_ascii=False)

    print(f"\n所有测试数据处理完成，共 {len(existing_data)} 条记录已写入: {OUTPUT_FILE}")
    return OUTPUT_FILE
import os
import json
import re
from typing import List, Dict, Any
from collections import defaultdict

def get_k_value(preprocessing_file_path: str) -> int:
    """
    从 pre-processing.py 文件中读取常量 K_VALUE 的值。
    """
    k_value = 5  # 默认值
    try:
        with open(preprocessing_file_path, "r", encoding="utf-8") as f:
            content = f.read()
            match = re.search(r"K_VALUE\s*=\s*(\d+)", content)
            if match:
                k_value = int(match.group(1))
            else:
                print("[WARNING] 未找到 K_VALUE，使用默认值 5")
    except Exception as e:
        print(f"[ERROR] 无法读取 K_VALUE: {e}")
    return k_value


def parse_cluster_file(directory_path: str, dataset_id: int):
    """
    解析聚类结果文件 (位于指定目录下的 repaired_{dataset_id}.txt 文件)，
    从中提取:
      - Best parameters (返回一个 dict)
      - Final Combined Score (返回浮点数)
    """
    best_params = {}
    final_score = 0.0
    try:
        # 构建具体文件路径
        file_path = os.path.join(directory_path, f"repaired_{dataset_id}.txt")
        if not os.path.isfile(file_path):
            print(f"[ERROR] 文件 {file_path} 不存在")
            return best_params, final_score

        # 解析文件内容
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().splitlines()

        for line in content:
            # 解析 Best parameters
            if "Best parameters" in line:
                parts = line.split(":", 1)
                if len(parts) == 2:
                    param_str = parts[1].strip()
                    param_pairs = param_str.split(",")
                    for pp in param_pairs:
                        key, value = pp.strip().split("=")
                        try:
                            best_params[key] = float(value)
                        except ValueError:
                            best_params[key] = value
            # 解析 Final Combined Score
            if "Final Combined Score" in line:
                match_score = re.search(r"Final Combined Score\s*:\s*([\d\.]+)", line)
                if match_score:
                    final_score = float(match_score.group(1))
    except Exception as e:
        print(f"[ERROR] 无法解析文件 {file_path}: {e}")

    return best_params, final_score


def save_analyzed_results(
        preprocessing_file_path: str,
        eigenvectors_path: str,
        clustered_results_path: str,
        output_path: str
):
    """
    1) 从 pre-processing.py 中获取 k_value
    2) 遍历所有 dataset_id (通过 eigenvectors.json 获取)
    3) 对每个 dataset_id, 在 clustered_results.json 中查找所有方法,
       解析 clustered_file_path, 获取 best_params, final_score
    4) 将所有策略按 final_score 排序, 取 top-K（同时过滤掉 final_score >= 3.0 的数据）
    5) 以指定格式保存到 output_path
    """

    # 1) 获取 K 值
    k_value = get_k_value(preprocessing_file_path)
    print(f"[INFO] Top-K 值: {k_value}")

    # 2) 读取 eigenvectors.json
    try:
        with open(eigenvectors_path, "r", encoding="utf-8") as f:
            eigenvectors_list = json.load(f)
    except Exception as e:
        print(f"[ERROR] 无法读取 {eigenvectors_path}: {e}")
        return

    dataset_ids = [item["dataset_id"] for item in eigenvectors_list]

    # 3) 读取 clustered_results.json
    try:
        with open(clustered_results_path, "r", encoding="utf-8") as f:
            clustered_results = json.load(f)
    except Exception as e:
        print(f"[ERROR] 无法读取 {clustered_results_path}: {e}")
        return

    dataset_methods = defaultdict(list)
    for method_info in clustered_results:
        dataset_id = method_info.get("dataset_id")
        if dataset_id is not None:
            dataset_methods[dataset_id].append(method_info)

    # 4) 遍历每个 dataset_id 的方法，过滤掉 final_score >= 3.0 的记录
    analyzed_results = []
    for dataset_id in dataset_ids:
        if dataset_id not in dataset_methods:
            print(f"[WARNING] dataset_id {dataset_id} 在 clustered_results.json 中未找到记录，跳过。")
            continue

        strategy_list = []
        for method_info in dataset_methods[dataset_id]:
            cleaning_alg = method_info.get("cleaning_algorithm", "unknown_cleaning")
            clustering_alg = method_info.get("clustering_name", "unknown_clustering")
            directory_path = method_info.get("clustered_file_path", "")

            # 使用 dataset_id 定位具体的 repaired 文件
            best_params, final_score = parse_cluster_file(directory_path, dataset_id)
            # 过滤掉综合得分大于等于 3.0 的数据
            if final_score is not None and final_score >= 3.0:
                continue
            strategy_list.append([cleaning_alg, clustering_alg, best_params, final_score])

        strategy_list_sorted = sorted(strategy_list, key=lambda x: x[3] if x[3] is not None else -1, reverse=True)
        top_k = strategy_list_sorted[:k_value]

        analyzed_results.append({
            "dataset_id": dataset_id,
            "top_k": top_k
        })

    # 5) 保存结果
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(analyzed_results, f, ensure_ascii=False, indent=4)
        print(f"[INFO] 分析结果已保存到 {output_path}")
    except Exception as e:
        print(f"[ERROR] 无法保存分析结果: {e}")

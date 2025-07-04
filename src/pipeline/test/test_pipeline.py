import os
import json
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from enum import Enum
import multiprocessing

from pre_processing_test import process_test_datasets
from test_classify import run_test_classification
from function_back import function_back

# 从 test_error_correction 模块中导入运行清洗任务的函数
from src.pipeline.test.test_error_correction import run_test_error_correction
from src.pipeline.train.cluster_methods import run_clustering
from src.pipeline.test.test_analysis import save_test_analyzed_results

# 全局配置
BASE_DIR = os.path.dirname(__file__)
DATASETS_DIR = os.path.join(BASE_DIR, "..", "..", "..", "datasets", "test")
RESULTS_DIR = os.path.join(BASE_DIR, "..", "..", "..", "results")

TEST_DATA_FILE = os.path.join(RESULTS_DIR, "test_eigenvectors.json")
MODEL_FILE = os.path.join(RESULTS_DIR, "xgboost_multilabel_model.joblib")
BINARIZER_FILE = os.path.join(RESULTS_DIR, "multilabel_binarizer.joblib")
PREDICTIONS_OUTPUT_FILE = os.path.join(RESULTS_DIR, "test_predictions.json")
STRATEGIES_OUTPUT_FILE = os.path.join(RESULTS_DIR, "test_strategies.json")

TEST_CLEANED_RESULTS_PATH = os.path.join(RESULTS_DIR, "test_cleaned_results.json")
TEST_CLUSTERED_RESULTS_PATH = os.path.join(RESULTS_DIR, "test_clustered_results.json")
TEST_ANALYZED_RESULTS_PATH = os.path.join(RESULTS_DIR, "test_analyzed_results.json")


# 定义聚类方法的枚举
class ClusterMethod(Enum):
    AP = 0         # Affinity Propagation
    DBSCAN = 1     # Density-Based Spatial Clustering
    GMM = 2        # Gaussian Mixture Model
    HC = 3         # Hierarchical Clustering
    KMEANS = 4     # K-means Clustering
    OPTICS = 5     # Ordering Points To Identify Cluster Structure


def try_run_error_correction(dataset_path, dataset_id, algorithm_id, clean_csv_path, output_dir, max_retries=2):
    for attempt in range(max_retries + 1):
        try:
            new_file_path, runtime = run_test_error_correction(
                dataset_path=dataset_path,
                dataset_id=dataset_id,
                algorithm_id=algorithm_id,
                clean_csv_path=clean_csv_path,
                output_dir=output_dir,
            )
            if new_file_path and runtime is not None:
                return new_file_path, runtime
            else:
                print(f"[WARNING] 第 {attempt+1} 次清洗返回空结果。", flush=True)
        except Exception as e:
            print(f"[WARNING] 第 {attempt+1} 次清洗异常: {e}", flush=True)
        time.sleep(1)
    return None, None


def process_dataset(record, work_dir):

    try:
        dataset_id = record.get("dataset_id")
        dataset_name = record.get("dataset_name")
        csv_file = record.get("csv_file")
        strategies = record.get("top_r", [])

        print(f"[INFO] [DatasetID={dataset_id}] 处理数据集: {dataset_name}, 文件: {csv_file}", flush=True)

        dataset_folder = os.path.join(work_dir, "datasets", "test", dataset_name)
        csv_path = os.path.join(dataset_folder, csv_file)
        clean_csv_path = os.path.join(dataset_folder, "clean.csv")

        if not os.path.exists(csv_path) or not os.path.exists(clean_csv_path):
            print(f"[WARNING] [DatasetID={dataset_id}] 数据文件不存在，跳过。", flush=True)
            return [], []

        # 按清洗算法分组（同一算法只运行一次）
        group = {}
        for strat in strategies:
            cleaning_algo = strat[0]
            group.setdefault(cleaning_algo, []).append(strat)

        dataset_cleaned_results = []
        dataset_clustered_results = []

        for cleaning_algo, strat_list in group.items():
            algorithm_id = 2 if cleaning_algo.lower() == "raha_baran" else 1
            print(f"[INFO] [DatasetID={dataset_id}] 运行清洗算法: {cleaning_algo}", flush=True)
            output_dir_clean = os.path.join(work_dir, "results", dataset_name, cleaning_algo)
            cleaned_file_path, cleaning_runtime = try_run_error_correction(
                dataset_path=csv_path,
                dataset_id=dataset_id,
                algorithm_id=algorithm_id,
                clean_csv_path=clean_csv_path,
                output_dir=output_dir_clean,
                max_retries=2
            )
            if not cleaned_file_path or cleaning_runtime is None:
                print(f"[ERROR] [DatasetID={dataset_id}] 清洗算法 {cleaning_algo} 运行失败", flush=True)
                continue
            print(f"[INFO] [DatasetID={dataset_id}] 清洗完成（{cleaning_algo}）：文件={cleaned_file_path}, 运行时间={cleaning_runtime:.2f} 秒", flush=True)
            dataset_cleaned_results.append({
                "dataset_id": dataset_id,
                "dataset_name": dataset_name,
                "cleaning_algorithm": cleaning_algo,
                "algorithm_id": algorithm_id,
                "cleaned_file_path": cleaned_file_path,
                "cleaning_runtime": cleaning_runtime
            })

            # 针对该清洗结果，依次运行各个聚类策略
            for strat in strat_list:
                clustering_algo = strat[1]
                try:
                    cluster_method_id = ClusterMethod[clustering_algo.upper()].value
                except KeyError:
                    cluster_method_id = 0
                print(f"[INFO] [DatasetID={dataset_id}] 使用清洗结果运行聚类：算法={clustering_algo}, cluster_method_id={cluster_method_id}", flush=True)
                cluster_output_dir, cluster_runtime = run_clustering(
                    dataset_id=dataset_id,
                    algorithm=cleaning_algo,
                    cluster_method_id=cluster_method_id,
                    cleaned_file_path=cleaned_file_path
                )
                if cluster_output_dir and cluster_runtime is not None:
                    dataset_clustered_results.append({
                        "dataset_id": dataset_id,
                        "dataset_name": dataset_name,
                        "cleaning_algorithm": cleaning_algo,
                        "cleaning_runtime": cleaning_runtime,
                        "clustering_algorithm": clustering_algo,
                        "cluster_method_id": cluster_method_id,
                        "clustering_runtime": cluster_runtime,
                        "clustered_file_path": cluster_output_dir,
                    })
                    print(f"[INFO] [DatasetID={dataset_id}] 聚类完成：算法={clustering_algo}, 运行时间={cluster_runtime:.2f} 秒", flush=True)
                else:
                    print(f"[ERROR] [DatasetID={dataset_id}] 聚类算法 {clustering_algo} 运行失败", flush=True)
                print("-" * 60, flush=True)
        return dataset_cleaned_results, dataset_clustered_results
    except Exception as e:
        print(f"[ERROR] 处理数据集时发生异常: {e}", flush=True)
        return [], []


def main():

    work_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

    # Step 1: 预处理测试数据集
    print("[STEP 1] 开始预处理测试数据集...", flush=True)
    output_file = process_test_datasets()
    if not output_file or not os.path.exists(output_file):
        print("[ERROR] 测试数据集预处理失败，流程中止。", flush=True)
        exit(1)
    print("[STEP 1] 测试数据集预处理完成！", flush=True)

    # Step 2: 分类测试数据
    print("[STEP 2] 开始分类测试数据...", flush=True)
    try:
        run_test_classification(
            test_data_path=TEST_DATA_FILE,
            model_path=MODEL_FILE,
            binarizer_path=BINARIZER_FILE,
            output_path=PREDICTIONS_OUTPUT_FILE,
            top_r=5
        )
        print("[STEP 2] 分类测试数据完成！", flush=True)
    except Exception as e:
        print(f"[ERROR] 分类测试数据失败: {e}", flush=True)
        exit(1)

    # Step 3: 映射预测结果到策略
    print("[STEP 3] 开始映射预测结果到策略...", flush=True)
    try:
        function_back(
            predictions_file=PREDICTIONS_OUTPUT_FILE,
            strategies_file=STRATEGIES_OUTPUT_FILE,
            eigenvectors_file=TEST_DATA_FILE
        )
        print("[STEP 3] 映射完成！", flush=True)
    except Exception as e:
        print(f"[ERROR] 映射预测结果失败: {e}", flush=True)
        exit(1)

    # Step 4: 执行清洗 & 聚类流程（使用多进程并行处理）
    print("[STEP 4] 开始执行测试清洗 & 聚类流程...", flush=True)
    if os.path.exists(STRATEGIES_OUTPUT_FILE):
        with open(STRATEGIES_OUTPUT_FILE, "r", encoding="utf-8") as f:
            test_strategies = json.load(f)
    else:
        print("[ERROR] 未找到策略记录文件，流程中止。", flush=True)
        exit(1)

    print(f"[INFO] 共 {len(test_strategies)} 条策略记录待处理。", flush=True)
    all_cleaned_results = []
    all_clustered_results = []

    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=2, mp_context=ctx) as executor:
        futures = [
            executor.submit(process_dataset, record, work_dir)
            for record in test_strategies
        ]
        for future in futures:
            try:
                cleaned, clustered = future.result()
                all_cleaned_results.extend(cleaned)
                all_clustered_results.extend(clustered)
            except Exception as e:
                print(f"[ERROR] 处理数据集时发生异常: {e}", flush=True)

    with open(TEST_CLEANED_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_cleaned_results, f, ensure_ascii=False, indent=4)
    print(f"[STEP 4] 测试清洗结果已保存到 {TEST_CLEANED_RESULTS_PATH}", flush=True)

    with open(TEST_CLUSTERED_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_clustered_results, f, ensure_ascii=False, indent=4)
    print(f"[STEP 4] 测试聚类结果已保存到 {TEST_CLUSTERED_RESULTS_PATH}", flush=True)

    # Step 5: 分析聚类结果
    print("[STEP 5] 开始分析测试聚类结果...", flush=True)
    try:
        save_test_analyzed_results(
            eigenvectors_path=TEST_DATA_FILE,
            clustered_results_path=TEST_CLUSTERED_RESULTS_PATH,
            output_path=TEST_ANALYZED_RESULTS_PATH
        )
        print(f"[STEP 5] 测试聚类分析结果已保存到 {TEST_ANALYZED_RESULTS_PATH}", flush=True)
    except Exception as e:
        print(f"[ERROR] 分析测试聚类结果时发生错误: {e}", flush=True)

    print("[INFO] test_pipeline 全部流程完成！", flush=True)


if __name__ == "__main__":
    main()
import os
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from src.pipeline.train.cluster_methods import ClusterMethod
from src.pipeline.train.error_correction import run_error_correction
from src.pipeline.train.cluster_methods import run_clustering
from src.pipeline.train.clustered_analysis import save_analyzed_results
from src.pipeline.train.classifier_preparation import generate_training_data

def process_record(record_idx, record, work_dir):
    cleaned_results = []
    clustered_results = []

    dataset_id = record_idx
    dataset_name = record["dataset_name"]
    csv_file = record["csv_file"]
    error_rate = record["error_rate"]

    print(f"[INFO] 准备处理数据集: {dataset_name} (CSV: {csv_file}, error_rate={error_rate}%)")

    if abs(error_rate - 0.01) < 1e-12:
        print(f"[INFO] 检测到 clean 数据集 {dataset_name}，跳过清洗和聚类")
        print("=" * 50)
        return cleaned_results, clustered_results

    dataset_folder = os.path.join(work_dir, "datasets", "train", dataset_name)
    csv_path = os.path.join(dataset_folder, csv_file)
    clean_csv_path = os.path.join(dataset_folder, "clean.csv")

    if not os.path.exists(csv_path) or not os.path.exists(clean_csv_path):
        print(f"数据集 {dataset_name} 的文件路径不存在，跳过。")
        return cleaned_results, clustered_results

    # 将 strategies 修改为一个 dict: {算法编号: "算法名称"}
    strategies = {
        1: "mode",
        2: "raha_baran",
        3: "holoclean",
        4: "bigdansing",
        5: "boostclean",
        6: "horizon",
        7: "scared",
        8: "Unified",
        9: "google_gemini"
    }

    # 遍历字典的键和值
    for algo_id, algo_name in strategies.items():
        print(f"[INFO] 正在运行清洗策略: {algo_name}")
        new_file_path, runtime = run_error_correction(
            dataset_path=csv_path,
            dataset_id=dataset_id,
            algorithm_id=algo_id,  # 直接使用 algo_id
            clean_csv_path=clean_csv_path,
            output_dir=os.path.join(work_dir, "results", dataset_name, algo_name),
        )

        if new_file_path and runtime:
            print(f"清洗完成: Dataset={dataset_name}, Algo={algo_name}")
            print(f"结果文件路径: {new_file_path}")
            print(f"运行时间: {runtime:.2f} 秒")

            cleaned_results.append({
                "dataset_id": dataset_id,
                "algorithm": algo_name,
                "algorithm_id": algo_id,
                "cleaned_file_path": new_file_path,
                "runtime": runtime
            })

            for cluster_method_id in range(4):
                cluster_output_dir, cluster_runtime = run_clustering(
                    dataset_id=dataset_id,
                    algorithm=algo_name,
                    cluster_method_id=cluster_method_id,
                    cleaned_file_path=new_file_path
                )

                if cluster_output_dir and cluster_runtime:
                    clustered_results.append({
                        "dataset_id": dataset_id,
                        "cleaning_algorithm": algo_name,
                        "cleaning_runtime": runtime,
                        "clustering_algorithm": cluster_method_id,
                        "clustering_name": ClusterMethod(cluster_method_id).name,
                        "clustering_runtime": cluster_runtime,
                        "clustered_file_path": cluster_output_dir,
                    })
                    print(f"[INFO] 聚类完成: {ClusterMethod(cluster_method_id).name}, 运行时间: {cluster_runtime:.2f} 秒")
                else:
                    print(f"[ERROR] 聚类算法 {ClusterMethod(cluster_method_id).name} 运行失败")
        print("=" * 50)

    return cleaned_results, clustered_results

def main():
    preprocessing_file_path = os.path.join("pre-processing.py")
    work_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    eigenvectors_path = os.path.join(work_dir, "results", "eigenvectors.json")
    cleaned_results_path = os.path.join(work_dir, "results", "cleaned_results.json")
    clustered_results_path = os.path.join(work_dir, "results", "clustered_results.json")
    analyzed_results_path = os.path.join(work_dir, "results", "analyzed_results.json")

    if not os.path.exists(eigenvectors_path):
        print(f"未找到 {eigenvectors_path}, 请先运行 pre-processing.py.")
        return

    with open(eigenvectors_path, "r", encoding="utf-8") as f:
        all_records = json.load(f)

    if not all_records:
        print("eigenvectors.json 文件为空或没有记录.")
        return

    # 或者测试第一个数据集：
    # all_records = all_records[:1]

    cleaned_results = []
    clustered_results = []

    with ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context("spawn")) as executor:
        futures = [
            executor.submit(process_record, record_idx, record, work_dir)
            for record_idx, record in enumerate(all_records)
        ]

        for future in futures:
            try:
                result_cleaned, result_clustered = future.result()
                cleaned_results.extend(result_cleaned)
                clustered_results.extend(result_clustered)
            except Exception as e:
                print(f"[ERROR] 处理数据集时发生异常: {e}", flush=True)

    with open(cleaned_results_path, "w", encoding="utf-8") as f:
        json.dump(cleaned_results, f, ensure_ascii=False, indent=4)

    print(f"清洗结果已保存到 {cleaned_results_path}")

    with open(clustered_results_path, "w", encoding="utf-8") as f:
        json.dump(clustered_results, f, ensure_ascii=False, indent=4)

    print(f"聚类结果已保存到 {clustered_results_path}")

    print("[INFO] 开始分析聚类结果")
    save_analyzed_results(
        preprocessing_file_path=preprocessing_file_path,
        eigenvectors_path=eigenvectors_path,
        clustered_results_path=clustered_results_path,
        output_path=analyzed_results_path
    )
    print(f"[INFO] 分析结果已保存到 {analyzed_results_path}")

    print("[INFO] 开始生成训练数据")
    try:
        generate_training_data()
        print("[INFO] 训练数据已成功生成并保存")
    except Exception as e:
        print(f"[ERROR] 生成训练数据时发生错误: {e}")

if __name__ == "__main__":
    main()

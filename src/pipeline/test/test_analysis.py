import json
from collections import defaultdict
from src.pipeline.train.clustered_analysis import parse_cluster_file

def save_test_analyzed_results(
        eigenvectors_path: str,
        clustered_results_path: str,
        output_path: str
):
    # 1) 获取 r 值（这里 r_value 用于初始候选数，但最终输出固定为 5 项）
    r_value = 5
    print(f"[INFO] Top-r 值: {r_value}")

    # 2) 读取 eigenvectors.json
    try:
        with open(eigenvectors_path, "r", encoding="utf-8") as f:
            eigenvectors_list = json.load(f)
    except Exception as e:
        print(f"[ERROR] 无法读取 {eigenvectors_path}: {e}")
        return

    # 这里假设 eigenvectors_list 中的每个元素都有 "dataset_id" 键
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

    # 4) 遍历每个 dataset_id 的方法，构造 analyzed_results
    analyzed_results = []
    for dataset_id in dataset_ids:
        if dataset_id not in dataset_methods:
            print(f"[WARNING] dataset_id {dataset_id} 在 clustered_results 中未找到记录，跳过。")
            continue

        strategy_list = []
        for method_info in dataset_methods[dataset_id]:
            cleaning_alg = method_info.get("cleaning_algorithm", "unknown_cleaning")
            clustering_alg = method_info.get("clustering_algorithm", "unknown_clustering")
            directory_path = method_info.get("clustered_file_path", "")

            # 使用 dataset_id 定位具体的 repaired 文件，获取最佳参数和综合得分
            best_params, final_score = parse_cluster_file(directory_path, dataset_id)
            strategy_list.append([cleaning_alg, clustering_alg, best_params, final_score])

        # 对策略根据综合得分进行降序排序
        strategy_list_sorted = sorted(strategy_list, key=lambda x: x[3], reverse=True)

        # 筛选时：若策略的综合得分 >= 3.0，则忽略（最多忽略两个），否则加入候选列表
        selected = []
        ignored_count = 0
        for s in strategy_list_sorted:
            if s[3] >= 3.0 and ignored_count < 2:
                ignored_count += 1
                continue
            selected.append(s)
            if len(selected) == 5:
                break

        # 如果经过过滤后不足 5 项，则补充未加入候选的策略（不再过滤）
        if len(selected) < 5:
            for s in strategy_list_sorted:
                if s not in selected:
                    selected.append(s)
                    if len(selected) == 5:
                        break

        # 如果依然不足 5 项，则用默认值进行填充（默认值可根据实际需求调整）
        while len(selected) < 5:
            selected.append(["unknown_cleaning", "unknown_clustering", {}, 0])

        # ========== 在这里增加“去重并只保留3个”的步骤 ==========

        # 1) 对已经选出的 5 个策略做去重：
        #    将清洗算法、聚类算法、参数(字典)三者视为组合的唯一key
        deduped = []
        seen_keys = set()
        for s in selected:
            cleaning_alg, clustering_alg, params, score = s
            # 将 params 转成 JSON 字符串作为 hash key（保证 dict 顺序不会影响）
            params_str = json.dumps(params, sort_keys=True)
            unique_key = (cleaning_alg, clustering_alg, params_str)
            if unique_key not in seen_keys:
                deduped.append(s)
                seen_keys.add(unique_key)

        # 2) 只保留前3条（如果你的业务需要继续按 score 降序，可在去重前或后都行
        #    这里selected本就按score排序，所以 deduped 仍保持降序的顺序
        top_3 = deduped[:3]

        # 3) 如果去重后不足3条，则用默认值补足
        while len(top_3) < 3:
            top_3.append(["unknown_cleaning", "unknown_clustering", {}, 0])

        # 将这3条写到结果
        analyzed_results.append({
            "dataset_id": dataset_id,
            "top_r": top_3  # 只有3个
        })

    # 5) 保存结果到 output_path
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(analyzed_results, f, ensure_ascii=False, indent=4)
        print(f"[INFO] 分析结果已保存到", output_path)
    except Exception as e:
        print(f"[ERROR] 无法保存分析结果: {e}")

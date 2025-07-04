import os
import json


def load_json(file_path):
    """加载 JSON 文件并返回数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def compute_average_score(results, key_name):
    score_dict = {}
    for item in results:
        dataset_id = item.get("dataset_id")
        score_list = item.get(key_name, [])
        if score_list:
            total_score = sum(entry[3] for entry in score_list)
            avg_score = total_score / len(score_list)
            score_dict[dataset_id] = avg_score
    return score_dict

def compute_total_time(results):
    time_dict = {}
    for record in results:
        dataset_id = record.get("dataset_id")
        cleaning_runtime = record.get("cleaning_runtime", 0)
        clustering_runtime = record.get("clustering_runtime", 0)
        total_runtime = cleaning_runtime + clustering_runtime
        time_dict[dataset_id] = time_dict.get(dataset_id, 0) + total_runtime
    return time_dict


def main():
    # 获取当前脚本所在目录，并确定 results 目录的路径
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, "../../../results")

    # 文件路径
    test_analyzed_results_path = os.path.join(results_dir, "test_analyzed_results.json")
    analyzed_results_path = os.path.join(results_dir, "analyzed_results.json")
    clustered_results_path = os.path.join(results_dir, "clustered_results.json")
    test_clustered_results_path = os.path.join(results_dir, "test_clustered_results.json")

    # 加载数据
    test_analyzed_results = load_json(test_analyzed_results_path)
    analyzed_results = load_json(analyzed_results_path)
    clustered_results = load_json(clustered_results_path)
    test_clustered_results = load_json(test_clustered_results_path)

    # 计算平均分
    # avg_score2：优化后的 test_analyzed_results.json 中 "top_r" 得分平均值
    avg_score2 = compute_average_score(test_analyzed_results, "top_r")
    # avg_score1：优化前的 analyzed_results.json 中 "top_k" 得分平均值
    avg_score1 = compute_average_score(analyzed_results, "top_k")

    # 计算运行时间总和
    # time1：clustered_results.json 中的总运行时间
    time1 = compute_total_time(clustered_results)
    # time2：test_clustered_results.json 中的总运行时间
    time2 = compute_total_time(test_clustered_results)

    # 取四个文件中都存在的 dataset_id
    common_dataset_ids = set(avg_score1.keys()) & set(avg_score2.keys()) & set(time1.keys()) & set(time2.keys())

    computed_results = []
    for dataset_id in common_dataset_ids:
        s1 = avg_score1[dataset_id]
        s2 = avg_score2[dataset_id]
        t1 = time1[dataset_id]
        t2 = time2[dataset_id]

        # 防止除以 0
        if s1 == 0:
            loss_rate = 0
        else:
            loss_rate = (s1 - s2) / s1

        acceleration = (1 - loss_rate) * (t1 / t2) if t2 != 0 else 0

        computed_results.append({
            "dataset_id": dataset_id,
            "avg_score1": s1,
            "avg_score2": s2,
            "time1": t1,
            "time2": t2,
            "loss_rate": loss_rate,
            "acceleration": acceleration
        })

    # 将计算结果写入 computed_results.json
    output_path = os.path.join(results_dir, "computed_results.json")
    with open(output_path, "w", encoding='utf-8') as f:
        json.dump(computed_results, f, indent=4, ensure_ascii=False)

    print("计算完成，结果已保存到:", output_path)


if __name__ == "__main__":
    main()

import json
import os


def generate_training_data():
    """
    从 analyzed_results.json 和 eigenvectors.json 中提取训练数据，
    包括每个 dataset_id 的特征向量 x 和对应的标签集合 L。

    返回:
        list: 包含训练数据的列表，每个元素是一个字典，包含 dataset_id, x, 和 L。
    """
    # 文件路径
    base_path = "../../../results/"
    analyzed_results_path = os.path.join(base_path, "analyzed_results.json")
    eigenvectors_path = os.path.join(base_path, "eigenvectors.json")
    output_path = os.path.join(base_path, "training_labels.json")

    # 读取 JSON 数据
    with open(analyzed_results_path, "r", encoding="utf-8") as f:
        analyzed_results = json.load(f)

    with open(eigenvectors_path, "r", encoding="utf-8") as f:
        eigenvectors = json.load(f)

    # 创建字典便于快速查找 eigenvectors 中的特征向量
    eigenvector_dict = {item["dataset_id"]: item for item in eigenvectors}

    training_data = []

    # 遍历 analyzed_results
    for item in analyzed_results:
        dataset_id = item["dataset_id"]
        top_k = item["top_k"]

        # 获取 eigenvectors 中的特征向量
        if dataset_id in eigenvector_dict:
            vector_info = eigenvector_dict[dataset_id]
            x = [
                vector_info["error_rate"],
                vector_info["missing_rate"],
                vector_info["noise_rate"],
                vector_info["m"],
                vector_info["n"],
            ]
        else:
            # 如果 eigenvectors 中没有对应的 dataset_id，跳过
            print(f"警告：未找到 dataset_id={dataset_id} 的特征向量")
            continue

        # 构造训练数据条目
        training_data.append({
            "dataset_id": dataset_id,
            "x": x,
            "L": top_k
        })

    # 保存训练数据到文件
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(training_data, f, ensure_ascii=False, indent=4)

    print(f"训练数据已生成并保存到文件：{output_path}")

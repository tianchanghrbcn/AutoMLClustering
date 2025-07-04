import os
import json


def load_predictions(predictions_file):
    """
    加载预测结果 JSON 文件，并增加异常处理。
    """
    try:
        with open(predictions_file, "r", encoding="utf-8") as f:
            predictions = json.load(f)
        return predictions
    except FileNotFoundError:
        print(f"[ERROR] 预测文件 {predictions_file} 未找到。")
        return []
    except json.JSONDecodeError as e:
        print(f"[ERROR] 解析 {predictions_file} 时发生 JSONDecodeError: {e}")
        return []


def save_strategies(strategies, output_file):
    """
    保存映射后的策略到 JSON 文件，并增加异常处理。
    """
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(strategies, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"[ERROR] 无法保存策略到文件 {output_file}: {e}")


def parse_ap_parameters(ap_param, dataset_id):
    """
    解析 AP 参数字符串，格式应为 'damp_X-pref_Y'。
    增加容错处理：当参数格式不符合预期时，返回未知值，并输出警告。
    """
    try:
        # 检查字符串中是否包含 '-' 且同时包含 "damp" 和 "pref"
        if "-" in ap_param and "damp" in ap_param and "pref" in ap_param:
            damping_label, preference_label = ap_param.split("-")
            damping = "0.5-0.7" if damping_label == "damp_Low" else "0.7-0.9"
            preference = "-500 to -300" if preference_label == "pref_Low" else "-300 to -100"
            return {"damping": damping, "preference": preference}
        else:
            raise ValueError(f"AP 参数格式不符合预期: {ap_param}")
    except Exception as e:
        print(f"[WARNING] 无法解析 AP 参数：{ap_param} (dataset_id={dataset_id}) - {e}")
        return {"damping": "未知", "preference": "未知"}


def map_predictions_to_strategies(predictions):
    """
    将预测的标签映射回具体的策略配置，增加容错处理和健壮性检查。
    """
    strategies = []

    for prediction in predictions:
        dataset_id = prediction.get("dataset_id")
        top_labels = prediction.get("top_labels", [])

        dataset_strategies = {
            "dataset_id": dataset_id,
            "top_r": []
        }

        for label in top_labels:
            try:
                # 如果标签不是字符串，则跳过处理
                if not isinstance(label, str):
                    print(f"[WARNING] 非字符串格式的标签将被跳过: {label}")
                    continue

                parts = label.split("-")
                # 检查预测标签格式是否满足至少包含3部分（如：cleaning-算法-参数）
                if len(parts) < 3:
                    print(f"[WARNING] 预测标签格式错误: {label}")
                    continue

                cleaning_algorithm = parts[0]
                clustering_algorithm = parts[1]
                hyperparams = {}

                if clustering_algorithm == "KMEANS":
                    # 例如：cleaning-KMEANS-k_bin1
                    if parts[2] == "k_bin1":
                        hyperparams["k"] = "≤ sqrt(n)/2"
                    elif parts[2] == "k_bin2":
                        hyperparams["k"] = "(sqrt(n)/2, sqrt(n)]"
                    else:
                        hyperparams["k"] = "> sqrt(n)"
                elif clustering_algorithm == "AP":
                    # 对于 AP，后续参数可能包含多个 '-'，直接 join 拼接
                    ap_param = "-".join(parts[2:])
                    hyperparams = parse_ap_parameters(ap_param, dataset_id)
                elif clustering_algorithm == "DBSCAN":
                    # 期望格式：cleaning-DBSCAN-eps_Low-minS_Low
                    if len(parts) >= 4:
                        eps = parts[2]
                        min_samples = parts[3]
                        hyperparams["eps"] = "0.1-1.0" if eps == "eps_Low" else "1.0-2.0"
                        hyperparams["min_samples"] = "5-25" if min_samples == "minS_Low" else "25-50"
                    else:
                        print(f"[WARNING] DBSCAN 参数格式错误: {label}")
                elif clustering_algorithm == "OPTICS":
                    # 期望格式：cleaning-OPTICS-minS_Low-xi_Low
                    if len(parts) >= 4:
                        min_samples = parts[2]
                        xi = parts[3]
                        hyperparams["min_samples"] = "5-15" if min_samples == "minS_Low" else "15-30"
                        hyperparams["xi"] = "0.01-0.05" if xi == "xi_Low" else "0.05-0.1"
                    else:
                        print(f"[WARNING] OPTICS 参数格式错误: {label}")
                elif clustering_algorithm == "HC":
                    # 例如：cleaning-HC-k_bin2
                    if parts[2] == "k_bin1":
                        hyperparams["k"] = "≤ sqrt(n)/2"
                    elif parts[2] == "k_bin2":
                        hyperparams["k"] = "(sqrt(n)/2, sqrt(n)]"
                    else:
                        hyperparams["k"] = "> sqrt(n)"
                elif clustering_algorithm == "GMM":
                    # 例如：cleaning-GMM-k_bin1-cov=full
                    if parts[2] == "k_bin1":
                        hyperparams["k"] = "≤ sqrt(n)/2"
                    elif parts[2] == "k_bin2":
                        hyperparams["k"] = "(sqrt(n)/2, sqrt(n)]"
                    else:
                        hyperparams["k"] = "> sqrt(n)"
                    if len(parts) >= 4 and "=" in parts[3]:
                        hyperparams["covariance_type"] = parts[3].split("=")[1]
                    else:
                        hyperparams["covariance_type"] = "未知"
                else:
                    print(f"[WARNING] 未知的聚类算法: {clustering_algorithm} in label {label}")
                    continue

                dataset_strategies["top_r"].append([cleaning_algorithm, clustering_algorithm, hyperparams])
            except Exception as e:
                print(f"[ERROR] 处理标签 {label} 时发生错误: {e}")
                continue

        strategies.append(dataset_strategies)

    return strategies


def function_back(predictions_file, strategies_file, eigenvectors_file):
    """
    主函数：将预测标签映射回具体策略，并添加 dataset_name 和 csv_file 字段。
    增加异常处理以确保整个流程的健壮性。
    """
    print("[INFO] 加载预测结果...")
    predictions = load_predictions(predictions_file)
    if not predictions:
        print("[ERROR] 没有加载到预测结果，程序终止。")
        return

    print("[INFO] 加载特征向量...")
    try:
        with open(eigenvectors_file, "r", encoding="utf-8") as f:
            eigenvectors = json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] 特征向量文件 {eigenvectors_file} 未找到。")
        return
    except json.JSONDecodeError as e:
        print(f"[ERROR] 解析 {eigenvectors_file} 时发生 JSONDecodeError: {e}")
        return

    # 创建 dataset_id 到 dataset_name 和 csv_file 的映射
    id_to_metadata = {}
    for record in eigenvectors:
        dataset_id = record.get("dataset_id")
        dataset_name = record.get("dataset_name")
        csv_file = record.get("csv_file")
        if dataset_id is not None:
            id_to_metadata[dataset_id] = {"dataset_name": dataset_name, "csv_file": csv_file}
        else:
            print(f"[WARNING] 特征向量记录缺少 dataset_id: {record}")

    print("[INFO] 映射预测标签到具体策略...")
    strategies = map_predictions_to_strategies(predictions)

    # 添加 dataset_name 和 csv_file 到策略中
    for strategy in strategies:
        dataset_id = strategy.get("dataset_id")
        metadata = id_to_metadata.get(dataset_id)
        if metadata:
            strategy["dataset_name"] = metadata.get("dataset_name", "未知")
            strategy["csv_file"] = metadata.get("csv_file", "未知")
        else:
            print(f"[WARNING] 未找到 dataset_id {dataset_id} 对应的元数据。")
            strategy["dataset_name"] = "未知"
            strategy["csv_file"] = "未知"

    print("[INFO] 保存映射后的策略...")
    save_strategies(strategies, strategies_file)
    print(f"[INFO] 策略映射完成，结果已保存到 {strategies_file}")


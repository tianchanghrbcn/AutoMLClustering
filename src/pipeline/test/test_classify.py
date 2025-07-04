import os
import json
import numpy as np
from joblib import load

# ========== 函数定义 ==========

def load_test_dataset(file_path):
    """
    加载测试数据集。
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_classification_model_and_binarizer(model_path, binarizer_path):
    """
    加载分类器模型和标签 Binarizer。
    """
    model = load(model_path)
    binarizer = load(binarizer_path)
    return model, binarizer

def classify_top_labels(model, binarizer, features, top_r):
    """
    对输入特征进行分类并返回 Top R 标签和对应概率。
    """
    predicted_probabilities = model.predict_proba(features)  # 概率预测
    label_classes = binarizer.classes_  # 所有可能标签

    predictions = []
    for index, probabilities in enumerate(predicted_probabilities):
        top_indices = np.argsort(probabilities)[-top_r:][::-1]  # 获取 Top R 索引
        top_labels = label_classes[top_indices]  # Top R 标签
        top_confidences = probabilities[top_indices]  # Top R 概率

        predictions.append({
            "dataset_id": index,
            "top_labels": top_labels.tolist(),
            "top_confidences": top_confidences.tolist(),
        })
    return predictions

def save_classification_results(results, output_path):
    """
    保存分类结果到 JSON 文件。
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

def run_test_classification(test_data_path, model_path, binarizer_path, output_path, top_r):
    """
    测试阶段主函数：加载测试数据、模型、进行预测并保存结果。
    """
    print("[INFO] 正在加载测试数据...")
    test_data = load_test_dataset(test_data_path)
    test_features = np.array([entry["x"] for entry in test_data])

    print("[INFO] 正在加载模型和 Binarizer...")
    model, binarizer = load_classification_model_and_binarizer(model_path, binarizer_path)

    print("[INFO] 开始分类并提取 Top R 标签...")
    classification_results = classify_top_labels(model, binarizer, test_features, top_r)

    print("[INFO] 正在保存分类结果...")
    save_classification_results(classification_results, output_path)
    print(f"[INFO] 分类完成，结果已保存到 {output_path}")

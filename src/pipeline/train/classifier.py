#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import math
import numpy as np
import time
from concurrent.futures import ProcessPoolExecutor

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import hamming_loss, f1_score
from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier
from joblib import dump

base_path = "../../../results/"
training_labels_path = os.path.join(base_path, "training_labels.json")

def load_training_data(json_path):
    """
    从 JSON 文件中读取训练数据，并返回列表结构。
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# ========== (1) 定义辅助函数：根据你的策略对参数做分段 ==========

def bin_k(k_value, sqrt_half, sqrt_n):
    """
    将聚类簇数 k 分为三段：
      1) k <= sqrt(n)/2
      2) sqrt(n)/2 < k <= sqrt(n)
      3) k > sqrt(n)
    返回 'k_bin1' / 'k_bin2' / 'k_bin3'
    """
    if k_value <= sqrt_half:
        return "k_bin1"
    elif k_value <= sqrt_n:
        return "k_bin2"
    else:
        return "k_bin3"

def bin_ap_params(damping, preference):
    """
    AP算法参数分段：
      - damping ∈ [0.5,0.7], [0.7,0.9]
      - preference ∈ [-500,-300], [-300,-100]
    返回字符串如 'damp_Low-pref_High'。
    """
    # damping
    if 0.5 <= damping <= 0.7:
        d_label = "damp_Low"
    else:
        d_label = "damp_High"

    # preference
    if -500 <= preference <= -300:
        p_label = "pref_Low"
    else:
        p_label = "pref_High"

    return f"{d_label}-{p_label}"

def bin_dbscan_params(eps, min_samples):
    """
    DBSCAN参数分段：
      - eps ∈ [0.1,1.0], [1.0,2.0]
      - min_samples ∈ [5,25], [25,50]
    """
    # eps
    if 0.1 <= eps <= 1.0:
        e_label = "eps_Low"
    else:
        e_label = "eps_High"

    # min_samples
    if 5 <= min_samples <= 25:
        m_label = "minS_Low"
    else:
        m_label = "minS_High"

    return f"{e_label}-{m_label}"

def bin_optics_params(min_samples, xi):
    """
    OPTICS参数分段：
      - min_samples ∈ [5,15],[15,30]
      - xi ∈ [0.01,0.05],[0.05,0.1]
    """
    # min_samples
    if 5 <= min_samples <= 15:
        ms_label = "minS_Low"
    else:
        ms_label = "minS_High"

    # xi
    if 0.01 <= xi <= 0.05:
        xi_label = "xi_Low"
    else:
        xi_label = "xi_High"

    return f"{ms_label}-{xi_label}"

# ========== (2) parse_data_for_multilabel：对参数做分段并拼装标签 ==========

def parse_data_for_multilabel(training_data):
    """
    解析 training_labels.json 的数据结构，将其拆分为：
      - X: 特征矩阵 (n_samples x n_features)
      - Y: 多标签列表 (n_samples x ?) 每个样本是一组字符串标签
    根据自定义分段策略，对聚类算法的参数进行离散化并拼接到标签中。
    """

    X = []
    Y = []

    # n = len(training_data) 用来做 k 的分段阈值
    n = len(training_data)
    sqrt_n = math.sqrt(n)
    sqrt_half = sqrt_n / 2.0

    for entry in training_data:
        # 1) 读取特征向量
        features = entry["x"]
        X.append(features)

        # 2) 构建多标签
        labels_for_this_sample = []

        for label_info in entry["L"]:
            method_type = label_info[0]
            method_name = label_info[1]
            hyperparams = label_info[2]

            # 处理 GMM 中可能出现的字段名差异 (covariance type -> covariance_type)
            if method_name == "GMM" and "covariance type" in hyperparams:
                hyperparams["covariance_type"] = hyperparams.pop("covariance type")

            if method_name == "KMEANS":
                k_val = hyperparams.get("k", 2)
                k_bin_label = bin_k(k_val, sqrt_half, sqrt_n)
                label_name = f"{method_type}-{method_name}-{k_bin_label}"

            elif method_name == "GMM":
                k_val = hyperparams.get("k", 2)
                k_bin_label = bin_k(k_val, sqrt_half, sqrt_n)
                cov_type = hyperparams.get("covariance_type", "full")
                label_name = f"{method_type}-{method_name}-{k_bin_label}-cov={cov_type}"

            elif method_name == "HC":
                k_val = hyperparams.get("k", 2)
                k_bin_label = bin_k(k_val, sqrt_half, sqrt_n)
                # linkage, metric 暂不区分
                label_name = f"{method_type}-{method_name}-{k_bin_label}"

            elif method_name == "AP":
                damping_val = hyperparams.get("damping", 0.5)
                preference_val = hyperparams.get("preference", -500)
                ap_label = bin_ap_params(damping_val, preference_val)
                label_name = f"{method_type}-{method_name}-{ap_label}"

            elif method_name == "DBSCAN":
                eps_val = hyperparams.get("eps", 0.1)
                ms_val = hyperparams.get("min_samples", 5)
                db_label = bin_dbscan_params(eps_val, ms_val)
                label_name = f"{method_type}-{method_name}-{db_label}"

            elif method_name == "OPTICS":
                ms_val = hyperparams.get("min_samples", 5)
                xi_val = hyperparams.get("xi", 0.01)
                optics_label = bin_optics_params(ms_val, xi_val)
                label_name = f"{method_type}-{method_name}-{optics_label}"

            else:
                # 其他算法未定义参数分段时，原样合并
                label_name = f"{method_type}-{method_name}"

            labels_for_this_sample.append(label_name)

        Y.append(labels_for_this_sample)

    # 转为 NumPy 数组便于后续处理
    return np.array(X), Y

def build_multilabel_binarizer(all_label_sets):
    """
    根据所有样本的标签集合构建并拟合 MultiLabelBinarizer。
    """
    mlb = MultiLabelBinarizer()
    mlb.fit(all_label_sets)
    return mlb

def train_multilabel_classifier(X, Y):
    """
    使用 XGBoost 完成多标签分类（OneVsRest）。
    - 不做 train_test_split，因为只有训练数据。
    - 在训练集上直接评估 Hamming Loss、F1-Score。
    """
    # 1) 构建 MultiLabelBinarizer
    mlb = build_multilabel_binarizer(Y)
    Y_bin = mlb.transform(Y)

    # 2) 训练模型（OneVsRest + XGB）
    model = OneVsRestClassifier(
        XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            eval_metric='logloss',
            random_state=42
        )
    )

    model.fit(X, Y_bin)

    # 3) 在同一训练集上进行预测并评估
    Y_pred_bin = model.predict(X)

    # -- 评估指标 --
    hloss = hamming_loss(Y_bin, Y_pred_bin)
    f1_micro = f1_score(Y_bin, Y_pred_bin, average="micro")
    f1_macro = f1_score(Y_bin, Y_pred_bin, average="macro")

    print(f"[Train] Hamming Loss: {hloss:.4f}")
    print(f"[Train] F1-Score (Micro): {f1_micro:.4f}")
    print(f"[Train] F1-Score (Macro): {f1_macro:.4f}")

    return model, mlb

def save_model_and_binarizer(model, mlb, directory):
    """
    将模型和 MultiLabelBinarizer 保存到指定目录下，以便后续测试使用。
    """
    model_path = os.path.join(directory, "xgboost_multilabel_model.joblib")
    mlb_path = os.path.join(directory, "multilabel_binarizer.joblib")

    dump(model, model_path)
    dump(mlb, mlb_path)

    print(f"Model saved to: {model_path}")
    print(f"MultiLabelBinarizer saved to: {mlb_path}")

def save_training_predictions(X, Y, model, mlb, directory, training_data):
    """
    生成对训练集的预测（多标签），以 JSON 格式写入指定目录。
    这里也演示如何将预测结果与原数据 (如 dataset_id) 对应保存。
    """
    Y_pred_bin = model.predict(X)
    Y_pred_labels = mlb.inverse_transform(Y_pred_bin)

    # 把预测结果和原始 dataset_id 绑定，以便后续使用
    results = []
    for entry, pred_labels in zip(training_data, Y_pred_labels):
        result = {
            "dataset_id": entry.get("dataset_id", None),
            "predicted_labels": list(pred_labels)
        }
        results.append(result)

    output_path = os.path.join(directory, "training_predictions.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Training predictions saved to: {output_path}")

def process_training_data(training_data):
    """
    单个进程的训练数据处理：解析数据、训练模型并保存结果。
    """
    start_time = time.time()
    X, Y = parse_data_for_multilabel(training_data)
    model, mlb = train_multilabel_classifier(X, Y)
    end_time = time.time()
    print(f"[INFO] Process completed in {end_time - start_time:.2f} seconds")
    return model, mlb, X, Y

def main():
    # 1) 读取 JSON 数据（全部为训练集）
    training_data = load_training_data(training_labels_path)

    # 使用多进程加速训练
    start_time = time.time()
    with ProcessPoolExecutor(max_workers=8) as executor:  # 根据机器核心数调整 max_workers
        future = executor.submit(process_training_data, training_data)
        model, mlb, X, Y = future.result()

    end_time = time.time()
    print(f"[INFO] Total execution time: {end_time - start_time:.2f} seconds")

    # 4) 保存模型和 MultiLabelBinarizer
    save_model_and_binarizer(model, mlb, base_path)

    # 5) 将训练集预测结果输出到 JSON 文件（同目录）
    save_training_predictions(X, Y, model, mlb, base_path, training_data)

    print("Training completed successfully.")

if __name__ == "__main__":
    main()

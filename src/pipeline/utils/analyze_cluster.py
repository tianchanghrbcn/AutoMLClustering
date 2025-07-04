#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import csv

# 配置常量
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
TRAIN_CONFIG_PATH = os.path.join(BASE_DIR, "src", "pipeline", "train", "comparison.json")
# 输出聚类分析csv存放目录
OUTPUT_DIR = os.path.join(BASE_DIR, "results", "analysis_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 任务和清洗方法列表（仅处理这几个任务和方法）
TASK_NAMES = ["beers", "flights", "hospital", "rayyan"]
CLEANING_METHODS = ["mode", "bigdansing", "boostclean", "holoclean", "horizon", "scared", "baran", "Unified", "GroundTruth"]
# 聚类算法列表
CLUSTER_METHODS = ["KMEANSNF", "GMM", "HC", "KMEANS", "KMEANSPPS", "DBSCAN"]

# 定义输出csv的字段顺序
CSV_FIELDS = [
    "task_name", "num", "dataset_id", "error_rate", "m", "n",
    "anomaly", "missing",
    "cleaning_method", "cluster_method", "parameters",
    "Silhouette Score", "Davies-Bouldin Score", "Combined Score"
]

def parse_cluster_file(filepath):

    params = ""
    combined = silhouette = davies = None
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("Best parameters:"):
                    # 提取冒号后面的部分
                    params = line.split(":", 1)[1].strip()
                elif line.startswith("Final Combined Score:"):
                    combined = float(line.split(":", 1)[1].strip())
                elif line.startswith("Final Silhouette Score:"):
                    silhouette = float(line.split(":", 1)[1].strip())
                elif line.startswith("Final Davies-Bouldin Score:"):
                    davies = float(line.split(":", 1)[1].strip())
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
    return params, silhouette, davies, combined

def main():
    # 读取 comparison.json 文件
    try:
        with open(TRAIN_CONFIG_PATH, "r", encoding="utf-8") as f:
            config_list = json.load(f)
    except Exception as e:
        print(f"Error reading {TRAIN_CONFIG_PATH}: {e}")
        return

    # 以 task_name 为 key 收集所有聚类结果行（列表）
    results_by_task = { task: [] for task in TASK_NAMES }

    for entry in config_list:
        task_name = entry.get("task_name")
        if task_name not in TASK_NAMES:
            continue  # 仅处理指定的任务

        num = entry.get("num")
        dataset_id = entry.get("dataset_id")
        error_rate = entry.get("error_rate")
        m_val = entry.get("m")
        n_val = entry.get("n")
        # 解析 details字段，提取 anomaly, missing, format, knowledge 信息
        details = entry.get("details", "")
        # 假设格式为 "anomaly=2.50%, missing=0.00%, format=2.50%, knowledge=0.00%"
        anomaly = missing = fmt = knowledge = None
        try:
            for part in details.split(","):
                part = part.strip()
                if part.startswith("anomaly="):
                    anomaly = float(part.split("=")[1].replace("%", "").strip())
                elif part.startswith("missing="):
                    missing = float(part.split("=")[1].replace("%", "").strip())
        except Exception as e:
            print(f"Error parsing details in entry {entry}: {e}")

        # 从 paths 中读取配置
        paths = entry.get("paths", {})
        # 此处 clean_csv 可能用不到，我们主要关注 dirty_csv和 repaired_paths
        repaired_paths = paths.get("repaired_paths", {})

        # 对于每个清洗方法
        for cleaning_method, repaired_csv in repaired_paths.items():
            # 确保该清洗方法在我们关注的列表中
            if cleaning_method not in CLEANING_METHODS:
                continue
            # 对于每个聚类算法
            for cluster_method in CLUSTER_METHODS:
                # 构造聚类结果文件路径：
                # AutoMLClustering\results\clustered_data\{cluster_method}\{cleaning_method}\clustered_{dataset_id}\repaired_{dataset_id}.txt
                cluster_file = os.path.join(
                    BASE_DIR, "results", "clustered_data", cluster_method,
                    cleaning_method, f"clustered_{dataset_id}",
                    f"repaired_{dataset_id}.txt"
                )
                if not os.path.exists(cluster_file):
                    # 可以打印警告，也可忽略
                    # print(f"Cluster result file not found: {cluster_file}")
                    continue

                # 解析聚类结果文件
                params, silhouette, davies, combined = parse_cluster_file(cluster_file)
                # 如果解析失败（比如文件内容为空），跳过
                if combined is None or silhouette is None or davies is None:
                    # print(f"Incomplete clustering scores in {cluster_file}")
                    continue

                # 构造一行记录
                row = {
                    "task_name": task_name,
                    "num": num,
                    "dataset_id": dataset_id,
                    "error_rate": error_rate,
                    "m": m_val,
                    "n": n_val,
                    "anomaly": anomaly,
                    "missing": missing,
                    "cleaning_method": cleaning_method,
                    "cluster_method": cluster_method,
                    "parameters": params,
                    "Silhouette Score": silhouette,
                    "Davies-Bouldin Score": davies,
                    "Combined Score": combined
                }
                results_by_task[task_name].append(row)

    # 对每个 task_name 输出一个 csv 文件
    for task in TASK_NAMES:
        output_file = os.path.join(OUTPUT_DIR, f"{task}_cluster.csv")
        rows = results_by_task.get(task, [])
        if not rows:
            print(f"No clustering results found for task: {task}")
            continue
        try:
            with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDS)
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)
            print(f"Wrote clustering analysis for {task} to {output_file}")
        except Exception as e:
            print(f"Error writing csv for task {task}: {e}")

if __name__ == "__main__":
    main()

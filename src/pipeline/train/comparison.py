#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import re

# （1）先定义关键路径 & 全局变量
EIGENVECTORS_PATH = "../../../results/eigenvectors.json"  # 3层向上到 results/eigenvectors.json
TASK_NAMES = ["beers", "flights", "hospital", "rayyan"]
CLEANING_METHODS = ["mode", "bigdansing", "boostclean", "holoclean", "horizon", "scared", "baran", "Unified", "GroundTruth"]


# （2）工具函数：解析 explanation.txt 内容
def parse_explanation_file(explanation_path):
    scenario_dict = {}

    if not os.path.isfile(explanation_path):
        return scenario_dict

    with open(explanation_path, 'r', encoding='utf-8') as f:
        current_num = None
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Match lines like "5 => anomaly=0%, missing=5%"
            match = re.match(r'^(\d+)\s*=>\s*(.*)$', line)
            if match:
                current_num = int(match.group(1))
                details = match.group(2).strip()
                scenario_dict[current_num] = {
                    "details": details,
                    "scenario_info": ""
                }
            else:
                # If a line is enclosed in parentheses, treat it as scenario_info
                match_info = re.match(r'^\((.*)\)$', line)
                if match_info and current_num is not None:
                    scenario_dict[current_num]["scenario_info"] = match_info.group(1).strip()

    return scenario_dict


def main():
    # （3） 读取 eigenvectors.json
    eigenvectors_fullpath = os.path.abspath(EIGENVECTORS_PATH)
    if not os.path.isfile(eigenvectors_fullpath):
        print(f"[ERROR] eigenvectors.json not found at {eigenvectors_fullpath}")
        return

    with open(eigenvectors_fullpath, 'r', encoding='utf-8') as f:
        eigen_data = json.load(f)  # eigen_data是一个list，每个元素是个dict

    file_map = {}
    for rec in eigen_data:
        task_name = rec["dataset_name"]  # e.g. "flights"
        csv_file = rec["csv_file"]  # e.g. "flights_5.csv"
        dataset_id = rec["dataset_id"]

        base_name = os.path.splitext(csv_file)[0]  # "flights_5"
        # 取最后的 '_' 后面部分当 num
        parts = base_name.split('_')
        if len(parts) < 2:
            # 说明不符合 "xxx_num" 格式
            continue
        try:
            num = int(parts[-1])  # e.g. 5
        except:
            continue

        key = f"{task_name}_{num}"
        if key not in file_map:
            file_map[key] = []

        file_map[key].append(rec)

    # （4） 构建一个大的列表, 里面每个元素包含整合信息
    comparison_list = []

    # 遍历TASK_NAMES
    for tname in TASK_NAMES:
        # 解析 explanation
        explanation_path = f"../../../datasets/train/{tname}/{tname}_explanation.txt"
        scenario_dict = parse_explanation_file(explanation_path)

        # 干净数据路径
        clean_csv_path = f"../../../datasets/train/{tname}/clean.csv"

        # scenario_dict 形如 {5: {"scenario_info":"xx","details":"yy"}, 7:{...}}
        for num_val, scenario_info in scenario_dict.items():
            key = f"{tname}_{num_val}"

            # 脏数据
            dirty_csv_path = f"../../../datasets/train/{tname}/{tname}_{num_val}.csv"

            # 可能在 eigenvectors.json 里出现多个record
            rec_list = file_map.get(key, [])

            # 如果在 eigenvectors.json没匹配到, 也可以生成一个"空"记录
            if not rec_list:
                # 可能 eigenvectors.json没包含  => 在comparisons写一个无 dataset_id
                empty_obj = {
                    "task_name": tname,
                    "num": num_val,
                    "dataset_id": None,
                    "error_rate": None,
                    "missing_rate": None,
                    "noise_rate": None,
                    "m": None,
                    "n": None,
                    "scenario_info": scenario_info.get("scenario_info", ""),
                    "details": scenario_info.get("details", ""),
                    "paths": {
                        "clean_csv": clean_csv_path,
                        "dirty_csv": dirty_csv_path,
                        "repaired_paths": {}  # 这个为空
                    }
                }
                comparison_list.append(empty_obj)
            else:
                # 对于 rec_list 里的每条 eigen record
                for rec in rec_list:
                    dataset_id = rec["dataset_id"]
                    error_rate = rec.get("error_rate", None)
                    missing_rate = rec.get("missing_rate", None)
                    noise_rate = rec.get("noise_rate", None)
                    m = rec.get("m", None)
                    n = rec.get("n", None)

                    # 构建 repaired_paths dict
                    # e.g. "results/cleaned_data/{method}/repaired_{id}.csv"
                    repaired_dict = {}
                    for method in CLEANING_METHODS:
                        repaired_path = f"../../../results/cleaned_data/{method}/repaired_{dataset_id}.csv"
                        repaired_dict[method] = repaired_path

                    # 整合
                    combined_obj = {
                        "task_name": tname,
                        "num": num_val,
                        "dataset_id": dataset_id,
                        "error_rate": error_rate,
                        "noise_rate": noise_rate,
                        "m": m,
                        "n": n,
                        "scenario_info": scenario_info.get("scenario_info", ""),
                        "details": scenario_info.get("details", ""),
                        "paths": {
                            "clean_csv": clean_csv_path,
                            "dirty_csv": dirty_csv_path,
                            "repaired_paths": repaired_dict
                        }
                    }
                    comparison_list.append(combined_obj)

    # （5） 生成 comparison.json
    # 如果想加更多内容或字段, 可在 combined_obj 里补充
    out_path = "comparison.json"
    with open(out_path, 'w', encoding='utf-8') as outf:
        json.dump(comparison_list, outf, indent=2, ensure_ascii=False)

    print(f"[INFO] comparison.json generated successfully! Total records: {len(comparison_list)}")


if __name__ == "__main__":
    main()

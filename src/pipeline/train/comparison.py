#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
comparison.py · 解析 eigenvectors.json + explanation.txt，生成 comparison.json

变动：
    • 解析新版 explanation.txt（行格式: “01 | Anom=0%, Miss=5% → …”）
    • 将 details 统一映射为 “anomaly=0%, missing=5%” 等小写长写形式
"""

import json
import os
import re

# ---------- 配置 ----------
EIGENVECTORS_PATH = "../../../results/eigenvectors.json"
TASK_NAMES = ["beers", "flights", "hospital", "rayyan"]
CLEANING_METHODS = ["mode", "bigdansing", "boostclean", "holoclean",
                    "horizon", "scared", "baran", "Unified", "GroundTruth"]

# ---------- 解析 explanation.txt ----------
def _normalize_details(details_raw: str) -> str:
    """
    将 “Anom=5%, Miss=10%” → “anomaly=5%, missing=10%”
    """
    m = re.match(r'(?i)Anom\s*=\s*([\d.]+%)\s*,\s*Miss\s*=\s*([\d.]+%)', details_raw)
    if m:
        return f"anomaly={m.group(1)}, missing={m.group(2)}"
    # 若未匹配，直接小写返回
    return details_raw.lower()

def parse_explanation_file(explanation_path):
    """
    返回: { num: {"details": "...", "scenario_info": "..."} }
    行样例:
        01 | Anom=0%, Miss=5%  →  r_anom=0.00%, r_miss=5.00%, r_tot=5.00
    """
    out = {}
    if not os.path.isfile(explanation_path):
        return out

    with open(explanation_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("Error‑injection"):
                continue
            m = re.match(r'^(\d+)\s*\|\s*(.*?)\s*→\s*(.*)$', line)
            if not m:
                continue
            num          = int(m.group(1))
            details_part = _normalize_details(m.group(2).strip())
            info_part    = m.group(3).strip()
            out[num] = {"details": details_part, "scenario_info": info_part}
    return out

# ---------- 主流程 ----------
def main():
    # 1) 读取 eigenvectors.json
    eigenvectors_fullpath = os.path.abspath(EIGENVECTORS_PATH)
    if not os.path.isfile(eigenvectors_fullpath):
        print(f"[ERROR] eigenvectors.json not found at {eigenvectors_fullpath}")
        return
    with open(eigenvectors_fullpath, "r", encoding="utf-8") as f:
        eigen_data = json.load(f)

    # 2) 构建 task_num ➜ [records] 映射
    file_map = {}
    for rec in eigen_data:
        tname    = rec["dataset_name"]
        csv_file = rec["csv_file"]              # e.g. flights_3.csv
        try:
            num = int(os.path.splitext(csv_file)[0].split('_')[-1])
        except ValueError:
            continue
        key = f"{tname}_{num}"
        file_map.setdefault(key, []).append(rec)

    # 3) 整合 comparison_list
    comparison_list = []
    for tname in TASK_NAMES:
        exp_path = f"../../../datasets/train/{tname}/{tname}_explanation.txt"
        scenarios = parse_explanation_file(exp_path)
        clean_csv_path = f"../../../datasets/train/{tname}/clean.csv"

        for num_val, meta in scenarios.items():
            key = f"{tname}_{num_val}"
            dirty_csv_path = f"../../../datasets/train/{tname}/{tname}_{num_val}.csv"
            recs = file_map.get(key, [])

            if not recs:  # eigenvectors.json 无记录
                comparison_list.append({
                    "task_name": tname,
                    "num": num_val,
                    "dataset_id": None,
                    "error_rate": None,
                    "missing_rate": None,
                    "noise_rate": None,
                    "m": None,
                    "n": None,
                    "scenario_info": meta["scenario_info"],
                    "details": meta["details"],
                    "paths": {
                        "clean_csv": clean_csv_path,
                        "dirty_csv": dirty_csv_path,
                        "repaired_paths": {}
                    }
                })
            else:
                for rec in recs:
                    dataset_id   = rec["dataset_id"]
                    repaired_map = {
                        m: f"../../../results/cleaned_data/{m}/repaired_{dataset_id}.csv"
                        for m in CLEANING_METHODS
                    }
                    comparison_list.append({
                        "task_name": tname,
                        "num": num_val,
                        "dataset_id": dataset_id,
                        "error_rate": rec["error_rate"],
                        "missing_rate": rec["missing_rate"],
                        "noise_rate": rec["noise_rate"],
                        "m": rec["m"],
                        "n": rec["n"],
                        "scenario_info": meta["scenario_info"],
                        "details": meta["details"],
                        "paths": {
                            "clean_csv": clean_csv_path,
                            "dirty_csv": dirty_csv_path,
                            "repaired_paths": repaired_map
                        }
                    })

    # 4) 写出 comparison.json
    with open("comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison_list, f, indent=2, ensure_ascii=False)
    print(f"[INFO] comparison.json generated successfully! Total records: {len(comparison_list)}")

if __name__ == "__main__":
    main()

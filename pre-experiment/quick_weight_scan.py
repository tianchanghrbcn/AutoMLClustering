#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
grid_search_driver.py  ——  搜索 α ∈ [0,1]（20 等分）并汇总：
    • max_variance : 4 个数据集上 6 算法得分方差的最大值
    • median_avg   : 4 个数据集对应 6 算法得分中位数的平均值

运行示例
---------
python grid_search_driver.py               # 默认并行 4、Optuna 20 trial
python grid_search_driver.py --parallel 8  # 指定并行数
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------- 各算法脚本 ----------
SCRIPTS: Dict[str, Path] = {
    "DBSCAN":     Path("DBSCAN.py"),
    "GMM":        Path("GMM.py"),
    "HC":         Path("HC.py"),
    "KMEANS":     Path("KMEANS.py"),
    "KMEANSNF":   Path("KMEANSNF.py"),
    "KMEANSPPS":  Path("KMEANSPPS.py"),
}

RESULT_ROOT = Path("results") / "clustered_data"
LOG_DIR     = Path("outputs") / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ---------- 查找 summary ----------
def find_summary(algo: str, dataset: str) -> Path | None:
    algo_dir = RESULT_ROOT / algo
    if not algo_dir.exists():
        return None
    for p in algo_dir.glob(f"**/clustered_{dataset}/**/*_summary.json"):
        return p
    return None


# ---------- 子进程任务 ----------
def run_job(job_args: Tuple[Path, str, float, float, float, int]) -> dict:
    csv_path, algo, a, b, g, trials = job_args
    env = os.environ.copy()
    env.update({
        "CSV_FILE_PATH": str(csv_path.resolve()),
        "DATASET_ID":    csv_path.stem,
        "ALGO":          algo,
    })

    cmd = [
        sys.executable, str(SCRIPTS[algo]),
        "--alpha",  f"{a}",
        "--beta",   f"{b}",
        "--gamma",  f"{g}",
        "--trials", str(trials)
    ]

    t0      = datetime.now()
    proc    = subprocess.run(cmd, env=env,
                             capture_output=True, text=True)
    elapsed = (datetime.now() - t0).total_seconds()

    summary_fp = find_summary(algo, csv_path.stem)
    combined = None
    if summary_fp and summary_fp.exists():
        js = json.loads(summary_fp.read_text(encoding="utf-8"))
        # ① 顶层直接有 combined
        if "combined" in js and isinstance(js["combined"], (int, float)):
            combined = js["combined"]
        # ② nested raw / cleaned
        else:
            for v in js.values():
                if isinstance(v, dict) and "combined" in v:
                    combined = v["combined"]
                    break

    if combined is not None:
        return dict(status="OK", dataset=csv_path.stem, algo=algo,
                    alpha=a, combined=float(combined), elapsed=elapsed)

    # -------- 失败：写日志 --------
    log_name = f"{algo}_{csv_path.stem}_{a:.3f}.log".replace(" ", "")
    (LOG_DIR / log_name).write_text(
        "CMD: " + " ".join(cmd) +
        f"\nReturn code: {proc.returncode}\n\n--- STDOUT ---\n{proc.stdout}\n"
        f"--- STDERR ---\n{proc.stderr}",
        encoding="utf-8")
    tqdm.write(f"[FAIL] {algo}/{csv_path.stem} α={a:.3f} ({elapsed:.1f}s) → {log_name}")
    return dict(status="FAIL", dataset=csv_path.stem, algo=algo,
                alpha=a, elapsed=elapsed)


# ---------- 指标计算 ----------
def calc_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    for alpha, sub in df.groupby("alpha"):
        # 每个数据集 → 该 α 下 6 算法得分
        var_per_ds = sub.groupby("dataset")["combined"].var(ddof=0).fillna(0.0)
        med_per_ds = sub.groupby("dataset")["combined"].median()

        rows.append({
            "alpha":        alpha,
            "max_variance": var_per_ds.max(),
            "median_avg":   med_per_ds.mean()
        })
    return pd.DataFrame(rows).sort_values("alpha").reset_index(drop=True)


# ---------- 主入口 ----------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials",   type=int, default=20, help="Optuna trial 数")
    ap.add_argument("--parallel", type=int, default=4,  help="并行进程数")
    ap.add_argument("--out_dir",  default="outputs",    help="结果输出目录")
    args = ap.parse_args()

    # ------------ 数据集 ------------
    data_dir = Path("data")
    datasets = sorted(data_dir.glob("*.csv"))
    if not datasets:
        sys.exit("❌ data 目录没有 CSV 文件")

    # ------------ α 列表（20 等分） ------------
    alphas = np.linspace(0.0, 1.0, 20)      # 20 行结果
    print(f"搜索 α 数: {len(alphas)} → {np.round(alphas,4)}")

    jobs = [
        (csv, algo, float(a), float(1.0 - a), 0.0, args.trials)
        for a in alphas
        for csv in datasets
        for algo in SCRIPTS
    ]
    print(f"总任务数: {len(jobs)} "
          f"(数据集 {len(datasets)} × 算法 {len(SCRIPTS)} × α {len(alphas)})")

    ok, fail, results = 0, 0, []
    with mp.Pool(args.parallel) as pool:
        for res in tqdm(pool.imap_unordered(run_job, jobs),
                        total=len(jobs), ncols=95, desc="Running tasks"):
            if res["status"] == "OK":
                ok += 1; results.append(res)
            else:
                fail += 1
            tqdm.write(f"✔ {ok}  ✖ {fail}  / {len(jobs)}", end="\r")

    if not results:
        sys.exit("\n❌ 全部任务失败，请查看 logs")

    df_all        = pd.DataFrame(results)
    metrics_table = calc_metrics(df_all)

    out_dir = Path(args.out_dir); out_dir.mkdir(exist_ok=True)
    df_all.to_csv(out_dir / "all_runs.csv", index=False)
    metrics_table.to_csv(out_dir / "alpha_metrics.csv", index=False)

    print("\n===== 20 × 3 结果 =====")
    print(metrics_table.to_string(index=False, float_format="%.6f"))

    print(f"\n✔ 成功 {ok}  ✖ 失败 {fail}")
    if fail:
        print(f"查看失败日志 → {LOG_DIR.resolve()}")
    print(f"所有结果已保存至 → {out_dir.resolve()}")


if __name__ == "__main__":
    mp.freeze_support()
    main()

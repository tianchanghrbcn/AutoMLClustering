#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
grid_search_driver.py  ——  统一 α-β-γ 权重网格搜索（默认并行 4）

示例
-----
# 默认：step=0.15，20 次 Optuna 试验，4 进程并行
python grid_search_driver.py

# 指定并行数、Optuna 试验次数
python grid_search_driver.py --parallel 8 --trials 30

# 固定测试几组权重
python grid_search_driver.py --weights 0.3,0.3,0.4 0.5,0.2,0.3
"""
from __future__ import annotations   # 让 3.9 支持 -> 注释字符串中可写 X | Y 型

import argparse
import itertools
import json
import multiprocessing as mp
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------- 各算法脚本 ----------
SCRIPTS = {
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
def find_summary(algo: str, dataset: str) -> Optional[Path]:
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

    cmd = [sys.executable, str(SCRIPTS[algo]),
           "--alpha",  f"{a}", "--beta", f"{b}", "--gamma", f"{g}",
           "--trials", str(trials)]

    t0      = datetime.now()
    proc    = subprocess.run(cmd, env=env,
                             capture_output=True, text=True)
    elapsed = (datetime.now() - t0).total_seconds()

    summary_fp = find_summary(algo, csv_path.stem)
    if summary_fp and summary_fp.exists():
        js  = json.loads(summary_fp.read_text(encoding="utf-8"))
        sil = js.get("best_silhouette") or js.get("silhouette") or 0.0
        db  = js.get("best_db")         or js.get("davies_bouldin") or 0.0
        sse = js.get("best_sse")        or js.get("sse")            or 0.0
        return dict(status="OK", dataset=csv_path.stem, algo=algo,
                    alpha=a, beta=b, gamma=g,
                    silhouette=sil,
                    db_inv=0.0 if db == 0 else 1.0 / db,
                    sse_term=0.0 if sse == 0 else 1.0 - sse / (sse + 1e-9),
                    elapsed=elapsed)

    # -------- 失败：写日志 --------
    log_name = f"{algo}_{csv_path.stem}_{a}_{b}_{g}.log".replace(" ", "")
    (LOG_DIR / log_name).write_text(
        "CMD: " + " ".join(cmd) +
        f"\nReturn code: {proc.returncode}\n\n--- STDOUT ---\n{proc.stdout}\n"
        f"--- STDERR ---\n{proc.stderr}",
        encoding="utf-8")

    tqdm.write(f"[FAIL] {algo}/{csv_path.stem} α={a} β={b} γ={g} "
               f"({elapsed:.1f}s) → {log_name}")
    return dict(status="FAIL", dataset=csv_path.stem, algo=algo,
                alpha=a, beta=b, gamma=g, elapsed=elapsed)


# ---------- 评分 ----------
def evaluate_grid(df: pd.DataFrame) -> List[dict]:
    for m in ["silhouette", "db_inv", "sse_term"]:
        df[m + "_n"] = df.groupby("dataset")[m].transform(
            lambda x: (x - x.min()) / (x.max() - x.min() + 1e-12))

    rows = []
    for (a, b, g), sub in df.groupby(["alpha", "beta", "gamma"]):
        score = (0.4 * sub["silhouette_n"].mean() +
                 0.4 * sub["db_inv_n"].mean()    +
                 0.2 * sub["sse_term_n"].mean())
        rows.append(dict(alpha=a, beta=b, gamma=g, score=float(score)))
    return sorted(rows, key=lambda r: r["score"], reverse=True)


# ---------- 主入口 ----------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=float, default=0.15,
                    help="网格步长 (默认 0.15)")
    ap.add_argument("--weights", nargs="+",
                    help="显式权重组合，如 0.3,0.3,0.4 0.4,0.4,0.2")
    ap.add_argument("--trials", type=int, default=20,
                    help="各脚本 Optuna trial 数")
    ap.add_argument("--parallel", type=int, default=4,
                    help="并行进程数 (默认 4)")
    ap.add_argument("--out_dir", default="outputs", help="结果输出目录")
    args = ap.parse_args()

    # ------------ 数据集 ------------
    data_dir = Path("data")
    datasets = sorted(data_dir.glob("*.csv"))
    if not datasets:
        sys.exit("❌ data 目录没有 CSV 文件")

    # ------------ 权重网格 ------------
    if args.weights:
        grid: List[Tuple[float, float, float]] = [
            tuple(map(float, w.split(","))) for w in args.weights]
    else:
        vals = np.round(np.arange(0.2, 0.801, args.step), 3)
        grid = [(a, b, round(1 - a - b, 3))
                for a, b in itertools.product(vals, repeat=2)
                if a + b <= 1 + 1e-9]
    print(f"搜索权重组合数: {len(grid)}")

    jobs = [(csv, algo, a, b, g, args.trials)
            for (a, b, g) in grid
            for csv in datasets
            for algo in SCRIPTS]
    print(f"总任务数: {len(jobs)} (数据集 {len(datasets)} × 算法 {len(SCRIPTS)} × 权重 {len(grid)})")

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

    df_all     = pd.DataFrame(results)
    best_grid  = evaluate_grid(df_all)
    best_combo = best_grid[0]

    out_dir = Path(args.out_dir); out_dir.mkdir(exist_ok=True)
    df_all.to_csv(out_dir / "all_runs.csv", index=False)
    pd.DataFrame(best_grid).to_csv(out_dir / "grid_scores.csv", index=False)
    (out_dir / "best_weights.json").write_text(
        json.dumps(best_combo, indent=4, ensure_ascii=False), encoding="utf-8")

    print("\n🟢 最优权重")
    print(f"   alpha={best_combo['alpha']:.3f}, "
          f"beta={best_combo['beta']:.3f}, "
          f"gamma={best_combo['gamma']:.3f}")
    print(f"   平均归一化得分 = {best_combo['score']:.4f}")
    print(f"✔ 成功 {ok}  ✖ 失败 {fail}")
    if fail:
        print(f"查看失败日志 → {LOG_DIR.resolve()}")
    print(f"结果已保存到 {out_dir.resolve()}")


if __name__ == "__main__":
    mp.freeze_support()   # Windows 兼容
    main()

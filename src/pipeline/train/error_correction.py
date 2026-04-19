#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import shutil
import time
import re
import subprocess
from pathlib import Path

# =========================================================
# 新增：自动推导项目根目录（替代硬编码 /home/changtian/Cleaning-Clustering）
# 你的文件路径是：.../Cleaning-Clustering/src/pipeline/train/error_correction.py
# 所以 parents[3] 正好是 Cleaning-Clustering 根目录
# 也支持通过环境变量 CLEANING_CLUSTERING_ROOT 手动覆盖
# =========================================================
PROJECT_ROOT = Path(
    os.environ.get("CLEANING_CLUSTERING_ROOT", Path(__file__).resolve().parents[3])
).resolve()

# =========================================================
# 新增：用 conda run 替代绝对 python 路径
# - 优先用环境变量 CONDA_EXE（conda 激活时一般会有）
# - 否则用 PATH 里的 conda
# =========================================================
def conda_run_python(env_name: str):
    conda_exe = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if not conda_exe:
        raise RuntimeError(
            "[ERROR] 找不到 conda 可执行文件（CONDA_EXE 未设置且 PATH 中无 conda）。\n"
            "请先执行：source ~/miniconda3/etc/profile.d/conda.sh\n"
            "或确保 conda 在 PATH 中，再运行该脚本。"
        )
    return [conda_exe, "run", "-n", env_name, "python"]


# =========================================================
# 新增：统一计时辅助函数
# =========================================================
def run_with_time(cmd, *, cwd=None, text=True, stream=False):
    """
    使用 /usr/bin/time 获取墙钟时间（秒）。返回 (CompletedProcess, runtime)
    - stream=False  : 一次性阻塞执行，捕获 stdout/stderr
    - stream=True   : 保留实时输出（不捕获），适用于不需要解析时间的流式子进程
    """
    time_cmd = ["/usr/bin/time", "-f", "RUNTIME=%e"] + cmd
    if stream:
        tic = time.perf_counter()
        proc = subprocess.run(time_cmd, cwd=cwd)
        toc = time.perf_counter()
        runtime = toc - tic
        return proc, runtime

    proc = subprocess.run(
        time_cmd, cwd=cwd, text=text, capture_output=True
    )
    m = re.search(r"RUNTIME=(\d+\.\d+)", proc.stderr)
    runtime = float(m.group(1)) if m else None

    if proc.stdout:
        print(proc.stdout.strip())
    if proc.stderr:
        cleaned_stderr = re.sub(r"RUNTIME=\d+\.\d+", "", proc.stderr).strip()
        if cleaned_stderr:
            print(cleaned_stderr)

    if proc.returncode != 0:
        raise subprocess.CalledProcessError(
            proc.returncode, cmd, output=proc.stdout, stderr=proc.stderr
        )
    return proc, runtime


def run_error_correction(dataset_path, dataset_id, algorithm_id,
                         clean_csv_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # ----------------- 算法名称映射 -----------------
    algo_map = {
        1: "mode",
        2: "baran",
        3: "holoclean",
        4: "bigdansing",
        5: "boostclean",
        6: "horizon",
        7: "scared",
        8: "Unified",
        9: "google_gemini",
    }
    algo_name = algo_map.get(algorithm_id)
    if algo_name is None:
        print(f"[ERROR] 未支持的算法 ID: {algorithm_id}")
        return None, None

    task_name_1 = f"dataset_{dataset_id}_algo_{algorithm_id}"
    task_name_2 = os.path.basename(os.path.dirname(clean_csv_path))

    # =========================================================
    # 原来这里是硬编码 /home/changtian/Cleaning-Clustering/...
    # 现在用 PROJECT_ROOT 自动拼出来（逻辑结果一致）
    # =========================================================
    rule_path = str(PROJECT_ROOT / "datasets" / "train" / task_name_2 / "dc_rules_holoclean.txt")
    rule_path_2 = str(PROJECT_ROOT / "datasets" / "train" / task_name_2 / "dc_rules-validate-fd-horizon.txt")

    # =========================================================
    # 生成 command（保持原逻辑）
    # =========================================================
    if algorithm_id == 1:
        command = [
            "python", "../../cleaning/mode/mode.py",
            "--clean_path", clean_csv_path,
            "--dirty_path", dataset_path,
            "--task_name", task_name_2
        ]
        _, runtime = run_with_time(command)
        print(f"[INFO] Mode 任务 `{task_name_2}` 运行成功！（{runtime:.2f}s）")

    elif algorithm_id == 2:
        # ------------------------- Baran ----------------------
        try:
            df = pd.read_csv(dataset_path)
            index_attribute = df.columns[0]
        except Exception as e:
            print(f"读取数据集时出错: {e}")
            return None, None

        command = [
            "python", "../../cleaning/baran/correction_with_baran.py",
            "--dirty_path", dataset_path,
            "--clean_path", clean_csv_path,
            "--task_name", task_name_1,
            "--output_path", output_dir,
            "--index_attribute", index_attribute
        ]

        print(f"运行清洗算法 {algorithm_id}（{algo_name}），数据集编号: {dataset_id}")
        tic = time.perf_counter()
        process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True)
        repaired_file = None
        stdout_lines = []

        while process.poll() is None:
            output = process.stdout.readline()
            if output:
                stdout_lines.append(output.strip())
                print(output.strip())
                if "Repaired data saved to" in output:
                    m = re.search(r"Repaired data saved to\s+(.+\.csv)", output)
                    if m:
                        repaired_file = m.group(1).strip()
                        process.terminate()
                        break

        stdout, stderr = process.communicate()
        stdout_lines.extend(stdout.splitlines())
        full_stdout = "\n".join(stdout_lines)
        if not repaired_file:
            m = re.search(r"Repaired data saved to\s+(.+\.csv)",
                          full_stdout, re.MULTILINE)
            if m:
                repaired_file = m.group(1).strip()
        if not repaired_file:
            print("[ERROR] 未检测到清洗结果文件路径，可能清洗未正常完成")
            return None, None

        # ---------- ★ 把真实结果复制到 Repaired_res 目录 ----------
        baran_res_dir = PROJECT_ROOT / "src" / "cleaning" / "Repaired_res" / "baran" / task_name_2
        os.makedirs(baran_res_dir, exist_ok=True)
        dest_path = baran_res_dir / os.path.basename(repaired_file)
        try:
            shutil.copy(repaired_file, str(dest_path))
            print(f"[INFO] 已将 Baran 结果复制到 {dest_path}")
        except Exception as e:
            print(f"[ERROR] 复制 Baran 结果文件失败: {e}")
            return None, None
        # ------------------------------------------------------

        toc = time.perf_counter()
        runtime = toc - tic
        print(f"[INFO] Baran 任务 `{task_name_2}` 运行成功！（{runtime:.2f}s）")

    elif algorithm_id == 3:
        # 原来：timeout 1d /home/.../envs/hc37/bin/python ...
        # 现在：timeout 1d conda run -n hc37 python ...
        command = (
            ["timeout", "1d"]
            + conda_run_python("hc37")
            + [
                str(PROJECT_ROOT / "src" / "cleaning" / "holoclean-master" / "holoclean_run.py"),
                "--dirty_path", dataset_path,
                "--clean_path", clean_csv_path,
                "--task_name", task_name_2,
                "--rule_path", rule_path,
                "--onlyed", "0",
                "--perfected", "0",
            ]
        )
        _, runtime = run_with_time(command)
        print(f"[INFO] HoloClean 任务 `{task_name_2}` 运行成功！（{runtime:.2f}s）")

    elif algorithm_id == 4:
        # 原来：/home/.../envs/torch110/bin/python ...
        # 现在：conda run -n torch110 python ...
        command = (
            conda_run_python("torch110")
            + [
                "../../cleaning/BigDansing_Holistic/bigdansing.py",
                "--task_name", task_name_2,
                "--rule_path", rule_path,
                "--onlyed", "0",
                "--perfected", "0",
                "--dirty_path", dataset_path,
                "--clean_path", clean_csv_path
            ]
        )
        _, runtime = run_with_time(command)
        print(f"[INFO] BigDansing 运行完成！（{runtime:.2f}s）")

    elif algorithm_id == 5:
        # 原来：/home/.../envs/activedetect/bin/python ...
        # 现在：conda run -n activedetect python ...
        command = (
            conda_run_python("activedetect")
            + [
                "../../cleaning/BoostClean/activedetect/experiments/Experiment.py",
                "--task_name", task_name_2,
                "--rule_path", rule_path,
                "--onlyed", "0",
                "--perfected", "0",
                "--dirty_path", dataset_path,
                "--clean_path", clean_csv_path
            ]
        )
        _, runtime = run_with_time(command)
        print(f"[INFO] BoostClean 运行完成！（{runtime:.2f}s）")

    elif algorithm_id == 6:
        command = (
            conda_run_python("torch110")
            + [
                "../../cleaning/horizon/horizon.py",
                "--task_name", task_name_2,
                "--rule_path", rule_path_2,
                "--onlyed", "0",
                "--perfected", "0",
                "--dirty_path", dataset_path,
                "--clean_path", clean_csv_path
            ]
        )
        _, runtime = run_with_time(command)
        print(f"[INFO] Horizon 运行完成！（{runtime:.2f}s）")

    elif algorithm_id == 7:
        command = (
            conda_run_python("torch110")
            + [
                "../../cleaning/SCAREd/scared.py",
                "--task_name", task_name_2,
                "--rule_path", rule_path_2,
                "--onlyed", "0",
                "--perfected", "0",
                "--dirty_path", dataset_path,
                "--clean_path", clean_csv_path
            ]
        )
        _, runtime = run_with_time(command)
        print(f"[INFO] Scared 运行完成！（{runtime:.2f}s）")

    elif algorithm_id == 8:
        command = (
            conda_run_python("torch110")
            + [
                "../../cleaning/Unified/Unified.py",
                "--task_name", task_name_2,
                "--rule_path", rule_path_2,
                "--onlyed", "0",
                "--perfected", "0",
                "--dirty_path", dataset_path,
                "--clean_path", clean_csv_path
            ]
        )
        _, runtime = run_with_time(command)
        print(f"[INFO] Unified 运行完成！（{runtime:.2f}s）")

    elif algorithm_id == 9:
        command = (
            conda_run_python("gm39")
            + [
                "gemini_cleaning.py",
                "--input_csv", dataset_path,
                "--output_dir", output_dir
            ]
        )
        _, runtime = run_with_time(
            command,
            cwd=str(PROJECT_ROOT / "src" / "cleaning" / "google_gemini")
        )
        print(f"[INFO] Google Gemini 运行完成！（{runtime:.2f}s）")

    # =========================================================
    # 后处理 & 结果文件复制（保持原逻辑）
    # =========================================================
    repaired_res_dir = PROJECT_ROOT / "src" / "cleaning" / "Repaired_res" / algo_name / task_name_2
    pattern = re.compile(r'^repaired_.*\.csv$')
    repaired_csv_name = next(
        (f for f in os.listdir(repaired_res_dir) if pattern.match(f)), None
    )

    if repaired_csv_name is None:
        print(f"[ERROR] 在 {repaired_res_dir} 未找到修复后文件")
        return None, None

    repaired_file_path = repaired_res_dir / repaired_csv_name
    cleaned_data_dir = os.path.join("../../../results/cleaned_data", algo_name)
    os.makedirs(cleaned_data_dir, exist_ok=True)
    new_file_path = os.path.join(cleaned_data_dir, f"repaired_{dataset_id}.csv")
    shutil.copy(str(repaired_file_path), new_file_path)

    print(f"[INFO] 算法 {algorithm_id}（{algo_name}）修复文件已复制到 {new_file_path}")
    return new_file_path, runtime

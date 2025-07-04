import os
import pandas as pd
import shutil
import time
import re  # 用于解析 stdout 输出
import subprocess


def run_error_correction(dataset_path, dataset_id, algorithm_id, clean_csv_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # 给每个算法固定起一个名称，方便后续处理
    if algorithm_id == 1:
        algo_name = "mode"
    elif algorithm_id == 2:
        algo_name = "baran"
    elif algorithm_id == 3:
        algo_name = "holoclean"
    elif algorithm_id == 4:
        algo_name = "bigdansing"
    elif algorithm_id == 5:
        algo_name = "boostclean"
    elif algorithm_id == 6:
        algo_name = "horizon"
    elif algorithm_id == 7:
        algo_name = "scared"
    elif algorithm_id == 8:
        algo_name = "Unified"
    elif algorithm_id == 9:
        algo_name = "google_gemini"

    else:
        print(f"[ERROR] 未支持的算法 ID: {algorithm_id}")
        return None, None

    task_name_1 = f"dataset_{dataset_id}_algo_{algorithm_id}"
    task_name_2 = os.path.basename(os.path.dirname(clean_csv_path))
    rule_path = f"/root/AutoMLClustering/datasets/train/{task_name_2}/dc_rules_holoclean.txt"
    rule_path_2 = f"/root/AutoMLClustering/datasets/train/{task_name_2}/dc_rules-validate-fd-horizon.txt"
    file_name = os.path.basename(dataset_path)

    # ========== 根据算法 ID 生成不同的 command ==========

    # 算法 1：mode
    if algorithm_id == 1:
        command = [
            "python", "../../cleaning/mode/mode.py",
            "--clean_path", clean_csv_path,
            "--dirty_path", dataset_path,
            "--task_name", task_name_2
        ]

        try:
            subprocess.run(command, check=True)
            print(f"[INFO] Mode 任务 `{task_name_2}` 运行成功！")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] 运行 Mode 时出错: {e}")


    # 算法 2：baran
    elif algorithm_id == 2:

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

        try:
            subprocess.run(command, check=True)
            print(f"[INFO] Baran 任务 `{task_name_2}` 运行成功！")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] 运行 Baran 时出错: {e}")

    elif algorithm_id == 3:

        command = [
            "timeout", "1d", "/root/miniconda3/envs/hc37/bin/python",
            "/root/AutoMLClustering/src/cleaning/holoclean-master/holoclean_run.py",
            "--dirty_path", dataset_path,
            "--clean_path", clean_csv_path,
            "--task_name", task_name_2,
            "--rule_path", rule_path,
            "--onlyed", "0",
            "--perfected", "0",
        ]

        try:
            # **直接在指定 conda 环境下运行 HoloClean**
            subprocess.run(command, check=True)
            print(f"[INFO] HoloClean 任务 `{task_name_2}` 运行成功！")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] 运行 HoloClean 时出错: {e}")


    elif algorithm_id == 4:

        command = [
            "/root/miniconda3/envs/torch110/bin/python", "../../cleaning/BigDansing_Holistic/bigdansing.py",
            "--task_name", task_name_2,
            "--rule_path", rule_path,
            "--onlyed", "0",
            "--perfected", "0",
            "--dirty_path", dataset_path,
            "--clean_path", clean_csv_path
        ]

        try:
            subprocess.run(command, check=True)
            print("[INFO] BigDansing 运行完成！")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] 运行 BigDansing 时出错: {e}")


    elif algorithm_id == 5:

        command = [
            "/root/miniconda3/envs/activedetect/bin/python",
            "../../cleaning/BoostClean/activedetect/experiments/Experiment.py",
            "--task_name", task_name_2,
            "--rule_path", rule_path,
            "--onlyed", "0",
            "--perfected", "0",
            "--dirty_path", dataset_path,
            "--clean_path", clean_csv_path
        ]

        try:
            subprocess.run(command, check=True)
            print("[INFO] BoostClean 运行完成！")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] 运行 BoostClean 时出错: {e}")

    # 算法 6：示例占位
    elif algorithm_id == 6:

        command = [
            "/root/miniconda3/envs/torch110/bin/python",
            "../../cleaning/horizon/horizon.py",
            "--task_name", task_name_2,
            "--rule_path", rule_path_2,
            "--onlyed", "0",
            "--perfected", "0",
            "--dirty_path", dataset_path,
            "--clean_path", clean_csv_path
        ]

        try:
            subprocess.run(command, check=True)
            print("[INFO] Horizon 运行完成！")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] 运行 Horizon 时出错: {e}")

    # 算法 7：示例占位
    elif algorithm_id == 7:

        command = [
            "/root/miniconda3/envs/torch110/bin/python",
            "../../cleaning/SCAREd/scared.py",
            "--task_name", task_name_2,
            "--rule_path", rule_path_2,
            "--onlyed", "0",
            "--perfected", "0",
            "--dirty_path", dataset_path,
            "--clean_path", clean_csv_path
        ]

        try:
            subprocess.run(command, check=True)
            print("[INFO] Scared 运行完成！")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] 运行 Scared 时出错: {e}")


    # 算法 8：示例占位
    elif algorithm_id == 8:

        command = [
            "/root/miniconda3/envs/torch110/bin/python",
            "../../cleaning/Unified/Unified.py",
            "--task_name", task_name_2,
            "--rule_path", rule_path_2,
            "--onlyed", "0",
            "--perfected", "0",
            "--dirty_path", dataset_path,
            "--clean_path", clean_csv_path
        ]

        try:
            subprocess.run(command, check=True)
            print("[INFO] Unified 运行完成！")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] 运行 Unified 时出错: {e}")


    elif algorithm_id == 9:

        try:
            subprocess.run(
                [
                    "/root/miniconda3/envs/gm39/bin/python",
                    "gemini_cleaning.py",
                    "--input_csv", dataset_path,
                    "--output_dir", output_dir
                ],
                check=True,
                cwd="/root/AutoMLClustering/src/cleaning/google_gemini"  # 切换目录
            )
            print("[INFO] Google Gemini 运行完成！")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] 运行 Google Gemini 时出错: {e}")


    else:
        # 理论上不会走到这里，因为前面已经做了判断
        print(f"[ERROR] 算法 ID {algorithm_id} 未定义执行命令")
        return None, None

    # ========== 执行命令并捕获输出 ==========

    if algorithm_id in [2]:

        try:
            print(f"运行清洗算法 {algorithm_id}（{algo_name}），数据集编号: {dataset_id}")
            start_time = time.time()

            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            repaired_file = None
            stdout_lines = []  # 用于存储完整的标准输出

            while process.poll() is None:
                output = process.stdout.readline()
                if output:
                    stdout_lines.append(output.strip())  # 实时保存输出
                    print(output.strip())  # 实时打印输出

                    # 检测到清洗结果路径时立即停止
                    if "Repaired data saved to" in output:
                        match = re.search(r"Repaired data saved to\s+(.+\.csv)", output)
                        if match:
                            repaired_file = match.group(1).strip()
                            print(f"[INFO] 清洗结果文件路径: {repaired_file}")
                            process.terminate()
                            break

            # 处理剩余的标准输出内容
            stdout, stderr = process.communicate()
            stdout_lines.extend(stdout.splitlines())
            full_stdout = "\n".join(stdout_lines)

            if not repaired_file:
                # 如果未检测到路径，尝试从完整输出中匹配
                match = re.search(r"Repaired data saved to\s+(.+\.csv)", full_stdout, re.MULTILINE)
                if match:
                    repaired_file = match.group(1).strip()

            if not repaired_file:
                print("[ERROR] 未检测到清洗结果文件路径，可能清洗未正常完成")
                return None, None

            # 正常终止时处理后续逻辑
            end_time = time.time()
            runtime = end_time - start_time

            cleaned_data_dir = os.path.join("../../../results/cleaned_data", algo_name)
            os.makedirs(cleaned_data_dir, exist_ok=True)
            new_file_path = os.path.join(cleaned_data_dir, f"repaired_{dataset_id}.csv")
            shutil.copy(repaired_file, new_file_path)

            return new_file_path, runtime

        except Exception as ex:
            process.kill()  # 确保子进程被终止
            print(f"[ERROR] 清洗算法执行中断: {ex}")
            return None, None

    else:

        start_time = time.time()

        BASE_DIR = "/root/AutoMLClustering"

        repaired_res_dir = os.path.join(BASE_DIR, "src", "cleaning", "Repaired_res", algo_name, task_name_2)
        # 正则匹配：以repaired_开头，以.csv结尾，中间可以是任意字符
        pattern = re.compile(r'^repaired_.*\.csv$')

        repaired_csv_name = None  # 先设为None，用来存储匹配到的文件名
        for filename in os.listdir(repaired_res_dir):
            if pattern.match(filename):
                repaired_csv_name = filename
                break  # 如果只需要第一个符合条件的文件，则可以break

        repaired_file_path = os.path.join(repaired_res_dir, repaired_csv_name)

        if not os.path.isfile(repaired_file_path):
            print(f"[ERROR] 在 {repaired_file_path} 未找到修复后文件")
            return None, None

        # 直接复制到 results 目录下
        cleaned_data_dir = os.path.join("../../../results/cleaned_data", algo_name)
        os.makedirs(cleaned_data_dir, exist_ok=True)

        new_file_path = os.path.join(cleaned_data_dir, f"repaired_{dataset_id}.csv")
        shutil.copy(repaired_file_path, new_file_path)

        end_time = time.time()
        runtime = end_time - start_time

        print(f"[INFO] 算法 {algorithm_id}（{algo_name}）修复文件已直接复制到 {new_file_path}")
        return new_file_path, runtime

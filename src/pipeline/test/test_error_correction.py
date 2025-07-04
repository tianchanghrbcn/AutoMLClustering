import os
import subprocess
import pandas as pd
import shutil
import time
import re  # 用于解析 stdout 输出


def run_test_error_correction(dataset_path, dataset_id, algorithm_id, clean_csv_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    task_name = f"dataset_{dataset_id}_algo_{algorithm_id}"
    algo_name = "mode" if algorithm_id == 1 else "raha_baran"

    if algorithm_id == 1:
        command = [
            "python", "../../cleaning/mode/correction_with_mode.py",
            "--clean_path", clean_csv_path,
            "--dirty_path", dataset_path,
            "--task_name", task_name
        ]
    elif algorithm_id == 2:
        try:
            df = pd.read_csv(dataset_path)
            index_attribute = df.columns[0]
        except Exception as e:
            print(f"[ERROR] 读取数据集时出错: {e}")
            return None, None

        command = [
            "python", "../../cleaning/baran/correction_with_baran.py",
            "--dirty_path", dataset_path,
            "--clean_path", clean_csv_path,
            "--task_name", task_name,
            "--output_path", output_dir,
            "--index_attribute", index_attribute
        ]
    else:
        print(f"[ERROR] 未支持的算法编号 {algorithm_id}，数据集编号: {dataset_id}")
        return None, None

    max_retries = 2
    for attempt in range(max_retries + 1):
        try:
            print(f"[INFO] 运行清洗算法 {algorithm_id}（{algo_name}），数据集编号: {dataset_id}，尝试 {attempt + 1}")
            start_time = time.time()
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            full_output = ""
            detected = False  # 标志是否检测到修复结果
            while True:
                line = process.stdout.readline()
                if line:
                    print(line, end="")  # 实时打印输出
                    full_output += line
                    # 对于 raha_baran 算法，检测到修复结果立即终止进程
                    if algorithm_id == 2 and "Repaired data saved to" in line:
                        print("[INFO] 检测到修复结果，立即终止进程。")
                        detected = True
                        process.terminate()
                        time.sleep(1)  # 等待进程终止
                        break
                elif process.poll() is not None:
                    break

            retcode = process.poll()
            if retcode is None:
                process.wait()
            # 如果检测到修复结果，则将退出码视为 0
            if detected:
                retcode = 0

            if retcode != 0:
                print(f"[ERROR] 清洗脚本退出码非 0。")
                continue  # 重试

            match = re.search(r"Repaired data saved to\s*[:\-]?\s*(.+\.csv)", full_output, re.IGNORECASE)
            if match:
                repaired_file = match.group(1).strip()
                print(f"[INFO] 清洗结果文件路径: {repaired_file}")
                if not os.path.exists(repaired_file):
                    print(f"[WARNING] 检测到的文件路径 {repaired_file} 不存在。")
                    continue
                end_time = time.time()
                runtime = end_time - start_time

                cleaned_data_dir = os.path.join("../../../results/test_cleaned_data", algo_name)
                os.makedirs(cleaned_data_dir, exist_ok=True)
                new_file_path = os.path.join(cleaned_data_dir, f"repaired_{dataset_id}.csv")
                shutil.copy(repaired_file, new_file_path)
                return new_file_path, runtime
            else:
                print(f"[WARNING] 尝试 {attempt + 1}: 未检测到清洗结果文件路径。")
        except Exception as ex:
            print(f"[WARNING] 尝试 {attempt + 1} 清洗异常: {ex}")
        time.sleep(2)
    print("[ERROR] 未检测到清洗结果文件路径，可能清洗未正常完成")
    return None, None

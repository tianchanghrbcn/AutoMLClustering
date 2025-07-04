import argparse
import os
import pandas as pd
import numpy as np
import random
import string


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Inject only 'anomaly' and 'missing' errors into a dataset, skipping the first column (primary key)."
    )
    parser.add_argument("--input", required=True, help="Path to the clean CSV file.")
    parser.add_argument("--output", required=True, help="Directory to store corrupted CSVs.")
    parser.add_argument("--task_name", required=True,
                        help="Used for naming: output => {task_name}_{num}.csv, plus a {task_name}_explanation.txt.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (optional).")
    return parser.parse_args()


def main():
    args = parse_arguments()
    input_path = args.input
    output_dir = args.output
    task_name = args.task_name
    seed = args.seed

    # 若指定随机种子, 则固定以便复现
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # 1. 读取数据
    df_clean = pd.read_csv(input_path)
    df_clean = df_clean.reset_index(drop=True)
    print(f"Loaded dataset from {input_path}, shape={df_clean.shape}")

    # 2. 第一列当作主键, 不做注入
    if df_clean.shape[1] < 2:
        print("Warning: <2 columns in dataset, can't do injection. Exiting.")
        return

    primary_key_col = df_clean.columns[0]
    print(f"Primary key column: '{primary_key_col}' (no injection).")

    # 其余列才可注入
    other_cols = df_clean.columns[1:].tolist()
    print(f"Target columns (for injection): {other_cols}")

    # ----------------------------------------------------------------------
    # 额外添加：将待注入列中若为整数类型的列，显式转换成 float
    # 这样后续若插入浮点数，就不会产生 "incompatible dtype" 的警告
    for c in other_cols:
        if pd.api.types.is_integer_dtype(df_clean[c]):
            df_clean[c] = df_clean[c].astype(float)
    # ----------------------------------------------------------------------

    # 3. 定义四种错误比例: 0%, 5%, 10%, 15%
    #   两种错误: anomalyRate & missingRate => 4×4=16 组合, 跳过(0,0) => 15个输出
    rates = [0.0, 0.05, 0.10, 0.15]

    # 准备输出目录 & 说明文件
    os.makedirs(output_dir, exist_ok=True)
    explanation_path = os.path.join(output_dir, f"{task_name}_explanation.txt")
    explanation_lines = []
    combo_counter = 0

    # 4. 遍历
    for anomalyRate in rates:
        for missingRate in rates:
            # 跳过(anomaly=0, missing=0) => 不输出
            if anomalyRate == 0.0 and missingRate == 0.0:
                continue

            combo_counter += 1

            # 拷贝一份干净数据
            df_corrupted = df_clean.copy()

            # 注入
            inject_anomaly_and_missing(df_corrupted, other_cols,
                                       anomalyRate, missingRate)

            # 输出文件: "{task_name}_{combo_counter}.csv"
            out_filename = f"{task_name}_{combo_counter}.csv"
            out_path = os.path.join(output_dir, out_filename)
            df_corrupted.to_csv(out_path, index=False)

            # 记录说明
            desc = f"{combo_counter} => anomaly={int(anomalyRate*100)}%, missing={int(missingRate*100)}%"
            explanation_lines.append(desc)
            print(f"[{combo_counter:02d}] Generated: {out_filename} | {desc}")

    # 5. 写明细到 explanation.txt
    with open(explanation_path, "w", encoding="utf-8") as f:
        f.write(f"Explanation of error combos for task: {task_name}\n\n")

        for line in explanation_lines:
            # line 例: "1 => anomaly=5%, missing=10%"
            combo_idx_str, detail = line.split(" => ", 1)
            # detail 类似: "anomaly=5%, missing=10%"
            f.write(line + "\n")
            f.write("   (These rates are applied independently to different cells)\n\n")

    print(f"\nAll {combo_counter} corrupted CSVs saved to: {output_dir}")
    print(f"Explanation file: {explanation_path}")


def inject_anomaly_and_missing(df, cols, anomalyRate, missingRate):
    """
    在 df 的指定 cols 中, 分别注入 anomalyRate 与 missingRate 的错误.
    - anomalyRate => e.g. 0.10 => 10% cells变异常
    - missingRate => e.g. 0.05 => 5% cells变缺失
    不允许重叠: 先选 anomaly,再选 missing,如有重复则跳过.
    """
    total_cells = df.shape[0] * len(cols)

    n_anomaly = int(total_cells * anomalyRate)
    n_missing = int(total_cells * missingRate)
    if n_anomaly <= 0 and n_missing <= 0:
        return

    # 先注入 anomaly
    anomaly_cells = pick_random_cells_no_overlap(df, cols, n_anomaly, forbid=None)
    for (r_i, c) in anomaly_cells:
        orig_val = df.at[r_i, c]
        df.at[r_i, c] = generate_anomaly_value(orig_val)

    # 后注入 missing
    forbid_set = set(anomaly_cells)  # 不与 anomaly 重叠
    missing_cells = pick_random_cells_no_overlap(df, cols, n_missing, forbid=forbid_set)
    for (r_i, c) in missing_cells:
        df.at[r_i, c] = np.nan


def pick_random_cells_no_overlap(df, cols, count, forbid=None):
    """
    从指定列中挑选count个(row_idx,col)不与forbid集合重叠
    """
    if count <= 0:
        return []

    candidates = []
    for c in cols:
        for r in df.index:
            pair = (r, c)
            if forbid and pair in forbid:
                continue
            candidates.append(pair)

    if count > len(candidates):
        count = len(candidates)
    return random.sample(candidates, count)


def generate_anomaly_value(orig_val):
    """
    简易异常: 对可转float的放大3~6倍; 否则向字符串插入怪字符
    """
    if pd.isnull(orig_val):
        orig_val = 1.0
    try:
        fval = float(orig_val)
        factor = random.uniform(3, 6)
        return fval * factor
    except:
        s = str(orig_val)
        return insert_weird_chars(s)


def insert_weird_chars(s):
    specs = ["#", "$", "%", "??", "@@"]
    times = random.randint(1, 2)
    for _ in range(times):
        ch = random.choice(specs)
        pos = random.randint(0, len(s))
        s = s[:pos] + ch + s[pos:]
    return s


if __name__ == "__main__":
    main()

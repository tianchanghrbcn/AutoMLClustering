#!/usr/bin/env python3
# cleaned_points_by_segment_2d.py —— age-income 平面绘制清洗后数据 (真值着色)
# -----------------------------------------------------------------------
# • 读取 demo_cleaned/{algo}/repaired_*.csv，与 clean_withseg.csv 匹配 segment
# • 读取 demo_dirty/rayyan_explanation.txt 解析错误注入比例，用中文标题标注
# • 每文件单张 PDF；每算法一张汇总网格 PDF (figures/cleaned_clusters/{algo}/)
# -----------------------------------------------------------------------

import warnings, os, re, argparse, math, glob
warnings.filterwarnings("ignore", message="Glyph")

import matplotlib
matplotlib.rcParams["font.family"]     = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Noto Sans CJK SC"]
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["pdf.fonttype"]    = 42

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---------- 路径与算法配置 ---------------------------------------------
BASE_DIR      = "demo_results"                     # cleaned_data 根目录在其中
DIRTY_DIR     = "demo_dirty"                       # explanation.txt 所在目录
EXPLAIN_FILE  = "rayyan_explanation.txt"
CLEAN_FILE    = "clean_withseg.csv"                # 真值
OUTPUT_ROOT   = "figures/cleaned_clusters"         # 输出根目录
SEED          = 42

STRATEGIES = {
    1: "mode",
    2: "baran",
    3: "holoclean",
    4: "bigdansing",
    5: "boostclean",
    6: "horizon",
    7: "scared",
    8: "Unified"
}

# ---------- 颜色与顺序 ---------------------------------------------------
SEG_COLOR = {"A": "#1f77b4", "B": "#ffbf00", "C": "#d62728",
             "D": "#17becf", "E": "#7f7f7f"}
SEG_ORDER = ["A", "B", "C", "D", "E"]

# ---------- 解析 explanation.txt ----------------------------------------
def parse_explanation(path: str) -> dict[int, str]:
    """
    匹配行例如：
    06 | Anom=5%, Miss=10%  →  r_anom=5.00%, r_miss=10.00%, r_tot=15.00%
    返回 {6: "异常=5%, 缺失=10%, 总=15%"}
    """
    pat = re.compile(r"(\d{2})\s*\|\s*Anom=([\d.]+%)\s*,\s*Miss=([\d.]+%).*?r_tot=([\d.]+%)")
    out = {}
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                m = pat.search(line)
                if m:
                    idx = int(m.group(1))
                    out[idx] = f"异常率={m.group(2)}，缺失率={m.group(3)}，总错误率={m.group(4)}"
    return out

# ---------- 主流程 -------------------------------------------------------
def main():
    # 0) 真值读取 & 坐标范围
    if not os.path.isfile(CLEAN_FILE):
        raise SystemExit(f"❌ 未找到 {CLEAN_FILE}")
    clean_df = pd.read_csv(CLEAN_FILE)[["ID", "age", "income", "segment"]]
    xmin, xmax = clean_df["age"].min(),    clean_df["age"].max()
    ymin, ymax = clean_df["income"].min(), clean_df["income"].max()
    seg_map = clean_df.set_index("ID")["segment"]

    # 解析错误注入说明
    explain_map = parse_explanation(os.path.join(DIRTY_DIR, EXPLAIN_FILE))

    # 每个算法单独处理
    for algo in STRATEGIES.values():
        in_dir  = Path(BASE_DIR) / "cleaned_data" / algo
        if not in_dir.exists():
            print(f"[跳过] {algo}: 目录不存在")
            continue

        out_dir = Path(OUTPUT_ROOT) / algo
        out_dir.mkdir(parents=True, exist_ok=True)

        csv_list = sorted(in_dir.glob("repaired_*.csv"))
        if not csv_list:
            print(f"[跳过] {algo}: 无 repaired_*.csv")
            continue

        # 准备汇总图（4×4）
        cols = 4
        rows = math.ceil(len(csv_list) / cols)
        fig_all, axes = plt.subplots(rows, cols, figsize=(12, 3*rows))
        fig_all.subplots_adjust(hspace=0.42, wspace=0.17)
        ax_iter = iter(axes.flatten()) if rows*cols>1 else iter([axes])

        for fp in csv_list:
            stem = fp.stem                      # repaired_07 等
            idx_match = re.search(r"(\d+)$", stem)
            idx = int(idx_match.group(1)) if idx_match else None
            detail = explain_map.get(idx, "（无注入说明）")

            df = pd.read_csv(fp)

            # 合并真值 segment
            if "ID" not in df.columns:
                print(f"[跳过] {stem}: 缺 ID"); continue
            df = df.merge(seg_map, on="ID", how="left")
            if "segment" not in df.columns:
                print(f"[跳过] {stem}: 无 segment"); continue

            # 数值清洗
            for col in ["age", "income"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df[col] = df[col].fillna(df[col].median())
            df.dropna(subset=["age", "income"], how="all", inplace=True)

            # -------- 单图 --------
            plt.figure(figsize=(5, 4))
            for seg in SEG_ORDER:
                mask = df["segment"].str.startswith(seg)
                plt.scatter(df.loc[mask, "age"], df.loc[mask, "income"],
                            s=18, color=SEG_COLOR[seg], alpha=0.75, label=seg)
            plt.xlim(xmin, xmax); plt.ylim(ymin, ymax)
            plt.xlabel("年龄 (age)"); plt.ylabel("年收入 (income)")
            plt.title(f"{algo} | {stem} | {detail}", fontsize=10)
            plt.legend(title="原始簇", fontsize=8, ncol=3)
            plt.tight_layout()
            single_fp = out_dir / f"{stem}_byseg.pdf"
            plt.savefig(single_fp, dpi=300); plt.close()
            print(f"[✓] {algo}: 保存 {single_fp}")

            # -------- 汇总子图 --------
            ax = next(ax_iter)
            for seg in SEG_ORDER:
                mask = df["segment"].str.startswith(seg)
                ax.scatter(df.loc[mask, "age"], df.loc[mask, "income"],
                           s=10, color=SEG_COLOR[seg], alpha=0.75)
            ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{stem}\n{detail}", fontsize=7)

        # 隐藏空子图
        for ax in ax_iter:
            ax.axis("off")

        ov_fp = out_dir / f"{algo}_overview.pdf"
        fig_all.suptitle(f"{algo} 清洗后簇分布汇总", fontsize=14)
        fig_all.tight_layout(rect=[0,0,1,0.96])
        fig_all.savefig(ov_fp, dpi=300); plt.close()
        print(f"[✓] {algo}: 汇总图已保存 {ov_fp}\n")

# ---------- CLI ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="清洗后 age-income 二维真值着色可视化（含注入比例）")
    main()

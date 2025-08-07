#!/usr/bin/env python3
# dirty_points_by_segment_2d.py —— 直接按 age-income 坐标绘制脏数据 + 真值着色
# -----------------------------------------------------------------------
# • 读取 demo_dirty/*.csv，与 clean_withseg.csv (含 segment) 通过 ID 匹配
# • 去掉降维，只画 age vs. income；坐标轴范围固定为 clean 数据的 min~max
# • 颜色映射：A→蓝, B→亮黄, C→红, D→青, E→灰
# • 输出：每文件一张 PDF + 汇总 4×4 PDF （输出目录 figures/dirty_clusters）

import warnings, os, re, glob, argparse
warnings.filterwarnings("ignore", message="Glyph")

import matplotlib
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Noto Sans CJK SC"]
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["pdf.fonttype"] = 42

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---------- 路径与基本配置 --------------------------------------------
DIRTY_DIR    = "demo_dirty"
CLEAN_FILE   = "clean_withseg.csv"
EXPLAIN_FILE = "rayyan_explanation.txt"
OUTPUT_DIR   = "figures/dirty_clusters"
SEED         = 42

# ---------- 颜色与顺序 -------------------------------------------------
SEG_COLOR = {"A": "#1f77b4", "B": "#ffbf00", "C": "#d62728",
             "D": "#17becf", "E": "#7f7f7f"}
SEG_ORDER = ["A", "B", "C", "D", "E"]

# ---------- 解析 explanation.txt --------------------------------------
def parse_explanation(path):
    pat = re.compile(r"(\d{2}).*?Anom=([\d.]+%).*?Miss=([\d.]+%).*?r_tot=([\d.]+%)")
    out = {}
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                m = pat.search(line)
                if m:
                    idx = int(m.group(1))
                    out[idx] = f"异常率={m.group(2)}，缺失率={m.group(3)}，总错误率={m.group(4)}"
    return out

# ---------- 主流程 ------------------------------------------------------
def main():
    # 0) 读取真值并获取坐标轴范围
    if not os.path.isfile(CLEAN_FILE):
        raise SystemExit(f"❌ 未找到 {CLEAN_FILE}")
    clean_df = pd.read_csv(CLEAN_FILE)[["ID", "age", "income", "segment"]]
    xmin, xmax = clean_df["age"].min(),    clean_df["age"].max()
    ymin, ymax = clean_df["income"].min(), clean_df["income"].max()

    seg_map = clean_df.set_index("ID")["segment"]
    explain = parse_explanation(os.path.join(DIRTY_DIR, EXPLAIN_FILE))
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # 汇总图准备
    fig_all, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig_all.subplots_adjust(hspace=0.42, wspace=0.17)
    ax_iter = iter(axes.flatten())

    for fp in sorted(glob.glob(os.path.join(DIRTY_DIR, "*.csv"))):
        stem   = Path(fp).stem
        idx_m  = re.search(r"_(\d+)$", stem)
        idx    = int(idx_m.group(1)) if idx_m else None
        demo   = f"示例 {idx:02d}" if idx else stem.replace("rayyan", "示例")

        df = pd.read_csv(fp)
        if "ID" not in df.columns:
            print(f"[跳过] {stem} 缺失 ID"); continue

        # 合并 segment 真值
        df = df.merge(seg_map, on="ID", how="left")
        if "segment" not in df.columns:
            print(f"[跳过] {stem} 无 segment"); continue

        # 数值转换 & 缺失填补
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
        detail = explain.get(idx, "（无注入说明）")
        plt.title(f"{demo} | {detail}", fontsize=10)
        plt.legend(title="原始簇", fontsize=8, ncol=3)
        plt.tight_layout()
        single_fp = Path(OUTPUT_DIR) / f"{demo.replace(' ', '')}_byseg.pdf"
        plt.savefig(single_fp, dpi=300); plt.close()
        print(f"[✓] 保存: {single_fp}")

        # -------- 汇总子图 --------
        ax = next(ax_iter)
        for seg in SEG_ORDER:
            mask = df["segment"].str.startswith(seg)
            ax.scatter(df.loc[mask, "age"], df.loc[mask, "income"],
                       s=10, color=SEG_COLOR[seg])
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{demo}\n{detail}", fontsize=7)

    # 隐藏空子图
    for ax in ax_iter: ax.axis("off")

    ov_fp = Path(OUTPUT_DIR) / "dirty_points_overview.pdf"
    fig_all.suptitle("基于原始簇的错误数据投影（age-income 平面）", fontsize=14)
    fig_all.tight_layout(rect=[0,0,1,0.96])
    fig_all.savefig(ov_fp, dpi=300); plt.close()
    print(f"[✓] 汇总图已保存: {ov_fp}")

# ---------- CLI ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="age-income 二维真值着色可视化")
    main()

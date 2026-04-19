#!/usr/bin/env python3
# dirty_points_by_segment_2d.py —— 基于 Clean 对比标注缺失/异常 (Clip+Arrow)
# --------------------------------------------------------------------
#   clean_withseg.csv           # baseline, 含 ID, age, income, segment
#   demo_dirty/rayyan_*.csv     # 脏数据
#   demo_dirty/rayyan_explanation.txt (可选)  # 错误率说明
# 输出：
#   figures/dirty_clusters/<示例>.pdf  +  overview.pdf
# --------------------------------------------------------------------

import warnings, os, re, argparse, math  # NEW: math
warnings.filterwarnings("ignore", message="Glyph")

import matplotlib
matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["SimHei", "Microsoft YaHei", "Noto Sans CJK SC"],
    "axes.unicode_minus": False,
    "pdf.fonttype": 42
})

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---------- 路径 -------------------------------------------------------
DIRTY_DIR    = "demo_dirty"
CLEAN_FILE   = "clean_withseg.csv"
EXPLAIN_FILE = "rayyan_explanation.txt"
OUTPUT_DIR   = "figures/dirty_clusters"

# ---------- 颜色 -------------------------------------------------------
SEG_COLOR = {"A": "#1f77b4", "B": "#ffbf00", "C": "#d62728",
             "D": "#17becf", "E": "#7f7f7f"}
SEG_ORDER = ["A", "B", "C", "D", "E"]

# ---------- 解析 explanation.txt --------------------------------------
def parse_explanation(path: str) -> dict[int, str]:
    pat = re.compile(r"(\d{2}).*?Anom=([\d.]+%).*?Miss=([\d.]+%).*?r_tot=([\d.]+%)")
    out = {}
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                m = pat.search(line)
                if m:
                    idx = int(m.group(1))
                    out[idx] = f"异常={m.group(2)} 缺失={m.group(3)} 总={m.group(4)}"
    return out

# ---------- 绘制函数 ---------------------------------------------------
def scatter_clip_arrow(ax, df, xmin, xmax, ymin, ymax, x_thr, y_thr):
    dx = 0.02*(xmax-xmin); dy = 0.02*(ymax-ymin)
    x_out, y_out = xmin-dx, ymin-dy

    # 正常点
    norm = df.err_type == "normal"
    for seg in SEG_ORDER:
        m = norm & df.segment.str.startswith(seg)
        ax.scatter(df.loc[m,'age'], df.loc[m,'income'],
                   s=18, c=SEG_COLOR[seg], lw=0, alpha=.8)

    # 缺失维度 —— X / Y
    miss_x = (df.err_type=="missing") & df.age.isna() & df.income.notna()
    miss_y = (df.err_type=="missing") & df.income.isna() & df.age.notna()
    ax.scatter(np.full(miss_x.sum(), x_out), df.loc[miss_x,'income'],
               marker='x', c='k', s=28, lw=1.2)
    ax.scatter(df.loc[miss_y,'age'], np.full(miss_y.sum(), y_out),
               marker='x', c='k', s=28, lw=1.2)

    # 异常 —— Clip+Arrow
    anom = df.err_type == "anomaly"
    for seg in SEG_ORDER:
        m = anom & df.segment.str.startswith(seg)
        for _, r in df.loc[m].iterrows():
            x = min(r.age, x_thr*.99) if not np.isnan(r.age) else x_out
            y = min(r.income, y_thr*.99) if not np.isnan(r.income) else y_out
            ax.scatter(x, y, s=18, c=SEG_COLOR[seg], lw=0, alpha=.8)
            if r.age > x_thr:
                ax.annotate("", xy=(x, y), xytext=(x_thr*.9, y),
                            arrowprops=dict(arrowstyle="->", color=SEG_COLOR[seg], lw=.8))
            if r.income > y_thr:
                ax.annotate("", xy=(x, y), xytext=(x, y_thr*.9),
                            arrowprops=dict(arrowstyle="->", color=SEG_COLOR[seg], lw=.8))

    ax.set_xlim(xmin-dx*1.5, x_thr)
    ax.set_ylim(ymin-dy*1.5, y_thr)

# ---------- 主流程 ------------------------------------------------------
def main():
    # --- Clean 真值 ---
    if not os.path.isfile(CLEAN_FILE):
        raise SystemExit("❌ 缺少 clean_withseg.csv")
    clean = pd.read_csv(CLEAN_FILE)[["ID","age","income","segment"]]
    for c in ("age","income"):
        clean[c] = pd.to_numeric(clean[c], errors="coerce")
    clean.set_index("ID", inplace=True)

    xmin,xmax = clean.age.min(),   clean.age.max()
    ymin,ymax = clean.income.min(),clean.income.max()
    x_thr = np.percentile(clean.age,   75)+1.5*np.subtract(*np.percentile(clean.age,   [75,25]))
    y_thr = np.percentile(clean.income,75)+1.5*np.subtract(*np.percentile(clean.income,[75,25]))

    explain = parse_explanation(Path(DIRTY_DIR)/EXPLAIN_FILE)
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # --- 收集并按编号排序文件（数值序） ----------------------------------  # NEW
    files = []
    for p in Path(DIRTY_DIR).glob("rayyan_*.csv"):
        m = re.search(r"rayyan_(\d+)$", p.stem)
        if m:
            files.append((int(m.group(1)), p))
    files.sort(key=lambda t: t[0])  # 数值从小到大

    if not files:
        raise SystemExit("❌ 未找到 demo_dirty/rayyan_*.csv")

    # --- 动态总览画布：横向填充（行优先） --------------------------------  # NEW
    n_cols = 4
    n_rows = math.ceil(len(files) / n_cols)
    fig_all, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows))
    fig_all.subplots_adjust(hspace=.42, wspace=.17)
    ax_iter = iter(np.array(axes).flatten())

    # --- 遍历（按编号升序） ---
    for idx, dirty_path in files:
        demo = f"示例数据集 {idx:02d}"

        dirty = pd.read_csv(dirty_path).set_index("ID")
        for c in ("age","income"):
            dirty[c] = pd.to_numeric(dirty[c], errors="coerce")
        df = dirty.join(clean[["age","income","segment"]].rename(
                        columns={"age":"age_true","income":"income_true"}))
        df.reset_index(inplace=True)

        # 标记错误类型
        miss = df.age.isna() | df.income.isna()
        valid_truth = df[["age_true","income_true"]].notna().all(axis=1)
        anom = valid_truth & (~miss) & ((df.age != df.age_true) | (df.income != df.income_true))
        df["err_type"] = np.select([miss, anom], ["missing","anomaly"], default="normal")

        # 标题：explanation 优先，否则用实测
        n = len(df)
        miss_rate  = (miss).sum() / max(n,1)
        anom_rate  = (anom).sum() / max(n,1)
        total_rate = ((miss | anom)).sum() / max(n,1)
        detail_auto = f"异常={anom_rate:.1%} 缺失={miss_rate:.1%} 总={total_rate:.1%}"
        detail = explain.get(idx, detail_auto)

        # --- 单图 PDF ---
        fig, ax = plt.subplots(figsize=(5,4))
        scatter_clip_arrow(ax, df, xmin,xmax,ymin,ymax,x_thr,y_thr)
        ax.set_xlabel("年龄 (age)"); ax.set_ylabel("年收入 (income)")
        ax.set_title(f"{demo} | {detail}", fontsize=10)
        handles = [
            plt.Line2D([0], [0], marker='x', color='k', ls='', label='缺失维度', markersize=7),
            plt.Line2D([0, 1], [0, 0], color='k', lw=1.2, marker='>', markersize=6, markevery=[1], label='异常维度')
        ]
        ax.legend(handles=handles, loc='upper right', fontsize=8, frameon=False)

        plt.tight_layout()
        single = Path(OUTPUT_DIR)/f"{demo.replace(' ','')}.pdf"
        fig.savefig(single,dpi=600); plt.close()
        print("✓ 已保存", single)

        # --- 总览子图（横向填充） ---
        ax_small = next(ax_iter)
        scatter_clip_arrow(ax_small, df, xmin,xmax,ymin,ymax,x_thr,y_thr)
        ax_small.set_xticks([]); ax_small.set_yticks([])
        ax_small.set_title(f"{demo} | {detail}", fontsize=7)

    # 把未用到的子图隐藏
    for ax in ax_iter:
        ax.axis("off")

    fig_all.suptitle("脏数据集投影（→ 异常值, × 缺失值）", fontsize=14)
    fig_all.tight_layout(rect=[0,0,1,0.96])
    ov = Path(OUTPUT_DIR)/"dirty_points_overview.pdf"
    fig_all.savefig(ov,dpi=300); plt.close()
    print("✓ 总览图已保存", ov)

# ---------------- CLI ----------------
if __name__ == "__main__":
    argparse.ArgumentParser(description="基于 Clean 对比标注缺失/异常").parse_args()
    main()

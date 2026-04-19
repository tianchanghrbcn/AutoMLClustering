#!/usr/bin/env python3
# clean_dirty_cleaned_pentaptych.py —— 单图版（相对坐标 & y轴千元）: Clean | Dirty | Mode | HoloClean | Baran
# 本版仅输出 5 张单图（PDF + PNG），并增大散点尺寸
# ------------------------------------------------------------------------------------------
import warnings, json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from pathlib import Path
from matplotlib.ticker import FuncFormatter, MaxNLocator, AutoMinorLocator

warnings.filterwarnings("ignore", message="Glyph")

# 字体与导出为矢量 PDF
warnings.filterwarnings("ignore", category=UserWarning)
matplotlib.rc('font', family='Times New Roman')
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# ---------- 常量 ----------
BASE_DIR    = "demo_results"
DIRTY_DIR   = "demo_dirty"
EIGENV_PATH = Path(BASE_DIR) / "eigenvectors.json"
CLEAN_FILE  = "clean_withseg.csv"
OUTPUT_DIR  = "figures/pentaptych"

# 仅画示例数据集 #02
TARGET_ID   = 2

# 分段上色
SEG_COLOR = {"A":"#1f77b4","B":"#ffbf00","C":"#d62728",
             "D":"#17becf","E":"#7f7f7f"}
SEG_ORDER = ["A","B","C","D","E"]

# 三种清洗算法
CLEAN_STRATEGIES = ["mode", "holoclean", "baran"]

# 视觉参数（单图）
TICK_SIZE   = 19        #（单图不显示标题，仅刻度数字；此项仅在空图兜底用）
POINT_SIZE  = 180       # 散点尺寸（调大）
YSCALE_K    = 1000.0    # y 轴按“千”为单位（直接去掉 3 个 0）

# —— 单图专用：更密的刻度 & 更大的字号 ——
SINGLE_TICK_SIZE     = 24   # 单图刻度数字字号
SINGLE_MAJOR_TICKS   = 8    # 单图主刻度最大个数（x/y 各）
SINGLE_MINOR_DIVISOR = 4    # 两主刻度之间的次刻度等分数

# —— 异常值箭头样式（更粗更大） ——
ARROW_LW = 1.8          # 线宽
ARROW_MS = 16           # mutation_scale，控制箭头头部大小

# ---------- 工具 ----------
def read_eigen(p:Path):
    mp={}
    with open(p,encoding="utf-8") as f:
        for x in json.load(f):
            mp[x["dataset_id"]]={"csv":x["csv_file"],
              "detail":f"缺失率：{x['missing_rate']*100:.0f}% 异常率：{x['noise_rate']*100:.0f}% 总错误率：{x['error_rate']:.0f}%"}
    return mp

def tag_err(df, ref):
    """基于 clean 基线判断缺失/异常：缺失→missing；与基线不同→anomaly；否则 normal。"""
    miss = df.age.isna() | df.income.isna()
    anom = (~miss) & ((df.age!=ref.age) | (df.income!=ref.income))
    return np.select([miss, anom], ["missing","anomaly"], "normal")

def scatter_relative(ax, df,
                     x0, y0,
                     x_left_abs, x_right_abs, y_bottom_abs, y_top_abs,
                     x_out_rel, y_out_rel, x_thr_rel, y_thr_rel,
                     s=POINT_SIZE):
    """
    相对坐标绘制：以 (x0,y0) 为原点，x 使用原单位（相对差值），y 使用“千”为单位（相对差值/1000）。
    传入的边界/阈值均为未缩放的相对值（与 clean 一致），函数内对 y 做 /1000 处理。
    """
    # 坐标范围（相对；y 做千单位缩放）
    x_min_rel = x_left_abs  - x0
    x_max_rel = x_right_abs - x0
    y_min_rel = (y_bottom_abs - y0) / YSCALE_K
    y_max_rel = (y_top_abs    - y0) / YSCALE_K

    # normal 点
    norm = df.err_type=="normal"
    for seg in SEG_ORDER:
        m = norm & df.segment.astype(str).str.startswith(seg, na=False)
        if m.any():
            ax.scatter(df.loc[m, "age"].to_numpy() - x0,
                       (df.loc[m, "income"].to_numpy() - y0) / YSCALE_K,
                       c=SEG_COLOR[seg], s=s, lw=0, alpha=.85)

    # missing：x 或 y 缺失，落在边界外（相对坐标的固定位置）
    mx = (df.err_type=="missing") & df.age.isna() & df.income.notna()
    my = (df.err_type=="missing") & df.income.isna() & df.age.notna()
    if mx.any():
        ax.scatter(np.full(mx.sum(), x_out_rel),
                   (df.loc[mx, "income"].to_numpy() - y0) / YSCALE_K,
                   marker='x', c='k', s=s+20, lw=1.2)
    if my.any():
        ax.scatter(df.loc[my, "age"].to_numpy() - x0,
                   np.full(my.sum(), y_out_rel / YSCALE_K),
                   marker='x', c='k', s=s+20, lw=1.2)

    # anomaly：超阈值的点截断到 0.99*阈值位置，并用箭头指回阈值
    anom = df.err_type=="anomaly"
    for seg in SEG_ORDER:
        m = anom & df.segment.astype(str).str.startswith(seg, na=False)
        for _, r in df.loc[m].iterrows():
            xr = (r.age - x0) if not np.isnan(r.age) else x_out_rel
            yr = ((r.income - y0) / YSCALE_K) if not np.isnan(r.income) else (y_out_rel / YSCALE_K)
            # 截断到阈值内侧
            if not np.isnan(r.age):    xr = min(xr, x_thr_rel*0.99)
            if not np.isnan(r.income): yr = min(yr, (y_thr_rel/YSCALE_K)*0.99)
            ax.scatter(xr, yr, c=SEG_COLOR[seg], s=s, lw=0, alpha=.85)
            # 箭头（仅当原始值超阈才画）
            if (not np.isnan(r.age)) and (r.age > (x0 + x_thr_rel)):
                ax.annotate("", xy=(xr, yr), xytext=(x_thr_rel*0.9, yr),
                            arrowprops=dict(arrowstyle="->", color=SEG_COLOR[seg],
                                            lw=ARROW_LW, mutation_scale=ARROW_MS))
            if (not np.isnan(r.income)) and (r.income > (y0 + y_thr_rel)):
                ax.annotate("", xy=(xr, yr), xytext=(xr, (y_thr_rel/YSCALE_K)*0.9),
                            arrowprops=dict(arrowstyle="->", color=SEG_COLOR[seg],
                                            lw=ARROW_LW, mutation_scale=ARROW_MS))

    # 轴范围与刻度美化（无坐标标题，仅保留数值；y 轴按千显示为整数）
    ax.set_xlim(x_min_rel, x_max_rel)
    ax.set_ylim(y_min_rel, y_max_rel)
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.tick_params(direction="in", labelsize=TICK_SIZE)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{int(np.round(v))}"))
    # 如需 x 轴也显示整数，取消下一行注释
    # ax.xaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{int(np.round(v))}"))

def safe_load_cleaned(algo:str, seg_map:pd.Series):
    """读取 cleaned_data/{algo}/repaired_2.csv 并合并 segment；缺文件返回空 DF。"""
    fp = Path(BASE_DIR)/"cleaned_data"/algo/f"repaired_{TARGET_ID}.csv"
    if not fp.is_file():
        return pd.DataFrame(columns=["ID","age","income","segment","err_type"])
    df = pd.read_csv(fp)[["ID","age","income"]].merge(seg_map, on="ID", how="left")
    for c in ("age","income"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# ---------- 主 ----------
def main():
    # 基线 clean
    if not Path(CLEAN_FILE).is_file():
        raise SystemExit("❌ 缺 clean_withseg.csv")
    base = pd.read_csv(CLEAN_FILE)[["ID","age","income","segment"]]
    for c in ("age","income"):
        base[c] = pd.to_numeric(base[c], errors="coerce")

    # 用 Clean 面板确定统一坐标框与中心
    xmin, xmax = base.age.min(), base.age.max()
    ymin, ymax = base.income.min(), base.income.max()
    # IQR 上阈
    x_thr = np.percentile(base.age,75) + 1.5*np.subtract(*np.percentile(base.age,[75,25]))
    y_thr = np.percentile(base.income,75) + 1.5*np.subtract(*np.percentile(base.income,[75,25]))
    # 与原脚本一致的边界外延
    dx = 0.02*(xmax-xmin)
    dy = 0.02*(ymax-ymin)
    x_left_abs   = xmin - 1.5*dx
    y_bottom_abs = ymin - 1.5*dy
    # 统一中心点（Clean 面板的绘图边界中心）
    x0 = 0.5*(x_left_abs + x_thr)
    y0 = 0.5*(y_bottom_abs + y_thr)
    # 相对坐标下的“缺失点落点”（仍位于可见范围的左/下侧，y 后续会在函数内除以1000）
    x_out_rel  = (xmin - dx) - x0
    y_out_rel  = (ymin - dy) - y0
    # 相对阈值位置（用于截断与箭头；y 后续会在函数内除以1000）
    x_thr_rel  = x_thr - x0
    y_thr_rel  = y_thr - y0

    seg_map = base.set_index("ID")["segment"]
    base["err_type"] = "normal"               # 基线全设 normal
    base_ref = base.set_index("ID")           # 用于判断 dirty/cleaned 的 missing/anomaly

    # 读取 eigenvectors.json → 找到 dirty 源文件
    eigen = read_eigen(EIGENV_PATH)
    if TARGET_ID not in eigen:
        raise SystemExit(f"❌ eigenvectors.json 中没有 dataset_id={TARGET_ID}")
    dirty_path = Path(DIRTY_DIR)/eigen[TARGET_ID]["csv"]
    if not dirty_path.is_file():
        raise SystemExit(f"❌ 脏数据文件不存在：{dirty_path}")

    # 脏数据
    dirty = pd.read_csv(dirty_path)[["ID","age","income"]].merge(seg_map, on="ID", how="left")
    for c in ("age","income"):
        dirty[c] = pd.to_numeric(dirty[c], errors="coerce")
    dirty.set_index("ID", inplace=True)
    dirty["err_type"] = tag_err(dirty, base_ref)
    dirty.reset_index(inplace=True)

    # 三种清洗算法的数据
    cleaned_frames = {}
    for algo in CLEAN_STRATEGIES:
        dfc = safe_load_cleaned(algo, seg_map)
        if dfc.empty:
            cleaned_frames[algo] = dfc
            continue
        dfc.set_index("ID", inplace=True)
        dfc["err_type"] = tag_err(dfc, base_ref)
        dfc.reset_index(inplace=True)
        cleaned_frames[algo] = dfc

    # --------- 仅保存单图（7×5.5）---------
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    panels = [
        ("clean", base),
        ("dirty", dirty),
        ("mode", cleaned_frames.get("mode", pd.DataFrame())),
        ("holoclean", cleaned_frames.get("holoclean", pd.DataFrame())),
        ("baran", cleaned_frames.get("baran", pd.DataFrame())),
    ]

    for name, df_plot in panels:
        fig_single, ax_single = plt.subplots(1, 1, figsize=(7, 5.5))
        if df_plot.empty:
            ax_single.text(0.5, 0.5, "文件缺失", ha="center", va="center", fontsize=12, transform=ax_single.transAxes)
            ax_single.set_xlabel(None)
            ax_single.set_ylabel(None)
            ax_single.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{int(np.round(v))}"))
        else:
            scatter_relative(ax_single, df_plot,
                             x0, y0,
                             x_left_abs, x_thr, y_bottom_abs, y_thr,
                             x_out_rel, y_out_rel, x_thr_rel, y_thr_rel,
                             s=POINT_SIZE)

        # —— 单图：增加刻度密度 & 放大字号 ——
        ax_single.xaxis.set_major_locator(MaxNLocator(nbins=SINGLE_MAJOR_TICKS))
        ax_single.yaxis.set_major_locator(MaxNLocator(nbins=SINGLE_MAJOR_TICKS, integer=True))
        ax_single.xaxis.set_minor_locator(AutoMinorLocator(SINGLE_MINOR_DIVISOR))
        ax_single.yaxis.set_minor_locator(AutoMinorLocator(SINGLE_MINOR_DIVISOR))
        ax_single.tick_params(which="major", direction="in",
                              labelsize=SINGLE_TICK_SIZE, length=6, width=1.2)
        ax_single.tick_params(which="minor", direction="in",
                              length=3, width=0.8)
        # 如需 x 轴也显示整数，取消下一行注释：
        # ax_single.xaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{int(np.round(v))}"))

        plt.tight_layout()
        save_single = Path(OUTPUT_DIR) / f"pentaptych_rel_{TARGET_ID:02d}_{name}.pdf"
        fig_single.savefig(save_single, dpi=600)
        fig_single.savefig(save_single.with_suffix(".png"), dpi=600)
        plt.close(fig_single)
        print(f"✓ 保存单图 {name} ->", save_single)

if __name__ == "__main__":
    main()

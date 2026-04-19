#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_cleaned_scatter_by_cluster.py —— 统一版式/统一配色/统一坐标框
- 同一数据集 did 下，对 {baran, mode, holoclean}：
  * 统一二维投影（优先 age/income；否则对拼接数据做 PCA2，并复用）
  * 统一坐标边界（全局 0.5%～99.5% 分位 + 2% 边距），相对坐标 + y 轴“千”单位
  * 统一调色卡（簇 ID 全局收集、排序后映射到 tab20）
- 输出：每算法一张单图（PDF + PNG），无标题、无轴标签，仅刻度
"""

from pathlib import Path
from typing import Optional, Tuple, List, Dict, Callable
import json, warnings, math, re
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator, AutoMinorLocator

warnings.filterwarnings("ignore", category=UserWarning)
matplotlib.rc('font', family='Times New Roman')
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# ------- 配置 -------
ROOT_CLEANED   = Path("demo_results/cleaned_data")
ROOT_CLUSTERED = Path("demo_results/clustered_data/HC")
ALGORITHMS     = ["baran", "mode", "holoclean"]
DATASET_IDS    = [2, 5]

# 视觉参数（与单图风格一致）
YSCALE_K = 1000.0
SINGLE_TICK_SIZE     = 24
SINGLE_MAJOR_TICKS   = 8
SINGLE_MINOR_DIVISOR = 4
POINT_SIZE           = 80     # 簇点大小（可按需调）
FIGSIZE              = (7, 5.5)

# 分位数边界（尽量包含绝大多数点）
LOW_Q, HIGH_Q = 0.5, 99.5

# ------- 工具 -------
def ensure_outdir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def savefig(fig: plt.Figure, out: Path, dpi=600):
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=dpi)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)

def find_cluster_dir(algo: str, did: int) -> Path:
    return ROOT_CLUSTERED / algo / f"clustered_{did}"

def find_best_state(cluster_dir: Path, stem: str) -> Optional[str]:
    cleaned = cluster_dir / f"{stem}_cleaned_clusters.csv"
    raw     = cluster_dir / f"{stem}_raw_clusters.csv"
    if cleaned.exists(): return "cleaned"
    if raw.exists():     return "raw"
    any_csv = list(cluster_dir.glob(f"{stem}_*_clusters.csv"))
    if any_csv:
        m = re.search(rf"{re.escape(stem)}_(.+?)_clusters\.csv$", any_csv[0].name)
        return m.group(1) if m else None
    return None

def load_labels(cluster_dir: Path, stem: str, state: str) -> Optional[pd.DataFrame]:
    csv_path = cluster_dir / f"{stem}_{state}_clusters.csv"
    if csv_path.exists():
        dfc = pd.read_csv(csv_path, usecols=lambda c: c.lower() in ("orig_index","cluster"))
        if {"orig_index","cluster"}.issubset(dfc.columns):
            return dfc[["orig_index","cluster"]].copy()
    lab_path = cluster_dir / f"{stem}_{state}_labels.json"
    if lab_path.exists():
        with open(lab_path, "r", encoding="utf-8") as f:
            mp = json.load(f)
        rows = []
        for k, idxs in mp.items():
            c = int(k)
            for ix in idxs:
                try: rows.append({"orig_index": int(ix), "cluster": c})
                except: pass
        if rows: return pd.DataFrame(rows)
    return None

def pick_two_features(columns: List[str]) -> Optional[Tuple[str, str]]:
    lo = {c.lower(): c for c in columns}
    if "age" in lo and "income" in lo:
        return lo["age"], lo["income"]
    return None

def intersect_numeric_columns(dfs: List[pd.DataFrame]) -> List[str]:
    num_sets = []
    for df in dfs:
        num_df = df.select_dtypes(include=[np.number]).copy()
        drop = [c for c in num_df.columns
                if ("id" in c.lower()) or ("index" in c.lower())
                or ("cluster" in c.lower()) or ("unnamed" in c.lower())]
        num_df.drop(columns=drop, errors="ignore", inplace=True)
        num_sets.append(set(num_df.columns))
    if not num_sets: return []
    cols = set.intersection(*num_sets)
    return sorted(cols)

def fit_global_pca2(dfs: List[pd.DataFrame], cols: List[str]):
    from sklearn.decomposition import PCA
    Xs = []
    for df in dfs:
        X = df[cols].to_numpy(dtype=float)
        # 列均值填充
        col_means = np.nanmean(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_means, inds[1])
        Xs.append(X)
    X_all = np.vstack(Xs)
    pca = PCA(n_components=2, random_state=0)
    pca.fit(X_all)
    return pca

def make_palette_for_labels(all_labels: List[int]) -> Dict[int, Tuple[float,float,float]]:
    uniq = sorted(set(int(x) for x in all_labels))
    cmap = list(plt.get_cmap("tab20").colors)
    reps = math.ceil(len(uniq) / len(cmap))
    colors = (cmap * reps)[:len(uniq)]
    return {lab: colors[i] for i, lab in enumerate(uniq)}

def compute_global_frame(X_concat: np.ndarray):
    """基于所有算法的二维点，统一坐标边界 & 相对中心"""
    x_all, y_all = X_concat[:,0], X_concat[:,1]
    # 分位边界（尽量包含绝大多数点）
    x_lo, x_hi = np.percentile(x_all, [LOW_Q, HIGH_Q])
    y_lo, y_hi = np.percentile(y_all, [LOW_Q, HIGH_Q])
    # 边距（2%）
    dx = 0.02 * max(1e-9, x_hi - x_lo)
    dy = 0.02 * max(1e-9, y_hi - y_lo)
    x_left_abs,  x_right_abs  = x_lo - 0.5*dx, x_hi + 0.5*dx
    y_bottom_abs, y_top_abs   = y_lo - 0.5*dy, y_hi + 0.5*dy
    # 中心
    x0 = 0.5 * (x_left_abs + x_right_abs)
    y0 = 0.5 * (y_bottom_abs + y_top_abs)
    return dict(x0=x0, y0=y0,
                x_left_abs=x_left_abs, x_right_abs=x_right_abs,
                y_bottom_abs=y_bottom_abs, y_top_abs=y_top_abs)

# ------- 主流程（按 did 统一，再画各算法） -------
def process_dataset(did: int):
    # 1) 读入三个算法的清洗数据 + 标签（仅保留有标签的）
    per_algo_df: Dict[str, pd.DataFrame] = {}
    per_algo_labels: Dict[str, np.ndarray] = {}
    for algo in ALGORITHMS:
        cleaned_csv = ROOT_CLEANED / algo / f"repaired_{did}.csv"
        cluster_dir = find_cluster_dir(algo, did)
        stem = f"repaired_{did}"
        if not cleaned_csv.exists() or not cluster_dir.exists():
            continue
        state = find_best_state(cluster_dir, stem)
        if state is None:
            continue
        lab = load_labels(cluster_dir, stem, state)
        if lab is None or lab.empty:
            continue
        dfc = pd.read_csv(cleaned_csv).reset_index(drop=False).rename(columns={"index":"orig_index"})
        if "orig_index" not in dfc.columns:
            dfc.insert(0, "orig_index", np.arange(len(dfc), dtype=int))
        dfm = dfc.merge(lab, on="orig_index", how="inner")
        if dfm.empty:
            continue
        per_algo_df[algo] = dfm
        per_algo_labels[algo] = dfm["cluster"].to_numpy()

    if not per_algo_df:
        print(f"[跳过] did={did} 无可用数据/标签")
        return

    # 2) 统一二维投影（优先 age/income；否则全局 PCA2）
    any_df = next(iter(per_algo_df.values()))
    feat = pick_two_features(list(any_df.columns))
    projector: Callable[[pd.DataFrame], np.ndarray]
    proj_names: Tuple[str,str]

    if feat is not None:
        xcol, ycol = feat
        projector = lambda df: df[[xcol, ycol]].to_numpy(dtype=float)
        proj_names = (xcol, ycol)
    else:
        # 取在各算法间的“数值列交集”，对拼接数据拟合 PCA2
        joint_cols = intersect_numeric_columns(list(per_algo_df.values()))
        if len(joint_cols) < 1:
            print(f"[跳过] did={did} 数值列交集不足")
            return
        pca = fit_global_pca2(list(per_algo_df.values()), joint_cols)
        projector = lambda df: pca.transform(
            np.nan_to_num(df[joint_cols].to_numpy(dtype=float),
                          nan=np.nanmean(df[joint_cols].to_numpy(dtype=float), axis=0))
        )
        proj_names = ("PCA1","PCA2")

    # 3) 统一坐标边界（基于所有算法的投影点）
    X_all = []
    for algo, df in per_algo_df.items():
        X_all.append(projector(df))
    X_concat = np.vstack(X_all)
    frame = compute_global_frame(X_concat)
    x0, y0 = frame["x0"], frame["y0"]

    # 4) 统一调色卡（全算法簇 ID 汇总）
    all_labels = np.concatenate(list(per_algo_labels.values()))
    color_map = make_palette_for_labels(all_labels)

    # 5) 逐算法绘图（共享 frame & color_map）
    for algo, df in per_algo_df.items():
        X = projector(df)
        x_rel = X[:,0] - x0
        y_rel = (X[:,1] - y0) / YSCALE_K     # k 单位
        labels = df["cluster"].to_numpy()

        fig = plt.figure(figsize=FIGSIZE)
        ax = plt.gca()
        for c in sorted(set(labels)):
            m = (labels == c)
            ax.scatter(x_rel[m], y_rel[m],
                       s=POINT_SIZE, lw=0, color=color_map[int(c)], alpha=0.85)

        # 坐标范围（与所有算法一致）
        ax.set_xlim(frame["x_left_abs"] - x0, frame["x_right_abs"] - x0)
        ax.set_ylim((frame["y_bottom_abs"] - y0) / YSCALE_K,
                    (frame["y_top_abs"]    - y0) / YSCALE_K)

        # 无标题、无轴标签，仅刻度
        ax.set_title(None); ax.set_xlabel(None); ax.set_ylabel(None)

        # 更密的主/次刻度 & 大字号；y 轴整数刻度（千）
        ax.xaxis.set_major_locator(MaxNLocator(nbins=SINGLE_MAJOR_TICKS))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=SINGLE_MAJOR_TICKS, integer=True))
        ax.xaxis.set_minor_locator(AutoMinorLocator(SINGLE_MINOR_DIVISOR))
        ax.yaxis.set_minor_locator(AutoMinorLocator(SINGLE_MINOR_DIVISOR))
        ax.tick_params(which="major", direction="in",
                       labelsize=SINGLE_TICK_SIZE, length=6, width=1.2)
        ax.tick_params(which="minor", direction="in",
                       length=3, width=0.8)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{int(np.round(v))}"))
        # 如需 x 轴也显示整数，取消下一行：
        # ax.xaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{int(np.round(v))}"))

        out_dir = ensure_outdir(find_cluster_dir(algo, did) / "figs")
        stem = f"repaired_{did}"
        out_name = f"{stem}_unified_cluster_scatter_relk_{proj_names[0]}_vs_{proj_names[1]}"
        out_path = out_dir / out_name
        savefig(fig, out_path, dpi=600)
        print(f"[OK] did={did} algo={algo} -> {out_path.with_suffix('.pdf')}")

# ------- 入口 -------
def main():
    for did in DATASET_IDS:
        process_dataset(did)
    print("✅ 完成：统一配色+统一坐标框的聚类散点已导出。")

if __name__ == "__main__":
    main()

# make_all_figs.py
from pathlib import Path
from typing import Optional
import json, math, warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager

# ============================== 中文字体与全局样式 ==============================
def pick_cn_font() -> str:
    """从已安装字体中挑一个可用的中文字体，避免乱码。"""
    preferred = [
        "SimSun", "Microsoft YaHei", "SimHei",
        "Noto Sans CJK SC", "Source Han Sans SC",
        "STSong", "FangSong"
    ]
    available = {f.name for f in fontManager.ttflist}
    for name in preferred:
        if name in available:
            return name
    # 兜底：退回到 DejaVu Sans（一般不会乱码，但不保证所有中文）
    return "DejaVu Sans"

CN_FONT_NAME = pick_cn_font()

# 全局：英文默认 Times New Roman，中文用 FontProperties 指定
matplotlib.rc('font', family='Times New Roman')
matplotlib.rcParams['axes.unicode_minus'] = False    # 修复负号乱码
matplotlib.rcParams['pdf.fonttype'] = 42             # 更好地嵌入字体
matplotlib.rcParams['ps.fonttype'] = 42

# 你要求的风格：标题更小、更短
TITLE_SIZE  = 12
LABEL_SIZE  = 12
LEGEND_SIZE = 11
NOTE_SIZE   = 10   # 图内角落小字（Sil/DB等）

cn_font_title  = FontProperties(family=CN_FONT_NAME, size=TITLE_SIZE)
cn_font_label  = FontProperties(family=CN_FONT_NAME, size=LABEL_SIZE)
cn_font_legend = FontProperties(family=CN_FONT_NAME, size=LEGEND_SIZE)
cn_font_note   = FontProperties(family=CN_FONT_NAME, size=NOTE_SIZE)

warnings.filterwarnings("ignore", category=UserWarning)

# ============================== 配置（按你的目录） ==============================
ROOT_DIR = Path("demo_results/clustered_data/HC")
CLEANING_ALGOS = ["baran", "mode", "holoclean"]
DATASET_IDS = [2, 5]           # 你只做了 2 和 5
STEM_PREFIX = "repaired_"      # 前缀保持不变
# ============================================================================

# ------------------------ 通用 I/O 小工具 -------------------------
def _paths(base_dir: Path, stem: str, state: str):
    p = lambda suf: base_dir / f"{stem}_{state}_{suf}"
    return {
        "summary": p("summary.json"),
        "clusters": p("clusters.csv"),
        "x_scaled": p("X_scaled.npy"),
        "columns": p("columns.json"),
        "trials_csv": p("trials.csv"),
        "tree_profile": p("tree_profile.csv"),
        "tree_npz": p("tree.npz"),
    }

def _ensure_outdir(d: Path) -> Path:
    out = d / "figs"
    out.mkdir(parents=True, exist_ok=True)
    return out

def _savefig(fig: plt.Figure, out_path: Path, dpi=300):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=dpi)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)

def _load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _detect_stems_states(folder: Path):
    """
    在 folder 下查找 *_summary.json，返回 {stem: [states...]}。
    如 repaired_2_raw_summary.json -> stem='repaired_2', state='raw'
    """
    mapping = {}
    for f in folder.glob("*_summary.json"):
        name = f.name[:-len("_summary.json")]
        if "_" not in name:
            continue
        stem, state = name.rsplit("_", 1)
        mapping.setdefault(stem, [])
        if state not in mapping[stem]:
            mapping[stem].append(state)
    return mapping

def _prefer_state(states):
    """优先 cleaned，其次 raw；若都无，返回 None。"""
    if "cleaned" in states:
        return "cleaned"
    if "raw" in states:
        return "raw"
    return None

# -------------------------- 绘图工具函数 --------------------------
def _fit_pca_2d(X: np.ndarray, fit_on: Optional[np.ndarray] = None, random_state: int = 0):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=random_state)
    if fit_on is None:
        return pca, pca.fit_transform(X)
    else:
        pca.fit(fit_on)
        return pca, pca.transform(X)

# 1) 二维散点 + 决策背景（最近质心，贴合HC）
def plot_scatter_with_decision_bg(base_dir: Path, stem: str, state: str,
                                  display_name: str, fit_on_state: Optional[str] = "cleaned"):
    paths = _paths(base_dir, stem, state)
    if not (paths["x_scaled"].exists() and paths["clusters"].exists() and paths["summary"].exists()):
        print(f"[跳过] 散点图：缺文件 -> {paths}")
        return
    X = np.load(paths["x_scaled"])
    dfc = pd.read_csv(paths["clusters"])
    labels = dfc["cluster"].to_numpy()
    sm = _load_json(paths["summary"])

    # 轮廓系数和DB指标调整
    sil = sm["silhouette"] * (1 + sm["silhouette"]) / 2  # 调整 Silhouette
    db = 1 / (1 + sm["davies_bouldin"])  # 调整 DB

    # 为了前后对比公平：若有 cleaned，则用 cleaned 的 X 来拟合 PCA
    p_fit = None
    if fit_on_state is not None:
        fit_paths = _paths(base_dir, stem, fit_on_state)
        if fit_paths["x_scaled"].exists():
            p_fit = np.load(fit_paths["x_scaled"])

    _, X2d = _fit_pca_2d(X, fit_on=p_fit)

    uniq = np.unique(labels)
    centers = np.vstack([X2d[labels == c].mean(axis=0) for c in uniq])

    x_min, x_max = X2d[:, 0].min()-1, X2d[:, 0].max()+1
    y_min, y_max = X2d[:, 1].min()-1, X2d[:, 1].max()+1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    assign = ((grid[:, None, :] - centers[None, :, :]) ** 2).sum(-1).argmin(1).reshape(xx.shape)

    fig = plt.figure(figsize=(4.2, 3.6))
    ax = plt.gca()
    ax.contourf(xx, yy, assign, alpha=0.15)  # 背景
    for c in uniq:
        idx = labels == c
        ax.scatter(X2d[idx, 0], X2d[idx, 1], s=14)
    ax.scatter(centers[:, 0], centers[:, 1], marker="X", s=60, edgecolors="k", linewidths=0.6)

    ax.set_xlabel("主成分1", fontproperties=cn_font_label)
    ax.set_ylabel("主成分2", fontproperties=cn_font_label)
    # 简短标题 + 小角注（保证不被裁切）
    ax.set_title(f"{display_name}", fontproperties=cn_font_title)
    ax.text(0.01, 0.99, f"Sil*={sil:.3f}   DB*={db:.3f}",
            transform=ax.transAxes, va='top', ha='left', fontproperties=cn_font_note)

    # 不绘制图例
    # ax.legend(frameon=False, fontsize=LEGEND_SIZE, ncol=2, prop=cn_font_legend)

    _savefig(fig, _ensure_outdir(base_dir) / f"{stem}_{state}_scatter_decision")

# 2) 指标条形（Sil / 1/(1+DB) / Combined）
def plot_metrics_bars(base_dir: Path, stem: str, states: list[str], display_name: str):
    rows = []
    for st in states:
        path = _paths(base_dir, stem, st)["summary"]
        if not path.exists():
            continue
        sm = _load_json(path)
        rows.append({
            "state": st,
            "silhouette": sm["silhouette"],
            "db_inv": 1.0 / (1.0 + sm["davies_bouldin"]),
            "combined": sm["combined"],
        })
    if not rows:
        print(f"[跳过] 指标条形图：无可用 summary -> {base_dir}, {stem}")
        return
    df = pd.DataFrame(rows)
    outdir = _ensure_outdir(base_dir)

    metric_map = {
        "silhouette": "轮廓系数（越大越好）",
        "db_inv": "1/(1+DB)（越大越好）",
        "combined": "综合得分（越大越好）",
    }
    for metric_key, ylabel_cn in metric_map.items():
        fig = plt.figure(figsize=(3.8, 3.2))
        ax = plt.gca()
        ax.bar(df["state"], df[metric_key])
        ax.set_ylabel(ylabel_cn, fontproperties=cn_font_label)
        ax.set_xlabel("状态", fontproperties=cn_font_label)
        ax.set_title(f"{display_name}·{ylabel_cn}", fontproperties=cn_font_title)
        _savefig(fig, outdir / f"{stem}_metrics_{metric_key}")

# 3) HC 过程曲线（SSE 收敛 & 合并距离）
def plot_hc_process_curves(base_dir: Path, stem: str, state: str, display_name: str):
    paths = _paths(base_dir, stem, state)
    if not (paths["tree_profile"].exists() and paths["summary"].exists()):
        print(f"[跳过] 过程曲线：缺文件 -> {paths}")
        return
    prof = pd.read_csv(paths["tree_profile"])
    sm = _load_json(paths["summary"])
    k_star = sm.get("best_k", None)

    # A: sse_ratio vs n_clusters
    fig = plt.figure(figsize=(4.2, 3.2))
    ax = plt.gca()
    ax.semilogy(prof["n_clusters"], prof["sse_ratio"])
    ax.invert_xaxis()
    if k_star is not None:
        ax.axvline(k_star, linestyle="--", linewidth=1)
    ax.set_xlabel("聚类数 k", fontproperties=cn_font_label)
    ax.set_ylabel("总SSE比例（相对单簇）", fontproperties=cn_font_label)
    ax.set_title(f"{display_name}·SSE收敛", fontproperties=cn_font_title)
    _savefig(fig, _ensure_outdir(base_dir) / f"{stem}_{state}_sse_profile")

    # B: merge_distance vs step
    fig = plt.figure(figsize=(4.2, 3.2))
    ax = plt.gca()
    ax.plot(prof["step"], prof["merge_distance"])
    ax.set_xlabel("合并步数", fontproperties=cn_font_label)
    ax.set_ylabel("合并距离", fontproperties=cn_font_label)
    ax.set_title(f"{display_name}·合并距离", fontproperties=cn_font_title)
    _savefig(fig, _ensure_outdir(base_dir) / f"{stem}_{state}_merge_distance")

# 4) 层次聚类树（Dendrogram）
def plot_dendrogram(base_dir: Path, stem: str, state: str, display_name: str):
    try:
        from scipy.cluster.hierarchy import dendrogram
    except Exception:
        print("[提示] 未安装 scipy，跳过树状图。")
        return
    paths = _paths(base_dir, stem, state)
    if not (paths["tree_npz"].exists() and paths["summary"].exists()):
        print(f"[跳过] 树状图：缺文件 -> {paths}")
        return
    npz = np.load(paths["tree_npz"])
    children = npz["children"]
    distances = np.nan_to_num(npz["distances"].astype(float), nan=0.0)
    counts = npz["counts"].astype(float)
    sm = _load_json(paths["summary"])
    k_star = sm.get("best_k", None)

    n_samples = children.shape[0] + 1
    Z = np.zeros((children.shape[0], 4), dtype=float)
    for i, (a, b) in enumerate(children):
        Z[i, 0] = a
        Z[i, 1] = b
        Z[i, 2] = distances[i] if i < len(distances) else 0.0
        Z[i, 3] = counts[n_samples + i] if (n_samples + i) < len(counts) else 2.0

    color_th = None
    if k_star is not None:
        step_cut = max(n_samples - k_star - 1, 0)
        if 0 <= step_cut < Z.shape[0]:
            color_th = Z[step_cut, 2]

    fig = plt.figure(figsize=(6.2, 3.6))
    ax = plt.gca()
    dendrogram(Z, color_threshold=color_th, no_labels=True, count_sort=True, ax=ax)

    # 设置字体大小
    if color_th is not None:
        ax.axhline(color_th, linestyle="--", linewidth=1)

    # 增大横纵坐标轴标题和总标题的字体大小
    ax.set_xlabel("样本", fontproperties=cn_font_label, fontsize=14)  # 横坐标标题字体
    ax.set_ylabel("合并距离", fontproperties=cn_font_label, fontsize=14)  # 纵坐标标题字体
    ax.set_title(f"{display_name}·树状图（k*={k_star}）", fontproperties=cn_font_title, fontsize=16)  # 总标题字体

    # 增大坐标轴数据的字体大小
    ax.tick_params(axis='both', which='major', labelsize=12)  # 坐标轴刻度数据字体

    _savefig(fig, _ensure_outdir(base_dir) / f"{stem}_{state}_dendrogram")


# 5) 超参数试验概览（轨迹 / Score-k / 配置热力）
def plot_trials_overview(base_dir: Path, stem: str, state: str, display_name: str):
    paths = _paths(base_dir, stem, state)
    if not paths["trials_csv"].exists():
        print(f"[跳过] 试验概览：缺文件 -> {paths['trials_csv']}")
        return
    trials = pd.read_csv(paths["trials_csv"])
    if trials.empty:
        print("[跳过] 试验概览：空文件"); return

    # A) 轨迹
    best_idx = trials["combined_score"].idxmax()
    best = trials.loc[best_idx]

    fig = plt.figure(figsize=(4.2, 3.2))
    ax = plt.gca()
    ax.scatter(trials["trial_number"], trials["combined_score"], s=14)
    ax.scatter([best["trial_number"]], [best["combined_score"]],
               marker="*", s=100, edgecolors="k")
    ax.set_xlabel("试验序号", fontproperties=cn_font_label)
    ax.set_ylabel("综合得分", fontproperties=cn_font_label)
    ax.set_title(f"{display_name}·试验轨迹", fontproperties=cn_font_title)
    _savefig(fig, _ensure_outdir(base_dir) / f"{stem}_{state}_trials_trajectory")

    # B) Score vs k
    fig = plt.figure(figsize=(4.2, 3.2))
    ax = plt.gca()
    ax.scatter(trials["n_clusters"], trials["combined_score"], s=14)
    ax.set_xlabel("聚类数 k", fontproperties=cn_font_label)
    ax.set_ylabel("综合得分", fontproperties=cn_font_label)
    ax.set_title(f"{display_name}·得分-聚类数", fontproperties=cn_font_title)
    _savefig(fig, _ensure_outdir(base_dir) / f"{stem}_{state}_trials_score_vs_k")

    # C) (linkage, metric) 最佳得分热力
    pivot = (trials.groupby(["linkage","metric"])["combined_score"]
                   .max().reset_index()
                   .pivot(index="linkage", columns="metric", values="combined_score"))
    fig = plt.figure(figsize=(4.6, 3.4))
    ax = plt.gca()
    im = ax.imshow(pivot.values, aspect="auto")
    ax.set_xticks(range(pivot.shape[1])); ax.set_xticklabels(list(pivot.columns))
    ax.set_yticks(range(pivot.shape[0])); ax.set_yticklabels(list(pivot.index))
    plt.colorbar(im, fraction=0.046, pad=0.04)
    ax.set_title(f"{display_name}·配置最佳得分", fontproperties=cn_font_title)
    # 数值标注
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=NOTE_SIZE)
    _savefig(fig, _ensure_outdir(base_dir) / f"{stem}_{state}_trials_heatmap")

# 6) 簇大小分布
def plot_cluster_sizes(base_dir: Path, stem: str, state: str, display_name: str):
    paths = _paths(base_dir, stem, state)
    if not paths["summary"].exists():
        print(f"[跳过] 簇大小：缺 summary"); return
    sm = _load_json(paths["summary"])
    sizes = sm.get("cluster_sizes", None)
    if sizes is None:
        if not paths["clusters"].exists():
            print("[跳过] 簇大小：无 sizes 且缺 clusters.csv"); return
        labels = pd.read_csv(paths["clusters"])["cluster"].to_numpy()
        uniq = np.unique(labels)
        sizes = [int(np.sum(labels == c)) for c in uniq]
    fig = plt.figure(figsize=(4.2, 3.2))
    ax = plt.gca()
    ax.bar([f"簇{i}" for i in range(len(sizes))], sizes)
    ax.set_xlabel("簇", fontproperties=cn_font_label)
    ax.set_ylabel("样本数量", fontproperties=cn_font_label)
    ax.set_title(f"{display_name}·簇大小", fontproperties=cn_font_title)
    _savefig(fig, _ensure_outdir(base_dir) / f"{stem}_{state}_cluster_sizes")

# 7) 簇中心热力图（标准化空间，Top-K 特征）
def plot_centers_heatmap(base_dir: Path, stem: str, state: str, display_name: str, topk_features: int = 20):
    paths = _paths(base_dir, stem, state)
    if not (paths["summary"].exists() and paths["columns"].exists()):
        print(f"[跳过] 簇中心热力：缺文件 -> {paths}")
        return
    sm = _load_json(paths["summary"])
    cols = _load_json(paths["columns"]).get("columns", [])
    centers = sm.get("cluster_centers", None)

    # 兜底：从 X_scaled + labels 计算
    if centers is None:
        if not (paths["x_scaled"].exists() and paths["clusters"].exists()):
            print("[跳过] 簇中心热力：无 centers 且缺 x_scaled/clusters"); return
        X = np.load(paths["x_scaled"])
        labels = pd.read_csv(paths["clusters"])["cluster"].to_numpy()
        uniq = np.unique(labels)
        centers = [X[labels == c].mean(axis=0).tolist() for c in uniq]

    centers = np.asarray(centers)  # (k, d)
    d = centers.shape[1]
    if not cols or len(cols) != d:
        cols = [f"f{i}" for i in range(d)]

    # 取平均绝对值较大的 Top-K 特征
    importance = np.abs(centers).mean(axis=0)
    idx = np.argsort(importance)[::-1][:min(topk_features, d)]
    centers_sel = centers[:, idx]
    cols_sel = [cols[i] for i in idx]

    fig = plt.figure(figsize=(min(6.5, 1.0 + 0.35*len(cols_sel)), 3.6))
    ax = plt.gca()
    im = ax.imshow(centers_sel, aspect="auto")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(cols_sel))); ax.set_xticklabels(cols_sel, rotation=60, ha="right")
    ax.set_yticks(range(centers_sel.shape[0])); ax.set_yticklabels([f"簇{i}" for i in range(centers_sel.shape[0])])
    ax.set_title(f"{display_name}·簇中心热力", fontproperties=cn_font_title)
    _savefig(fig, _ensure_outdir(base_dir) / f"{stem}_{state}_centers_heatmap")

# 8) raw → cleaned 的 Sankey（可选，需要 plotly）
def plot_sankey_raw_to_cleaned(base_dir: Path, stem: str, display_name_left: str, display_name_right: str):
    try:
        import plotly.graph_objects as go
    except Exception:
        print("[提示] 未安装 plotly，跳过 Sankey。")
        return
    left = _paths(base_dir, stem, "raw")["clusters"]
    right = _paths(base_dir, stem, "cleaned")["clusters"]
    if not (left.exists() and right.exists()):
        print("[跳过] Sankey：需要同时存在 raw 与 cleaned。"); return

    L = pd.read_csv(left)[["orig_index", "cluster"]].rename(columns={"cluster": "left"})
    R = pd.read_csv(right)[["orig_index", "cluster"]].rename(columns={"cluster": "right"})
    M = L.merge(R, on="orig_index", how="inner")

    left_cats = sorted(M["left"].unique())
    right_cats = sorted(M["right"].unique())
    lmap = {c: i for i, c in enumerate(left_cats)}
    rmap = {c: len(left_cats) + j for j, c in enumerate(right_cats)}

    ct = M.groupby(["left", "right"]).size().reset_index(name="n")
    sources = [lmap[i] for i in ct["left"]]
    targets = [rmap[j] for j in ct["right"]]
    values = ct["n"].tolist()
    labels = [f"{display_name_left} 簇{c}" for c in left_cats] + \
             [f"{display_name_right} 簇{c}" for c in right_cats]

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(label=labels, pad=20, thickness=14),
        link=dict(source=sources, target=targets, value=values)
    ))
    fig.update_layout(title=f"{stem}: {display_name_left} → {display_name_right}")
    outdir = _ensure_outdir(base_dir)
    html_path = outdir / f"{stem}_sankey_raw_to_cleaned.html"
    fig.write_html(str(html_path))
    try:
        fig.write_image(str(outdir / f"{stem}_sankey_raw_to_cleaned.png"))
        fig.write_image(str(outdir / f"{stem}_sankey_raw_to_cleaned.pdf"))
    except Exception:
        print("[提示] 若需静态导出，请安装 kaleido： pip install -U kaleido")
    print(f"[OK] Sankey 保存：{html_path}")

# ----------------------- 跨算法对比（每 dataset） -----------------------
def compare_algorithms_metrics(root_dir: Path, dataset_id: int):
    """
    聚合 {baran, mode, holoclean} 各自的最佳可用 state（优先 cleaned），
    输出 Sil / 1/(1+DB) / Combined 三张条形图到 compare_figs_dataset_{id}/
    """
    rows = []
    for algo in CLEANING_ALGOS:
        base = root_dir / algo / f"clustered_{dataset_id}"
        if not base.exists():
            continue
        mapping = _detect_stems_states(base)
        if not mapping:
            continue
        stem = sorted(mapping.keys())[0]
        st = _prefer_state(mapping[stem])
        if st is None:
            continue
        sm = _load_json(_paths(base, stem, st)["summary"])
        rows.append({
            "algo": algo,
            "state": st,
            "silhouette": sm["silhouette"],
            "db_inv": 1.0 / (1.0 + sm["davies_bouldin"]),
            "combined": sm["combined"],
        })
    if not rows:
        print(f"[跳过] 跨算法对比：数据集 {dataset_id} 无可比较项目")
        return

    cmp_dir = root_dir / f"compare_figs_dataset_{dataset_id}"
    cmp_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)

    metric_map = {
        "silhouette": "轮廓系数（越大越好）",
        "db_inv": "1/(1+DB)（越大越好）",
        "combined": "综合得分（越大越好）",
    }

    for metric_key, ylabel_cn in metric_map.items():
        fig = plt.figure(figsize=(4.6, 3.4))
        ax = plt.gca()
        ax.bar(df["algo"], df[metric_key])
        ax.set_ylabel(ylabel_cn, fontproperties=cn_font_label)
        ax.set_xlabel("清洗算法", fontproperties=cn_font_label)
        ax.set_title(f"数据集 {dataset_id}·{ylabel_cn}", fontproperties=cn_font_title)
        _savefig(fig, cmp_dir / f"dataset{dataset_id}_compare_{metric_key}")

# ----------------------------- 主调度 -----------------------------
def process_folder_for_algo_dataset(algo: str, dataset_id: int):
    base_dir = ROOT_DIR / algo / f"clustered_{dataset_id}"
    if not base_dir.exists():
        print(f"[跳过] 不存在：{base_dir}")
        return
    mapping = _detect_stems_states(base_dir)
    if not mapping:
        print(f"[跳过] 无 *_summary.json：{base_dir}")
        return

    for stem, states in mapping.items():
        display_name = f"{algo}·数据集{dataset_id}"

        # 1) 指标条形（按 state）
        plot_metrics_bars(base_dir, stem, sorted(states), display_name)

        # 2) 按 state 出其余图；散点图若有 cleaned，则用 cleaned 拟合PCA
        fit_on_state = "cleaned" if "cleaned" in states else None
        for st in sorted(states):
            plot_scatter_with_decision_bg(base_dir, stem, st, display_name, fit_on_state=fit_on_state)
            plot_hc_process_curves(base_dir, stem, st, display_name)
            plot_dendrogram(base_dir, stem, st, display_name)
            plot_trials_overview(base_dir, stem, st, display_name)
            plot_cluster_sizes(base_dir, stem, st, display_name)
            plot_centers_heatmap(base_dir, stem, st, display_name, topk_features=20)

        # 3) 若 raw & cleaned 同时存在，画 Sankey
        if "raw" in states and "cleaned" in states:
            plot_sankey_raw_to_cleaned(base_dir, stem,
                                       f"{algo}-未清洗", f"{algo}-清洗后")

def main():
    print(f"[信息] 选用中文字体：{CN_FONT_NAME}")
    for algo in CLEANING_ALGOS:
        for did in DATASET_IDS:
            process_folder_for_algo_dataset(algo, did)
    for did in DATASET_IDS:
        compare_algorithms_metrics(ROOT_DIR, did)
    print("✅ 绘图完成。输出位于各目录 figs/ 以及 compare_figs_dataset_{num}/ 。")

if __name__ == "__main__":
    main()

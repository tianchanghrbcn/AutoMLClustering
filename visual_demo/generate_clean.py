#!/usr/bin/env python3
# build_5clusters_3D_tsne.py —— 分离度“中等-稍弱”的 5 簇示例 + t-SNE 投影
# ------------------------------------------------------------------------
# 1. 生成 300 行 (age, income, saving)               2. 双 CSV 输出
# 3. t-SNE↓2D 可视化，保存 reference_clusters_tsne.pdf
# ------------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

# ---------- 参数 --------------------------------------------------------
SEED            = 42
N_CLUSTERS      = 5
N_PER_CLUSTER   = 60
OUT_WITH_SEG    = "clean_3D_tsne_withseg.csv"
OUT_NO_SEG      = "clean_3D_tsne_noseg.csv"
OUT_FIG_PDF     = "reference_clusters_tsne.pdf"

# 分布配置：在「中等」的基础上把均值拉近、方差放大 → 分离度稍弱
DIST = {
    "A 学生":    {"age": (22, 5, 16, 30),
                  "inc": (20_000, 7_000,  8_000, 35_000),
                  "sav": ( 3_000, 2_000,      0,  8_000)},
    "B 初入职场": {"age": (30, 5, 22, 40),
                  "inc": (35_000, 8_000, 18_000, 52_000),
                  "sav": (12_000, 4_000,  4_000, 22_000)},
    "C 中产":     {"age": (39, 5, 29, 49),
                  "inc": (57_000, 8_000, 35_000, 75_000),
                  "sav": (28_000, 7_000, 12_000, 45_000)},
    "D 高管":     {"age": (50, 5, 40, 60),
                  "inc": (82_000,10_000, 55_000,105_000),
                  "sav": (55_000,12_000, 30_000, 80_000)},
    "E 退休":     {"age": (63, 4, 55, 71),
                  "inc": (52_000, 8_000, 30_000, 70_000),
                  "sav": (70_000,10_000, 45_000, 95_000)},
}

# ---------- 生成单簇 -----------------------------------------------------
def _gen_cluster(label: str, n: int) -> pd.DataFrame:
    cfg = DIST[label]
    age = np.random.normal(*cfg["age"][:2], n)\
            .clip(*cfg["age"][2:]).round().astype(int)
    inc = np.random.normal(*cfg["inc"][:2], n)\
            .clip(*cfg["inc"][2:]).round(-2).astype(int)
    sav = np.random.normal(*cfg["sav"][:2], n)\
            .clip(*cfg["sav"][2:]).round(-2).astype(int)
    return pd.DataFrame({"age": age, "income": inc, "saving": sav,
                         "segment": label})

# ---------- 主流程 -------------------------------------------------------
def main() -> None:
    np.random.seed(SEED)

    labels_full = list(DIST)[:N_CLUSTERS]
    df = pd.concat([_gen_cluster(lab, N_PER_CLUSTER) for lab in labels_full],
                   ignore_index=True)

    # 打乱并加主键
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    df.insert(0, "ID", np.arange(1, len(df) + 1))

    # ---------------- 保存 CSV ----------------
    df.to_csv(OUT_WITH_SEG, index=False, encoding="utf-8-sig")
    df.drop(columns=["segment"]).to_csv(OUT_NO_SEG, index=False,
                                        encoding="utf-8-sig")
    print(f"[OK] 生成 {len(df)} 行 → {OUT_WITH_SEG} / {OUT_NO_SEG}")

    # ---------------- t-SNE 降维 ----------------
    X = df[["age", "income", "saving"]].values
    X_std = StandardScaler().fit_transform(X)

    tsne = TSNE(n_components=2,
                perplexity=30,        # 300 行数据经验值
                random_state=SEED,
                init="random",
                learning_rate=200)
    X_tsne = tsne.fit_transform(X_std)

    # ---------------- 绘图 --------------------
    plt.rcParams["font.sans-serif"] = ["SimHei"]   # 显示中文
    plt.rcParams["axes.unicode_minus"] = False
    cmap = plt.cm.get_cmap("Set1", N_CLUSTERS)
    legend_map = {lab: chr(ord('A') + i) for i, lab in enumerate(labels_full)}

    plt.figure(figsize=(6, 4))
    for idx, lab in enumerate(labels_full):
        mask = df["segment"] == lab
        plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                    s=30, color=cmap(idx), label=legend_map[lab])

    plt.xlabel("t-SNE 维度 1")
    plt.ylabel("t-SNE 维度 2")
    plt.title("t-SNE 投影下的五簇（分离度：中等稍弱）")
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT_FIG_PDF, dpi=300)
    plt.close()
    print(f"[OK] 图形已保存: {OUT_FIG_PDF}")

if __name__ == "__main__":
    main()

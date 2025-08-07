#!/usr/bin/env python3
# build_5clusters_2D_segcolor.py —— 按指定 5 色绘制 age-income 二维五簇示例
# ------------------------------------------------------------------
# 1. 生成 300 行 (age, income)            2. 双 CSV 输出
# 3. age-income 平面散点图，保存 reference_clusters_2d.pdf
# ------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- 固定参数 ----------------------------------------------------
SEED            = 42
N_CLUSTERS      = 5
N_PER_CLUSTER   = 60
OUT_WITH_SEG    = "clean_withseg.csv"
OUT_NO_SEG      = "clean_noseg.csv"
OUT_FIG_PDF     = "reference_clusters.pdf"

# ---------- 指定颜色与顺序 ---------------------------------------------
SEG_COLOR = {"A": "#1f77b4",  # 蓝
             "B": "#ffbf00",  # 亮黄
             "C": "#d62728",  # 红
             "D": "#17becf",  # 青
             "E": "#7f7f7f"}  # 灰
SEG_ORDER = ["A", "B", "C", "D", "E"]

# ---------- 生成分布配置 -----------------------------------------------
DIST = {
    #     μ_age, σ_age,  min, max     μ_inc, σ_inc,  min,  max
    "A": {"age": (22,    4,     16,  30),
          "inc": (22_000,6_000,  8_000, 32_000)},  # ↓σ，μ_inc -3k

    "B": {"age": (31,    4,     23,  39),          # μ_age +1
          "inc": (38_000,6_000, 18_000, 50_000)},  # μ_inc +3k, σ↓2k

    "C": {"age": (40,    4,     32,  48),          # μ_age +1
          "inc": (61_000,6_000, 40_000, 78_000)},  # μ_inc +4k, σ↓2k

    "D": {"age": (50,    4,     42,  58),
          "inc": (86_000,8_000, 60_000,108_000)},  # σ_inc ↓2k

    "E": {"age": (63,    3,     56,  70),          # σ_age ↓1
          "inc": (54_000,6_000, 33_000, 72_000)},  # σ_inc ↓2k, μ_inc +2k
}


# ---------- 生成单簇 -----------------------------------------------------
def _gen_cluster(label: str, n: int) -> pd.DataFrame:
    cfg = DIST[label]
    age = np.random.normal(*cfg["age"][:2], n)\
            .clip(*cfg["age"][2:]).round().astype(int)
    inc = np.random.normal(*cfg["inc"][:2], n)\
            .clip(*cfg["inc"][2:]).round(-2).astype(int)
    return pd.DataFrame({"age": age, "income": inc, "segment": label})

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

    # ---------------- 绘图 --------------------
    plt.rcParams["font.sans-serif"] = ["SimHei"]   # 显示中文
    plt.rcParams["axes.unicode_minus"] = False
    legend_map = {lab: SEG_ORDER[i] for i, lab in enumerate(labels_full)}

    plt.figure(figsize=(6, 4))
    for idx, lab in enumerate(labels_full):
        seg_letter = legend_map[lab]           # A/B/C/D/E
        color = SEG_COLOR[seg_letter]
        mask = df["segment"] == lab
        plt.scatter(df.loc[mask, "age"], df.loc[mask, "income"],
                    s=30, color=color, label=seg_letter)

    plt.xlabel("年龄")
    plt.ylabel("年收入")
    plt.title("年龄-年收入平面上的五簇（分离度：中等稍弱）")
    plt.legend(loc="best", fontsize=9, title="簇标识")
    plt.tight_layout()
    plt.savefig(OUT_FIG_PDF, dpi=300)
    plt.close()
    print(f"[OK] 图形已保存: {OUT_FIG_PDF}")

if __name__ == "__main__":
    main()

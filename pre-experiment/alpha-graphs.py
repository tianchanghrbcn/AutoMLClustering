#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_alpha_metrics.py  ── 绘制 α 网格结果（仅 PDF，中文标题正常）
输出：
  outputs/alpha_vs_median.pdf
  outputs/alpha_vs_variance.pdf
"""
import pathlib, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import font_manager as fm

# ---------- A. 注册并启用中文字体 ---------------------------------------
FONT_PATH = r"C:\Windows\Fonts\simsun.ttc"      # ⚠️ 改成本机可用的中文字体文件
if pathlib.Path(FONT_PATH).is_file():
    fm.fontManager.addfont(FONT_PATH)
    matplotlib.rcParams["font.family"] = fm.FontProperties(fname=FONT_PATH).get_name()
else:
    print(f"[WARN] 找不到字体文件: {FONT_PATH}\n→ 中文可能仍会缺字")
matplotlib.rcParams["axes.unicode_minus"] = False     # 负号正常显示
# -----------------------------------------------------------------------

# ---------- 1. 读取 ----------------------------------------------------
csv = pathlib.Path("outputs") / "alpha_metrics.csv"
if not csv.exists():
    raise SystemExit(f"❌ 找不到 {csv.resolve()}")
df = pd.read_csv(csv).sort_values("alpha").reset_index(drop=True)

# ---------- 2. 差分 ----------------------------------------------------
df["d1_m"] = np.gradient(df["median_avg"], df["alpha"])
df["d2_m"] = np.gradient(df["d1_m"], df["alpha"])
df["d1_v"] = np.gradient(df["max_variance"], df["alpha"])
df["d2_v"] = np.gradient(df["d1_v"], df["alpha"])

plt.rcParams.update({
    "pdf.fonttype": 42,             # 嵌入 TrueType，防止方块字
    "axes.titlesize": 13,
    "axes.labelsize": 13,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})
out_dir = pathlib.Path("outputs")
out_dir.mkdir(exist_ok=True)

# ---------- 3. α 对 m_avg ---------------------------------------------
fig, ax1 = plt.subplots(figsize=(6, 4))
ln1, = ax1.plot(df["alpha"], df["median_avg"], marker="o",
                color="tab:blue", label=r"$m_{\mathrm{avg}}$")
ax1.set_xlabel(r"$\alpha$")
ax1.set_ylabel(r"平均中位数 $m_{\mathrm{avg}}$（↑ 越大越好）")
ax1.grid(True, alpha=0.4)

ax2 = ax1.twinx()
ln2, = ax2.plot(df["alpha"], df["d2_m"], linestyle="--", marker="x",
                color="tab:green", label=r"$m_{\mathrm{avg}}''$")
ax2.set_ylabel(r"$m_{\mathrm{avg}}$ 的二阶导数")
ax2.axhline(0, linestyle=":", linewidth=0.8)

ax1.legend([ln1, ln2], [l.get_label() for l in (ln1, ln2)], loc="upper left")

fig.tight_layout()
fig.savefig(out_dir / "alpha_vs_median.pdf", bbox_inches="tight")
plt.close(fig)

# ---------- 4. α 对 v_max ---------------------------------------------
fig, ax1 = plt.subplots(figsize=(6, 4))
ln1, = ax1.plot(df["alpha"], df["max_variance"], marker="o",
                color="tab:blue", label=r"$v_{\max}$")
ax1.set_xlabel(r"$\alpha$")
ax1.set_ylabel(r"最大方差 $v_{\max}$（↓ 越小越好）")
ax1.grid(True, alpha=0.4)

ax2 = ax1.twinx()
ln2, = ax2.plot(df["alpha"], -df["d2_v"], linestyle="--", marker="x",
                color="tab:green", label=r"$-\;v_{\max}''$")
ax2.set_ylabel(r"$-\;v_{\max}$ 的二阶导数")
ax2.axhline(0, linestyle=":", linewidth=0.8)

ax1.legend([ln1, ln2], [l.get_label() for l in (ln1, ln2)], loc="upper right")

fig.tight_layout()
fig.savefig(out_dir / "alpha_vs_variance.pdf", bbox_inches="tight")
plt.close(fig)

print("✓ 图像已保存为 PDF：alpha_vs_median.pdf  alpha_vs_variance.pdf")

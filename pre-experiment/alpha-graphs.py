#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_alpha_metrics.py  ── 绘制 α 网格结果
输出：
  outputs/alpha_vs_median.{pdf,eps}
  outputs/alpha_vs_variance.{pdf,eps}
"""
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- 1. 读取 ----------
csv = pathlib.Path("outputs") / "alpha_metrics.csv"
if not csv.exists():
    raise SystemExit(f"❌ 找不到 {csv.resolve()}")
df = pd.read_csv(csv).sort_values("alpha").reset_index(drop=True)

# ---------- 2. 差分 ----------
df["d1_m"] = np.gradient(df["median_avg"], df["alpha"])
df["d2_m"] = np.gradient(df["d1_m"], df["alpha"])
df["d1_v"] = np.gradient(df["max_variance"], df["alpha"])
df["d2_v"] = np.gradient(df["d1_v"], df["alpha"])

plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.titlesize": 13,
    "axes.labelsize": 13,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})
out_dir = pathlib.Path("outputs")
out_dir.mkdir(exist_ok=True)

# ---------- 3. α vs m_avg ----------
fig, ax1 = plt.subplots(figsize=(6, 4))
ln1, = ax1.plot(df["alpha"], df["median_avg"], marker="o",
                color="tab:blue", label=r"$m_{\mathrm{avg}}$")
ax1.set_xlabel(r"$\alpha$")
ax1.set_ylabel(r"$m_{\mathrm{avg}}$  (↑ better)")
ax1.grid(True, alpha=0.4)

ax2 = ax1.twinx()
ln2, = ax2.plot(df["alpha"], df["d2_m"], linestyle="--", marker="x",
                color="tab:green", label=r"$m_{\mathrm{avg}}''$")
ax2.set_ylabel("2nd derivative of $m_{avg}$")
ax2.axhline(0, linestyle=":", linewidth=0.8)

# 合并图例
lines = [ln1, ln2]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc="upper left")

plt.tight_layout()
for ext in ("pdf", "eps"):
    fig.savefig(out_dir / f"alpha_vs_median.{ext}", bbox_inches="tight")
plt.close(fig)

# ---------- 4. α vs v_max ----------
fig, ax1 = plt.subplots(figsize=(6, 4))
ln1, = ax1.plot(df["alpha"], df["max_variance"], marker="o",
                color="tab:blue", label=r"$v_{\max}$")
ax1.set_xlabel(r"$\alpha$")
ax1.set_ylabel(r"$v_{\max}$  (↓ better)")
ax1.grid(True, alpha=0.4)

ax2 = ax1.twinx()
ln2, = ax2.plot(df["alpha"], -df["d2_v"], linestyle="--", marker="x",
                color="tab:green", label=r"$-\;v_{\max}''$")
ax2.set_ylabel(r"$-\,$2nd derivative of $v_{\max}$")
ax2.axhline(0, linestyle=":", linewidth=0.8)

lines = [ln1, ln2]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc="upper right")

plt.tight_layout()
for ext in ("pdf", "eps"):
    fig.savefig(out_dir / f"alpha_vs_variance.{ext}", bbox_inches="tight")
plt.close(fig)

print("✓ 图像已保存到 outputs/: alpha_vs_median.*  alpha_vs_variance.*")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_alpha_metrics_vldb.py
Generate VLDB-ready α-grid plots (English + Chinese) as vector PDFs.

Input:
  outputs/alpha_metrics.csv
    required columns: alpha, median_avg, max_variance

Output (4 files):
  outputs/alpha_vs_median_en.pdf
  outputs/alpha_vs_variance_en.pdf
  outputs/alpha_vs_median_zh.pdf
  outputs/alpha_vs_variance_zh.pdf

Updates (per latest request):
1) Remove α* label/arrow. Add a vertical dashed line at α=0.47 and a horizontal dashed line at y=0
   (both in light green) in BOTH plots.
2) Two curves use different colors (blue/red).
3) Remove internal grid.
4) Left y-axis of median plot shown in 10^-3 scale (values ×1000), annotate ×10^-3 outside top-left.
5) Left y-axis of variance plot shown in 10^-4 scale (values ×10000), annotate ×10^-4 outside top-left.
6) Variance plot right y-axis uses the unscaled second-derivative curve (no extra scaling).

Notes:
- The script will try to pick an available CJK font automatically for the Chinese version.
  If your environment has no Chinese font, set env ZH_FONT_PATH to a .ttf/.ttc file path.
"""

from __future__ import annotations

import os
import pathlib
from typing import Optional, List

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm


# ------------------------- Font helpers -------------------------
def _register_font_from_path(font_path: Optional[str]) -> Optional[str]:
    """Register a font file and return its family name if successful."""
    if not font_path:
        return None
    p = pathlib.Path(font_path)
    if not p.is_file():
        return None
    try:
        fm.fontManager.addfont(str(p))
        prop = fm.FontProperties(fname=str(p))
        return prop.get_name()
    except Exception:
        return None


def _pick_installed_font(candidates: List[str]) -> Optional[str]:
    """Pick the first candidate that exists in the current Matplotlib font list."""
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name
    return None


def _configure_matplotlib(lang: str, zh_font_path: Optional[str] = None) -> None:
    """Set global rcParams for a VLDB-style figure."""
    matplotlib.rcParams.update({
        # Vector PDF with embedded TrueType
        "pdf.fonttype": 42,
        "ps.fonttype": 42,

        # Sizes (tuned for single-column figure width)
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,

        # Lines/markers
        "lines.linewidth": 1.2,
        "lines.markersize": 3.5,

        # Minus sign
        "axes.unicode_minus": False,
    })

    if lang == "en":
        matplotlib.rcParams["font.family"] = "serif"
        matplotlib.rcParams["font.serif"] = [
            "Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"
        ]
        matplotlib.rcParams["mathtext.fontset"] = "stix"
        return

    # Chinese (best-effort)
    zh_name = _register_font_from_path(zh_font_path) if zh_font_path else None
    if zh_name is None:
        zh_name = _pick_installed_font([
            "SimSun", "Songti SC", "STSong",
            "Noto Serif CJK SC", "Noto Sans CJK SC",
            "Source Han Serif SC", "Source Han Sans SC",
            "Microsoft YaHei", "PingFang SC"
        ])

    if zh_name is not None:
        matplotlib.rcParams["font.family"] = zh_name
        matplotlib.rcParams["mathtext.fontset"] = "stix"
    else:
        matplotlib.rcParams["font.family"] = "DejaVu Sans"
        matplotlib.rcParams["mathtext.fontset"] = "stix"
        print("[WARN] No Chinese font found. Chinese PDFs may show missing glyphs. "
              "Set env ZH_FONT_PATH to a valid .ttf/.ttc path to fix it.")


# ------------------------- Plot helpers -------------------------
def _add_reference_lines(ax1, ax2, alpha_star: Optional[float],
                         color: str = "lightgreen") -> None:
    """Add a vertical line at alpha_star and a horizontal line at y=0 (on the right axis)."""
    if alpha_star is None:
        return
    for ax in (ax1, ax2):
        ax.axvline(alpha_star, linestyle="--", linewidth=0.8,
                   color=color, alpha=0.9, zorder=0)
    ax2.axhline(0.0, linestyle="--", linewidth=0.8,
                color=color, alpha=0.9, zorder=0)


def _plot_alpha_vs_median(df: pd.DataFrame, out_path: pathlib.Path, lang: str,
                          alpha_star: Optional[float]) -> None:
    # Left axis scaling: show values ×1000, annotate ×10^-3 outside
    y_scale = 1e3
    scale_txt = r"$\times 10^{-3}$"

    if lang == "en":
        labels = {
            "title": r"Weight selection ($\alpha^\star = 0.47$)",
            "y_left": r"Median of best score $m_{\mathrm{avg}}$",
            "y_right": r"Second derivative of $m_{\mathrm{avg}}$",
            "legend_left": r"$m_{\mathrm{avg}}$",
            "legend_right": r"$m_{\mathrm{avg}}''$",
        }
    else:
        labels = {
            "title": r"权重选择（$\alpha^\star = 0.47$）",
            "y_left": r"最优得分中位数的平均值 $m_{\mathrm{avg}}$",
            "y_right": r"$m_{\mathrm{avg}}$ 的二阶导数",
            "legend_left": r"$m_{\mathrm{avg}}$",
            "legend_right": r"$m_{\mathrm{avg}}''$",
        }

    fig, ax1 = plt.subplots(figsize=(3.6, 2.4))

    y_left = df["median_avg"] * y_scale
    ln1, = ax1.plot(
        df["alpha"], y_left,
        marker="o", linestyle="-",
        color="tab:blue",
        label=labels["legend_left"]
    )
    ax1.set_xlabel(r"$\alpha$")
    ax1.set_ylabel(labels["y_left"])
    ax1.set_title(labels["title"])
    ax1.grid(False)  # remove grid

    # annotate scaling factor outside top-left
    ax1.text(
        0.00, 1.02, scale_txt,
        transform=ax1.transAxes,
        ha="left", va="bottom",
        clip_on=False
    )

    ax2 = ax1.twinx()
    ln2, = ax2.plot(
        df["alpha"], df["d2_m"],
        marker="x", linestyle="--",
        color="tab:red",
        label=labels["legend_right"]
    )
    ax2.set_ylabel(labels["y_right"])

    _add_reference_lines(ax1, ax2, alpha_star, color="gray")

    ax1.legend([ln1, ln2], [ln1.get_label(), ln2.get_label()],
               loc="upper left", frameon=False)

    fig.tight_layout(pad=0.2)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_alpha_vs_variance(df: pd.DataFrame, out_path: pathlib.Path, lang: str,
                            alpha_star: Optional[float],
                            scale: float = 1e4) -> None:
    # Left axis scaling: show values ×10000, annotate ×10^-4 outside
    scale_txt = r"$\times 10^{-4}$"

    if lang == "en":
        labels = {
            "title": r"Weight selection ($\alpha^\star = 0.47$)",
            "y_left": r"Max variance $v_{\max}$",
            "y_right": r"Second derivative of $v_{\max}$",
            "legend_left": r"$v_{\max}$",
            "legend_right": r"$-\,v_{\max}''$",
        }
    else:
        labels = {
            "title": r"权重选择（$\alpha^\star = 0.47$）",
            "y_left": r"最大方差 $v_{\max}$",
            "y_right": r"$-\;v_{\max}$ 的二阶导数",
            "legend_left": r"$v_{\max}$",
            "legend_right": r"$-\;v_{\max}''$",
        }

    fig, ax1 = plt.subplots(figsize=(3.6, 2.4))

    y_left = df["max_variance"] * scale
    ln1, = ax1.plot(
        df["alpha"], y_left,
        marker="o", linestyle="-",
        color="tab:blue",
        label=labels["legend_left"]
    )
    ax1.set_xlabel(r"$\alpha$")
    ax1.set_ylabel(labels["y_left"])
    ax1.set_title(labels["title"])
    ax1.grid(False)  # remove grid

    # annotate scaling factor outside top-left
    ax1.text(
        0.00, 1.02, scale_txt,
        transform=ax1.transAxes,
        ha="left", va="bottom",
        clip_on=False
    )

    ax2 = ax1.twinx()
    # NOTE: right axis is NOT scaled
    ln2, = ax2.plot(
        df["alpha"], (-df["d2_v"]),
        marker="x", linestyle="--",
        color="tab:red",
        label=labels["legend_right"]
    )
    ax2.set_ylabel(labels["y_right"])

    _add_reference_lines(ax1, ax2, alpha_star, color="gray")

    ax1.legend([ln1, ln2], [ln1.get_label(), ln2.get_label()],
               loc="upper right", frameon=False)

    fig.tight_layout(pad=0.2)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ------------------------- Main -------------------------
def main() -> None:
    in_csv = pathlib.Path("outputs") / "alpha_metrics.csv"
    if not in_csv.exists():
        raise SystemExit(f"Cannot find input CSV: {in_csv.resolve()}")

    df = pd.read_csv(in_csv).sort_values("alpha").reset_index(drop=True)

    required = {"alpha", "median_avg", "max_variance"}
    if not required.issubset(df.columns):
        raise SystemExit(f"alpha_metrics.csv must contain columns: {sorted(required)}")

    # Derivatives for visualization
    df["d1_m"] = np.gradient(df["median_avg"], df["alpha"])
    df["d2_m"] = np.gradient(df["d1_m"], df["alpha"])
    df["d1_v"] = np.gradient(df["max_variance"], df["alpha"])
    df["d2_v"] = np.gradient(df["d1_v"], df["alpha"])

    out_dir = pathlib.Path("outputs")
    out_dir.mkdir(exist_ok=True)

    # Chosen alpha (for reference lines only)
    alpha_star = 0.47

    # Optional: provide a Chinese font file path via env (useful on Linux servers)
    zh_font_path = os.environ.get("ZH_FONT_PATH", None)

    for lang in ("en", "zh"):
        _configure_matplotlib(lang, zh_font_path=zh_font_path)

        _plot_alpha_vs_median(
            df,
            out_dir / f"alpha_vs_median_{lang}.pdf",
            lang=lang,
            alpha_star=alpha_star
        )
        _plot_alpha_vs_variance(
            df,
            out_dir / f"alpha_vs_variance_{lang}.pdf",
            lang=lang,
            alpha_star=alpha_star,
            scale=1e4  # 10^-4 scaling on left axis only
        )

    print("Done. PDFs saved to:", out_dir.resolve())


if __name__ == "__main__":
    main()

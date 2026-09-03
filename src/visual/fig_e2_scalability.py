# -*- coding: utf-8 -*-
"""Fig 4: E2 scalability. Energy / latency vs K with log y-axis. English only."""

import os, sys, numpy as np, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE); ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)
sys.path.insert(0, "/root/.zcode/skills/nature-figure/scripts")
from audit_panel_alignment import require_matplotlib_panel_alignment
from config import RESULTS_DIR
from matplotlib.ticker import FuncFormatter

def _plain_log(v, pos=None):
    return ('%g' % v) if v >= 0.1 else ('%.2f' % v)
from visual.palette import set_paper_style, method_color, method_marker, save_figure

set_paper_style()
OUT_DIR = os.path.join(ROOT, "latex", "figure")
os.makedirs(OUT_DIR, exist_ok=True)
INPUT_NPZ = os.path.join(RESULTS_DIR, "e2_scalability.npz")
PDF_DPI = TIFF_DPI = 1200

METHODS = ['Greedy','AllLocal','AllEdge','Random',
           'SAC','DDPG','HMA-Distill','HMA-Hybrid']

# Baselines share gray family, HMA pops:
import visual.palette as P

# P1-8: same-figure markers must be unique — palette defaults collide for
# SAC/HMA-Distill ('o') and Random/HMA-Hybrid ('D'), so override here
E2_MARKER_OVERRIDE = {'SAC': 'X', 'HMA-Hybrid': '*'}


def _marker(m):
    return E2_MARKER_OVERRIDE.get(m, P.method_marker(m))


def main():
    if not os.path.exists(INPUT_NPZ):
        fig, ax = plt.subplots(figsize=(7.16, 3.0))
        ax.text(0.5, 0.5, "TBD - waiting for E2 data",
                ha='center', va='center', fontsize=17)
        ax.axis('off')
        save_figure(fig, "fig_e2_scalability", OUT_DIR)
        return

    npz = np.load(INPUT_NPZ, allow_pickle=False)
    pairs = {}
    for k in npz.files:
        if k.endswith("__mean"):
            base, metric, _ = k.rsplit("__", 2)
            Kstr, m = base.split("__", 1)
            K = int(Kstr[1:])
            pairs.setdefault(K, {}).setdefault(m, {})[metric] = float(npz[k][0])

    Ks = sorted(pairs.keys())
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.0))
    title_labs = ['(a) Energy (kJ)', '(b) Latency (s)']
    for ax, metric, ylab, tlab in zip(axes, ('energy','latency'),
                                       ('Energy (kJ)','Latency (s)'),
                                       title_labs):
        for m in METHODS:
            if m in pairs.get(Ks[0], {}):
                ys = [pairs[K][m].get(metric, np.nan) for K in Ks]
                ax.plot(Ks, ys, marker=_marker(m), ms=4,
                        color=method_color(m), label=m, linewidth=1.3,
                        markeredgewidth=0.6, markeredgecolor='black')
        ax.set_xlabel('Number of users $K$', fontsize=14)
        ax.set_ylabel(ylab, fontsize=14)
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(FuncFormatter(_plain_log))
        ax.tick_params(labelsize=13)
        ax.grid(True, alpha=0.3)
        ax.set_title(tlab, pad=10, fontsize=15, weight='bold')
    # 显式刻度: 保证 log 轴标签为明文数字 (避免 offset 形式 "4 x 10^-1")
    axes[0].set_ylim(0.01, 100)
    axes[0].set_yticks([0.01, 0.1, 1, 10, 100])
    axes[1].set_ylim(0.1, 10)
    axes[1].set_yticks([0.1, 0.2, 0.5, 1, 2, 5, 10])
    # 图例移出绘图区: 图内图例 (ncol=2, lower right) 宽于 (b) 轴宽, 会左溢出
    # 压住 y 刻度标签 '0.2'/'1' 与旋转 y 轴标签 'Latency (s)', 且数据线/网格线
    # 从图例文字下穿过 (碰撞审计 text-stroke FAIL)。改放两子图下方空白居中。
    plt.tight_layout()
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center',
               bbox_to_anchor=(0.5, -0.06), ncol=3, fontsize=11.5,
               frameon=False, columnspacing=1.4, handlelength=1.6,
               handletextpad=0.6, borderaxespad=0.1)
    require_matplotlib_panel_alignment(
        fig,
        json_out=os.path.join(OUT_DIR, "fig_e2_scalability.alignment.json"),
        overlay_svg=os.path.join(OUT_DIR, "fig_e2_scalability.alignment.svg"),
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True,
    )
    from audit_panel_alignment import matplotlib_layout_manifest
    manifest = matplotlib_layout_manifest(fig)
    with open(os.path.join(OUT_DIR, "fig_e2_scalability.layout.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    save_figure(fig, "fig_e2_scalability", OUT_DIR)
    print("  saved: fig_e2_scalability.pdf/.tiff")


if __name__ == "__main__":
    main()
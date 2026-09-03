# -*- coding: utf-8 -*-
"""Fig 8: E6 robustness. Latency/SLA time series before/after perturbation."""

import os, sys, numpy as np, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE); ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)
sys.path.insert(0, "/root/.zcode/skills/nature-figure/scripts")
from audit_panel_alignment import require_matplotlib_panel_alignment
from config import RESULTS_DIR
from visual.palette import set_paper_style, perturb_color, PALETTE, save_figure

set_paper_style()
OUT_DIR = os.path.join(ROOT, "latex", "figure")
os.makedirs(OUT_DIR, exist_ok=True)
INPUT_NPZ = os.path.join(RESULTS_DIR, "e6_robust.npz")
PDF_DPI = TIFF_DPI = 1200
PERTURB_STEP = 100


def main():
    if not os.path.exists(INPUT_NPZ):
        fig, ax = plt.subplots(figsize=(7.16, 3.0))
        ax.text(0.5, 0.5, "TBD - waiting for E6 data",
                ha='center', va='center', fontsize=17)
        ax.axis('off')
        save_figure(fig, "fig_e6_robust", OUT_DIR)
        return

    npz = np.load(INPUT_NPZ, allow_pickle=False)
    ptypes = sorted({k.split("__")[0] for k in npz.files})
    # P1-8: distinct marker per perturbation type (color alone is not
    # sufficient); markevery thins markers so they do not occlude the curve
    MK = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']
    pmk = {pt: MK[i % len(MK)] for i, pt in enumerate(ptypes)}
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.2))
    title_labs = ['(a) Latency (s)', '(b) High-priority SLA (%)']
    for pt in ptypes:
        lk, sk = f"{pt}__latency", f"{pt}__sla"
        c = perturb_color(pt)
        if lk in npz.files:
            arr = npz[lk]
            axes[0].plot(np.arange(len(arr)), arr, label=pt, color=c,
                          linewidth=1.3, marker=pmk[pt], ms=3,
                          markevery=max(1, len(arr) // 40),
                          markeredgewidth=0.3, markeredgecolor='black')
        if sk in npz.files:
            arr = npz[sk] * 100
            axes[1].plot(np.arange(len(arr)), arr, label=pt, color=c,
                          linewidth=1.3, marker=pmk[pt], ms=3,
                          markevery=max(1, len(arr) // 40),
                          markeredgewidth=0.3, markeredgecolor='black')
    for ax, ylab, tlab in zip(axes, ('Latency (s)', 'High-priority SLA (%)'),
                                title_labs):
        ax.axvline(PERTURB_STEP, color=PALETTE['sac'], linestyle=':',
                    linewidth=1.0, label='perturbation')
        ax.set_xlabel('Step', fontsize=14)
        ax.set_ylabel(ylab, fontsize=14)
        ax.tick_params(labelsize=13)
        ax.grid(True, alpha=0.3)
        ax.set_title(tlab, pad=10, fontsize=15, weight='bold')
    # 图例移出绘图区: loc='best' 的图内图例压住曲线与网格线, 线从文字下穿过
    # (碰撞审计 text-stroke FAIL)。改放两子图下方空白居中 (含 perturbation
    # 参考虚线条目; 去重同一 label 的重复 artist)。
    plt.tight_layout()
    hmap = {}
    for h, l in zip(*axes[0].get_legend_handles_labels()):
        hmap.setdefault(l, h)          # 保留先出现的 artist (曲线在前, 虚线在后)
    fig.legend(list(hmap.values()), list(hmap.keys()),
               loc='upper center', bbox_to_anchor=(0.5, -0.06),
               ncol=3, fontsize=11.5, frameon=False, columnspacing=1.4,
               handlelength=1.8, handletextpad=0.6, borderaxespad=0.1)
    require_matplotlib_panel_alignment(
        fig,
        json_out=os.path.join(OUT_DIR, "fig_e6_robust.alignment.json"),
        overlay_svg=os.path.join(OUT_DIR, "fig_e6_robust.alignment.svg"),
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True,
    )
    from audit_panel_alignment import matplotlib_layout_manifest
    manifest = matplotlib_layout_manifest(fig)
    with open(os.path.join(OUT_DIR, "fig_e6_robust.layout.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    save_figure(fig, "fig_e6_robust", OUT_DIR)
    print("  saved: fig_e6_robust.pdf/.tiff")


if __name__ == "__main__":
    main()
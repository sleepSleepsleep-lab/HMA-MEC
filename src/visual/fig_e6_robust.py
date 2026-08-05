# -*- coding: utf-8 -*-
"""Fig 8: E6 robustness. Latency/SLA time series before/after perturbation."""

import os, sys, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE); ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)
from config import RESULTS_DIR
from visual.palette import set_paper_style, perturb_color, PALETTE

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
                ha='center', va='center', fontsize=11)
        ax.axis('off')
        fig.savefig(os.path.join(OUT_DIR, "fig_e6_robust.pdf"),
                    format="pdf", dpi=PDF_DPI, bbox_inches='tight')
        return

    npz = np.load(INPUT_NPZ, allow_pickle=False)
    ptypes = sorted({k.split("__")[0] for k in npz.files})
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.2))
    title_labs = ['(a) Latency (s)', '(b) High-priority SLA (%)']
    for pt in ptypes:
        lk, sk = f"{pt}__latency", f"{pt}__sla"
        c = perturb_color(pt)
        if lk in npz.files:
            arr = npz[lk]
            axes[0].plot(np.arange(len(arr)), arr, label=pt, color=c,
                          linewidth=1.3, marker='o', ms=2,
                          markeredgewidth=0.3, markeredgecolor='black')
        if sk in npz.files:
            arr = npz[sk] * 100
            axes[1].plot(np.arange(len(arr)), arr, label=pt, color=c,
                          linewidth=1.3, marker='o', ms=2,
                          markeredgewidth=0.3, markeredgecolor='black')
    for ax, ylab, tlab in zip(axes, ('Latency (s)', 'High-priority SLA (%)'),
                                title_labs):
        ax.axvline(PERTURB_STEP, color=PALETTE['sac'], linestyle=':',
                    linewidth=1.0, label='perturbation')
        ax.set_xlabel('Step', fontsize=8)
        ax.set_ylabel(ylab, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_title(tlab, pad=10, fontsize=9, weight='bold')
    axes[0].legend(fontsize=7, frameon=True, framealpha=0.85, loc='best')

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_e6_robust.pdf"),
                format="pdf", dpi=PDF_DPI, bbox_inches='tight')
    fig.savefig(os.path.join(OUT_DIR, "fig_e6_robust.tiff"),
                format="tiff", dpi=TIFF_DPI, bbox_inches='tight',
                pil_kwargs={"compression": "tiff_lzw"})
    print("  saved: fig_e6_robust.pdf/.tiff")


if __name__ == "__main__":
    main()
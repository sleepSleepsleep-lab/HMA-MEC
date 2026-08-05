# -*- coding: utf-8 -*-
"""Fig 4: E2 scalability. Energy / latency vs K with log y-axis. English only."""

import os, sys, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE); ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)
from config import RESULTS_DIR
from visual.palette import set_paper_style, method_color, method_marker

set_paper_style()
OUT_DIR = os.path.join(ROOT, "latex", "figure")
os.makedirs(OUT_DIR, exist_ok=True)
INPUT_NPZ = os.path.join(RESULTS_DIR, "e2_scalability.npz")
PDF_DPI = TIFF_DPI = 1200

METHODS = ['Greedy','AllLocal','AllEdge','Random',
           'SAC','DDPG','HMA-Distill','HMA-Hybrid']

# Baselines share gray family, HMA pops:
import visual.palette as P


def main():
    if not os.path.exists(INPUT_NPZ):
        fig, ax = plt.subplots(figsize=(7.16, 3.0))
        ax.text(0.5, 0.5, "TBD - waiting for E2 data",
                ha='center', va='center', fontsize=11)
        ax.axis('off')
        fig.savefig(os.path.join(OUT_DIR, "fig_e2_scalability.pdf"),
                    format="pdf", dpi=PDF_DPI, bbox_inches='tight')
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
                ax.plot(Ks, ys, marker=method_marker(m), ms=4,
                        color=method_color(m), label=m, linewidth=1.3,
                        markeredgewidth=0.6, markeredgecolor='black')
        ax.set_xlabel('Number of users $K$', fontsize=8)
        ax.set_ylabel(ylab, fontsize=8)
        ax.set_yscale('log')
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_title(tlab, pad=10, fontsize=9, weight='bold')
    axes[0].legend(fontsize=7, loc='best', frameon=True, framealpha=0.85)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_e2_scalability.pdf"),
                format="pdf", dpi=PDF_DPI, bbox_inches='tight')
    fig.savefig(os.path.join(OUT_DIR, "fig_e2_scalability.tiff"),
                format="tiff", dpi=TIFF_DPI, bbox_inches='tight',
                pil_kwargs={"compression": "tiff_lzw"})
    print("  saved: fig_e2_scalability.pdf/.tiff")


if __name__ == "__main__":
    main()
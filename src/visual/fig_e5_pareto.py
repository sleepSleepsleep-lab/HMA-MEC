# -*- coding: utf-8 -*-
"""Fig 7: E5 Pareto front. English only."""

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
INPUT_NPZ = os.path.join(RESULTS_DIR, "e5_pareto.npz")
PDF_DPI = TIFF_DPI = 1200


def main():
    if not os.path.exists(INPUT_NPZ):
        fig, ax = plt.subplots(figsize=(7.16, 3.0))
        ax.text(0.5, 0.5, "TBD - waiting for E5 data",
                ha='center', va='center', fontsize=11)
        ax.axis('off')
        fig.savefig(os.path.join(OUT_DIR, "fig_e5_pareto.pdf"),
                    format="pdf", dpi=PDF_DPI, bbox_inches='tight')
        return

    npz = np.load(INPUT_NPZ, allow_pickle=False)
    fig, ax = plt.subplots(figsize=(7.16, 3.4))
    for m in ('HMA','SAC','DDPG'):
        e_key, t_key = f"{m}__energy", f"{m}__latency"
        if e_key in npz.files and t_key in npz.files:
            e, t = npz[e_key], npz[t_key]
            order = np.argsort(e)
            ax.plot(e[order], t[order], '-',
                    marker=method_marker(m), ms=4,
                    color=method_color(m), label=m,
                    linewidth=1.4, markeredgewidth=0.6,
                    markeredgecolor='black')
    ax.set_xlabel(r'Energy $E_{\text{total}}$ (kJ)', fontsize=9)
    ax.set_ylabel(r'Latency $\bar{T}$ (s)', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, frameon=False)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_e5_pareto.pdf"),
                format="pdf", dpi=PDF_DPI, bbox_inches='tight')
    fig.savefig(os.path.join(OUT_DIR, "fig_e5_pareto.tiff"),
                format="tiff", dpi=TIFF_DPI, bbox_inches='tight',
                pil_kwargs={"compression": "tiff_lzw"})
    print("  saved: fig_e5_pareto.pdf/.tiff")


if __name__ == "__main__":
    main()
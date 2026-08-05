# -*- coding: utf-8 -*-
"""Fig 5: E3 ablation dual-bar chart. English only."""

import os, sys, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE); ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)
from config import RESULTS_DIR
from visual.palette import set_paper_style, PALETTE

set_paper_style()
OUT_DIR = os.path.join(ROOT, "latex", "figure")
os.makedirs(OUT_DIR, exist_ok=True)
INPUT_NPZ = os.path.join(RESULTS_DIR, "e3_ablation.npz")
PDF_DPI = TIFF_DPI = 1200

BASELINE_COLOR = PALETTE['hma_primary']
SECONDARY_COLOR = PALETTE['hma_hybrid']


def main():
    if not os.path.exists(INPUT_NPZ):
        fig, ax = plt.subplots(figsize=(7.16, 3.0))
        ax.text(0.5, 0.5, "TBD - waiting for E3 data",
                ha='center', va='center', fontsize=11)
        ax.axis('off')
        fig.savefig(os.path.join(OUT_DIR, "fig_e3_ablation.pdf"),
                    format="pdf", dpi=PDF_DPI, bbox_inches='tight')
        return

    npz = np.load(INPUT_NPZ, allow_pickle=False)
    variants = []
    for k in npz.files:
        if k.endswith("__energy__mean"):
            v = k.rsplit("__",2)[0]
            if v not in variants: variants.append(v)
    variants = sorted(variants)
    es = [float(npz[f"{v}__energy__mean"][0])
          if f"{v}__energy__mean" in npz.files else 0.0
          for v in variants]
    ss = [float(npz[f"{v}__success_rate__mean"][0]) * 100
          if f"{v}__success_rate__mean" in npz.files else 0.0
          for v in variants]

    # Highlight P_Full / Full variant with primary color, others muted
    full_idx = next((i for i, v in enumerate(variants)
                      if v in ('P_Full', 'P_Full')),
                    None)
    energy_colors = [PALETTE['hma_primary'] if i == full_idx
                      else PALETTE['neutral_med'] for i in range(len(variants))]
    succ_colors = [PALETTE['hma_hybrid'] if i == full_idx
                     else PALETTE['hma_fullllm'] for i in range(len(variants))]

    fig, ax = plt.subplots(figsize=(7.16, 3.2))
    x = np.arange(len(variants))
    width = 0.35
    ax.bar(x - width/2, es, width, label='Energy (kJ)',
           color=energy_colors, edgecolor='black', linewidth=0.4)
    ax2 = ax.twinx()
    ax2.bar(x + width/2, ss, width, label='Success rate (%)',
            color=succ_colors, edgecolor='black', linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=40, ha='right', fontsize=6.5)
    ax.set_ylabel('Energy (kJ)', fontsize=8)
    ax2.set_ylabel('Success rate (%)', fontsize=8)
    ax.tick_params(labelsize=7); ax2.tick_params(labelsize=7)
    ax.grid(True, axis='y', alpha=0.3)
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.02),
               ncol=2, fontsize=7, frameon=False)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_e3_ablation.pdf"),
                format="pdf", dpi=PDF_DPI, bbox_inches='tight')
    fig.savefig(os.path.join(OUT_DIR, "fig_e3_ablation.tiff"),
                format="tiff", dpi=TIFF_DPI, bbox_inches='tight',
                pil_kwargs={"compression": "tiff_lzw"})
    print("  saved: fig_e3_ablation.pdf/.tiff")


if __name__ == "__main__":
    main()
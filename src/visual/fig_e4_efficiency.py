# -*- coding: utf-8 -*-
"""Fig 6: E4 efficiency. Box plots of per-step latency + c_min histogram."""

import os, sys, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE); ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)
from config import RESULTS_DIR
from visual.palette import set_paper_style, PALETTE, method_color

set_paper_style()
OUT_DIR = os.path.join(ROOT, "latex", "figure")
os.makedirs(OUT_DIR, exist_ok=True)
INPUT_NPZ = os.path.join(RESULTS_DIR, "e4_efficiency.npz")
PDF_DPI = TIFF_DPI = 1200

# Use unified palette (HMA-Distill, HMA-Hybrid)
MODE_COLORS = {'Distill': method_color('HMA-Distill'),
                'Hybrid':   method_color('HMA-Hybrid'),
                'FullLLM':  method_color('HMA-FullLLM')}


def main():
    if not os.path.exists(INPUT_NPZ):
        fig, ax = plt.subplots(figsize=(7.16, 3.0))
        ax.text(0.5, 0.5, "TBD - waiting for E4 data",
                ha='center', va='center', fontsize=11)
        ax.axis('off')
        for ext in ('.pdf', '.tiff'):
            fig.savefig(os.path.join(OUT_DIR, f"fig_e4_efficiency{ext}"),
                        format=ext.strip('.'), dpi=PDF_DPI,
                        bbox_inches='tight',
                        pil_kwargs={"compression": "tiff_lzw"} if ext=='.tiff' else {})
        plt.close(fig); print("  placeholder saved"); return

    npz = np.load(INPUT_NPZ, allow_pickle=False)
    modes = sorted({k.split("__")[0] for k in npz.files
                     if k.endswith("__latencies_ms")})

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.0))
    # panel (a): per-step latency box plot
    data = [npz[f"{m}__latencies_ms"] for m in modes
             if f"{m}__latencies_ms" in npz.files]
    bp = axes[0].boxplot(data, sym='.', showmeans=True, patch_artist=True,
                          widths=0.55)
    for patch, m in zip(bp['boxes'], modes):
        patch.set_facecolor(MODE_COLORS.get(m, PALETTE['neutral_med']))
        patch.set_alpha(0.7)
        patch.set_edgecolor('black')
    for median_line in bp['medians']:
        median_line.set_color('black'); median_line.set_linewidth(1.2)
    for mean_point in bp['means']:
        mean_point.set_marker('D'); mean_point.set_markerfacecolor('red')
        mean_point.set_markersize(3)
    axes[0].set_xticklabels(modes, fontsize=7)
    axes[0].set_ylabel('Per-step latency (ms)', fontsize=8)
    axes[0].set_yscale('log')
    axes[0].grid(True, axis='y', alpha=0.3)
    axes[0].tick_params(labelsize=7)
    axes[0].set_title('(a) Latency distribution (ms)', pad=10,
                          fontsize=9, weight='bold')

    # panel (b): c_min histogram
    for m in modes:
        key = f"{m}__conf_min"
        if key in npz.files:
            arr = npz[key]
            axes[1].hist(arr, bins=30, alpha=0.55,
                         color=MODE_COLORS.get(m, PALETTE['neutral_med']),
                         label=m, edgecolor='black', linewidth=0.3)
    axes[1].axvline(0.3, color=PALETTE['sac'], linestyle='--',
                     linewidth=0.8, label=r'$\tau_{\text{low}}=0.3$')
    axes[1].set_xlabel(r'Confidence $c_{\min}$', fontsize=8)
    axes[1].set_ylabel('Count', fontsize=8)
    axes[1].legend(fontsize=7, frameon=False)
    axes[1].tick_params(labelsize=7)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title(r'(b) Confidence $c_{\min}$', pad=10,
                          fontsize=9, weight='bold')

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_e4_efficiency.pdf"),
                format="pdf", dpi=PDF_DPI, bbox_inches='tight')
    fig.savefig(os.path.join(OUT_DIR, "fig_e4_efficiency.tiff"),
                format="tiff", dpi=TIFF_DPI, bbox_inches='tight',
                pil_kwargs={"compression": "tiff_lzw"})
    print("  saved: fig_e4_efficiency.pdf/.tiff")


if __name__ == "__main__":
    main()
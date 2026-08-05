# -*- coding: utf-8 -*-
"""Fig 9: E7 hypersensitivity sweeps. 4-panel grid (tau_c, beta, eps_c, delta_v)."""

import os, sys, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE); ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)
from config import RESULTS_DIR
from visual.palette import (set_paper_style, PALETTE,
                              method_color)

set_paper_style()
OUT_DIR = os.path.join(ROOT, "latex", "figure")
os.makedirs(OUT_DIR, exist_ok=True)
INPUT_NPZ = os.path.join(RESULTS_DIR, "e7_sensitivity.npz")
PDF_DPI = TIFF_DPI = 1200

PARAM_META = {
    "tau_c":   dict(label=r"Confidence threshold $\tau_c$",
                   xlabel=r"$\tau_c$",
                   ref=0.6, ref_name="default"),
    "beta":    dict(label=r"Confidence sensitivity $\beta$",
                   xlabel=r"$\beta$",
                   ref=2, ref_name="default"),
    "eps_c":   dict(label=r"Consensus tolerance $\epsilon_c$",
                   xlabel=r"$\epsilon_c$",
                   ref=0.05, ref_name="default"),
    "delta_v": dict(label=r"Verifier tolerance $\delta_v$",
                   xlabel=r"$\delta_v$",
                   ref=0.15, ref_name="default"),
}

METRIC_META = [
    ("energy",  "Energy (kJ)", False),
    ("latency", "Latency (s)", False),
    ("success", "Success rate", True),     # convert to %
    ("sla",     "High-pri SLA", True),
]


def main():
    if not os.path.exists(INPUT_NPZ):
        fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.0))
        for ax in axes.ravel():
            ax.text(0.5, 0.5, "TBD - waiting for E7 data",
                    ha='center', va='center', fontsize=11)
            ax.axis('off')
        fig.savefig(os.path.join(OUT_DIR, "fig_e7_sensitivity.pdf"),
                    format="pdf", dpi=PDF_DPI, bbox_inches='tight')
        fig.savefig(os.path.join(OUT_DIR, "fig_e7_sensitivity.tiff"),
                    format="tiff", dpi=TIFF_DPI, bbox_inches='tight',
                    pil_kwargs={"compression": "tiff_lzw"})
        plt.close(fig)
        print("  placeholder saved"); return

    npz = np.load(INPUT_NPZ, allow_pickle=False)
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.0))
    panel_titles = ['(a)', '(b)', '(c)', '(d)']

    for ax, (param, meta), tlab in zip(axes.ravel(), PARAM_META.items(),
                                         panel_titles):
        xs = npz[f"{param}__values"]
        # 四条曲线: 能耗/时延/成功率/SLA; 用双 y 轴
        ax1 = ax
        # 能耗 (主轴, 蓝色)
        e_arr = npz[f"{param}__energy"]
        l_arr = npz[f"{param}__latency"]
        s_arr = npz[f"{param}__success"] * 100.0
        sla_arr = npz[f"{param}__sla"] * 100.0

        # drop NaN to keep lines aligned
        line_e,   = ax1.plot(xs, e_arr, '-', marker='o', ms=3,
                              color=PALETTE['hma_primary'],
                              linewidth=1.5, label='Energy (kJ)')
        line_l,   = ax1.plot(xs, l_arr, '-', marker='s', ms=3,
                              color=PALETTE['hma_hybrid'],
                              linewidth=1.5, label='Latency (s)')
        ax1.set_xlabel(meta['xlabel'], fontsize=8)
        ax1.set_ylabel(r'Energy/latency', fontsize=8)
        ax1.tick_params(axis='y', labelsize=7)
        ax1.tick_params(axis='x', labelsize=7)
        ax1.grid(True, alpha=0.3)

        # secondary axis: success/sla (%)
        ax2 = ax1.twinx()
        line_s,   = ax2.plot(xs, s_arr, '--', marker='^', ms=3,
                              color=PALETTE['sac'],
                              linewidth=1.2, label='Success (%)')
        line_sla, = ax2.plot(xs, sla_arr, '--', marker='D', ms=3,
                              color=PALETTE['hma_fullllm'],
                              linewidth=1.2, label='SLA (%)')
        ax2.set_ylabel(r'Success / SLA (%)', fontsize=8)
        ax2.tick_params(axis='y', labelsize=7)
        ax2.grid(False)

        # vertical line at default value
        try:
            ax1.axvline(meta['ref'], color=PALETTE['neutral_med'],
                         linestyle=':', linewidth=0.8, alpha=0.85)
        except Exception:
            pass

        ax1.set_title(f"{tlab} {param}", pad=6, fontsize=9, weight='bold')

        # legend only in last panel
        if tlab == '(d)':
            lines = [line_e, line_l, line_s, line_sla]
            labels = [l.get_label() for l in lines]
            fig.legend(lines, labels, loc='upper center',
                        bbox_to_anchor=(0.5, 1.02), ncol=4,
                        fontsize=7, frameon=False)

    plt.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(os.path.join(OUT_DIR, "fig_e7_sensitivity.pdf"),
                format="pdf", dpi=PDF_DPI, bbox_inches='tight')
    fig.savefig(os.path.join(OUT_DIR, "fig_e7_sensitivity.tiff"),
                format="tiff", dpi=TIFF_DPI, bbox_inches='tight',
                pil_kwargs={"compression": "tiff_lzw"})
    print("  saved: fig_e7_sensitivity.pdf/.tiff")


if __name__ == "__main__":
    main()
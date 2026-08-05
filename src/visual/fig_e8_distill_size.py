# -*- coding: utf-8 -*-
"""Fig 10: E8 dataset size ablation. Dual-axis: metrics + training time."""

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
INPUT_NPZ = os.path.join(RESULTS_DIR, "e8_distill_size.npz")
PDF_DPI = TIFF_DPI = 1200


def main():
    if not os.path.exists(INPUT_NPZ):
        fig, ax = plt.subplots(figsize=(7.16, 3.2))
        ax.text(0.5, 0.5, "TBD - waiting for E8 data",
                ha='center', va='center', fontsize=11)
        ax.axis('off')
        fig.savefig(os.path.join(OUT_DIR, "fig_e8_distill_size.pdf"),
                    format="pdf", dpi=PDF_DPI, bbox_inches='tight')
        fig.savefig(os.path.join(OUT_DIR, "fig_e8_distill_size.tiff"),
                    format="tiff", dpi=TIFF_DPI, bbox_inches='tight',
                    pil_kwargs={"compression": "tiff_lzw"})
        plt.close(fig); print("  placeholder saved"); return

    npz = np.load(INPUT_NPZ, allow_pickle=False)
    sizes = npz['size']
    fig, ax1 = plt.subplots(figsize=(7.16, 3.4))
    # 主轴: 能耗+时延
    e = npz['energy']; l = npz['latency']
    s = npz['success'] * 100; sla = npz['sla'] * 100
    line_e, = ax1.plot(sizes, e, '-', marker='o', ms=4,
                       color=PALETTE['hma_primary'], linewidth=1.6,
                       label='Energy (kJ)')
    line_l, = ax1.plot(sizes, l, '-', marker='s', ms=4,
                       color=PALETTE['hma_hybrid'], linewidth=1.6,
                       label='Latency (s)')
    ax1.set_xlabel(r'Size of distillation dataset $|\mathcal{D}_\mathrm{debate}|$',
                     fontsize=9)
    ax1.set_ylabel(r'Energy (kJ) / Latency (s)', fontsize=9)
    ax1.tick_params(labelsize=8)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    line_s,   = ax2.plot(sizes, s,   '--', marker='^', ms=4,
                          color=PALETTE['sac'], linewidth=1.3,
                          label='Success (%)')
    line_sla, = ax2.plot(sizes, sla, '--', marker='D', ms=4,
                          color=PALETTE['hma_fullllm'], linewidth=1.3,
                          label='SLA (%)')
    ax2.set_ylabel('Success / SLA (%)', fontsize=9)
    ax2.tick_params(labelsize=8)
    ax2.grid(False)

    # 训练时间在第三 y 轴 (略) -- 用不同 marker 风格 + 文本注释
    try:
        dt = npz['train_dt']
    except KeyError:
        dt = None
    if dt is not None:
        # 把训练时间在主轴右上以文字注释呈现，避免拥挤
        for x, d in zip(sizes, dt):
            ax1.annotate(f"{d:.0f}s", xy=(x, max(e.max(), l.max()) * 1.05),
                           ha='center', fontsize=6.5,
                           color=PALETTE['neutral_dark'])

    ax1.set_title('Distillation dataset size ablation (E8)', pad=8,
                    fontsize=9, weight='bold')
    lines = [line_e, line_l, line_s, line_sla]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5, -0.3),
                ncol=4, fontsize=7, frameon=False)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_e8_distill_size.pdf"),
                format="pdf", dpi=PDF_DPI, bbox_inches='tight')
    fig.savefig(os.path.join(OUT_DIR, "fig_e8_distill_size.tiff"),
                format="tiff", dpi=TIFF_DPI, bbox_inches='tight',
                pil_kwargs={"compression": "tiff_lzw"})
    print("  saved: fig_e8_distill_size.pdf/.tiff")


if __name__ == "__main__":
    main()
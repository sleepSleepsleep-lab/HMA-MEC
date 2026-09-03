# -*- coding: utf-8 -*-
"""Fig 10: E8 dataset size ablation. Dual-axis: metrics + training time."""

import os, sys, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE); ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)
from config import RESULTS_DIR
from visual.palette import set_paper_style, PALETTE, save_figure

set_paper_style()
OUT_DIR = os.path.join(ROOT, "latex", "figure")
os.makedirs(OUT_DIR, exist_ok=True)
INPUT_NPZ = os.path.join(RESULTS_DIR, "e8_distill_size.npz")
PDF_DPI = TIFF_DPI = 1200


def main():
    if not os.path.exists(INPUT_NPZ):
        fig, ax = plt.subplots(figsize=(7.16, 3.2))
        ax.text(0.5, 0.5, "TBD - waiting for E8 data",
                ha='center', va='center', fontsize=17)
        ax.axis('off')
        save_figure(fig, "fig_e8_distill_size", OUT_DIR)
        save_figure(fig, "fig_e8_distill_size", OUT_DIR)
        plt.close(fig); print("  placeholder saved"); return

    npz = np.load(INPUT_NPZ, allow_pickle=False)
    sizes = npz['size']
    fig, ax1 = plt.subplots(figsize=(7.16, 3.4))
    # 主轴: 能耗+时延
    e = npz['energy']; l = npz['latency']
    s = npz['success'] * 100; sla = npz['sla'] * 100
    ax1.set_xscale('log')

    ax1.set_xticks([1000, 2000, 5000, 10000, 20000])
    ax1.set_xticklabels(['1000', '2000', '5000', '10000', '20000'])
    line_e, = ax1.plot(sizes, e, '-', marker='o', ms=4,
                       color=PALETTE['hma_primary'], linewidth=1.6,
                       label='Energy (kJ)')
    line_l, = ax1.plot(sizes, l, '-', marker='s', ms=4,
                       color=PALETTE['hma_hybrid'], linewidth=1.6,
                       label='Latency (s)')
    ax1.set_xlabel(r'Size of distillation dataset $|D_{\mathrm{debate}}|$',
                     fontsize=15)
    ax1.set_ylabel(r'Energy (kJ) / Latency (s)', fontsize=15)
    ax1.tick_params(labelsize=14)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    line_s,   = ax2.plot(sizes, s,   '--', marker='^', ms=4,
                          color=PALETTE['sac'], linewidth=1.3,
                          label='Success (%)')
    line_sla, = ax2.plot(sizes, sla, '--', marker='D', ms=4,
                          color=PALETTE['hma_fullllm'], linewidth=1.3,
                          label='SLA (%)')
    ax2.set_ylabel('Success / SLA (%)', fontsize=15, labelpad=24)
    ax2.tick_params(labelsize=14, pad=8, length=0)
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
                           ha='center', fontsize=12.5,
                           color=PALETTE['neutral_dark'])

    # 图内不设大标题: 标题由论文题注给出 (学术排版规范)
    # 图例移出绘图区: 原先 ax1 图例 (center left, anchor 1.08) 溢出右侧,
    # 其句柄线横穿 ax2 旋转 y 轴标签 'Success / SLA (%)' 文字带 (碰撞审计
    # text-stroke FAIL)。改为 figure 级图例, 置于绘图区下方空白居中。
    lines = [line_e, line_l, line_s, line_sla]
    labels = [l.get_label() for l in lines]

    plt.tight_layout()
    fig.legend(lines, labels, loc='upper center',
               bbox_to_anchor=(0.5, -0.06), ncol=2, fontsize=12,
               frameon=False, columnspacing=2.0, handlelength=1.8,
               handletextpad=0.6, borderaxespad=0.1)
    save_figure(fig, "fig_e8_distill_size", OUT_DIR)
    save_figure(fig, "fig_e8_distill_size", OUT_DIR)
    print("  saved: fig_e8_distill_size.pdf/.tiff")


if __name__ == "__main__":
    main()
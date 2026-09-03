# -*- coding: utf-8 -*-
"""Fig 5: E3 component ablation — 4 variants.
Data: results/e3_component_ablation.json
  HMA-Full        : distilled policy + plan-refiner (full method)
  HMA-NoRefiner   : distilled policy only (no verifier-driven refinement)
  MPC             : heuristic seed + same refiner (no LLM prior)
  HMA-RandomSeed  : random seed + same refiner (prior quality lower bound)
English labels only.
"""

import os, sys, json
import numpy as np
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
INPUT_JSON = os.path.join(RESULTS_DIR, "e3_component_ablation.json")
PDF_DPI = TIFF_DPI = 1200

ORDER = ['HMA-Full', 'HMA-NoRefiner', 'MPC', 'HMA-RandomSeed']
LABELS = ['HMA-Full', 'NoRefiner', 'MPC', 'RandSeed']


def main():
    if not os.path.exists(INPUT_JSON):
        fig, ax = plt.subplots(figsize=(7.16, 3.0))
        ax.text(0.5, 0.5, "TBD - waiting for E3 data",
                ha='center', va='center', fontsize=17)
        ax.axis('off')
        save_figure(fig, "fig_e3_ablation", OUT_DIR)
        return

    d = json.load(open(INPUT_JSON, encoding='utf-8'))
    nms = [v for v in ORDER if v in d]
    ts = [d[v]['mean']['latency'] for v in nms]
    ss = [d[v]['mean']['success_rate'] * 100 for v in nms]
    es = [d[v]['mean']['energy'] for v in nms]
    sl = [d[v]['mean']['priority_sla'] * 100 for v in nms]
    # P1-8: error bars from json std (npz e3_ablation.npz uses different
    # variant names P_Full/P-NoA2-*, so std is taken from the json)
    ts_e = [d[v]['std']['latency'] for v in nms]
    ss_e = [d[v]['std']['success_rate'] * 100 for v in nms]
    full_idx = nms.index('HMA-Full') if 'HMA-Full' in nms else 0

    palette_e = [PALETTE['hma_primary'] if i == full_idx else PALETTE['neutral_med']
                 for i in range(len(nms))]
    palette_s = [PALETTE['hma_hybrid'] if i == full_idx else PALETTE['hma_fullllm']
                 for i in range(len(nms))]

    fig, ax = plt.subplots(figsize=(7.16, 3.4))
    x = np.arange(len(nms))
    width = 0.38
    ax.bar(x - width / 2, ts, width, label='Latency (s)',
           color=palette_e, edgecolor='black', linewidth=0.4,
           yerr=ts_e, capsize=2.0,
           error_kw={'elinewidth': 0.6, 'ecolor': '#3A3F45'})
    ax.set_ylabel('Latency (s)', fontsize=14, labelpad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS[:len(nms)], fontsize=12.5)
    ax.tick_params(labelsize=13, pad=8)
    ax.grid(True, axis='y', alpha=0.3)
    # 柱顶数值在正文表 3, 双柱组内标签会互相重叠, 不再逐柱标注

    ax2 = ax.twinx()
    ax2.bar(x + width / 2, ss, width, label='Success (%)',
            color=palette_s, edgecolor='black', linewidth=0.4,
            yerr=ss_e, capsize=2.0,
            error_kw={'elinewidth': 0.6, 'ecolor': '#3A3F45'})
    ax2.set_ylabel('Success rate (%)', fontsize=14, labelpad=14)
    ax2.set_ylim(0, 100)
    ax2.tick_params(labelsize=13, pad=8)

    # 纵轴拉长: 把顶部 E/SLA 注释与数值标签完全囊括进子图框线内
    ax.set_ylim(0, max(ts) * 1.28)
    ax2.set_ylim(0, max(ss) * 1.18)
    # E/SLA 数值在正文表 3, 图内不重复注释 (避免 4 组横向重叠)

    # 左右双轴标签已直接标识两系列 (Latency/Success), 图例冗余, 按
    # 直接标签优先原则省略
    plt.tight_layout()
    save_figure(fig, "fig_e3_ablation", OUT_DIR)
    print(f"  saved: fig_e3_ablation.pdf/.tiff ({len(nms)} variants)")


if __name__ == "__main__":
    main()

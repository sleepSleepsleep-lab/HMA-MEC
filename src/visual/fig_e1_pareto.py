# -*- coding: utf-8 -*-
"""fig_e1_pareto.py — E1 多方法 × 多指标全景对比 (标注型热力图).

历次方案迭代: 3D 散点(点/标签重叠) -> 2D 气泡双面板(前沿堆叠) ->
本版: 行=指标(能耗/时延/成功率/SLA), 列=13 方法; 颜色按行归一化
(深绿=本指标内更优, 深红=更差), 单元格内写原始数值 —— 无点重叠、
无标签重叠, 一眼可见"谁在哪个指标上最优"。

输出: latex/figure/fig_e1_pareto.pdf / .tiff (1200 DPI)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/local")

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import RESULTS_DIR
from local.experiment_common import load_npz
from visual.palette import set_paper_style, method_color, save_figure
set_paper_style()

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "latex", "figure")

PATH = os.path.join(RESULTS_DIR, "e1_fresh.npz")

# 列顺序 (HMA 家族置底置后)
ORDER = ["Greedy", "AllLocal", "AllEdge", "Random", "SAC", "DDPG",
         "DQN", "MADDPG", "GA", "MPC", "LeDRL", "SingleLLM",
         "HMA-Distill", "HMA-Hybrid"]
DISPLAY = {"LeDRL": "LeDRL (B10)", "SingleLLM": "SingleLLM (B11)",
           "MADDPG": "MADDPG (B13)"}

# 行: (指标名, 显示名, 缩放, 格式, 越小越好?)
ROWS = [("energy", "Energy (kJ)", 1.0, "{:.2f}", True),
        ("latency", "Latency (s)", 1.0, "{:.3f}", True),
        ("success_rate", "Success (%)", 100.0, "{:.1f}", False),
        ("priority_sla", "SLA (%)", 100.0, "{:.1f}", False)]


def main():
    res = load_npz(PATH)
    methods = [m for m in ORDER if m in res and res[m]['mean'] is not None]
    if not methods:
        fig, ax = plt.subplots(figsize=(7.16, 3.0))
        ax.text(0.5, 0.5, "TBD - waiting for E1 data",
                ha='center', va='center', fontsize=17)
        ax.axis('off')
        save_figure(fig, "fig_e1_pareto", FIG_DIR)
        return

    # 原始值矩阵 (行×列)
    raw = np.zeros((len(ROWS), len(methods)))
    for i, (metric, _, scale, _, _) in enumerate(ROWS):
        for j, m in enumerate(methods):
            raw[i, j] = res[m]['mean'][metric] * scale

    # 每行归一化到 [0,1], 1 = 本行内最优 (即"越优越绿")
    norm = np.zeros_like(raw)
    for i, (_, _, _, _, lower_better) in enumerate(ROWS):
        r = raw[i]
        lo, hi = r.min(), r.max()
        if hi - lo < 1e-12:
            norm[i] = 0.5
        elif lower_better:
            norm[i] = (hi - r) / (hi - lo)
        else:
            norm[i] = (r - lo) / (hi - lo)

    fig, ax = plt.subplots(figsize=(7.16, 2.9))
    cmap = plt.cm.get_cmap('RdYlGn').copy()
    cmap.set_bad('white')
    im = ax.imshow(norm, cmap=cmap, vmin=0, vmax=1, aspect='auto')

    # 显式白色分隔线 (只画内部格线; 外框由坐标轴 spines 提供)。
    # 禁用 palette 默认灰色 major grid, 避免影像之下不可见 grid 层干扰审计。
    ax.grid(False)
    for i in range(len(ROWS) - 1):
        ax.axhline(i + 0.5, color='white', lw=1.2)
    for j in range(len(methods) - 1):
        ax.axvline(j + 0.5, color='white', lw=1.2)

    # 单元格数值 —— 字号自适应: 最宽取值串在单元内两侧留有 >= 2 pt 间隙
    cell_fmt = [[fmt.format(raw[i, j]) for j in range(len(methods))]
                for i, (_, _, _, fmt, _) in enumerate(ROWS)]
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    wmax = 0.0
    for row in cell_fmt:
        for s in set(row):
            probe = ax.text(0, 0, s, fontsize=10.0)
            bb = probe.get_window_extent(renderer=rend)
            wmax = max(wmax, bb.width * 72.0 / fig.dpi)
            probe.remove()
    cell_fs = min(10.5, max(6.0, 23.5 * 10.0 / wmax))
    for i, row in enumerate(cell_fmt):
        for j, s in enumerate(row):
            bg = norm[i, j]
            color = 'white' if bg > 0.62 or bg < 0.20 else 'black'
            ax.text(j, i, s, ha='center', va='center',
                    fontsize=cell_fs, color=color)

    # 列/行刻度标签
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=90, ha='center', fontsize=12.8)
    for lab, m in zip(ax.get_xticklabels(), methods):
        if m.startswith('HMA'):
            lab.set_fontweight('bold')
    ax.set_yticks(range(len(ROWS)))
    ax.set_yticklabels([r[1] for r in ROWS], fontsize=14)
    ax.tick_params(axis='x', labelsize=12.8)
    ax.tick_params(axis='y', labelsize=14)

    # HMA 列(末两列)整体加粗边框 —— 画在白色格线之上, 数值与框线无相交
    ax.add_patch(plt.Rectangle((len(methods) - 2 - 0.5, -0.42), 2.0,
                               len(ROWS) - 0.16, fill=False, ec='black',
                               lw=1.4))

    fig.tight_layout()
    save_figure(fig, "fig_e1_pareto", FIG_DIR)
    print(f"  saved: fig_e1_pareto.pdf/.tiff (heatmap {len(ROWS)}×{len(methods)})")


if __name__ == "__main__":
    main()
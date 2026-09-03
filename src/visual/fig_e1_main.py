# -*- coding: utf-8 -*-
"""Fig: E1 main comparison bar charts (3 rows × 1 column). English only.

垂直三行子图 (a) Energy / (b) Latency / (c) Success rate:
  - 全部方法一张图, 柱间距单行更宽, 消除横轴拥挤
  - 柱顶数值标注带防重叠: 数值太近时自动垂直错开
  - x 轴标签仅保留在末行 (共享轴), 图内无大标题 (题注承担)
"""

import os, sys, numpy as np, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE); ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)
sys.path.insert(0, "/root/.zcode/skills/nature-figure/scripts")
from audit_panel_alignment import require_matplotlib_panel_alignment
from config import RESULTS_DIR
from visual.palette import set_paper_style, method_color, save_figure

set_paper_style()

OUT_DIR = os.path.join(ROOT, "latex", "figure")
INPUT_NPZ = os.path.join(RESULTS_DIR, "e1_fresh.npz")


def _extract_per_method(npz, methods):
    out = {}
    for m in methods:
        keys = [k for k in npz.files if k.startswith(m + "__")]
        if not keys:
            continue
        means = {}
        stds = {}
        for metric in ('energy', 'latency', 'success_rate', 'priority_sla'):
            key = f"{m}__{metric}__mean"
            if key in npz.files:
                means[metric] = float(npz[key][0])
            skey = f"{m}__{metric}__std"
            if skey in npz.files:
                stds[metric] = float(npz[skey][0])
        out[m] = {'mean': means, 'std': stds}
    return out


def annotate_bars(ax, xs, vals, fmt, scale=1.0, fs=12, gap_ratio=0.02,
                  errs=None):
    """柱顶数值标注, 自动防重叠: 与已放置相邻标签距离 < gap 时交替上下偏移.
    errs 非空时标签置于误差棒顶端之上, 避免数值被 cap 线穿过."""
    ymax = max(v + (e if errs is not None else 0.0)
               for v, e in zip(vals, errs)) if errs is not None else max(vals)
    gap = max(ymax, 1e-9) * gap_ratio
    placed = {}          # x -> 当前该柱已用偏移(不计, 同柱只标一个)
    used_y = []          # (label_y) 用于全局去重
    for x, v, in zip(xs, vals):
        e = errs[x] if errs is not None else 0.0
        base = v + e + gap * 0.6
        dy = 0.0
        # 同组(相邻柱)冲突检测
        for (ux, uy) in used_y:
            if abs(ux - x) < 0.8 and abs(uy - base) < gap * 1.4:
                dy = gap * (1 + 0.6 * (len(used_y) % 2))
                break
        label = fmt.format(v / scale if scale != 1.0 else v)
        ax.text(x, base + dy, label, ha='center', va='bottom', fontsize=fs)
        used_y.append((x, base + dy))
    return ymax


def main():
    if not os.path.exists(INPUT_NPZ):
        fig, ax = plt.subplots(figsize=(7.16, 3.0))
        ax.text(0.5, 0.5, "TBD - waiting for E1 data",
                ha='center', va='center', fontsize=17)
        ax.axis('off')
        save_figure(fig, "fig_e1_main", OUT_DIR)
        return
    npz = np.load(INPUT_NPZ, allow_pickle=False)
    methods = sorted({k.split("__")[0] for k in npz.files})
    order = ['Greedy', 'AllLocal', 'AllEdge', 'Random', 'SAC', 'DDPG',
             'DQN', 'MADDPG', 'GA', 'MPC', 'LeDRL', 'SingleLLM',
             'HMA-Distill', 'HMA-Hybrid']
    methods = [m for m in order if m in methods] + \
              [m for m in methods if m not in order]
    data = _extract_per_method(npz, methods)
    names = list(data.keys())

    fig, axes = plt.subplots(3, 1, figsize=(7.16, 9.6))
    metrics = [('energy', 'Energy (kJ)', 1.0, '{:.3f}'),
               ('latency', 'Latency (s)', 1.0, '{:.3f}'),
               ('success_rate', 'Success rate (%)', 100.0, '{:.1f}')]
    title_labs = ['(a) Energy (kJ)', '(b) Latency (s)', '(c) Success rate (%)']

    for ax, (metric, label, scale, fmt), tlab in zip(axes, metrics, title_labs):
        vals = [data[m]['mean'].get(metric, 0.0) * scale for m in names]
        errs = [data[m]['std'].get(metric, 0.0) * scale for m in names]
        cs = [method_color(m) for m in names]
        x = np.arange(len(names))
        ax.bar(x, vals, color=cs, edgecolor='black', linewidth=0.4,
               yerr=errs, capsize=2.0, error_kw={'elinewidth': 0.6, 'ecolor': '#3A3F45'})
        # 14 方法 x 3 指标: 数值已在正文表 1, 图内不做逐柱标注 (避免 42 个
        # 标签在单栏密度下互相碰撞); 误差棒与共享图例承担量化信息
        tops = [v + e for v, e in zip(vals, errs)]
        ax.set_ylabel(label, fontsize=14)
        ax.tick_params(labelsize=13)
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_title(tlab, pad=6, fontsize=15, weight='bold')
        ax.set_ylim(0, max(tops) * 1.12)
        # 14 个方法名不宜作 x 轴刻度 (旋转后仍互相重叠): 只保留刻度线,
        # 方法-颜色对应关系由底部共享图例承担 (图例行=方法顺序)
        ax.set_xticks(x)
        ax.set_xticklabels([''] * len(names))
        ax.tick_params(axis='x', length=0)

    fig.subplots_adjust(hspace=0.5)
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=method_color(m), edgecolor='black',
                     linewidth=0.4) for m in names]
    fig.legend(handles, names, loc='lower center', ncol=7,
               bbox_to_anchor=(0.5, -0.02), fontsize=12, frameon=False,
               handlelength=1.2, handleheight=1.0, columnspacing=1.2)
    require_matplotlib_panel_alignment(
        fig,
        json_out=os.path.join(OUT_DIR, "fig_e1_main.alignment.json"),
        overlay_svg=os.path.join(OUT_DIR, "fig_e1_main.alignment.svg"),
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True,
    )
    from audit_panel_alignment import matplotlib_layout_manifest
    manifest = matplotlib_layout_manifest(fig)
    with open(os.path.join(OUT_DIR, "fig_e1_main.layout.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    save_figure(fig, "fig_e1_main", OUT_DIR)
    print(f"  saved: fig_e1_main.pdf/.tiff ({len(names)} methods × 3 rows)")


if __name__ == "__main__":
    main()
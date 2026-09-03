# -*- coding: utf-8 -*-
"""Fig: E20(i)/E21 验证器精化基线扩展 (boxplot, per-episode n=250).

English only. Boxes: 4 DRL/MARL baselines × {raw (gray), refined (teal)}
+ HMA-Distill reference (E1, n=250, navy). Shows that the verifier pulls
all learned policies to one cluster slightly below HMA.

Collision-audit notes:
  - ylim must contain every whisker/flier extreme (raw minima ~0.125 and
    caps at 1.0); otherwise whiskers poking below the axes cross the x tick
    labels.
  - Only horizontal gridlines (y axis) are drawn; the shared style enables
    x gridlines by default, so the x grid is switched off explicitly.
  - The legend lives below the axes; per-group "+x.xpp" delta labels sit in
    the empty band above the whisker caps (y>1.0), clear of every stroke.
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
DPI = 1200

RAW_COLOR = PALETTE['neutral_med']      # gray (raw policy)
REF_COLOR = PALETTE['hma_fullllm']      # teal-light (refined)
HMA_COLOR = PALETTE['hma_primary']      # navy (HMA-Distill)


def main():
    npz = np.load(os.path.join(RESULTS_DIR, "e20_fresh.npz"),
                  allow_pickle=False)
    methods = ["SAC", "DDPG", "DQN", "MADDPG"]
    raw = [np.asarray(npz[f"{m}__success_rate__raw__vals"], dtype=float)
           for m in methods]
    ref = [np.asarray(npz[f"{m}__success_rate__refined__vals"], dtype=float)
           for m in methods]
    hma = np.asarray(np.load(os.path.join(RESULTS_DIR, "e1_fresh.npz"),
                             allow_pickle=False)["HMA-Distill__success_rate__vals"],
                     dtype=float)

    fig, ax = plt.subplots(figsize=(7.16, 3.6))
    pos_raw = [1 + i * 3 - 0.38 for i in range(4)]
    pos_ref = [1 + i * 3 + 0.38 for i in range(4)]
    bp_raw = ax.boxplot(raw, positions=pos_raw, widths=0.62,
                        patch_artist=True,
                        medianprops=dict(color='#444444', lw=1.1),
                        whiskerprops=dict(color='#666666', lw=0.9),
                        capprops=dict(color='#666666', lw=0.9))
    bp_ref = ax.boxplot(ref, positions=pos_ref, widths=0.62,
                        patch_artist=True,
                        medianprops=dict(color='#1E3A5F', lw=1.1),
                        whiskerprops=dict(color='#2E5E7E', lw=0.9),
                        capprops=dict(color='#2E5E7E', lw=0.9))
    for patch in bp_raw['boxes']:
        patch.set_facecolor(RAW_COLOR); patch.set_alpha(0.85)
    for patch in bp_ref['boxes']:
        patch.set_facecolor(REF_COLOR); patch.set_alpha(0.9)
    bp_hm = ax.boxplot([hma], positions=[13.0], widths=0.72,
                       patch_artist=True,
                       medianprops=dict(color='#FFFFFF', lw=1.2),
                       whiskerprops=dict(color=PALETTE['hma_hybrid'], lw=0.9),
                       capprops=dict(color=PALETTE['hma_hybrid'], lw=0.9))
    for patch in bp_hm['boxes']:
        patch.set_facecolor(HMA_COLOR)

    # P1-8: double encoding for boxplots — overlay mean markers with
    # distinct shapes per group (gray o = raw, teal s = refined,
    # navy D = HMA) so the chart is readable without color alone
    ax.scatter(pos_raw, [r.mean() for r in raw], marker='o', s=22,
               color=RAW_COLOR, edgecolors='black', linewidths=0.5, zorder=5)
    ax.scatter(pos_ref, [r.mean() for r in ref], marker='s', s=22,
               color=REF_COLOR, edgecolors='black', linewidths=0.5, zorder=5)
    ax.scatter([13.0], [hma.mean()], marker='D', s=26,
               color=HMA_COLOR, edgecolors='black', linewidths=0.5, zorder=5)

    ax.axhline(0.889, color=PALETTE['hma_hybrid'], lw=1.0, ls='--', alpha=0.7)

    # y window: contains every whisker cap (top = 1.0) and flier (bottom
    # ~0.125) so no whisker stroke leaves the axes and crosses tick labels.
    ax.set_ylim(0.04, 1.18)
    ax.set_xlim(-0.7, 14.7)

    # "+x.xpp" delta annotations in the empty band above the whisker caps
    # (all caps end at y = 1.0; nothing is drawn above y = 1.0).
    ax.text(1.38, 1.045, "+%.1fpp" % ((ref[0].mean() - raw[0].mean()) * 100),
            fontsize=11, color=PALETTE['hma_primary'], ha='center', va='bottom')
    ax.text(4.38, 1.045, "+%.1fpp" % ((ref[1].mean() - raw[1].mean()) * 100),
            fontsize=11, color=PALETTE['hma_primary'], ha='center', va='bottom')
    ax.text(7.38, 1.045, "+%.1fpp" % ((ref[2].mean() - raw[2].mean()) * 100),
            fontsize=11, color=PALETTE['hma_primary'], ha='center', va='bottom')
    ax.text(10.38, 1.045, "+%.1fpp" % ((ref[3].mean() - raw[3].mean()) * 100),
            fontsize=11, color=PALETTE['hma_primary'], ha='center', va='bottom')

    ax.set_xticks([1, 4, 7, 10, 13])
    ax.set_xticklabels(methods + ["HMA-Distill"], fontsize=12)
    ax.set_ylabel("Episode success rate", fontsize=13, labelpad=8)
    ax.set_yticks(np.arange(0.2, 1.01, 0.2))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, p: f"{v:.1f}"))
    ax.tick_params(axis='y', labelsize=11)
    # Horizontal gridlines only (shared style enables x-gridlines by default).
    ax.grid(True, axis='y', ls=':', lw=0.6, alpha=0.6)
    ax.grid(False, axis='x')
    ax.set_axisbelow(True)

    # Legend below the axes (clear of all boxplot strokes and tick labels).
    handles = [bp_raw['boxes'][0], bp_ref['boxes'][0], bp_hm['boxes'][0]]
    labels = ["raw policy", "+ verifier refine", "HMA-Distill (E1)"]
    from matplotlib.lines import Line2D
    handles.append(Line2D([], [], color=PALETTE['hma_hybrid'], lw=1.0,
                          ls='--', alpha=0.7))
    labels.append("HMA mean 88.9% (E1 rerun)")
    fig.subplots_adjust(bottom=0.14, top=0.965, left=0.085, right=0.97)
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.055),
               ncol=2, fontsize=11, frameon=False, columnspacing=1.8,
               handlelength=1.6, handleheight=1.0)

    save_figure(fig, "fig_e20_refined", OUT_DIR)
    print("  saved: fig_e20_refined.pdf/.tiff")


if __name__ == "__main__":
    main()

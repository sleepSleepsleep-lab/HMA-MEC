# -*- coding: utf-8 -*-
"""
================================================================
Fig 1: HMA-MEC three-layer architecture and CW-Debate protocol
================================================================
Generates fig_arch.{pdf,tiff}. No in-figure title. English only.
1200 DPI TrueType-embedded PDF for IEEE submission, plus LZW TIFF
for supplementary. Run: python visual/fig_arch.py
================================================================
"""

import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
ROOT = os.path.dirname(SRC)
OUT_DIR = os.path.join(ROOT, "latex", "figure")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

PDF_DPI = TIFF_DPI = 1200

COLORS = {
    'env':'#7BA7D9', 'ua':'#71C58B', 'ea':'#8DCCC2',
    'oa':'#EAC49A', 'va':'#EF8979', 'distill':'#8A6C87',
    'arrow':'#555555', 'arrow_h':'#AA3333', 'fill':'#F7F9FB',
    'border':'#3A3F45',
}


def rbox(ax, x, y, w, h, label, color, edge=None, fs=9, weight='normal'):
    edge = edge or COLORS['border']
    p = FancyBboxPatch((x,y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.0, edgecolor=edge, facecolor=color, alpha=0.85)
    ax.add_patch(p)
    ax.text(x+w/2, y+h/2, label, ha='center', va='center',
            fontsize=fs, weight=weight)


def arrow(ax, x0, y0, x1, y1, color=None, lw=0.9, style='-|>'):
    color = color or COLORS['arrow']
    a = FancyArrowPatch((x0,y0),(x1,y1), arrowstyle=style,
        mutation_scale=10, linewidth=lw, color=color, shrinkA=2, shrinkB=2)
    ax.add_patch(a)


def draw_arch():
    fig, ax = plt.subplots(figsize=(7.16, 5.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 8); ax.set_aspect('equal')
    ax.axis('off')

    # Layer: Distill
    rbox(ax, 0.6, 6.3, 8.8, 0.7,
         r"Distill layer: online $\pi^{agent}_\theta(s)\to a$ (zero API) + "
         r"hard-state fallback to online debate",
         COLORS['distill'], fs=9, weight='bold')

    # Layer: Multi-agent debate
    rbox(ax, 0.6, 2.4, 8.8, 3.6, "", COLORS['fill'])
    ax.text(0.8, 5.85, "Multi-agent debate layer", fontsize=10,
            weight='bold', color=COLORS['border'], va='top')

    round_labels = ["R0 Broadcast","R1 Propose","R2 Critique",
                    "R3 Arbitrate","R4 Verify & R5 Consensus"]
    for i, lab in enumerate(round_labels):
        cx = 1.4 + i * 1.55
        ax.text(cx, 5.55, lab, ha='center', va='center', fontsize=8,
                color=COLORS['border'], style='italic')

    for i in range(4):
        cx = 1.4 + i * 1.55
        rbox(ax, cx-0.55, 4.6, 1.1, 0.6,
             f"UA-{i}\n($a_k, m_k, c_k$)", COLORS['ua'], fs=7.5)
    ax.text(1.0, 4.9, "UA x K", ha='center', va='center', fontsize=8,
            weight='bold', color=COLORS['border'])

    for i in range(3):
        cx = 1.4 + i * 2.0 + 0.7
        rbox(ax, cx-0.55, 3.85, 1.1, 0.55,
             f"EA-{i}\nCap / admit", COLORS['ea'], fs=7.5)
    ax.text(1.0, 4.13, "EA x M", ha='center', va='center', fontsize=8,
            weight='bold', color=COLORS['border'])

    rbox(ax, 4.8, 3.05, 2.6, 0.55,
         r"OA: arbitrate + pref $\omega_t$ + ToM $\hat{c}^{(r)}$",
         COLORS['oa'], fs=7.8, weight='bold')
    rbox(ax, 7.7, 3.05, 1.6, 0.55, "VA: counterfactual + reject",
         COLORS['va'], fs=7.8, weight='bold')

    arrow(ax, 1.95, 4.6, 2.1, 4.4, lw=0.8)
    arrow(ax, 6.1, 4.15, 5.7, 3.6, lw=0.8)
    arrow(ax, 7.4, 3.05, 7.7, 3.3, color=COLORS['arrow_h'], lw=0.9)
    arrow(ax, 8.5, 3.6, 6.4, 4.4, color=COLORS['arrow_h'], lw=0.7)
    arrow(ax, 9.3, 3.32, 9.85, 3.32, lw=0.9)
    ax.text(9.85, 3.55, "output $a_t$", ha='center', va='bottom', fontsize=8)

    rbox(ax, 0.6, 1.2, 8.8, 0.85, "", COLORS['env'])
    ax.text(1.1, 1.62, r"MEC env: state $s_t$ / step($a_t$)",
            ha='left', va='center', fontsize=8.5, weight='bold')
    ax.text(1.1, 1.31, r"$\Phi(\cdot)\to \tilde{s}_t, \mathrm{txt}(s_t)\to \mathcal{P}_t$",
            ha='left', va='center', fontsize=8)
    ax.text(5.3, 1.62,
            r"counterfactual $\mathrm{Sim}(a_t|s_t) \to (E^{act}, \bar{T}^{act})$",
            ha='left', va='center', fontsize=8.5, weight='bold')
    ax.text(5.3, 1.31, "does not modify real state (A4 constraint)",
            ha='left', va='center', fontsize=8, style='italic')

    arrow(ax, 5.0, 2.4, 5.0, 2.05, color=COLORS['arrow_h'], lw=1.2)
    arrow(ax, 5.5, 2.05, 5.5, 2.4, color=COLORS['arrow_h'], lw=1.2)
    arrow(ax, 4.0, 6.3, 4.0, 6.05, lw=1.0)
    arrow(ax, 6.0, 6.05, 6.0, 6.3, lw=1.0)
    ax.text(4.0, 6.18, "distill D_debate", ha='center', va='center',
            fontsize=7, style='italic')
    ax.text(6.0, 6.18, "hard state fallback", ha='center', va='center',
            fontsize=7, style='italic')

    legend_items = [
        Line2D([0],[0], marker='s', color='w', markerfacecolor=COLORS['distill'],
               markersize=10, label='Distill'),
        Line2D([0],[0], marker='s', color='w', markerfacecolor=COLORS['ua'],
               markersize=10, label='UA'),
        Line2D([0],[0], marker='s', color='w', markerfacecolor=COLORS['ea'],
               markersize=10, label='EA'),
        Line2D([0],[0], marker='s', color='w', markerfacecolor=COLORS['oa'],
               markersize=10, label='OA'),
        Line2D([0],[0], marker='s', color='w', markerfacecolor=COLORS['va'],
               markersize=10, label='VA'),
        Line2D([0],[0], marker='s', color='w', markerfacecolor=COLORS['env'],
               markersize=10, label='MEC env'),
    ]
    ax.legend(handles=legend_items, loc='lower center',
              bbox_to_anchor=(0.5, -0.08), ncol=6, frameon=False, fontsize=7.5,
              handletextpad=0.4, columnspacing=1.2)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_arch.pdf"),
                format="pdf", dpi=PDF_DPI, bbox_inches='tight')
    fig.savefig(os.path.join(OUT_DIR, "fig_arch.tiff"),
                format="tiff", dpi=TIFF_DPI, bbox_inches='tight',
                pil_kwargs={"compression": "tiff_lzw"})
    print(f"[OK] fig_arch.pdf / .tiff saved to {OUT_DIR}")


if __name__ == "__main__":
    draw_arch()
# -*- coding: utf-8 -*-
"""
================================================================
Fig 1: HMA-MEC three-layer architecture and CW-Debate protocol
================================================================
Generates fig_arch.{pdf,tiff}. No in-figure title. English only.
Every label's rendered width is verified against its container at
draw time (see [OK]/[OVER] printout); re-tune font sizes if any
line reports OVER. Line-box math: 1.4 em per text line, i.e.
half-line ~= 0.018 * fontsize in data units at this canvas scale.
Run: python visual/fig_arch.py
================================================================
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

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
    'env': '#7BA7D9', 'ua': '#71C58B', 'ea': '#8DCCC2',
    'oa': '#EAC49A', 'va': '#EF8979', 'distill': '#8A6C87',
    'arrow': '#555555', 'fill': '#F7F9FB',
    'border': '#3A3F45', 'text': '#22262B',
}

CHECKS = []


def check(key, s, fs, x0, max_w):
    CHECKS.append((key, s, fs, x0, max_w))


def rbox(ax, x, y, w, h, color, l1=None, l2=None, fs=10, fs2=None, tcolor=None,
         edge=None, weight='bold', key='box'):
    """Rounded box; one centered line (l1) or two stacked lines (l1/l2)."""
    edge = edge or COLORS['border']
    tcolor = tcolor or COLORS['text']
    fs2 = fs2 if fs2 is not None else fs
    p = FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.0, edgecolor=edge, facecolor=color, alpha=0.9)
    ax.add_patch(p)
    if l1 is None:
        return
    cx = x + w / 2.0
    if l2 is None:
        ax.text(cx, y + h / 2.0, l1, ha='center', va='center',
                fontsize=fs, weight=weight, color=tcolor)
    else:
        pad = 0.05
        y1 = y + h - pad - 0.018 * fs          # head line center
        y2 = y + pad + 0.018 * fs2             # sub line center
        ax.text(cx, y1, l1, ha='center', va='center',
                fontsize=fs, weight=weight, color=tcolor)
        ax.text(cx, y2, l2, ha='center', va='center',
                fontsize=fs2, color=tcolor)
        check(f"{key}::h", l1, fs, x, w - 0.12)
        check(f"{key}::s", l2, fs2, x, w - 0.12)


def arrow(ax, x0, y0, x1, y1, color=None, lw=1.0, style='-|>'):
    color = color or COLORS['arrow']
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
        mutation_scale=9, linewidth=lw, color=color, shrinkA=1, shrinkB=1)
    ax.add_patch(a)


def draw_arch():
    fig, ax = plt.subplots(figsize=(7.16, 5.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 8); ax.set_aspect('equal')
    ax.axis('off')

    # ================= Env layer =================
    rbox(ax, 0.7, 0.65, 8.6, 1.1, COLORS['env'], l1=None, key='envbox')
    # line1: state interface (left) | counterfactual simulator (right)
    ax.text(0.95, 1.44, r"MEC env: state $s_t$ / step($a_t$)",
            ha='left', va='center', fontsize=11, weight='bold',
            color=COLORS['text'])
    check("env::L1a", r"MEC env: state $s_t$ / step($a_t$)", 11, 0.95, 4.0)
    ax.text(4.95, 1.44,
            r"$\mathrm{Sim}(a_t|s_t)$ $\to$ $(E^{\mathrm{act}}, \bar{T}^{\mathrm{act}})$",
            ha='left', va='center', fontsize=11, weight='bold',
            color=COLORS['text'])
    check("env::L1b",
          r"$\mathrm{Sim}(a_t|s_t)$ $\to$ $(E^{\mathrm{act}}, \bar{T}^{\mathrm{act}})$",
          11, 4.95, 4.15)
    # line2: feature embedding (left) | non-invasive constraint (right)
    ax.text(0.95, 0.97, r"$\Phi(\cdot)\to \tilde{s}_t$;  $\mathrm{txt}(s_t)\to \mathcal{P}_t$",
            ha='left', va='center', fontsize=11.5, color=COLORS['text'])
    check("env::L2a", r"$\Phi(\cdot)\to \tilde{s}_t$;  $\mathrm{txt}(s_t)\to \mathcal{P}_t$",
          11.5, 0.95, 3.1)
    ax.text(4.15, 0.97, "does not modify real state (A4)",
            ha='left', va='center', fontsize=11.5, style='italic',
            color=COLORS['text'])
    check("env::L2b", "does not modify real state (A4)", 11.5, 4.15, 5.0)

    # ================= Debate layer =================
    rbox(ax, 0.7, 2.0, 8.6, 4.2, COLORS['fill'], l1=None, key='debatebox')

    # layer title (top-left, inside box)
    ax.text(0.92, 6.02, "Multi-agent debate layer", fontsize=13,
            weight='bold', color=COLORS['border'], va='top')
    check("title", "Multi-agent debate layer", 13, 0.92, 7.0)

    # CW-Debate stage flow
    stage = (r"$\mathrm{Broadcast}\to\mathrm{Propose}\to\mathrm{Critique}\to"
             r"\mathrm{Arbitrate}\to\mathrm{Verify}\to\mathrm{Consensus}$")
    ax.text(5.0, 5.24, stage, ha='center', va='center', fontsize=9.0,
            color='#6A6F76')
    check("stage", stage, 9.0, 0.9, 8.2)

    # UA row (K users)
    UA_Y, UA_H, UA_W = 4.08, 0.88, 1.55
    ua_cx = [2.0, 3.95, 5.9, 7.85]
    for i, cx in enumerate(ua_cx):
        rbox(ax, cx - UA_W / 2, UA_Y, UA_W, UA_H, COLORS['ua'],
             l1=f"UA-{i}", l2=r"$(\alpha_k, m_k, c_k)$",
             fs=11.5, fs2=9.5, key=f'ua{i}')

    # EA row (M servers)
    EA_Y, EA_H, EA_W = 3.07, 0.88, 1.9
    ea_cx = [2.6, 5.05, 7.5]
    for i, cx in enumerate(ea_cx):
        rbox(ax, cx - EA_W / 2, EA_Y, EA_W, EA_H, COLORS['ea'],
             l1=f"EA-{i}", l2="Cap / admit", fs=11, fs2=10, key=f'ea{i}')

    # OA / VA row
    rbox(ax, 3.9, 2.15, 2.8, 0.82, COLORS['oa'],
         l1=r"OA: arbitrate + $\omega_t$", l2=r"ToM $\hat{c}^{(r)}$ + consensus",
         fs=9.5, fs2=9.5, key='oa')
    rbox(ax, 6.95, 2.15, 2.3, 0.82, COLORS['va'],
         l1="VA: counterfactual", l2="simulate + reject",
         fs=9.0, fs2=9.0, key='va')

    # internal flow arrows
    arrow(ax, ua_cx[0], UA_Y, ea_cx[0], EA_Y + EA_H, lw=0.9)   # UA0 -> EA0
    arrow(ax, ua_cx[3], UA_Y, ea_cx[2], EA_Y + EA_H, lw=0.9)   # UA3 -> EA2
    arrow(ax, ea_cx[1], EA_Y, 5.35, 2.97, lw=0.9)              # EA1 -> OA
    arrow(ax, 6.7, 2.56, 6.95, 2.56, lw=0.9)                   # OA -> VA
    arrow(ax, 9.25, 2.56, 9.62, 2.56, lw=1.0)                  # VA -> a_t
    ax.text(9.88, 2.56, r"$a_t$", ha='center', va='center',
            fontsize=11, style='italic', color=COLORS['text'])

    # ================= Distill layer =================
    rbox(ax, 0.7, 6.65, 8.6, 1.15, COLORS['distill'],
         l1=r"Distill layer: online $\pi^{\mathrm{agent}}_\theta(s)\to a$ (zero API)",
         l2="hard-state fallback to online debate",
         fs=12.5, fs2=12.5, tcolor='white', key='distill')

    # debate <-> distill channel
    ax.text(7.6, 5.88, r"distill $D_{debate}$", ha='center', va='center',
            fontsize=10, style='italic', color=COLORS['text'])
    check("ch::up", r"distill $D_{debate}$", 10, 6.6, 2.0)
    arrow(ax, 7.6, 6.2, 7.6, 6.65, lw=1.0)     # D_debate up
    arrow(ax, 4.6, 6.65, 4.6, 6.2, lw=1.0)     # fallback down

    # ================= env <-> debate arrows =================
    arrow(ax, 5.0, 1.75, 5.0, 2.0, lw=1.0)     # s_t up
    arrow(ax, 5.5, 2.0, 5.5, 1.75, lw=1.0)     # a_t down

    # ---------- width verification ----------
    fig.canvas.draw()
    du_px = ax.bbox.width / 10.0
    over = 0
    for key, s, fs, x0, max_w in CHECKS:
        t = ax.text(0.05, 0.05, s, fontsize=fs)
        fig.canvas.draw()
        bb = t.get_window_extent(fig.canvas.get_renderer())
        wu = bb.width / du_px
        t.remove()
        if wu > max_w:
            over += 1
            print(f"[OVER] {key:12s} fs={fs:4.1f} width={wu:5.2f}u "
                  f"(limit {max_w:5.2f}u from x={x0:4.2f})")
    fig.canvas.draw()
    if over:
        print(f"WARNING: {over} label(s) exceed container limit")
    else:
        print("[OK] all labels fit their containers")

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_arch.pdf"),
                format="pdf", dpi=PDF_DPI, bbox_inches='tight')
    fig.savefig(os.path.join(OUT_DIR, "fig_arch.tiff"),
                format="tiff", dpi=TIFF_DPI, bbox_inches='tight',
                pil_kwargs={"compression": "tiff_lzw"})
    print(f"[OK] fig_arch.pdf / .tiff saved to {OUT_DIR}")


if __name__ == "__main__":
    draw_arch()

# -*- coding: utf-8 -*-
"""Fig 2: Distill-Agent policy network architecture. English only."""

import os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE); ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42; plt.rcParams['ps.fonttype'] = 42

OUT_DIR = os.path.join(ROOT, "latex", "figure")
os.makedirs(OUT_DIR, exist_ok=True)
PDF_DPI = TIFF_DPI = 1200

C_BOX = {'input':'#7BA7D9','shared':'#8DCCC2','head':'#EAC49A',
         'conf':'#8A6C87','output':'#71C58B'}


def rbox(ax, x, y, w, h, label, color, fs=8):
    p = FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.0, edgecolor='black', facecolor=color, alpha=0.85)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, label, ha='center', va='center',
            fontsize=fs, weight='bold')


def arrow(ax, x0, y0, x1, y1):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='-|>', lw=0.9, color='#555'))


def main():
    fig, ax = plt.subplots(figsize=(7.16, 3.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.set_aspect('equal')
    ax.axis('off')

    rbox(ax, 0.3, 2.0, 1.4, 0.9, r"$s_t\in\mathbb{R}^{4K+2M}$", C_BOX['input'], 9)
    rbox(ax, 2.2, 2.0, 1.4, 0.9,
         r"LN -> LkReLU"
         "\n"
         r"-> 256 -> LN -> LkReLU -> 256",
         C_BOX['shared'], 7)
    arrow(ax, 1.7, 2.45, 2.2, 2.45)

    head_xs = [4.3, 4.3, 4.3, 4.3]
    head_ys = [4.0, 3.1, 1.7, 0.5]
    head_labels = [
        r"$\mu_\alpha=\sigma(W_\alpha h)$",
        r"$\sigma_\alpha=\exp(\mathrm{clip}(W_\sigma h))$",
        r"$\ell_{\text{server}}\in\mathbb{R}^{K\times M}$",
        r"$c_\text{min}=\sigma(w_c^\top \mathrm{LkReLU}(W_c h))$",
    ]
    for x, y, lab in zip(head_xs, head_ys, head_labels):
        rbox(ax, x, y, 2.5, 0.7, lab, C_BOX['head'], 7)
        arrow(ax, 3.6, 2.45, x, y + 0.35)

    rbox(ax, 7.4, 3.55, 2.3, 0.7,
         r"$\alpha_k\sim\mathrm{Beta}(\mu_\alpha\kappa,(1-\mu_\alpha)\kappa)$"
         "\n" r"$\to \alpha^\star \in [0.01, 1.0]$",
         C_BOX['output'], 7)
    rbox(ax, 7.4, 2.65, 2.3, 0.7,
         r"$\kappa = 1/\sigma_\alpha^2 - 1$",
         C_BOX['output'], 7)
    rbox(ax, 7.4, 1.25, 2.3, 0.7,
         r"$m_k=\arg\max\mathrm{softmax}(\ell_{\text{server},k})$",
         C_BOX['output'], 7)
    rbox(ax, 7.4, 0.05, 2.3, 0.7,
         r"Hybrid trigger: $c_\text{min} < \tau_\text{low}$",
         C_BOX['conf'], 7)
    for y in [3.9, 3.0, 1.6, 0.4]:
        arrow(ax, 6.8, y, 7.4, y)

    # single figure, no subplot label needed

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_distill_net.pdf"),
                format="pdf", dpi=PDF_DPI, bbox_inches='tight')
    fig.savefig(os.path.join(OUT_DIR, "fig_distill_net.tiff"),
                format="tiff", dpi=TIFF_DPI, bbox_inches='tight',
                pil_kwargs={"compression": "tiff_lzw"})
    print("[OK] fig_distill_net.pdf / .tiff")


if __name__ == "__main__":
    main()
"""Fig: Distill-Agent training history (loss curves)."""

import os, sys, json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE); ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)
sys.path.insert(0, "/root/.zcode/skills/nature-figure/scripts")
from visual.palette import set_paper_style, PALETTE
from audit_panel_alignment import (require_matplotlib_panel_alignment,
                                   matplotlib_layout_manifest)

set_paper_style()
OUT_DIR = os.path.join(ROOT, "latex", "figure")
os.makedirs(OUT_DIR, exist_ok=True)
PDF_DPI = TIFF_DPI = 1200

HIST_PATH = os.path.join(ROOT, "results", "checkpoints",
                          "distilled_policy_history.npy")


def main():
    if not os.path.exists(HIST_PATH):
        fig, ax = plt.subplots(figsize=(7.16, 3.5))
        ax.text(0.5, 0.5, "TBD - run train_and_save_history.py first",
                ha='center', va='center', fontsize=17)
        ax.axis('off')
        fig.savefig(os.path.join(OUT_DIR, "fig_train_history.pdf"),
                    format="pdf", dpi=PDF_DPI, bbox_inches='tight')
        fig.savefig(os.path.join(OUT_DIR, "fig_train_history.tiff"),
                    format="tiff", dpi=TIFF_DPI, bbox_inches='tight',
                    pil_kwargs={"compression": "tiff_lzw"})
        plt.close(fig); print("  placeholder saved"); return

    h = np.load(HIST_PATH)
    epochs = h[:, 0]
    la = h[:, 1]   # Laplace NLL
    ls = h[:, 2]   # Server CrossEntropy
    lc = h[:, 3]   # Confidence MSE
    lv = h[:, 4]   # Validation loss

    best_idx = int(np.argmin(lv))
    best_ep = int(epochs[best_idx])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 3.5))

    # Left subplot: training losses (P1-8: distinct markers so lines are
    # separable in grayscale; markevery thins markers on dense curves)
    ax1.plot(epochs, la, '-', marker='o', ms=3, markevery=4,
             color=PALETTE['hma_primary'],
             linewidth=1.3, label=r'$\mathcal{L}_\alpha$ (Laplace NLL)')
    ax1.plot(epochs, ls, '-', marker='s', ms=3, markevery=4,
             color=PALETTE['sac'],
             linewidth=1.3, label=r'$\lambda_s \mathcal{L}_s$ (server CE)')
    ax1.plot(epochs, lc, '-', marker='^', ms=3, markevery=4,
             color=PALETTE['hma_hybrid'],
             linewidth=1.3, label=r'$\lambda_c \mathcal{L}_c$ (conf MSE)')
    ax1.set_xlabel('Epoch', fontsize=13)
    ax1.set_ylabel('Loss', fontsize=13)
    ax1.tick_params(labelsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.grid(False, axis='x')
    ax1.set_title('(a) Training loss components', pad=9, fontsize=13,
                  weight='bold')

    # Right subplot: validation loss
    ax2.plot(epochs, lv, '-', color=PALETTE['hma_primary'],
             linewidth=1.3, label='Validation loss')
    # Mark best epoch
    ax2.axvline(best_ep, color=PALETTE['sac'], linestyle='--',
                linewidth=0.8, alpha=0.7,
                label=f'Best epoch = {best_ep}')
    ax2.scatter([best_ep], [lv[best_idx]], color=PALETTE['sac'],
                s=40, zorder=5, marker='D', edgecolors='black',
                linewidths=0.5)
    ax2.set_xlabel('Epoch', fontsize=13)
    ax2.set_ylabel('Validation loss', fontsize=13)
    ax2.tick_params(labelsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.grid(False, axis='x')
    ax2.set_title('(b) Validation loss', pad=9, fontsize=13, weight='bold')

    plt.tight_layout()

    # Shared legend under both panels, out of the way of every curve/gridline.
    handles, labels = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    handles += h2          # validation-loss line + dashed best-epoch marker
    labels += l2
    fig.legend(handles, labels, loc='upper center',
               bbox_to_anchor=(0.5, -0.085), ncol=3, fontsize=11.5,
               frameon=False, columnspacing=1.6, handlelength=1.7,
               handleheight=1.0)

    # Multi-panel alignment gate (measured on the final rendered layout).
    require_matplotlib_panel_alignment(
        fig,
        json_out=os.path.join(OUT_DIR, "fig_train_history.alignment.json"),
        overlay_svg=os.path.join(OUT_DIR, "fig_train_history.alignment.svg"),
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True,
    )
    manifest = matplotlib_layout_manifest(fig)
    with open(os.path.join(OUT_DIR, "fig_train_history.layout.json"),
              "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    fig.savefig(os.path.join(OUT_DIR, "fig_train_history.pdf"),
                format="pdf", dpi=PDF_DPI, bbox_inches='tight')
    fig.savefig(os.path.join(OUT_DIR, "fig_train_history.tiff"),
                format="tiff", dpi=TIFF_DPI, bbox_inches='tight',
                pil_kwargs={"compression": "tiff_lzw"})
    print("  saved: fig_train_history.pdf/.tiff")


if __name__ == "__main__":
    main()

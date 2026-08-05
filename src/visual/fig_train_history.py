"""Fig: Distill-Agent training history (loss curves)."""

import os, sys, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE); ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)
from visual.palette import set_paper_style, PALETTE

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
                ha='center', va='center', fontsize=11)
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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 3.2))

    # Left subplot: training losses
    ax1.plot(epochs, la, '-', color=PALETTE['hma_primary'],
             linewidth=1.3, label=r'$\mathcal{L}_\alpha$ (Laplace NLL)')
    ax1.plot(epochs, ls, '-', color=PALETTE['sac'],
             linewidth=1.3, label=r'$\lambda_s \mathcal{L}_s$ (server CE)')
    ax1.plot(epochs, lc, '-', color=PALETTE['hma_hybrid'],
             linewidth=1.3, label=r'$\lambda_c \mathcal{L}_c$ (conf MSE)')
    ax1.set_xlabel('Epoch', fontsize=8)
    ax1.set_ylabel('Loss', fontsize=8)
    ax1.tick_params(labelsize=7)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=7, frameon=False)
    ax1.set_title('(a) Training loss components', pad=8, fontsize=9, weight='bold')

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
    ax2.set_xlabel('Epoch', fontsize=8)
    ax2.set_ylabel('Validation loss', fontsize=8)
    ax2.tick_params(labelsize=7)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=7, frameon=False)
    ax2.set_title('(b) Validation loss', pad=8, fontsize=9, weight='bold')

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_train_history.pdf"),
                format="pdf", dpi=PDF_DPI, bbox_inches='tight')
    fig.savefig(os.path.join(OUT_DIR, "fig_train_history.tiff"),
                format="tiff", dpi=TIFF_DPI, bbox_inches='tight',
                pil_kwargs={"compression": "tiff_lzw"})
    print("  saved: fig_train_history.pdf/.tiff")


if __name__ == "__main__":
    main()
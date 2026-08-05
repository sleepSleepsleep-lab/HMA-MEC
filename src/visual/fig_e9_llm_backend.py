# -*- coding: utf-8 -*-
"""Fig 11: E9 LLM backend comparison bar charts."""

import os, sys, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE); ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)
from config import RESULTS_DIR
from visual.palette import set_paper_style

set_paper_style()
OUT_DIR = os.path.join(ROOT, "latex", "figure")
os.makedirs(OUT_DIR, exist_ok=True)
INPUT_NPZ = os.path.join(RESULTS_DIR, "e9_llm_backend.npz")
PDF_DPI = TIFF_DPI = 1200

DEFAULT_BACKENDS = ['DeepSeek', 'Qwen3.7-Plus', 'Qwen3.7-Max']
BACKEND_COLORS = ['#2E5E7E', '#588B8B', '#8FB992']


def main():
    backends = DEFAULT_BACKENDS[:]
    cs = BACKEND_COLORS[:]
    npz = None
    if os.path.exists(INPUT_NPZ):
        npz = np.load(INPUT_NPZ, allow_pickle=False)
        if 'backends' in npz.files and len(npz['backends']) > 0:
            backends = list(npz['backends'])
            cs = [BACKEND_COLORS[i % len(BACKEND_COLORS)] for i in range(len(backends))]

    def _get(key):
        if npz is None: return None
        if key not in npz.files: return None
        v = npz[key]
        return v if len(v) == len(backends) else None

    fig, axes = plt.subplots(1, 4, figsize=(7.16, 2.8))
    title_labs = ['(a) Energy (kJ)', '(b) Latency (s)',
                  '(c) Success (%)', '(d) SLA (%)']
    metric_keys = ['energy', 'latency', 'success', 'sla']
    scales = [1.0, 1.0, 100.0, 100.0]
    for ax, tlab, mk, sc in zip(axes, title_labs, metric_keys, scales):
        vals = _get(mk)
        if vals is None or len(vals) == 0:
            ax.text(0.5, 0.5, "TBD - run on GPU server",
                    ha='center', va='center', fontsize=10,
                    transform=ax.transAxes)
            ax.set_xticks(range(len(backends)))
            ax.set_xticklabels(backends, rotation=20, fontsize=7)
            ax.set_yticks([])
            ax.set_title(tlab, pad=6, fontsize=9, weight='bold')
            ax.tick_params(axis='x', labelsize=7)
            continue
        vals = vals * sc
        bars = ax.bar(range(len(backends)), vals, color=cs,
                      edgecolor='black', linewidth=0.4)
        for i, v in enumerate(vals):
            ax.text(i, v, f'{v / sc:.3g}', ha='center', va='bottom',
                     fontsize=7)
        ax.set_xticks(range(len(backends)))
        ax.set_xticklabels(backends, rotation=20, fontsize=7)
        ax.set_title(tlab, pad=6, fontsize=9, weight='bold')
        ax.tick_params(axis='y', labelsize=7)
        ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_e9_llm_backend.pdf"),
                format="pdf", dpi=PDF_DPI, bbox_inches='tight')
    fig.savefig(os.path.join(OUT_DIR, "fig_e9_llm_backend.tiff"),
                format="tiff", dpi=TIFF_DPI, bbox_inches='tight',
                pil_kwargs={"compression": "tiff_lzw"})
    print("  saved: fig_e9_llm_backend.pdf/.tiff")


if __name__ == "__main__":
    main()
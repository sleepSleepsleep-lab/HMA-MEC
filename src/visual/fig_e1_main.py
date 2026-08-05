# -*- coding: utf-8 -*-
"""Fig 3: E1 main comparison bar charts. English only."""

import os, sys, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE); ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)
from config import RESULTS_DIR
from visual.palette import set_paper_style, method_color, method_marker

set_paper_style()

OUT_DIR = os.path.join(ROOT, "latex", "figure")
os.makedirs(OUT_DIR, exist_ok=True)
INPUT_NPZ = os.path.join(RESULTS_DIR, "e1_comparison.npz")
PDF_DPI = TIFF_DPI = 1200


def _extract_per_method(npz, methods):
    out = {}
    for m in methods:
        keys = [k for k in npz.files if k.startswith(m + "__")]
        if not keys:
            continue
        means = {}
        for metric in ('energy','latency','success_rate','priority_sla'):
            key = f"{m}__{metric}__mean"
            if key in npz.files:
                means[metric] = float(npz[key][0])
        out[m] = means
    return out


# Method display order: baselines first, HMA variants last
METHOD_ORDER = ['Greedy', 'AllLocal', 'AllEdge', 'Random',
                'SAC', 'DDPG', 'HMA-Distill', 'HMA-Hybrid']


def main():
    if not os.path.exists(INPUT_NPZ):
        fig, ax = plt.subplots(figsize=(7.16, 3.0))
        ax.text(0.5, 0.5, "TBD - waiting for E1 data",
                ha='center', va='center', fontsize=11)
        ax.axis('off')
        fig.savefig(os.path.join(OUT_DIR, "fig_e1_main.pdf"),
                    format="pdf", dpi=PDF_DPI, bbox_inches='tight')
        return
    print(f"  using: {INPUT_NPZ}")
    npz = np.load(INPUT_NPZ, allow_pickle=False)
    methods = sorted({k.split("__")[0] for k in npz.files})
    # Re-order so HMA variants appear last & adjacent (best contrast)
    methods = [m for m in METHOD_ORDER if m in methods] + \
              [m for m in methods if m not in METHOD_ORDER]
    data = _extract_per_method(npz, methods)

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6))
    metrics = [('energy',      'Energy (kJ)',         1.0,   '{:.4f}'),
               ('latency',     'Latency (s)',         1.0,   '{:.3f}'),
               ('success_rate','Success rate (%)',  100.0,  '{:.1f}')]

    title_labs = ['(a) Energy (kJ)', '(b) Latency (s)', '(c) Success rate (%)']
    for ax, (metric, label, scale, fmt), tlab in zip(axes, metrics, title_labs):
        names = list(data.keys())
        vals = [data[m].get(metric, 0.0) * scale for m in names]
        cs = [method_color(m) for m in names]
        x = np.arange(len(names))
        ax.bar(x, vals, color=cs, edgecolor='black', linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha='right', fontsize=7.5)
        ax.set_ylabel(label, fontsize=7.5)
        ax.tick_params(axis='y', labelsize=7)
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_title(tlab, pad=8, fontsize=8.5, weight='bold')
        for i, v in enumerate(vals):
            ax.text(i, v + max(vals)*0.03, fmt.format(v / scale if scale != 1.0 else v),
                    ha='center', va='bottom', fontsize=7, rotation=0)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_e1_main.pdf"),
                format="pdf", dpi=PDF_DPI, bbox_inches='tight')
    fig.savefig(os.path.join(OUT_DIR, "fig_e1_main.tiff"),
                format="tiff", dpi=TIFF_DPI, bbox_inches='tight',
                pil_kwargs={"compression": "tiff_lzw"})
    print(f"  saved: {OUT_DIR}/fig_e1_main.pdf/.tiff")


if __name__ == "__main__":
    main()
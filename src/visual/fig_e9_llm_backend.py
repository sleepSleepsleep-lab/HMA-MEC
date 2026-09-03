# -*- coding: utf-8 -*-
"""Fig 11: E9 LLM backend comparison bar charts.
Data: results/e9_multi_llm_results.json (HMA-Distill per backend).
"""

import os, sys, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE); ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)
sys.path.insert(0, "/root/.zcode/skills/nature-figure/scripts")
from audit_panel_alignment import require_matplotlib_panel_alignment
from config import RESULTS_DIR
from visual.palette import set_paper_style, save_figure

set_paper_style()
OUT_DIR = os.path.join(ROOT, "latex", "figure")
os.makedirs(OUT_DIR, exist_ok=True)
INPUT_JSON = os.path.join(RESULTS_DIR, "e9_multi_llm_results.json")
PDF_DPI = TIFF_DPI = 1200

# 固定顺序: 论文 E9 的三个开源 7B/8B 后端
BACKEND_ORDER = ['Qwen2.5-7B', 'Llama-3.1-8B', 'Mistral-7B']
BACKEND_COLORS = ['#2E5E7E', '#588B8B', '#8FB992']


def _load_recs():
    """P1-1: 优先读对齐实验新数据 (e9_llm_backend.json, 三后端统一 479 条),
    缺失时回退旧的多模型结果 (e9_multi_llm_results.json)."""
    p_new = os.path.join(RESULTS_DIR, "e9_llm_backend.json")
    if os.path.exists(p_new):
        try:
            nd = json.load(open(p_new, encoding='utf-8'))
        except Exception:
            nd = {}
        if nd.get('backend'):
            recs = []
            for i, b in enumerate(nd['backend']):
                if b in BACKEND_ORDER:
                    recs.append((b, {
                        'energy': nd['energy'][i],
                        'latency': nd['latency'][i],
                        'success_rate': nd['success'][i],
                        'priority_sla': nd['sla'][i],
                    }, {}))
            if len(recs) == len(BACKEND_ORDER):
                return sorted(recs, key=lambda r: BACKEND_ORDER.index(r[0]))
    d = json.load(open(INPUT_JSON, encoding='utf-8'))
    recs = []
    for k, v in d.items():
        if not k.startswith('HMA-Distill'):
            continue
        backend = k.split(' (')[1].rstrip(')') if ' (' in k else k
        if 'mean' in v:
            recs.append((backend, v['mean'], v.get('std', {})))
    recs = [r for r in recs if r[0] in BACKEND_ORDER]
    return sorted(recs, key=lambda r: BACKEND_ORDER.index(r[0]))


def main():
    recs = _load_recs()
    backends = [r[0] for r in recs]
    cs = [BACKEND_COLORS[i % len(BACKEND_COLORS)] for i in range(len(backends))]

    def _get(mk):
        return [r[1][mk] for r in recs]

    # P1-8: error bars from json std (fallback 0 if a backend lacks std)
    def _get_std(mk):
        return [r[2].get(mk, 0.0) for r in recs]

    fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.4))
    axes = axes.ravel()
    title_labs = ['(a) Energy (kJ)', '(b) Latency (s)',
                  '(c) Success (%)', '(d) SLA (%)']
    metric_keys = ['energy', 'latency', 'success_rate', 'priority_sla']
    scales = [1.0, 1.0, 100.0, 100.0]
    for ax, tlab, mk, sc in zip(axes, title_labs, metric_keys, scales):
        if len(backends) == 0:
            ax.text(0.5, 0.5, "TBD - run on GPU server",
                    ha='center', va='center', fontsize=16,
                    transform=ax.transAxes)
            ax.axis('off')
            ax.set_title(tlab, pad=6, fontsize=15, weight='bold')
            continue
        vals = np.asarray(_get(mk)) * sc
        errs = np.asarray(_get_std(mk)) * sc
        ax.bar(range(len(backends)), vals, color=cs,
               edgecolor='black', linewidth=0.4,
               yerr=errs, capsize=2.0,
               error_kw={'elinewidth': 0.6, 'ecolor': '#3A3F45'})
        # 数值在正文表 e9, 图内不做逐柱标注 (3 组 x 4 指标在单栏密度下
        # 数值标签与误差棒互相干扰)
        ax.set_xticks(range(len(backends)))
        ax.set_xticklabels([''] * len(backends))
        ax.tick_params(axis='x', length=0)
        ax.set_title(tlab, pad=6, fontsize=15, weight='bold')
        ax.tick_params(axis='y', labelsize=13)
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_ylim(0, max(v + e for v, e in zip(vals, errs)) * 1.12)

    plt.tight_layout()
    from matplotlib.patches import Patch
    fig.legend([Patch(facecolor=c, edgecolor='black', linewidth=0.4)
                for c in cs], backends,
               loc='lower center', bbox_to_anchor=(0.5, 0.985),
               ncol=len(backends), fontsize=13, frameon=False,
               handlelength=1.2, columnspacing=1.5)
    fig.subplots_adjust(hspace=0.55, wspace=0.3, top=0.88)
    require_matplotlib_panel_alignment(
        fig,
        json_out=os.path.join(OUT_DIR, "fig_e9_llm_backend.alignment.json"),
        overlay_svg=os.path.join(OUT_DIR, "fig_e9_llm_backend.alignment.svg"),
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True,
    )
    from audit_panel_alignment import matplotlib_layout_manifest
    manifest = matplotlib_layout_manifest(fig)
    with open(os.path.join(OUT_DIR, "fig_e9_llm_backend.layout.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    save_figure(fig, "fig_e9_llm_backend", OUT_DIR)
    print(f"  saved: fig_e9_llm_backend.pdf/.tiff ({len(backends)} backends)")


if __name__ == "__main__":
    main()

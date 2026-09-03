# -*- coding: utf-8 -*-
"""Fig: E11 LLM teacher prior value — latency gain of LLM seed vs MPC
(Distill-teacher prior + same verifier refiner). English labels.
Data: results/e1_llm_necessity_v2.json  (_stats per scenario)

Layout notes (collision-audit driven):
  - Error bars are ~+-22..25pp (propagated per-episode latency spread), so the
    y window must fully contain every bar's whisker top/bottom, otherwise the
    stroked whiskers extend across tick labels / value labels.
  - Per-bar value labels sit just above each bar's own error-bar top;
    p-value labels sit in one row below every error-bar bottom. Both rows are
    placed between horizontal gridlines (ticks at -40/-20/0/20) so no text is
    crossed by a gridline.
  - Rotated x tick labels stay inside the bottom margin; the y label is short
    enough and the figure tall enough that the rotated label (whose PDF trace
    box the auditor inflates for rotated runs) stays inside the page.
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
INP = os.path.join(RESULTS_DIR, "e11_necessity_n20.json")

SCN = [('normal', 'normal'), ('link_fail', 'link fail'), ('channel_degrade', 'channel deg.'),
       ('dishonest', 'dishonest UA'), ('high_load', 'high load'),
       ('tight_deadline', 'tight deadline'), ('multi_fault', 'multi-fault')]


def main():
    d = json.load(open(INP, encoding='utf-8'))
    gains, ps, snames = [], [], []
    errs = []   # P1-8: error bars for the latency-gain bars
    for key, lab in SCN:
        st = d[key]['_stats']
        gains.append(st['llm_vs_mpc_latency_gain_pct'] * 100)   # 负值=LLM更优
        ps.append(st['wilcoxon_latency_p'])
        snames.append(lab)
        # Gain g = (mu_mpc - mu_llm) / mu_mpc (fraction, as stored in
        # _stats). No direct std of the gain exists, so propagate the
        # per-method latency std from the json:
        #   sigma_g = sqrt((sigma_l/mu_m)^2 + (mu_l*sigma_m/mu_m^2)^2)
        try:
            m_llm = d[key]['LLM种子(HMA)']
            m_mpc = d[key]['MPC(启发式)']
            mu_l, sg_l = m_llm['mean']['T'], m_llm['std']['T']
            mu_m, sg_m = m_mpc['mean']['T'], m_mpc['std']['T']
            if mu_m > 0:
                sg = (sg_l / mu_m) ** 2 + (mu_l * sg_m / mu_m ** 2) ** 2
                errs.append(float(np.sqrt(sg)) * 100)
            else:
                errs.append(0.0)
        except (KeyError, TypeError):
            errs.append(None)
    if any(e is None for e in errs):
        print("  note: std missing for some scenarios -> error bars "
              "skipped there")
    errs = [0.0 if e is None else e for e in errs]

    # Holm 校正 (7 场景时延检验族): 星号与着色基于校正后显著性
    ps_arr = np.array(ps)
    order = np.argsort(ps_arr)
    holm = np.ones_like(ps_arr)
    m = len(ps_arr)
    for rank, idx in enumerate(order):
        holm[idx] = min(ps_arr[idx] * (m - rank), 1.0)

    gains = np.asarray(gains)
    errs = np.asarray(errs)
    tops = gains + errs          # whisker top  (data)
    bots = gains - errs          # whisker bottom

    fig, ax = plt.subplots(figsize=(7.16, 3.7))
    colors = [PALETTE['hma_primary'] if h < 0.05 else PALETTE['neutral_med']
              for h in holm]
    ax.bar(range(len(gains)), gains, color=colors,
           edgecolor='black', linewidth=0.4,
           yerr=list(errs),
           capsize=2.0,
           error_kw={'elinewidth': 0.6, 'ecolor': '#3A3F45'})

    # Per-bar value label + significance stars, on one uniform row above every
    # error-bar top/cap. The row sits between the y=20 gridline and the top
    # spine, so no whisker, cap or gridline stroke crosses the text.
    vlab_fs = 12
    for i, (g, p, h) in enumerate(zip(gains, ps, holm)):
        star = '***' if h < 0.01 else ('**' if h < 0.05 else ('*' if h < 0.1 else ''))
        ax.text(i, 23.6, f'{g:.1f}%{star}', ha='center', va='bottom',
                fontsize=vlab_fs, color='black')

    # p-value labels in a single row below every error-bar bottom; the row is
    # clear of the whisker bottoms, of the bottom spine and of gridlines.
    ylim_top = max(tops) + 12.0          # headroom for the value-label row
    ylim_bot = min(bots) - 13.2          # headroom for the p-value row
    ax.set_ylim(ylim_bot, ylim_top)
    p_row_y = min(bots) - 4.2
    for i, p in enumerate(ps):
        ax.text(i, p_row_y, f'p={p:.3f}', ha='center', va='top',
                fontsize=12, color='dimgray')

    ax.axhline(0, color='gray', lw=0.8, ls='--')
    ax.set_yticks([-40, -20, 0, 20])
    ax.set_xticks(range(len(gains)))
    ax.set_xticklabels(snames, rotation=60, ha='right', fontsize=11)
    ax.set_ylabel('Latency gain vs MPC (%)', fontsize=12, labelpad=7)
    ax.tick_params(labelsize=11)
    ax.tick_params(axis='x', labelsize=11)
    # set_paper_style enables axes.grid for BOTH axes; keep only horizontal
    # gridlines (vertical x-gridlines at bar centres would cross the value
    # and p labels and fail the rendered collision audit).
    ax.grid(True, axis='y', alpha=0.3)
    ax.grid(False, axis='x')

    plt.tight_layout()
    save_figure(fig, "fig_e11_necessity", OUT_DIR)
    print(f"  saved: fig_e11_necessity.pdf/.tiff  (7 scenarios)")


if __name__ == "__main__":
    main()

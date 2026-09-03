# -*- coding: utf-8 -*-
"""fig_e5_pareto.py — E5 Pareto 前沿（2026-09-01 重绘版）.

数据源: results/e5_pareto_v2.npz (rerun_e5_pareto_v2.py 产物):
  蒸馏策略前向 + 注入偏好 ω 的验证器精化闭环, 每 ω 点 5 种子 × 3 episode
  × 100 步, 逐 episode 记录能耗/时延 -> 误差棒。
绘制: (E 均值 ± 标准差, T 均值 ± 标准差) 随 ω_e ∈ {0,0.25,0.5,0.75,1} 的
  能耗-时延权衡曲线, 数据点旁标注 ω_e, 附 E1 的 (HMA-Distill, SAC, DDPG)
  参考点。
"""

import os, sys, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE); ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)
from config import RESULTS_DIR
from visual.palette import set_paper_style, method_color, save_figure

set_paper_style()
OUT_DIR = os.path.join(ROOT, "latex", "figure")
INPUT_NPZ = os.path.join(RESULTS_DIR, "e5_pareto_v2.npz")
E1_NPZ = os.path.join(RESULTS_DIR, "e1_comparison_v2.npz")


def main():
    if not os.path.exists(INPUT_NPZ):
        fig, ax = plt.subplots(figsize=(7.16, 3.0))
        ax.text(0.5, 0.5, "TBD - waiting for E5 data",
                ha='center', va='center', fontsize=17)
        ax.axis('off')
        save_figure(fig, "fig_e5_pareto", OUT_DIR)
        return

    npz = np.load(INPUT_NPZ, allow_pickle=False)
    omegas = [0.0, 0.25, 0.5, 0.75, 1.0]
    E, T, E_std, T_std = [], [], [], []
    for w in omegas:
        key = f"w{w:g}" if f"w{w:g}__energy__vals" in npz.files else f"w{w}"
        e = np.asarray(npz[f"{key}__energy__vals"], dtype=float)
        t = np.asarray(npz[f"{key}__latency__vals"], dtype=float)
        E.append(e.mean()); T.append(t.mean())
        E_std.append(e.std()); T_std.append(t.std())

    fig, ax = plt.subplots(figsize=(7.16, 3.4))
    ax.errorbar(E, T, xerr=E_std, yerr=T_std, fmt='-o', ms=6,
                color=method_color('HMA-Distill'), linewidth=1.6,
                capsize=3, markeredgewidth=0.6, markeredgecolor='black',
                label='HMA (Distill + refiner, $\\omega$ sweep)')
    # 直接标注: 统一居中锚点(ha/va center), 偏移量(pt)按审计几何预留空隙,
    # 标签不与曲线/误差棒/刻度相交 (无 gridline 干扰, 见下)。dy>0 表示上移。
    offset_list = [(27, 10),        # omega_e = 0      (点 426.4,177.4)
                   (36, 16),        # omega_e = 0.25   (点 345.1,173.3)
                   (14, 20),        # omega_e = 0.5    (点 265.0,154.5)
                   (48, 12)]        # omega_e = 0.75,1 (点 93.9,37.9)
    labels = [f'$\\omega_e={w:g}$' for w in omegas[:3]]
    labels.append('$\\omega_e = 0.75, 1$')   # 0.75 与 1.0 数据点几乎重合, 合并标注
    for i, (lab, off) in enumerate(zip(labels, offset_list)):
        ax.annotate(lab, xy=(E[i], T[i]),
                    xytext=off, textcoords='offset points',
                    fontsize=13, color='black',
                    ha='center', va='center')
    if os.path.exists(E1_NPZ):
        z = np.load(E1_NPZ, allow_pickle=False)
        for m, mk, c in (('HMA-Distill', 'D', method_color('HMA-Distill')),
                         ('SAC', 's', method_color('SAC')),
                         ('DDPG', '^', method_color('DDPG'))):
            try:
                em = float(np.asarray(z[f"{m}__energy__mean"])[0])
                tm = float(np.asarray(z[f"{m}__latency__mean"])[0])
                ax.plot(em, tm, marker=mk, ms=7, color=c, mec='black',
                        mew=0.6, linestyle='None', label=f'{m} (E1, $n$=250)')
            except Exception:
                pass
    ax.set_xlabel(r'Energy $E_{\mathrm{total}}$ (kJ)', fontsize=15)
    ax.set_ylabel(r'Latency $\bar{T}$ (s)', fontsize=15)
    # 无网格线: 图例与直接标注位于数据区, 保留 grid 会让 0.4 pt 灰色网格线
    # 穿过这些文字并触发 text-stroke 审计失败 (网格为纯装饰, 已去除)。
    ax.grid(False)
    ax.legend(loc='upper right', fontsize=13.5, frameon=False,
              handlelength=1.6, borderaxespad=0.4)
    plt.tight_layout()
    save_figure(fig, "fig_e5_pareto", OUT_DIR)
    print("  saved: fig_e5_pareto.pdf/.tiff")


if __name__ == "__main__":
    main()

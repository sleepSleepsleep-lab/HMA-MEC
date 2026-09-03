# -*- coding: utf-8 -*-
"""
E21 显著性检验 (2026-08)
=======================
HMA-Distill (E1, n=250 per-episode) vs 精化后四基线 (E21, n=250 per-episode):
  - 跨方法比较: Mann-Whitney U (非配对, 独立样本)
  - 方法内 raw vs refined: Wilcoxon 符号秩 (配对, 同 seed 同 episode)
指标: suc (成功率), T (时延), E (能耗), sla (优先级 SLA)
"""
import json, os, sys
import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
ROOT = os.path.dirname(SRC)
RESULTS = os.path.join(ROOT, "results")

npz = np.load(os.path.join(RESULTS, "e1_comparison.npz"), allow_pickle=True)
e21 = json.load(open(os.path.join(RESULTS, "e21_refined_baselines_full.json")))

METRICS = {
    "suc": "success_rate",
    "T": "latency",
    "E": "energy",
    "sla": "priority_sla",
}


def hma_vals(metric):
    key = "HMA-Distill__{}__vals".format(METRICS[metric])
    return np.asarray(npz[key], dtype=float)


def base_vals(name, mode, metric):
    recs = e21[name]["per_episode"][mode]
    return np.asarray([r[metric] for r in recs], dtype=float)


def fmt_p(p):
    if p < 1e-4:
        return "p<1e-4"
    return "p=%.4f" % p


print("=" * 78)
print("  (A) 跨方法: HMA-Distill (E1) vs 精化后基线 (E21) — Mann-Whitney U (n=250 vs 250)")
print("=" * 78)
for metric, label in [("suc", "成功率"), ("T", "时延"), ("E", "能耗"), ("sla", "SLA")]:
    hma = hma_vals(metric)
    print("  --- %s ---" % label)
    for name in ["SAC", "DDPG", "DQN", "MADDPG"]:
        b = base_vals(name, "refined", metric)
        u, p = mannwhitneyu(hma, b, alternative="two-sided")
        print("    HMA vs %-7s refined: HMA=%.4f  %s=%.4f  U=%.0f  %s  %s"
              % (name, hma.mean(), name, b.mean(), u, fmt_p(p),
                 "HMA更优" if hma.mean() > b.mean() else "基线更优"))

print()
print("=" * 78)
print("  (B) 方法内: 原始策略 vs 验证器精化 (配对, 同 seed 同 episode) — Wilcoxon")
print("=" * 78)
for metric, label in [("suc", "成功率"), ("T", "时延"), ("E", "能耗"), ("sla", "SLA")]:
    print("  --- %s ---" % label)
    for name in ["SAC", "DDPG", "DQN", "MADDPG"]:
        raw = base_vals(name, "raw", metric)
        ref = base_vals(name, "refined", metric)
        w, p = wilcoxon(raw, ref, alternative="two-sided")
        print("    %-7s raw=%.4f refined=%.4f Δ=%.4f  W=%.0f  %s"
              % (name, raw.mean(), ref.mean(), ref.mean() - raw.mean(), w, fmt_p(p)))

print()
print("=" * 78)
print("  (C) 精化后基线两两 (Mann-Whitney): 四基线之间是否同簇")
print("=" * 78)
names = ["SAC", "DDPG", "DQN", "MADDPG"]
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        a = base_vals(names[i], "refined", "suc")
        b = base_vals(names[j], "refined", "suc")
        u, p = mannwhitneyu(a, b, alternative="two-sided")
        print("    %s vs %s (refined, suc): %.4f vs %.4f  U=%.0f  %s"
              % (names[i], names[j], a.mean(), b.mean(), u, fmt_p(p)))

print()
print("=" * 78)
print("  (D) 种子质量对照: MPC (E1, 搜索式种子) vs 精化后基线 (E21) — Mann-Whitney")
print("=" * 78)
for metric, label in [("suc", "成功率"), ("T", "时延"), ("E", "能耗")]:
    mpc = np.asarray(npz["MPC__{}__vals".format(METRICS[metric])], dtype=float)
    print("  --- %s ---" % label)
    for name in names:
        b = base_vals(name, "refined", metric)
        u, p = mannwhitneyu(mpc, b, alternative="two-sided")
        print("    MPC vs %-7s refined: MPC=%.4f  %s=%.4f  U=%.0f  %s  %s"
              % (name, mpc.mean(), name, b.mean(), u, fmt_p(p),
                 "MPC更优" if mpc.mean() > b.mean() else "基线更优"))
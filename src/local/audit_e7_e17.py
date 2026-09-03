# -*- coding: utf-8 -*-
"""全量实验结果审计 第二部分: E7-E17 + 统计检验 + E6 细粒度."""
import json, os, sys
import numpy as np

R = 'results'
PROBLEMS = []


def load(name):
    return json.load(open(os.path.join(R, name)))


def chk(cond, msg):
    tag = "OK " if cond else "!! "
    print(f"  [{tag}] {msg}")
    if not cond:
        PROBLEMS.append(msg)


# ============ E6 细粒度 ============
print("=" * 78)
print("E6 鲁棒性: 场景对比 (扰动后窗口)")
print("=" * 78)
d = load('e6_robust.json')
first = list(d.keys())[0]
meths = [k for k in d[first] if isinstance(d[first][k], list)]
print(f"  场景: {list(d.keys())}, 方法: {meths}")
for sc in d:
    for m in ['HMA', 'MPC', 'SAC', 'Greedy', 'AllEdge']:
        if m not in d[sc]:
            continue
        T = np.asarray(d[sc][m]['latency'])
        S = np.asarray(d[sc][m]['success_rate'])
        E = np.asarray(d[sc][m]['energy'])
        # 取后 1/3 (扰动生效窗口) 均值
        w = max(1, len(T) // 3)
        print(f"  {sc:14s} {m:8s} T后段={T[-w:].mean():.3f} suc后段={S[-w:].mean():.3f} "
              f"E后段={E[-w:].mean():.3f} (len={len(T)})")

# ============ E7 灵敏度 ============
print("=" * 78)
print("E7 灵敏度 (e7_sensitivity.json)")
print("=" * 78)
d = load('e7_sensitivity.json')
print("  ", json.dumps(d, ensure_ascii=False)[:800])

# ============ E8 蒸馏规模 ============
print("=" * 78)
print("E8 蒸馏规模 (e8_distill_size.json): 断言")
print("=" * 78)
d = load('e8_distill_size.json')
print("  ", json.dumps(d, ensure_ascii=False)[:600])

# ============ E9 多后端 ============
print("=" * 78)
print("E9 多后端 (e9_multi_llm_results.json + e9_llm_backend.json)")
print("=" * 78)
d = load('e9_multi_llm_results.json')
print("  ", json.dumps(d, ensure_ascii=False)[:700])
d = load('e9_llm_backend.json')
print("  ", json.dumps(d, ensure_ascii=False)[:400])

# ============ E10 对齐 ============
print("=" * 78)
print("E10 对齐 (e10_alignment_v2.json): 断言")
print("=" * 78)
d = load('e10_alignment_v2.json')
flat = {}
def rec(x, p=''):
    if isinstance(x, dict):
        for k, v in x.items():
            rec(v, p + k + '.')
    else:
        flat[p[:-1]] = x
rec(d)
for k in ['raw_align', 'refined_align', 'j_diff']:
    for kk, vv in flat.items():
        if k in kk:
            print(f"    {kk} = {vv}")
chk(any('0.75' in str(v) or (0.7 < v < 0.8) for v in flat.values() if isinstance(v, float)),
    "存在 raw_align≈0.755")
chk(any(0.85 < v < 0.90 for v in flat.values() if isinstance(v, float)),
    "存在 refined_align≈0.875")

# ============ E11 逆境价值 ============
print("=" * 78)
print("E11 逆境 (e1_llm_necessity_v2.json): 断言")
print("=" * 78)
d = load('e1_llm_necessity_v2.json')
print("  顶层键:", list(d.keys()))
for sc, v in d.items():
    if isinstance(v, dict):
        print(f"  {sc}: {json.dumps(v, ensure_ascii=False)[:250]}")
    else:
        print(f"  {sc}: {v}")

# ============ E12 ============
print("=" * 78)
print("E12 (e12_llm_pref_ga.json)")
print("=" * 78)
d = load('e12_llm_pref_ga.json')
print("  ", json.dumps(d, ensure_ascii=False)[:900])

# ============ E14 ============
print("=" * 78)
print("E14 云端 (e14_cloud_edge.json)")
print("=" * 78)
d = load('e14_cloud_edge.json')
print("  ", json.dumps(d, ensure_ascii=False)[:600])

# ============ E15 ============
print("=" * 78)
print("E15 (e15_debate_vs_single.json)")
print("=" * 78)
d = load('e15_debate_vs_single.json')
print("  ", json.dumps(d, ensure_ascii=False)[:1200])

# ============ E16 ============
print("=" * 78)
print("E16 (e16_seed_quality_n60.json + e16_pairwise_stats.json): 断言")
print("=" * 78)
d = load('e16_seed_quality_n60.json')
for g, a in d.items():
    print(f"  {g:12s} infeas={a['infeasible_frac']:.1%} J={a['mean_J']:.3f} "
          f"suc={a['mean_suc']:.1%} T={a['mean_T']:.3f} E={a['mean_E']:.3f} sla={a['mean_sla']:.1%}")
d = load('e16_pairwise_stats.json')
print("  配对", json.dumps(d, ensure_ascii=False)[:600])

# ============ E17 ============
print("=" * 78)
print("E17 (e17_feedback_v2.json): 断言")
print("=" * 78)
d = load('e17_feedback_v2.json')
fb, db, s1 = d['DebateFB'], d['Debate'], d['SingleLLM1']
chk(fb['mean_T'] < db['mean_T'] * 0.95,
    f"FB 时延 {fb['mean_T']:.3f} < Debate {db['mean_T']:.3f} * 0.95 (论文 -10.2%)")
chk(fb['infeasible_frac'] < db['infeasible_frac'] - 0.05,
    f"FB 拒绝率 {fb['infeasible_frac']:.1%} < Debate {db['infeasible_frac']:.1%} -5pp")
chk(db['mean_J'] < s1['mean_J'] + 0.05 and db['mean_J'] > s1['mean_J'] - 0.05,
    f"Debate vs SingleLLM1 J 持平 ({db['mean_J']:.3f} vs {s1['mean_J']:.3f})")
chk(fb['mean_E'] > db['mean_E'],
    f"FB 能耗 {fb['mean_E']:.3f} > Debate {db['mean_E']:.3f} (代价方向)")

# ============ 统计检验声明复算 ============
print("=" * 78)
print("关键统计检验复算 (Wilcoxon, 与论文声明一致?)")
print("=" * 78)
from scipy.stats import wilcoxon, mannwhitneyu

# E1: HMA-Distill vs GA 时延 (e1_comparison.npz? 用 per-step/episode vals?)
z = np.load(os.path.join(R, 'e1_comparison.npz'), allow_pickle=True)
keys = [k for k in z.keys() if 'HMA-Distill' in k]
print("  e1_comparison.npz 键样例:", keys[:8])

print()
print("=" * 78)
print(f"发现问题数: {len(PROBLEMS)}")
print("=" * 78)
for p in PROBLEMS:
    print("  !!", p)
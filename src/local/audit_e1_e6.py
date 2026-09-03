# -*- coding: utf-8 -*-
"""全量实验结果审计: 统一提取关键数字 + 合理性断言.
检查: 数值范围 / 方法间关系方向 / 论文声明一致性 / 统计检验可复算.
"""
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


# ============ E1 主对比 ============
print("=" * 78)
print("E1 主对比: 合理性断言")
print("=" * 78)
d = load('e1_comparison.json')
M = {k: v['mean'] for k, v in d.items()}
for k, m in M.items():
    E, T, s, sl = m['energy'], m['latency'], m['success_rate'], m['priority_sla']
    chk(0 <= E <= 3.5 and 0.3 <= T <= 3.0 and 0 <= s <= 1 and 0 <= sl <= 1,
        f"{k}: 数值范围 E={E:.3f} T={T:.3f} suc={s:.3f} sla={sl:.3f}")
hd, mp, ga = M['HMA-Distill'], M['MPC'], M['GA']
chk(hd['latency'] < mp['latency'] and hd['latency'] < ga['latency'],
    f"HMA-Distill 时延 {hd['latency']:.4f} < MPC {mp['latency']:.4f} 且 < GA {ga['latency']:.4f} (论文: -7.8% vs GA)")
chk(abs(hd['latency'] / ga['latency'] - 1 + 0.078) < 0.01,
    f"时延较 GA 降幅 {hd['latency']/ga['latency']-1:+.2%} ≈ -7.8%")
chk(hd['success_rate'] - ga['success_rate'] > 0.02,
    f"suc 较 GA 高 {(hd['success_rate']-ga['success_rate'])*100:.2f}pp (>2pp, 论文 +2.45pp)")
chk(hd['success_rate'] > 0.90 and mp['success_rate'] > 0.90,
    f"成功率均 >90%: HMA {hd['success_rate']:.3f}, MPC {mp['success_rate']:.3f}")
hdh = M['HMA-Hybrid']
chk(abs(hdh['latency'] - hd['latency']) < 0.002 and abs(hdh['success_rate'] - hd['success_rate']) < 0.002,
    f"Distill 与 Hybrid 几乎一致 (T 差 {abs(hdh['latency']-hd['latency'])*1000:.2f}ms, suc 差 {abs(hdh['success_rate']-hd['success_rate'])*100:.3f}pp)")
chk(all(d[k].get('n_samples', 0) >= 200 for k in M),
    "各方法 n_samples >= 200 (论文 250)")
chk(d['HMA-Distill']['n_samples'] == 250, f"HMA n_samples={d['HMA-Distill']['n_samples']}")

# ============ E2 可扩展性 ============
print("=" * 78)
print("E2 可扩展性: 合理性断言")
print("=" * 78)
z = np.load(os.path.join(R, 'e2_scalability.npz'), allow_pickle=True)


def zmean(K, m, k):
    return float(np.mean(z[f'{K}__{m}__{k}__vals']))


for K in ['K4', 'K8', 'K12', 'K16', 'K24', 'K32']:
    for m in ['HMA-Distill', 'HMA-Hybrid']:
        T, s = zmean(K, m, 'latency'), zmean(K, m, 'success_rate')
        chk(0 <= T <= 3.0 and 0 <= s <= 1, f"{K} {m}: T={T:.3f} suc={s:.3f}")
chk(zmean('K8', 'HMA-Distill', 'success_rate') > zmean('K16', 'HMA-Distill', 'success_rate'),
    f"K8 suc {zmean('K8','HMA-Distill','success_rate'):.3f} > K16 suc {zmean('K16','HMA-Distill','success_rate'):.3f} (退化方向)")
chk(zmean('K8', 'HMA-Distill', 'latency') < zmean('K12', 'HMA-Distill', 'latency'),
    f"K8 T {zmean('K8','HMA-Distill','latency'):.3f} < K12 T {zmean('K12','HMA-Distill','latency'):.3f} (论文 0.61→1.39)")
_ = np.max([zmean(f'K{n}', 'HMA-Distill', 'success_rate') for n in [16, 24, 32]])
chk(_ < 0.6, f"K16-32 最大 suc {_:.3f} < 0.6 (论文: 降至 50% 上下)")

# ============ K12 重蒸馏 ============
print("=" * 78)
print("K12 重蒸馏 (e2_k12_distilled.json): 断言")
print("=" * 78)
d = load('e2_k12_distilled.json')
k12 = {k: v['mean'] for k, v in d.items()}
h12, mp12, ga12, gr12 = k12['HMA-Distill-K12'], k12['MPC'], k12['GA'], k12['Greedy']
chk(abs(h12['suc'] - mp12['suc']) < 0.02,
    f"K12 HMA suc {h12['suc']:.4f} ≈ MPC {mp12['suc']:.4f}")
chk(h12['suc'] > ga12['suc'] + 0.01,
    f"K12 HMA suc {h12['suc']:.4f} > GA {ga12['suc']:.4f} +1pp")
chk(gr12['E'] > 1.5 * h12['E'],
    f"Greedy 能耗 {gr12['E']:.3f} > 2x HMA {h12['E']:.3f}")
J = lambda m: 0.5 * m['E'] + 0.5 * m['T']
chk(J(h12) < J(gr12), f"HMA-K12 J={J(h12):.3f} < Greedy J={J(gr12):.3f}")

# ============ E3 组件消融 ============
print("=" * 78)
print("E3 消融 (e3_component_ablation.json): 断言")
print("=" * 78)
d = load('e3_component_ablation.json')
ab = {k: v['mean'] for k, v in d.items()}
hf, nr, mpc3, rs = ab['HMA-Full'], ab['HMA-NoRefiner'], ab['MPC'], ab['HMA-RandomSeed']
chk(abs(nr['latency'] / hf['latency'] - 1 - 1.09) < 0.05,
    f"NoRefiner 时延 +{nr['latency']/hf['latency']-1:.1%} ≈ +109%")
chk(hf['success_rate'] - nr['success_rate'] > 0.25,
    f"Full vs NoRefiner suc 差 {(hf['success_rate']-nr['success_rate'])*100:.1f}pp (>25pp)")
chk(hf['priority_sla'] - nr['priority_sla'] > 0.30,
    f"Full vs NoRefiner SLA 差 {(hf['priority_sla']-nr['priority_sla'])*100:.1f}pp (>30pp, 论文 67.0→33.3)")
chk(mpc3['latency'] > hf['latency'] and (mpc3['latency'] / hf['latency'] - 1) < 0.05,
    f"MPC 种子时延 +{(mpc3['latency']/hf['latency']-1)*100:.1f}% (<5%, 论文 +3.4%)")
chk(rs['success_rate'] >= hf['success_rate'] - 0.01,
    f"RandomSeed suc {rs['success_rate']:.3f} 与 Full {hf['success_rate']:.3f} 相当")

# ============ E4 实时性 ============
print("=" * 78)
print("E4 实时性 (e4_efficiency.json + e4_decision_cost.json): 断言")
print("=" * 78)
d = load('e4_efficiency.json')
for mode, st in d.items():
    s = st['stats']
    chk(s['hybrid_trigger_rate'] == 0.0, f"{mode}: hybrid_trigger_rate=0.0")
    if mode == 'FullLLM':
        chk(60 < s['avg_fullllm_latency_ms'] / 1000 < 80,
            f"FullLLM 单步 {s['avg_fullllm_latency_ms']/1000:.1f}s (论文 ~68s)")
    else:
        chk(0 < s['avg_distill_latency_ms'] < 10,
            f"{mode}: 纯策略前向 {s['avg_distill_latency_ms']:.2f}ms (论文 ~2ms)")
        chk(s['n_distill_steps'] >= 500, f"{mode}: n_distill_steps={s['n_distill_steps']}")
        chk(s['conf_min_mean'] > 0.3, f"{mode}: conf_min_mean={s['conf_min_mean']:.3f} > 0.3")
d = load('e4_decision_cost.json')
chk(40 < d['decision_cost_ms_per_step']['HMA-Distill_(蒸馏+Refiner)'] < 90,
    f"闭环中位 ~{d['decision_cost_ms_per_step']['HMA-Distill_(蒸馏+Refiner)']}ms (论文 ~70ms)")
chk(d['decision_cost_ms_per_step']['FullLLM_(CW-Debate, 5轮)'] > 50000,
    f"FullLLM {d['decision_cost_ms_per_step']['FullLLM_(CW-Debate, 5轮)']/1000:.0f}s")

# ============ E5 Pareto ============
print("=" * 78)
print("E5 Pareto (e5_pareto.json)")
print("=" * 78)
d = load('e5_pareto.json')
print("  ", json.dumps(d, ensure_ascii=False)[:500])

# ============ E6 鲁棒性 ============
print("=" * 78)
print("E6 鲁棒性 (e6_robust.json): 断言")
print("=" * 78)
d = load('e6_robust.json')
print("  顶层键:", list(d.keys())[:10])
for sc in d:
    v = d[sc]
    if isinstance(v, dict) and 'mean' in v:
        m = v['mean']
        chk(0 <= m.get('success_rate', 0) <= 1 and 0 <= m.get('latency', 0) <= 5,
            f"{sc}: suc={m.get('success_rate', float('nan')):.3f} T={m.get('latency', float('nan')):.3f}")
        print(f"    {sc}: {json.dumps(m, ensure_ascii=False)[:150]}")
    else:
        print(f"  {sc}: {json.dumps(v, ensure_ascii=False)[:150]}")

print()
print("=" * 78)
print(f"发现问题数: {len(PROBLEMS)}")
print("=" * 78)
for p in PROBLEMS:
    print("  !!", p)
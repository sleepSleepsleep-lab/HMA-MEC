# -*- coding: utf-8 -*-
"""
================================================================
E17-v2 (2026-08): 仿真过程反馈是否激活辩论价值
================================================================
E15/E16 显示完整 CW-Debate 相对单次 LLM 无系统性原始质量增益。
本实验检验一个可修复的设计点: 辩论轮次之间缺乏客观仿真证据——
批判与仲裁只基于语言, 不基于仿真。为此在 CW-Debate 协议中注入
"仿真过程反馈" (process_feedback=True): 每一轮合并方案经 VA 反事实
仿真后, 把能耗/时延/成功率/SLA 与判定结果追加到下一轮 LLM 文本,
再进入下一轮提案-批判-仲裁。

对照 (同一批 30 个跨难度状态, 原始输出不经蒸馏精化, 反事实评估):
  A) CW-Debate                 (65 调/步, 原协议)
  B) CW-Debate + 仿真过程反馈   (65 调/步, 协议增强)
  C) 单次 LLM 直接决策           (1 调/步, 外部锚点)
指标与 E16 完全一致: VA 硬罚拒绝率 (suc<0.5 或 sla<0.5)、原始成功率、
时延、SLA、统一目标 J (+10 罚)。配对 Wilcoxon 检验 A vs B。

若 B 显著优于 A -> "客观仿真反馈激活辩论" 被直接证明, 作为框架增强写入论文;
若仍持平 -> 如实报告: 在本基准状态分布下, 辩论 (含反馈) 与单 LLM 等价,
论证辩论价值的证据确实不在原始方案质量层面。

依赖: 本地 vLLM。30 状态 × ~131 调用 ≈ 70 分钟。
================================================================
"""
import os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)
sys.path.insert(0, os.path.join(SRC, "local"))

from config import NUM_USERS, NUM_EDGE_SERVERS, SEED, RESULTS_DIR
from environment import MECEnvironment
from agent_define import make_agents
from cw_debate import cw_debate
from local.results_store import Recorder

K, M = NUM_USERS, NUM_EDGE_SERVERS
N_STATES = 30
OUT_JSON = os.path.join(RESULTS_DIR, "e17_feedback_v2.json")
RAW_JSON = os.path.join(RESULTS_DIR, "e17_feedback_v2_raw.json")

SYSTEM_PLAN_CN = (
    "你是移动边缘计算（MEC）卸载编排器。请根据当前系统状态，为每个用户 k 决定 "
    "本地执行比例 alpha_k（0=全部卸载到边缘，1=全部本地执行；时延约束紧的任务 "
    "应更多卸载）与目标服务器 server_k（取值 0..%d）。只输出 JSON："
    '{"alpha": [8 个 0~1 数字], "server": [8 个 0~%d 整数]}。' % (M - 1, M - 1))


def make_state(sd, deadline_scale):
    """构造跨难度状态: deadline 缩放产生易错/常规两类 (与 E16 相同)."""
    env = MECEnvironment(num_users=K, num_servers=M, seed=SEED * 7 + sd)
    env.reset()
    for t in env.tasks:
        t['tau'] *= deadline_scale
    return env


def single_plan(llm, env, n_samples=1):
    """单 LLM 直接决策 (与 E16 相同)."""
    txt = env.state_to_text()
    alphas, servers = [], []
    for _ in range(n_samples):
        r = llm.chat_json(SYSTEM_PLAN_CN, txt)
        a = np.asarray(r.get('alpha', []), dtype=float)
        s = np.asarray(r.get('server', []), dtype=int)
        if len(a) == K and len(s) == K:
            alphas.append(np.clip(a, 0.01, 1.0))
            servers.append(np.clip(s, 0, M - 1).astype(int))
        else:
            return None, None
    if not alphas:
        return None, None
    A = np.mean(alphas, axis=0)
    sd = np.stack(servers)
    S = np.array([np.bincount(sd[:, k], minlength=M).argmax() for k in range(K)])
    return A, S


def debate_plan(env, llm, process_feedback=False):
    """CW-Debate 完整五轮 (可选仿真过程反馈), 输出原始方案 (不经精化)."""
    agents = make_agents(env, with_va=True)
    out = cw_debate(env, agents, mode="FullLLM", llm=llm,
                    verbose=False, process_feedback=process_feedback)
    plan = out['plan']
    return np.asarray(plan['alpha'], dtype=float), \
           np.asarray(plan['server'], dtype=int)


def eval_plan(env, alpha, server):
    """反事实评估 + VA 硬罚 (与 E16 相同)."""
    sim = env.simulate({'alpha': alpha, 'server': server})
    E, T, suc, sla = sim['energy'], sim['latency'], sim['success_rate'], sim['priority_sla']
    J = 0.5 * E + 0.5 * T
    if suc < 0.5 or sla < 0.5:
        J += 10.0
    return dict(J=J, E=E, T=T, suc=suc, sla=sla,
                infeasible=int(suc < 0.5 or sla < 0.5))


def wilcoxon(a, b):
    """配对 Wilcoxon 符号秩检验; 返回 (stat, p); 全相等时 p=None."""
    from scipy.stats import wilcoxon
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    d = d[~np.isnan(d)]
    if len(d) == 0 or np.allclose(d, 0.0):
        return None, None
    try:
        return wilcoxon(d, zero_method="wilcox")
    except ValueError:
        return None, None


def main():
    from llm_client import get_llm_client
    llm = get_llm_client()
    print("=" * 64)
    print("  E17-v2: 仿真过程反馈是否激活辩论价值 (30 状态)")
    print("=" * 64)
    gen = {
        "Debate":  lambda env: debate_plan(env, llm, process_feedback=False),
        "DebateFB": lambda env: debate_plan(env, llm, process_feedback=True),
        "SingleLLM1": lambda env: single_plan(llm, env, 1),
    }
    recs = {g: [] for g in gen}
    n_fail = {g: 0 for g in gen}
    stats = {}          # 逐状态配对记录, 供 Wilcoxon
    per_state = []      # 逐状态原始记录 (含 scale, 供按难度对齐的配对分析)
    for g in gen:
        stats[g] = {'J': [], 'T': [], 'suc': [], 'sla': [], 'E': []}
    _rec = Recorder("e17")
    for i in range(N_STATES):
        scale = 0.3 + 0.7 * (i / max(N_STATES - 1, 1))
        env = make_state(i, scale)
        line = f"  [state{i}] scale={scale:.2f}"
        ps = {'scale': scale}
        for gname, fn in gen.items():
            t0 = time.time()
            try:
                a, s = fn(env)
            except Exception as ex:
                a, s = None, None
                print(f"    [{gname}] 异常: {ex}")
            dt = time.time() - t0
            if a is None:
                n_fail[gname] += 1
                print(f"    [{gname}] 解析失败 {dt:.0f}s")
                ps[gname] = None
                continue
            r = eval_plan(env, a, s)
            recs[gname].append(r)
            _rec.add(method=gname, seed=i, episode=None, metrics=r)
            for k in stats[gname]:
                stats[gname][k].append(r[k])
            ps[gname] = {kk: float(r[kk]) for kk in ['J', 'E', 'T', 'suc', 'sla']}
            line += (f" | {gname} J={r['J']:.2f} suc={r['suc']:.0%} "
                     f"T={r['T']:.2f} inf={r['infeasible']}")
        per_state.append(ps)
        print(line, flush=True)
    _rec.close()
    # ---- 聚合 ----
    out = {}
    print("=" * 64)
    for gname in gen:
        R = recs[gname]
        if not R:
            out[gname] = {"n_parsed": 0}
            print(f"  [{gname}] 全部解析失败!")
            continue
        agg = dict(
            n_parsed=len(R),
            n_parse_fail=n_fail[gname],
            infeasible_frac=float(np.mean([r['infeasible'] for r in R])),
            mean_J=float(np.mean([r['J'] for r in R])),
            mean_suc=float(np.mean([r['suc'] for r in R])),
            mean_T=float(np.mean([r['T'] for r in R])),
            mean_sla=float(np.mean([r['sla'] for r in R])),
            mean_E=float(np.mean([r['E'] for r in R])),
        )
        out[gname] = agg
        print(f"  => [{gname}] 解析失败={agg['n_parse_fail']}/{N_STATES} "
              f"VA硬罚拒绝={agg['infeasible_frac']:.1%} "
              f"原始J={agg['mean_J']:.3f} suc={agg['mean_suc']:.1%} "
              f"T={agg['mean_T']:.3f}s sla={agg['mean_sla']:.1%}")
    # ---- 配对检验: Debate vs DebateFB ----
    if len(recs['Debate']) == len(recs['DebateFB']) and recs['Debate']:
        print("=" * 64)
        for k in ['J', 'T', 'suc']:
            st, p = wilcoxon(stats['Debate'][k], stats['DebateFB'][k])
            md = float(np.mean(stats['Debate'][k]) - np.mean(stats['DebateFB'][k]))
            print(f"  Wilcoxon Debate-vs-DebateFB [{k}]: mean_diff="
                  f"{md:+.4f} stat={st} p={p if p is None else f'{p:.4f}'}")
        for k in ['J', 'T', 'suc']:
            st, p = wilcoxon(stats['Debate'][k], stats['SingleLLM1'][k])
            md = float(np.mean(stats['Debate'][k]) - np.mean(stats['SingleLLM1'][k]))
            print(f"  Wilcoxon Debate-vs-SingleLLM1 [{k}]: mean_diff="
                  f"{md:+.4f} stat={st} p={p if p is None else f'{p:.4f}'}")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    json.dump(out, open(OUT_JSON, "w"), indent=2, ensure_ascii=False)
    json.dump(per_state, open(RAW_JSON, "w"), indent=2, ensure_ascii=False)
    print(f"\n  保存 -> {OUT_JSON} (+ {RAW_JSON} 逐状态)")


if __name__ == "__main__":
    main()
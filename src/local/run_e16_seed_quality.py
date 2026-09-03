# -*- coding: utf-8 -*-
"""
================================================================
E16 (2026-08): 种子原始质量 —— 辩论 vs 单次/多次 LLM（不经精化）
================================================================
E15 中精化器把种子差异压缩殆尽, 测的是"精化后残留差异"。
本实验直接测量【原始输出质量】(不经 PlanRefiner):
  构造 24 个跨难度状态 (deadline 缩放 0.3~1.0, 制造 LLM 易错状态),
  对每个状态由三种生成器直接输出卸载方案:
    A) CW-Debate 完整五轮辩论      (65 次 LLM 调用)
    B) 单次 LLM 直接决策            (1 次调用)
    C) 单次 LLM × 5 采样多数投票     (5 次调用, 自洽性对照)
  用 env.simulate 评估原始方案:
    - VA 硬罚拒绝率 (suc<0.5 或 sla<0.5 的占比, 对应物理不可行)
    - 原始成功率 / 时延 / SLA / 统一目标 J
  若辩论的原始质量显著优于单 LLM -> 辩论的纠错/方案增益被直接证明;
  若持平 -> 诚实报告"在本基准状态分布下辩论无额外增益"。

依赖: 本地 vLLM。24 状态 × ~71 调用 ≈ 28 分钟。
================================================================
"""
import os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)
sys.path.insert(0, os.path.join(SRC, "local"))

from config import NUM_USERS, NUM_EDGE_SERVERS, SEED, RESULTS_DIR
from local.results_store import Recorder
from environment import MECEnvironment
from agent_runner import HMAAgentRunner

K, M = NUM_USERS, NUM_EDGE_SERVERS
N_STATES = 60
OUT_JSON = os.path.join(RESULTS_DIR, "e16_seed_quality_n60.json")

SYSTEM_PLAN_CN = (
    "你是移动边缘计算（MEC）卸载编排器。请根据当前系统状态，为每个用户 k 决定 "
    "本地执行比例 alpha_k（0=全部卸载到边缘，1=全部本地执行；时延约束紧的任务 "
    "应更多卸载）与目标服务器 server_k（取值 0..%d）。只输出 JSON："
    '{"alpha": [8 个 0~1 数字], "server": [8 个 0~%d 整数]}。' % (M - 1, M - 1))


def make_state(sd, deadline_scale):
    """构造跨难度状态: deadline 缩放产生易错/常规两类."""
    env = MECEnvironment(num_users=K, num_servers=M, seed=SEED * 7 + sd)
    env.reset()
    for t in env.tasks:
        t['tau'] *= deadline_scale
    return env


def single_plan(llm, env, n_samples=1):
    """单 LLM 直接决策; n_samples>1 时多次采样投票 (server 多数票, alpha 平均)."""
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
            # 单次解析失败时记录为无效
            return None, None
    if not alphas:
        return None, None
    A = np.mean(alphas, axis=0)
    sd = np.stack(servers)
    S = np.array([np.bincount(sd[:, k], minlength=M).argmax() for k in range(K)])
    return A, S


def debate_plan(env, llm):
    runner = HMAAgentRunner(env=env, mode="FullLLM", llm=llm,
                            agents=None, verbose=False, use_refiner=False)
    out = runner.run_step(state=env._get_state(), agents_reuse=False)
    return np.asarray(out['plan']['alpha'], dtype=float), \
           np.asarray(out['plan']['server'], dtype=int)


def eval_plan(env, alpha, server):
    sim = env.simulate({'alpha': alpha, 'server': server})
    E, T, suc, sla = sim['energy'], sim['latency'], sim['success_rate'], sim['priority_sla']
    J = 0.5 * E + 0.5 * T
    if suc < 0.5 or sla < 0.5:
        J += 10.0
    return dict(J=J, E=E, T=T, suc=suc, sla=sla,
                infeasible=int(suc < 0.5 or sla < 0.5))


def main():
    from llm_client import get_llm_client
    llm = get_llm_client()
    print("=" * 64)
    print("  E16: 种子原始质量（不经精化）—— 辩论 vs 单LLM×1 vs 单LLM×5")
    print("=" * 64)
    gen = {"Debate": debate_plan, "SingleLLM1": lambda env: single_plan(llm, env, 1),
           "SingleLLM5": lambda env: single_plan(llm, env, 5)}
    out = {}
    _rec = Recorder("e16")
    for gname in gen:
        recs = []
        n_parse_fail = 0
        for i in range(N_STATES):
            scale = 0.3 + 0.7 * (i / max(N_STATES - 1, 1))   # 0.3 -> 1.0 递增难度
            env = make_state(i, scale)
            t0 = time.time()
            try:
                if gname == "Debate":
                    a, s = gen[gname](env, llm)
                else:
                    a, s = gen[gname](env)
            except Exception as ex:
                a, s = None, None
                print(f"    [{gname} state{i}] 异常 {ex}")
            dt = time.time() - t0
            if a is None:
                n_parse_fail += 1
                print(f"    [{gname} state{i}] 解析失败 scale={scale:.2f}")
                continue
            r = eval_plan(env, a, s)
            recs.append(r)
            _rec.add(method=gname, seed=None, episode=i, metrics=r)
            print(f"    [{gname} state{i}] scale={scale:.2f} {dt:.0f}s "
                  f"J={r['J']:.3f} suc={r['suc']:.0%} T={r['T']:.3f} "
                  f"inf={r['infeasible']}")
        agg = dict(
            n_parsed=len(recs),
            n_parse_fail=n_parse_fail,
            infeasible_frac=float(np.mean([r['infeasible'] for r in recs])),
            mean_J=float(np.mean([r['J'] for r in recs])),
            mean_suc=float(np.mean([r['suc'] for r in recs])),
            mean_T=float(np.mean([r['T'] for r in recs])),
            mean_sla=float(np.mean([r['sla'] for r in recs])),
            mean_E=float(np.mean([r['E'] for r in recs])),
        )
        out[gname] = agg
        print(f"  => [{gname}] 失败率={agg['n_parse_fail']}/{N_STATES} "
              f"不可行={agg['infeasible_frac']:.1%} "
              f"原始J={agg['mean_J']:.3f} suc={agg['mean_suc']:.1%} "
              f"T={agg['mean_T']:.3f}s")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    json.dump(out, open(OUT_JSON, "w"), indent=2, ensure_ascii=False)
    print(f"\n  保存 -> {OUT_JSON}")


if __name__ == "__main__":
    main()
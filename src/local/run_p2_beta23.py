# -*- coding: utf-8 -*-
"""
P2-5 (2026-08): β₂/β₃ 置信度灵敏度系数扫描 (E7 扩展)
================================================================
三维置信度模型 c_k = σ(β1·Δ + β2·SINR + β3·(1-ρ)) 中, E7 只扫了 β1。
本实验以 FullLLM 辩论模式 (vLLM) 扫描 β₂ ∈ {0.5, 1.0, 1.5} 与
β₃ ∈ {1.0, 1.5, 2.0} (各一档 3 值), 每档 2 种子 x 20 步,
观察时延/成功率/SLA 变化, 确认平台区。
输出: results/p2_beta23.json
================================================================
"""
import os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)
sys.path.insert(0, os.path.join(SRC, "local"))

from config import (NUM_USERS, NUM_EDGE_SERVERS, SEED, RESULTS_DIR)
from environment import MECEnvironment
from local.plan_refiner import PlanRefiner
from local.experiment_common import compose_action
from local.results_store import Recorder

K, M = NUM_USERS, NUM_EDGE_SERVERS
N_SEEDS = 2
N_STEPS = 20
OUT_JSON = os.path.join(RESULTS_DIR, "p2_beta23.json")

B2_SWEEP = [0.5, 1.0, 1.5]
B3_SWEEP = [1.0, 1.5, 2.0]


def run_one(b2, b3, sd):
    """FullLLM 辩论 + 精化, 修改置信度模型的 beta2/beta3 (config 常量)."""
    from cw_debate import cw_debate
    from agent_define import make_agents
    from llm_client import get_llm_client
    import config as _cfg

    llm = get_llm_client()
    env = MECEnvironment(num_users=K, num_servers=M, seed=SEED + sd)
    env.reset()
    agents = make_agents(env)
    refiner = PlanRefiner(omega=(0.5, 0.5))
    # 临时覆盖置信度灵敏度系数 (cw_debate 运行期读取 config 常量)
    _cfg.CONFIDENCE_BETA_SINR = b2
    _cfg.CONFIDENCE_BETA_LOAD = b3
    E, T, S, SL = [], [], [], []
    for i in range(N_STEPS):
        st = env._get_state()
        out = cw_debate(env, agents, mode="FullLLM", llm=llm)
        plan = out['plan']
        a = np.asarray(plan['alpha'], dtype=float)
        s = np.asarray(plan['server'], dtype=int)
        ar, sr = refiner.refine(env, a.copy(), s.copy())
        act = np.zeros(2 * K, np.float32)
        act[0::2] = np.clip(ar, 0.01, 1.0)
        act[1::2] = (np.clip(sr, 0, M - 1) + 0.5) / M
        ns, _, d, info = env.step(act)
        E.append(info['energy']); T.append(info['latency'])
        S.append(info['success_rate']); SL.append(info['priority_sla'])
        if d:
            break
    return dict(E=float(np.mean(E)), T=float(np.mean(T)),
                suc=float(np.mean(S)), sla=float(np.mean(SL)))


def main():
    from llm_client import get_llm_client
    llm = get_llm_client()
    llm._respect_rate_limit = lambda: None   # 本地 vLLM 无速率限制
    print("=" * 60)
    print("  P2-5: β₂/β₃ 灵敏度扫描 (FullLLM 辩论 + 精化, n=%d x %d 步)"
          % (N_SEEDS, N_STEPS))
    print("=" * 60)
    _rec = Recorder("p2_beta23")
    out = {"beta2": {}, "beta3": {}}
    # β₂ 扫描 (β₃ 固定 1.5)
    for b2 in B2_SWEEP:
        rows = [run_one(b2, 1.5, sd) for sd in range(N_SEEDS)]
        mean = {k: float(np.mean([r[k] for r in rows]))
                for k in ('E', 'T', 'suc', 'sla')}
        out["beta2"][str(b2)] = mean
        print(f"  β₂={b2}: T={mean['T']:.4f} suc={mean['suc']:.1%} "
              f"sla={mean['sla']:.1%}")
        for sd, r in enumerate(rows):
            _rec.add(method=f"b2={b2}", seed=sd, episode=sd, metrics=r)
    # β₃ 扫描 (β₂ 固定 1.0)
    for b3 in B3_SWEEP:
        rows = [run_one(1.0, b3, sd) for sd in range(N_SEEDS)]
        mean = {k: float(np.mean([r[k] for r in rows]))
                for k in ('E', 'T', 'suc', 'sla')}
        out["beta3"][str(b3)] = mean
        print(f"  β₃={b3}: T={mean['T']:.4f} suc={mean['suc']:.1%} "
              f"sla={mean['sla']:.1%}")
        for sd, r in enumerate(rows):
            _rec.add(method=f"b3={b3}", seed=sd, episode=sd, metrics=r)
    _rec.close()
    json.dump(out, open(OUT_JSON, "w"), indent=2, ensure_ascii=False)
    print(f"  保存 -> {OUT_JSON}")


if __name__ == "__main__":
    main()

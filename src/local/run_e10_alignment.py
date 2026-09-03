# -*- coding: utf-8 -*-
"""
================================================================
实验 E10 推理质量分析 (local/run_e10_alignment.py)
================================================================
量化离线蒸馏引入的决策质量损失:
  1. Decision Alignment Rate rho_align : 蒸馏与 FullLLM 的服务器选择一致比例
  2. VA 拒绝率差异 : FullLLM vs Distill 方案被反事实验证器拒绝的比例
  3. alpha MAE       : 卸载比例决策的平均绝对误差

方法: 在同一批环境状态上分别用 FullLLM (CW-Debate) 与 Distill-Agent
做出决策, 逐状态比较。不影响真实环境 (决策取自 copy 状态, 不推进 env)。

依赖: 本地 vLLM 服务 (FullLLM 模式)。耗时: N_STATES × ~60s。
================================================================
"""
import os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)
sys.path.insert(0, os.path.join(SRC, "local"))

from config import NUM_USERS, NUM_EDGE_SERVERS, RESULTS_DIR, SEED, CHECKPOINT_DIR
from environment import MECEnvironment
from agent_runner import HMAAgentRunner
from local.plan_refiner import PlanRefiner
from local.experiment_common import compose_action

K, M = NUM_USERS, NUM_EDGE_SERVERS
N_STATES = 30                 # 对比状态数 (~30 分钟量级, FullLLM 每状态约 60s)
N_STEPS_FILL = 40            # 推进环境得到 N 个不同状态的步数
POLICY_PATH = os.path.join(CHECKPOINT_DIR, "distilled_policy.pth")
OUT_JSON = os.path.join(RESULTS_DIR, "e10_alignment.json")


def main():
    print("=" * 60)
    print(f"  E10 推理质量分析: FullLLM vs Distill 决策对齐 (K={K},M={M}, N={N_STATES})")
    print("=" * 60)

    from llm_client import get_llm_client
    llm = get_llm_client()

    env = MECEnvironment(num_users=K, num_servers=M, seed=SEED)
    env.reset()
    refiner = PlanRefiner(omega=(0.5, 0.5))
    distill_runner = HMAAgentRunner(env=env, mode="Distill",
                                    policy_path=POLICY_PATH, agents=None)
    full_runner = HMAAgentRunner(env=env, mode="FullLLM", llm=llm,
                                 agents=None, verbose=False)

    rows = []
    # 先用 Distill 快速推进环境到多样状态（仅产生状态快照，不影响后续对比）
    states = []
    for _ in range(N_STEPS_FILL):
        st = env._get_state()
        if len(states) < N_STATES:
            states.append(st)
        out = distill_runner.run_step(state=st, agents_reuse=True)
        a = compose_action(out['plan'], K, M)
        env.step(a)

    # 在保存的每个状态上分别进行 FullLLM 与 Distill 决策
    align_hits, align_total = 0, 0
    mae_alpha = []
    full_rej, distill_rej, n_full, n_dist = 0, 0, 0, 0
    for i, st in enumerate(states):
        env._set_state(st) if hasattr(env, '_set_state') else None

        t0 = time.time()
        of = full_runner.run_step(state=st, agents_reuse=False)
        dt_f = time.time() - t0
        od = distill_runner.run_step(state=st, agents_reuse=True)

        pf = of['plan']; pd = od['plan']
        af = np.asarray(pf['alpha']); ad = np.asarray(pd['alpha'])
        sf = np.asarray(pf['server'], dtype=int); sd = np.asarray(pd['server'], dtype=int)
        align_hits += int(np.sum(sf == sd))
        align_total += K
        mae_alpha.append(float(np.mean(np.abs(af - ad))))

        if of.get('va_accept') is not None:
            full_rej += int(not of['va_accept']); n_full += 1
        if od.get('va_accept') is not None:
            distill_rej += int(not od['va_accept']); n_dist += 1
        rows.append(dict(step=i, full=pf, distill=pd))
        print(f"  [state {i}] FullLLM {dt_f:.0f}s "
              f"align={align_hits/align_total:.3f} αMAE={mae_alpha[-1]:.3f}")

    rho_align = align_hits / max(align_total, 1)
    out = {
        'rho_align': rho_align,
        'alpha_mae': float(np.mean(mae_alpha)),
        'alpha_mae_std': float(np.std(mae_alpha)),
        'va_reject_full': full_rej / max(n_full, 1) if n_full else None,
        'va_reject_distill': distill_rej / max(n_dist, 1) if n_dist else None,
        'n_states': len(states),
        'n_full_va': n_full, 'n_distill_va': n_dist,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n  rho_align={rho_align:.4f}  αMAE={out['alpha_mae']:.4f}±{out['alpha_mae_std']:.4f}")
    print(f"  VA reject: FullLLM={out['va_reject_full']}  Distill={out['va_reject_distill']}")
    print(f"  已保存 -> {OUT_JSON}")


if __name__ == "__main__":
    main()

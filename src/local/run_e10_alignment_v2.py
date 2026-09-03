# -*- coding: utf-8 -*-
"""
================================================================
E10 v2 (2026-08 强化): 蒸馏 vs FullLLM 的最终方案一致性 + 性能等价
================================================================
在上一版 "原始决策对齐" (rho_align≈0.50) 基础上补充决定性证据:
  逐状态比较两个语义先验 (LLM 教师 / 蒸馏策略) 经【同一验证器精化】后
  得到的**最终方案**:
    - rho_align_ref : 精化后服务器选择一致率
    - J 等价        : 两个最终方案在统一目标 J=0.5E+0.5T(+硬罚) 下
                     经 env.simulate 的状态级差异分布 -> 性能等价性
结论预设: 语义先验分歧不损害最终性能(验证器主导), 且蒸馏保留连续先验。

依赖: 本地 vLLM (FullLLM 决策)。N_STATES × ~55s。
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
N_STATES = 24
N_STEPS_FILL = 40
POLICY_PATH = os.path.join(CHECKPOINT_DIR, "distilled_policy.pth")
OUT_JSON = os.path.join(RESULTS_DIR, "e10_alignment_v2.json")


def j_cost(env, plan, omega=(0.5, 0.5)):
    sim = env.simulate(plan)
    cost = omega[0] * sim['energy'] + omega[1] * sim['latency']
    if sim['success_rate'] < 0.5 or sim['priority_sla'] < 0.5:
        cost += 10.0
    return cost


def main():
    print("=" * 62)
    print(f"  E10 v2: 精化后方案一致性 + J 等价 (K={K},M={M}, N={N_STATES})")
    print("=" * 62)
    from llm_client import get_llm_client
    llm = get_llm_client()

    env = MECEnvironment(num_users=K, num_servers=M, seed=SEED)
    env.reset()
    refiner = PlanRefiner(omega=(0.5, 0.5))
    distill = HMAAgentRunner(env=env, mode="Distill",
                             policy_path=POLICY_PATH, agents=None,
                             use_refiner=False)
    full = HMAAgentRunner(env=env, mode="FullLLM", llm=llm,
                          agents=None, verbose=False, use_refiner=False)

    # 单遍推进 + 评估: 每个状态上做 FullLLM/Distill 成对决策、
    # 同一 refiner 精化、并在【该状态】反事实评估 J, 随后推进环境。
    # (审计修复 2026-08: 旧版在收集完 24 个状态后未再推进 env,
    #  导致 J 等价评估全部落在同一个末态上, 退化为 1 个有效样本。)
    rows = []
    hit_raw, hit_ref = 0, 0
    mae_raw, mae_ref = [], []
    j_diff = []
    for i in range(N_STATES):
        st = env._get_state()
        t0 = time.time()
        of = full.run_step(state=st, agents_reuse=False)
        od = distill.run_step(state=st, agents_reuse=True)
        dt = time.time() - t0

        pf = of['plan']; pd = od['plan']
        af_r = np.asarray(pf['alpha']); ad_r = np.asarray(pd['alpha'])
        sf_r = np.asarray(pf['server'], dtype=int); sd_r = np.asarray(pd['server'], dtype=int)
        hit_raw += int(np.sum(sf_r == sd_r))

        # 精化: 同一 refiner, 相同候选网格 (env 即当前状态, 反事实一致)
        af2, sf2 = refiner.refine(env, af_r.copy(), sf_r.copy())
        ad2, sd2 = refiner.refine(env, ad_r.copy(), sd_r.copy())
        hit_ref += int(np.sum(sf2 == sd2))
        mae_raw.append(float(np.mean(np.abs(af_r - ad_r))))
        mae_ref.append(float(np.mean(np.abs(af2 - ad2))))

        jf = j_cost(env, {'alpha': af2, 'server': sf2})
        jd = j_cost(env, {'alpha': ad2, 'server': sd2})
        j_diff.append(abs(jf - jd) / max(abs(jf), 1e-9))
        rows.append(dict(step=i, J_full=float(jf), J_distill=float(jd)))
        print(f"  [state {i}] {dt:.0f}s  raw_align={hit_raw/((i+1)*K):.3f} "
              f"ref_align={hit_ref/((i+1)*K):.3f} αMAE_raw={mae_raw[-1]:.3f} "
              f"αMAE_ref={mae_ref[-1]:.3f} Jdiff={j_diff[-1]*100:.1f}%")

        # 推进环境 (以蒸馏决策执行)
        a = compose_action(pd, K, M)
        env.step(a)

    n = N_STATES * K
    out = {
        'rho_align_raw': hit_raw / n,
        'rho_align_refined': hit_ref / n,
        'alpha_mae_raw': float(np.mean(mae_raw)),
        'alpha_mae_refined': float(np.mean(mae_ref)),
        'j_rel_diff_mean': float(np.mean(j_diff)),
        'j_rel_diff_median': float(np.median(j_diff)),
        'perf_equiv_frac_lt2pct': float(np.mean([d < 0.02 for d in j_diff])),
        'perf_equiv_frac_lt5pct': float(np.mean([d < 0.05 for d in j_diff])),
        'n_states': N_STATES,
        'per_state': rows,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n  raw   align={out['rho_align_raw']:.4f} αMAE={out['alpha_mae_raw']:.4f}")
    print(f"  ref   align={out['rho_align_refined']:.4f} αMAE={out['alpha_mae_refined']:.4f}")
    print(f"  J rel diff mean={out['j_rel_diff_mean']*100:.2f}%  "
          f"perf-equiv(<2%)={out['perf_equiv_frac_lt2pct']*100:.0f}%  "
          f"(<5%)={out['perf_equiv_frac_lt5pct']*100:.0f}%")
    print(f"  已保存 -> {OUT_JSON}")


if __name__ == "__main__":
    main()

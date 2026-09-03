# -*- coding: utf-8 -*-
"""
E5 Pareto 前沿重跑 (rerun_e5_pareto_v2.py)
==========================================
口径修正 (2026-09-01):
  - 旧版 run_e5_pareto.py 走 FullLLM 辩论且未经验证器精化, 数据与修正后物理模型
    不一致 (E1 中 HMA-Distill 在 ω=(0.5,0.5) 下 1.077 kJ / 0.668 s, 旧 E5 为
    0.555 kJ / 1.185 s)。
  - 本脚本按 E1 的 HMA-Distill 主线管线重跑: 蒸馏策略单次前向 (零 LLM 调用) 得到
    粗方案, 以注入偏好 ω=(ω_e, 1-ω_e) 的验证器精化闭环 (PlanRefiner, 适应度
    ω_e·E + ω_l·T + 达标硬罚) 逐用户精化, 覆盖能耗-时延权衡带。
  - 每 ω 点: 5 种子 × 3 episode × 100 步, 记录逐 episode 指标以画误差棒。
"""
import os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)
sys.path.insert(0, os.path.join(SRC, "local"))

from config import NUM_USERS, NUM_EDGE_SERVERS, SEED, RESULTS_DIR, CHECKPOINT_DIR
from environment import MECEnvironment
from distill_agent import PolicyAgentRunner
from local.plan_refiner import PlanRefiner
from local.experiment_common import compose_action

OMEGAS = [0.0, 0.25, 0.5, 0.75, 1.0]
N_SEEDS = 5
N_EPISODES = 3
N_STEPS = 100
POLICY_PATH = os.path.join(CHECKPOINT_DIR, "distilled_policy.pth")
OUT_NPZ = os.path.join(RESULTS_DIR, "e5_pareto_v2.npz")
OUT_JSON = os.path.join(RESULTS_DIR, "e5_pareto_v2.json")


def main():
    print("=" * 60)
    print("  E5 重跑: Distill + 验证器精化 (偏好注入 ω)")
    print(f"  {len(OMEGAS)} ω × {N_SEEDS} seeds × {N_EPISODES} ep × {N_STEPS} 步")
    print("=" * 60)
    runner = PolicyAgentRunner(model_path=POLICY_PATH)
    print("  策略网络加载完成:", POLICY_PATH)

    out = {}
    t0_all = time.time()
    for w_e in OMEGAS:
        omega = (float(w_e), float(1.0 - w_e))
        refiner = PlanRefiner(omega=omega)
        ep_energy, ep_latency, ep_suc, ep_sla = [], [], [], []
        for s in range(N_SEEDS):
            for e in range(N_EPISODES):
                env = MECEnvironment(num_users=NUM_USERS,
                                     num_servers=NUM_EDGE_SERVERS,
                                     seed=SEED + s * 7 + e * 13)
                env.reset()
                e_sum, t_sum, suc_sum, sla_sum = 0.0, 0.0, 0.0, 0.0
                for _ in range(N_STEPS):
                    state = env._get_state()
                    out_d = runner.infer(state, deterministic=False)
                    alpha, server = out_d['plan']['alpha'], out_d['plan']['server']
                    alpha_r, server_r = refiner.refine(env, alpha, server)
                    action = compose_action({'alpha': alpha_r,
                                             'server': server_r}, env.K, env.M)
                    ns, r, d, info = env.step(action)
                    e_sum += info['energy']
                    t_sum += info['latency']
                    suc_sum += info['success_rate']
                    sla_sum += info['priority_sla']
                ep_energy.append(e_sum / N_STEPS)
                ep_latency.append(t_sum / N_STEPS)
                ep_suc.append(suc_sum / N_STEPS)
                ep_sla.append(sla_sum / N_STEPS)
        out[w_e] = {
            'energy': ep_energy, 'latency': ep_latency,
            'success_rate': ep_suc, 'priority_sla': ep_sla,
        }
        print(f"  ω_e={w_e:.2f}: E={np.mean(ep_energy):.4f}±{np.std(ep_energy):.4f} kJ, "
              f"T={np.mean(ep_latency):.3f}±{np.std(ep_latency):.3f} s, "
              f"suc={np.mean(ep_suc)*100:.1f}%  "
              f"({time.time()-t0_all:.0f}s elapsed)")

    # 保存
    np.savez(OUT_NPZ, **{f"w{w_e}__{m}__vals": np.array(v)
                         for w_e, d in out.items() for m, v in d.items()})
    json_out = {str(w): {m: {'mean': float(np.mean(v)), 'std': float(np.std(v)),
                             'vals': list(np.round(v, 6))}
                         for m, v in d.items()} for w, d in out.items()}
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(json_out, f, ensure_ascii=False, indent=1)
    print(f"  已保存 -> {OUT_NPZ} / {OUT_JSON}")
    print(f"  总耗时 {time.time()-t0_all:.0f}s")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
================================================================
实验 E5 Pareto 前沿 (local/run_e5_pareto.py)
================================================================
在 K=8, M=4 基准场景下, 以 FullLLM 模式运行 CW-Debate,
向 OA 注入固定偏好权重 ω_e ∈ [0,1], 收集每 ω 的 (E, T) 均值,
绘制 HMA-MEC 的真实 Pareto 前沿.

相对于之前用 Distill 模式 + alpha 后处理的简化方案,
本实现通过实时 LLM 辩论 + OA 偏好仲裁获得真正的 Pareto 曲线.
代价: 每次 ω 需约 30 秒 (5 个 ω × 2 ep × 20 step × ~40 calls),
      总约 8000 次 API 调用 ≈ 15 元.

可通过编辑 OMEGAS / N_EPISODES / N_STEPS 三参数调节精度与成本.
================================================================
"""

import os
import sys
import time
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)

from config import (
    NUM_USERS, NUM_EDGE_SERVERS, MAX_STEPS, RESULTS_DIR, SEED,
    CHECKPOINT_DIR,
)
from environment import MECEnvironment
from agent_runner import HMAAgentRunner
from local.experiment_common import compose_action
from agent_define import make_agents
from llm_client import get_llm_client

# 显眼配置区
OMEGAS = np.linspace(0.0, 1.0, 5)    # 5 个 ω 点 (0, 0.25, 0.5, 0.75, 1)
N_EPISODES = 2                        # 每 ω 下 episode 数
N_STEPS = 30                          # 每 episode 步数 (C8: 20→30, Pareto 曲线更稳)
POLICY_PATH = os.path.join(CHECKPOINT_DIR, "distilled_policy.pth")
OUTPUT_NPZ = os.path.join(RESULTS_DIR, "e5_pareto.npz")
OUTPUT_JSON = os.path.join(RESULTS_DIR, "e5_pareto.json")


def main():
    print("=" * 60)
    print("  E5 Pareto 前沿实验 (FullLLM + OA 偏好注入)")
    print("=" * 60)
    print(f"  ω 点 × Eps × Steps = {len(OMEGAS)}×{N_EPISODES}×{N_STEPS}, "
          f"预计调用 {(len(OMEGAS)*N_EPISODES*N_STEPS*40):.0f} 次 API")
    print(f"  总耗时约 {len(OMEGAS)*N_EPISODES*N_STEPS*0.5:.0f}s")

    # 尝试获取 LLM client (若不可用, 回退到启发式, 此时 preference 也通过新逻辑生效)
    try:
        llm_client = get_llm_client()
        print(f"  LLM 客户端已就绪, 将使用 CW-Debate FullLLM 模式")
        using_llm = True
    except Exception as e:
        llm_client = None
        print(f"  LLM 客户端不可用 ({e}), 使用启发式 (preference 仍通过仲裁逻辑生效)")
        using_llm = False

    E_HMA, T_HMA = [], []
    CONF_HMA, FALLBACK_HMA = [], []
    for w_e in OMEGAS:
        omega = np.array([w_e, 1.0 - w_e], dtype=np.float32)
        e_sum, t_sum = 0.0, 0.0
        confs, fallbacks = [], []
        n_total = 0

        for ep in range(N_EPISODES):
            env = MECEnvironment(num_users=NUM_USERS,
                                  num_servers=NUM_EDGE_SERVERS,
                                  seed=SEED + ep)
            env.reset()
            agents = make_agents(env, with_va=True)
            t0 = time.time()

            for s in range(N_STEPS):
                runner = HMAAgentRunner(env=env, mode="FullLLM",
                                        llm=llm_client,
                                        agents=agents,
                                        preference=omega,
                                        verbose=False)
                out = runner.run_step(state=env._get_state(),
                                      agents_reuse=False)
                confs.append(float(out.get('conf_min', 0.0)))
                fallbacks.append(int(out.get('fallback_triggered', False)))
                a = compose_action(out['plan'], env.K, env.M)
                ns, _, d, info = env.step(a)
                e_sum += info['energy']
                t_sum += info['latency']
                n_total += 1
                if d:
                    break

            dt = time.time() - t0
            print(f"  ω_e={w_e:.2f} ep={ep}  "
                  f"time={dt:.1f}s  cur E={e_sum/max(n_total,1):.5f}")

        avg_e = e_sum / max(n_total, 1)
        avg_t = t_sum / max(n_total, 1)
        E_HMA.append(avg_e)
        T_HMA.append(avg_t)
        CONF_HMA.append(float(np.mean(confs)) if confs else 0.0)
        FALLBACK_HMA.append(float(np.mean(fallbacks)) if fallbacks else 0.0)
        print(f"  => ω_e={w_e:.2f}  E={avg_e:.5f}  T={avg_t:.3f}  "
              f"conf_min={CONF_HMA[-1]:.3f}  fb={FALLBACK_HMA[-1]:.3f}")

    # 保存
    flat = {
        "HMA__omegas": OMEGAS.astype(np.float32),
        "HMA__energy": np.array(E_HMA, dtype=np.float32),
        "HMA__latency": np.array(T_HMA, dtype=np.float32),
        "HMA__conf_min": np.array(CONF_HMA, dtype=np.float32),
        "HMA__fallback_rate": np.array(FALLBACK_HMA, dtype=np.float32),
    }
    os.makedirs(os.path.dirname(OUTPUT_NPZ), exist_ok=True)
    np.savez_compressed(OUTPUT_NPZ, **flat)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"HMA": {"omegas": OMEGAS.tolist(),
                            "energy": E_HMA, "latency": T_HMA,
                            "conf_min": CONF_HMA,
                            "fallback_rate": FALLBACK_HMA}},
                   f, ensure_ascii=False, indent=2)
    print(f"\n  npz 保存: {OUTPUT_NPZ}")
    print("=" * 60)


if __name__ == "__main__":
    main()
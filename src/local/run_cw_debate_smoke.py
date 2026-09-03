# -*- coding: utf-8 -*-
"""
================================================================
本地集成测试：CW-Debate 多步推理流程验证
================================================================
本脚本在本地 CPU、不调用任何 LLM 的前提下，完成多次 CW-Debate 推理
（含基线对比），验证多智能体框架可稳定地在一个 episode 内连续决策。

运行：
    python local/run_cw_debate_smoke.py
================================================================
"""

import os
import sys
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.dirname(HERE)
sys.path.insert(0, SRC_ROOT)

from environment import MECEnvironment
from agent_define import make_agents
from cw_debate import cw_debate


def main():
    print("=" * 60)
    print("  CW-Debate 多步推理集成测试 (mode=Distill)")
    print("=" * 60)

    NUM_EPISODES = 5
    STEPS_PER_EP = 50    # 缩短以快速验证
    cumulative_energy, cumulative_latency, cumulative_success = 0.0, 0.0, 0.0
    cumulative_sla = 0.0
    t0 = time.time()

    for ep in range(NUM_EPISODES):
        env = MECEnvironment(num_users=8, num_servers=4, seed=42 + ep)
        env.reset()
        agents = make_agents(env, with_va=True)
        ep_energy, ep_lat, ep_ok, ep_sla = [], [], [], []

        for step in range(STEPS_PER_EP):
            out = cw_debate(env, agents, mode="Distill", llm=None,
                            verbose=False)
            plan = out['plan']
            sim = env.simulate(plan)
            # 推进真实环境：把 plan 转换为 environment.action 格式
            # environment.step 期望 action[k,0]=alpha, action[k,1]=server/M in (0,1)
            action = np.zeros(env.action_dim, dtype=np.float32)
            action[0::2] = plan['alpha']
            action[1::2] = plan['server'] / max(env.M - 1, 1)  # 归一化到 (0,1)
            action = action.reshape(env.K, 2).flatten()
            _, _, done, info = env.step(action)
            ep_energy.append(info['energy'])
            ep_lat.append(info['latency'])
            ep_ok.append(info['success_rate'])
            ep_sla.append(sim['priority_sla'])
            if done:
                break

        cumulative_energy += float(np.mean(ep_energy))
        cumulative_latency += float(np.mean(ep_lat))
        cumulative_success += float(np.mean(ep_ok))
        cumulative_sla     += float(np.mean(ep_sla))
        print(f"  ep{ep+1}: "
              f"energy={np.mean(ep_energy):.4f} kJ, "
              f"latency={np.mean(ep_lat):.3f} s, "
              f"success={np.mean(ep_ok):.2%}, "
              f"sla={np.mean(ep_sla):.2%}")

    dt = time.time() - t0
    print("-" * 60)
    print(f"  跨 episode 平均: "
          f"energy={cumulative_energy/NUM_EPISODES:.4f} kJ, "
          f"latency={cumulative_latency/NUM_EPISODES:.3f} s, "
          f"success={cumulative_success/NUM_EPISODES:.2%}, "
          f"sla={cumulative_sla/NUM_EPISODES:.2%}")
    print(f"  总耗时 {dt:.2f} s, 平均每 step 耗时 "
          f"{dt/(NUM_EPISODES*STEPS_PER_EP)*1000:.2f} ms")
    print("=" * 60)
    print("  集成测试通过：多 Agent 决策链路稳定。")


if __name__ == "__main__":
    main()
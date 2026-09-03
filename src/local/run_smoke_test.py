# -*- coding: utf-8 -*-
"""
================================================================
本地自检脚本 (local/run_smoke_test.py)
================================================================
本脚本在本地 CPU 上完成一次最小化的端到端流程验证：
  1. 构造 MEC 环境
  2. 生成 A1 中全部四类 Agent
  3. UA / EA / OA / VA 各做一次 observe()
  4. 随机生成一个候选 plan，调用 VA.verify() 反事实仿真
  5. 打印结果，验证整个骨架可通

不依赖 GPU、不依赖 LLM API，可在任何安装 numpy 的本机直接运行：
    python local/run_smoke_test.py

运行成功后即可进入 P2 阶段，重点验证 CW-Debate 协议实现。
================================================================
"""

import os
import sys
import numpy as np

# 把 src 根目录加入 sys.path，便于直接 from config import *
HERE = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.dirname(HERE)
sys.path.insert(0, SRC_ROOT)

from environment import MECEnvironment
from agent_define import (
    make_agents, agent_topology_summary,
)
from verifier import evaluate_plan


def main():
    print("=" * 60)
    print("  HMA-MEC 骨架自检 (P1 阶段)")
    print("=" * 60)

    # 1) 构造环境
    env = MECEnvironment(num_users=8, num_servers=4, seed=42)
    state = env.reset()
    print(f"[1/5] 环境构造成功：K={env.K}, M={env.M}, "
          f"state_dim={env.state_dim}, action_dim={env.action_dim}")

    # 2) 生成 Agent
    agents = make_agents(env, with_va=True)
    topo = agent_topology_summary(agents)
    print(f"[2/5] Agent 生成成功：{topo}")

    # 3) 让所有 Agent 观察一次状态
    text = env.state_to_text()
    for ua in agents['UA']:
        ua.observe(state, text)
    for ea in agents['EA']:
        ea.observe(state, text)
    agents['OA'].observe(state, text)
    agents['VA'].observe(state, text)
    print("[3/5] 已为全部 Agent 注入 sys_state 缓存")
    print("      (UA[0] 画像) ->", agents['UA'][0].local_profile())
    print("      (EA[0] 剩余容量) ->", agents['EA'][0].capacity())

    # 4) 随机生成一个候选 plan，调用反事实仿真
    plan = {
        'alpha':  np.random.uniform(0.05, 0.95, env.K),
        'server': np.random.randint(0, env.M, env.K),
    }
    sim = evaluate_plan(env, plan)
    print(f"[4/5] 反事实仿真结果："
          f"energy={sim['energy']:.4f} kJ, latency={sim['latency']:.3f} s, "
          f"success={sim['success_rate']:.2%}, priority_sla={sim['priority_sla']:.2%}")

    # 5) 调用 VA.verify()，verify 中先给出一个虚构的「OA 自评估」
    oa_pred = {'energy': sim['energy'] * 1.10, 'latency': sim['latency'] * 1.10}
    va_out = agents['VA'].verify(plan, oa_prediction=oa_pred)
    print(f"[5/5] VA 验证结果：accept={va_out['accept']}, reason={va_out['reason']}")

    print("=" * 60)
    print("  自检通过，骨架运行正常")
    print("=" * 60)


if __name__ == "__main__":
    main()
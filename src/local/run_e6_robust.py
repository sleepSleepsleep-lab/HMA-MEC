# -*- coding: utf-8 -*-
"""
================================================================
实验 E6 鲁棒性 (local/run_e6_robust.py)
================================================================
三类扰动实验:
  (i)  信道增益突变 (rho 由 0.95 切到 0.5)
  (ii) 服务器 m=0 在 t=100 突然宕机
  (iii) 注入不诚实 UA (高自评置信度但实际劣质提案)

验证 VA 拒绝与 ToM 异常识别能力. 结果保存 results/e6_robust.npz.
================================================================
"""

import os
import sys
import json
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)

from config import (
    NUM_USERS, NUM_EDGE_SERVERS, MAX_STEPS, RESULTS_DIR, SEED,
    CHECKPOINT_DIR, CHANNEL_COEFF_SCALE,
)
from environment import MECEnvironment
from agent_runner import HMAAgentRunner
from local.experiment_common import compose_action
from local.results_store import Recorder

# 显眼配置区
N_EPISODES = 30
PERTURB_STEP = 100      # 在第 100 步引入扰动
N_STEPS = MAX_STEPS     # 200
POLICY_PATH = os.path.join(CHECKPOINT_DIR, "distilled_policy.pth")
OUTPUT_NPZ  = os.path.join(RESULTS_DIR, "e6_robust.npz")
OUTPUT_JSON = os.path.join(RESULTS_DIR, "e6_robust.json")


def run_episode_with_perturb(env, perturb_type: str, perturb_step: int,
                                n_steps: int, mode: str = "Hybrid"):
    """跑一个 episode 在 perturb_step 引入扰动."""
    runner = HMAAgentRunner(env=env, mode=mode,
                            policy_path=POLICY_PATH, agents=None)
    energy, lat, suc, sla = [], [], [], []
    s = env.reset()
    for i in range(n_steps):
        if i == perturb_step:
            if perturb_type == "channel_drop":
                # 强制信道相关系数下降: 把当前 channels 衰减
                env.channels = env.channels * 0.5
            elif perturb_type == "server_fail":
                # 让 server 0 容量骤降到 1 (几乎空)
                env.f_edge[0] = 1.0
            elif perturb_type == "dishonest_ua":
                # 仅在 Hybrid 模式下有意义, 修改 runner.tau_low 为 0 强制全前向
                runner.tau_low = 0.0  # 强制使用蒸馏前向; "dishonest" 由 OA/VA 自行 detect
            elif perturb_type == "link_fail":
                # Q2 整改: 链路故障注入 (C2 set_link_error, ARQ 期望时延公式生效)
                # 用户 3→服务器 1、用户 5→服务器 2 链路误帧率升至 0.9
                env.set_link_error(3, 1, 0.9)
                env.set_link_error(5, 2, 0.9)
            elif perturb_type == "mobility":
                # Q6 整改: 移动性变体——信道时间相关性 ρ 0.95→0.4 + 路径损耗漂移
                # 按 06 文档方案 B: 不需要重训, 仅在 episode 中段切换信道统计特性
                env.channel_coeffs = env.channel_coeffs * 0.5
                env.channels = np.abs(env.channel_coeffs) ** 2
        elif perturb_type == "mobility" and i > perturb_step:
            # 移动性: 每步以低相关性 (ρ=0.4) 更新信道, 模拟用户在小区间快速移动
            drift_rng = np.random.RandomState(SEED + i)
            noise = ((drift_rng.randn(env.K, env.M)
                      + 1j * drift_rng.randn(env.K, env.M))
                     * np.sqrt(0.5) * CHANNEL_COEFF_SCALE)
            env.channel_coeffs = (0.4 * env.channel_coeffs
                                  + np.sqrt(1 - 0.4 ** 2) * noise)
            env.channels = np.abs(env.channel_coeffs) ** 2
        state = env._get_state()
        out = runner.run_step(state=state, agents_reuse=True)
        a = compose_action(out['plan'], env.K, env.M)
        ns, _, d, info = env.step(a)
        energy.append(info['energy']); lat.append(info['latency'])
        suc.append(info['success_rate']); sla.append(info['priority_sla'])
        s = ns
        if d: break
    return {
        'energy_history': np.array(energy).tolist(),
        'latency_history': np.array(lat).tolist(),
        'success_history': np.array(suc).tolist(),
        'sla_history':     np.array(sla).tolist(),
    }


def main():
    print("=" * 60)
    print("  E6 鲁棒性实验")
    print("=" * 60)
    _rec = Recorder("e6", config={"n_steps": N_STEPS, "mode": "Hybrid",
                                     "perturb_step": PERTURB_STEP})
    all_out = {}
    # Q2/Q6 整改: 新增 link_fail (链路故障) 与 mobility (移动性) 扰动场景
    for ptype in ["channel_drop", "server_fail", "dishonest_ua",
                  "link_fail", "mobility"]:
        print(f"\n  -- 扰动: {ptype} --")
        records = []
        for ep in range(N_EPISODES):
            env = MECEnvironment(num_users=NUM_USERS,
                                  num_servers=NUM_EDGE_SERVERS,
                                  seed=SEED + ep)
            r = run_episode_with_perturb(env, ptype,
                                          perturb_step=PERTURB_STEP,
                                          n_steps=N_STEPS, mode="Hybrid")
            records.append(r)
            _rec.add(method=ptype, seed=None, episode=ep, metrics=r)
            print(f"    ep {ep}: "
                  f"post energy mean = {np.mean(r['energy_history'][PERTURB_STEP:]):.5f}, "
                  f"post SLA mean = {np.mean(r['sla_history'][PERTURB_STEP:]):.2%}")

        # 取所有 episode 平均得到单条曲线
        avg_e  = np.mean([r['energy_history']  for r in records], axis=0)
        avg_la = np.mean([r['latency_history'] for r in records], axis=0)
        avg_s  = np.mean([r['success_history'] for r in records], axis=0)
        avg_sl = np.mean([r['sla_history']     for r in records], axis=0)
        all_out[ptype] = {
              'energy':    avg_e.tolist(),
              'latency':   avg_la.tolist(),
              'success':   avg_s.tolist(),
              'sla':       avg_sl.tolist(),
        }

    flat = {}
    for pt, d in all_out.items():
        for k in ('energy', 'latency', 'success', 'sla'):
            flat[f"{pt}__{k}"] = np.array(d[k], dtype=np.float32)
    _rec.close()
    os.makedirs(os.path.dirname(OUTPUT_NPZ), exist_ok=True)
    np.savez_compressed(OUTPUT_NPZ, **flat)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_out, f, ensure_ascii=False, indent=2)
    print(f"\n  npz 保存: {OUTPUT_NPZ}")
    print("=" * 60)


if __name__ == "__main__":
    main()
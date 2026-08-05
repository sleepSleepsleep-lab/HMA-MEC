# -*- coding: utf-8 -*-
"""
================================================================
实验 E4 实时性与效率 (local/run_e4_efficiency.py)
================================================================
对比 HMA-MEC 三种在线推理模式 (Distill / Hybrid / FullLLM) 的
- token 数
- 单步推理时延
- 困难状态率 (Hybrid 模式触发在线辩论的比例)

结果保存 results/e4_efficiency.npz; 由 fig_e4_efficiency.py 绘图。
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

from config import NUM_USERS, NUM_EDGE_SERVERS, MAX_STEPS, MAX_EPISODES, RESULTS_DIR, SEED, CHECKPOINT_DIR
from environment import MECEnvironment
from agent_runner import HMAAgentRunner
from local.experiment_common import compose_action

# 显眼配置区
N_EPISODES = 3
N_STEPS    = MAX_STEPS          # 每 episode 步数
WARMUP_STEPS = 20              # 每个模式跑前先 warmup，防止初始化/缓存效应污染均值
MODES = ["Distill", "Hybrid"]   # FullLLM 仅作上限参考, 默认省略
POLICY_PATH = os.path.join(CHECKPOINT_DIR, "distilled_policy.pth")
OUTPUT_NPZ  = os.path.join(RESULTS_DIR, "e4_efficiency.npz")
OUTPUT_JSON = os.path.join(RESULTS_DIR, "e4_efficiency.json")


def run_one_mode(mode, n_episodes=N_EPISODES, n_steps=N_STEPS):
    """跑一种模式, 返回 (warmup 后) step 时延列表与 conf_min 列表与触发率.

    时延分两类记录:
      - latency_total_ms : run_step() 总耗时 (含 dispatch + reuse check + 前向 + 取决)
      - latency_infer_ms : 仅 distill 策略前向内部计时 (来自 runner.stats)
    """
    env = MECEnvironment(num_users=NUM_USERS, num_servers=NUM_EDGE_SERVERS,
                         seed=SEED)
    env.reset()
    runner = HMAAgentRunner(env=env, mode=mode, policy_path=POLICY_PATH,
                            agents=None)
    lat_total, lat_infer, confs, fallbacks = [], [], [], []
    for ep in range(n_episodes):
        env.reset()
        # ---- warmup 阶段：不计时 ----
        for _ in range(WARMUP_STEPS):
            state = env._get_state()
            t0 = time.time()
            out = runner.run_step(state=state, agents_reuse=True)
            _ = (time.time() - t0) * 1000  # discard
            a = compose_action(out['plan'], env.K, env.M)
            env.step(a)
        # ---- 计时阶段 ----
        for s in range(n_steps):
            state = env._get_state()
            t0 = time.time()
            out = runner.run_step(state=state, agents_reuse=True)
            dt = (time.time() - t0) * 1000
            lat_total.append(dt)
            confs.append(out['conf_min'])
            fallbacks.append(int(out.get('fallback_triggered', False)))
            a = compose_action(out['plan'], env.K, env.M)
            env.step(a)
    summary = runner.stats_summary()
    # 平均 inference 时延（来自内部 stats, 不是 dispatch 时延）
    if summary.get('avg_distill_latency_ms') is not None:
        lat_infer_array = np.full(len(lat_total),
                                   summary['avg_distill_latency_ms'],
                                   dtype=np.float32)
    else:
        lat_infer_array = np.zeros(len(lat_total), dtype=np.float32)
    return {
        'latencies_ms':     np.array(lat_total, dtype=np.float32),
        'infer_latencies_ms': lat_infer_array,
        'conf_min':         np.array(confs,        dtype=np.float32),
        'fallbacks':       np.array(fallbacks,    dtype=np.float32),
        'stats':             summary,
    }


def main():
    print("=" * 60)
    print("  E4 实时性实验 (warmup=%d, n_eps=%d, n_steps=%d)" %
          (WARMUP_STEPS, N_EPISODES, N_STEPS))
    print("=" * 60)
    all_out = {}
    for mode in MODES:
        print(f"\n  -- 模式: {mode} --")
        r = run_one_mode(mode)
        all_out[mode] = r
        s = r['stats']
        mean_total = float(r['latencies_ms'].mean())
        median_total = float(np.median(r['latencies_ms']))
        print(f"    avg run_step latency  = {mean_total:.3f} ms "
              f"(median {median_total:.3f})")
        print(f"    avg infer  latency    = {s['avg_distill_latency_ms']:.3f} ms")
        print(f"    avg fullllm latency   = {s['avg_fullllm_latency_ms']:.3f} ms")
        print(f"    hybrid_trigger_rate   = {s['hybrid_trigger_rate']:.2%}")
        print(f"    conf_min mean / min   = "
              f"{s['conf_min_mean']}  /  {s['conf_min_min']}")

    flat = {}
    for m, d in all_out.items():
        flat[f"{m}__latencies_ms"]      = d['latencies_ms']
        flat[f"{m}__infer_latencies_ms"]= d['infer_latencies_ms']
        flat[f"{m}__conf_min"]         = d['conf_min']
        flat[f"{m}__fallbacks"]         = d['fallbacks']
    os.makedirs(os.path.dirname(OUTPUT_NPZ), exist_ok=True)
    np.savez_compressed(OUTPUT_NPZ, **flat)
    summary_out = {m: {'stats': d['stats'],
                        'mean_run_step_ms': float(d['latencies_ms'].mean()),
                        'median_run_step_ms': float(np.median(d['latencies_ms']))}
                    for m, d in all_out.items()}
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary_out, f, ensure_ascii=False, indent=2)
    print(f"\n  npz 保存: {OUTPUT_NPZ}")
    print("=" * 60)


if __name__ == "__main__":
    main()
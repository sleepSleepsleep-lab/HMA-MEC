# -*- coding: utf-8 -*-
"""本机交叉环境计时核验（相对路径版，不依赖 AutoDL 绝对路径）。
核验对象：E4 三项时间声明 —— 蒸馏前向 (~2ms)、Distill 完整闭环 (~70ms)、GA 单步 (~0.3s)。
"""
import os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)
sys.path.insert(0, HERE)

from config import NUM_USERS, NUM_EDGE_SERVERS, SEED, CHECKPOINT_DIR
from environment import MECEnvironment
from agent_runner import HMAAgentRunner
from local.experiment_common import compose_action
from local.ga_baseline import GAOffloadBaseline
from local.baseline_mpc import MPCBaseline

K, M = NUM_USERS, NUM_EDGE_SERVERS
POLICY = os.path.join(CHECKPOINT_DIR, "distilled_policy.pth")
print(f"device: K={K}, M={M}, policy={POLICY} ({os.path.exists(POLICY)})")

def timed(label, fn, n=50, warmup=5):
    for _ in range(warmup): fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1000)
    ts = np.array(ts)
    print(f"  {label:34s} mean {ts.mean():8.2f} ms | median {np.median(ts):8.2f} ms | p90 {np.percentile(ts,90):8.2f} ms")
    return ts

print("\n== 1) HMA-Distill 完整闭环（策略前向 + 验证器逐用户精化）==")
env = MECEnvironment(num_users=K, num_servers=M, seed=SEED)
env.reset()
runner = HMAAgentRunner(env=env, mode="Distill", policy_path=POLICY, agents=None)
def distill_step():
    state = env._get_state()
    out = runner.run_step(state=state, agents_reuse=True)
    a = compose_action(out['plan'], env.K, env.M)
    env.step(a)
timed("run_step 总耗时 (论文声明 中位~70 ms)", distill_step, n=60, warmup=10)
s = runner.stats_summary()
if s.get('avg_distill_latency_ms') is not None:
    print(f"  {'runner 内部策略前向计时':34s} mean {s['avg_distill_latency_ms']:.3f} ms (论文声明 ~2 ms, CPU-only 口径)")

print("\n== 2) GA 单步离线搜索 (种群30x50代, 论文声明 ~0.3 s) ==")
env2 = MECEnvironment(num_users=K, num_servers=M, seed=SEED)
ga = GAOffloadBaseline(pop_size=30, generations=50, omega=(0.5, 0.5))
timed("GA pop30xgen50 predict", lambda: ga.predict(env2._get_state(), env2), n=10, warmup=2)

print("\n== 3) MPC (启发式种子+精化) 参照 ==")
env3 = MECEnvironment(num_users=K, num_servers=M, seed=SEED)
mpc = MPCBaseline()
timed("MPC predict", lambda: mpc.predict(env3._get_state(), env3), n=20, warmup=3)
print("\n完成。注意：本机为 Windows CPU-only，与论文 AutoDL-4090 环境存在 CPU 主频/库版本差异，"
      "仅用于量级交叉核验。")

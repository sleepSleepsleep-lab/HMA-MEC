# -*- coding: utf-8 -*-
"""实测各方法每步决策耗时 (论文实时性/计算开销表).
方法: HMA-Distill(含Refiner) / MPC / GA / SAC / DQN(if trained) / 启发式"""
import os, sys, time
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_REPO_ROOT)
_REPO_ROOT = os.path.dirname(_REPO_ROOT)

sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src", "local"))
import numpy as np
from environment import MECEnvironment
from agent_runner import HMAAgentRunner
from local.experiment_common import compose_action
from local.baseline_mpc import MPCBaseline
from baselines import SACAgent
from local.ga_baseline import GAOffloadBaseline
from config import SEED, NUM_USERS, NUM_EDGE_SERVERS

K, M = NUM_USERS, NUM_EDGE_SERVERS
env_ = MECEnvironment(num_users=K, num_servers=M, seed=SEED)

def bench(label, fn, n=20):
    # warmup
    for _ in range(2): fn()
    t0 = time.time()
    for _ in range(n): fn()
    per = (time.time() - t0) / n * 1000
    print(f"  {label:28s} {per:8.1f} ms/步")
    return per

print("== 决策开销实测 (K=8, M=4) ==")

env = MECEnvironment(num_users=K, num_servers=M, seed=SEED)
runner = HMAAgentRunner(env=env, mode="Distill")
bench("HMA-Distill (蒸馏+Refiner)", lambda: runner.run_step(state=env._get_state(), agents_reuse=True))

env = MECEnvironment(num_users=K, num_servers=M, seed=SEED)
mpc = MPCBaseline()
bench("MPC (启发式种子+Refiner)", lambda: mpc.predict(env._get_state(), env))

env = MECEnvironment(num_users=K, num_servers=M, seed=SEED)
ga = GAOffloadBaseline(pop_size=50, generations=100, omega=(0.5, 0.5))
bench("GA (pop50×gen100 离线搜索)", lambda: ga.predict(env._get_state(), env))

# SAC 用预训练快速训几步的实例测量 predict 开销
env = MECEnvironment(num_users=K, num_servers=M, seed=SEED)
sac = SACAgent(env)
sac.train(env, episodes=30, verbose=False)
bench("SAC (30ep 短训后决策)", lambda: sac.predict(env._get_state(), env))

print("  FullLLM (CW-Debate 5轮): ~60467 ms/步 (E4 实测)")

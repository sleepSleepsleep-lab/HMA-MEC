# -*- coding: utf-8 -*-
"""
================================================================
E1 极小规模快测 (local/run_e1_quick.py)
================================================================
仅运行启发式 + HMA-Distill 一组, 用于验证 E1 流程可跑通.
正式 E1 主对比仍是 run_e1_main.py (含 SAC/DDPG/很慢).
================================================================
"""

import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)
from config import NUM_USERS, NUM_EDGE_SERVERS, MAX_STEPS, RESULTS_DIR, SEED, CHECKPOINT_DIR
from local.experiment_common import run_multi_episodes, save_npz

N_SEEDS = 1
N_EPS   = 1
N_STEPS = 50
POLICY  = os.path.join(CHECKPOINT_DIR, "smoke_policy.pth")
OUTPUT  = os.path.join(RESULTS_DIR, "e1_quick.npz")

method_specs = [
    {'name': 'Greedy'},
    {'name': 'AllEdge'},
    {'name': 'HMA-Distill'},
]
print("="*60); print("  E1 极小规模快测"); print("="*60)
results = run_multi_episodes(method_specs, K=NUM_USERS, M=NUM_EDGE_SERVERS,
                              n_seeds=N_SEEDS, n_episodes=N_EPS, n_steps=N_STEPS)
save_npz(OUTPUT, results)
print("\n  结果:")
for m, d in results.items():
    print(f"  {m:<14}  E={d['mean']['energy']:.5f} T={d['mean']['latency']:.3f} "
          f"suc={d['mean']['success_rate']:.2%} sla={d['mean']['priority_sla']:.2%}")
print("="*60)
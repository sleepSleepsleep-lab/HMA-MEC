# -*- coding: utf-8 -*-
"""
MADDPG E1 重跑 (rerun_maddpg_fresh.py, 2026-09-01)
===================================================
- 训练种子: SEED + sd*100 (每 seed 一个 agent, 500 ep 训练, 与原 E1 同预算)
- 评估种子: SEED + sd*100 + ep (50 ep/seed, 250 个互不碰撞独立环境)
- 保存逐 episode 数据 (e1_fresh_maddpg.npz/.json), 与 rerun_e1_fresh.py 输出同格式
"""
import os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)
sys.path.insert(0, HERE)

from concurrent.futures import ProcessPoolExecutor
from config import (NUM_USERS, NUM_EDGE_SERVERS, RESULTS_DIR, SEED,
                    MAX_STEPS)
from environment import MECEnvironment
from local.baseline_maddpg import MADDPGAgent, evaluate, MADDPG_EPOCHS

N_SEEDS, N_EPISODES, N_STEPS = 5, 50, MAX_STEPS
K, M = NUM_USERS, NUM_EDGE_SERVERS


def run_seed(sd):
    env = MECEnvironment(num_users=K, num_servers=M, seed=SEED + sd * 100)
    agent = MADDPGAgent(env)
    t0 = time.time()
    hist = agent.train(env, episodes=MADDPG_EPOCHS, verbose=False)
    dt = time.time() - t0
    eps = []
    for ep in range(N_EPISODES):
        e = MECEnvironment(num_users=K, num_servers=M,
                           seed=SEED + sd * 100 + ep)
        r = evaluate(agent, e, n_steps=N_STEPS)
        eps.append(r)
        if (ep + 1) % 10 == 0:
            print(f"    [MADDPG] s{sd} ep{ep+1}/50 "
                  f"E={r[0]:.4f} T={r[1]:.3f} suc={r[2]:.2%}", flush=True)
    E = np.array([r[0] for r in eps])
    print(f"  [seed {sd}] 训练 {len(hist)}ep ({dt:.0f}s)  "
          f"E={E.mean():.4f} T={np.mean([r[1] for r in eps]):.3f} "
          f"suc={np.mean([r[2] for r in eps]):.2%}", flush=True)
    return eps


def main():
    print(f"  MADDPG 重跑: {N_SEEDS} seeds × {MADDPG_EPOCHS}ep 训练 "
          f"+ {N_EPISODES}ep 评估 (种子互不碰撞)", flush=True)
    t0 = time.time()
    ctx = __import__("multiprocessing").get_context("spawn")
    with ProcessPoolExecutor(max_workers=N_SEEDS, mp_context=ctx) as ex:
        per_seed = list(ex.map(run_seed, range(N_SEEDS)))
    all_eps = [r for seed_eps in per_seed for r in seed_eps]
    names = ['energy', 'latency', 'success_rate', 'priority_sla']
    agg = {names[i]: np.array([r[i] for r in all_eps]) for i in range(4)}
    out = {"MADDPG": {
        'mean': {k: float(v.mean()) for k, v in agg.items()},
        'std': {k: float(v.std()) for k, v in agg.items()},
        'n_samples': len(all_eps),
        'per_episode': {k: v.tolist() for k, v in agg.items()},
    }}
    npz_dict = {}
    for met, v in agg.items():
        npz_dict[f"MADDPG__{met}__vals"] = v.astype(np.float32)
        npz_dict[f"MADDPG__{met}__mean"] = np.array([v.mean()],
                                                    dtype=np.float32)
        npz_dict[f"MADDPG__{met}__std"] = np.array([v.std()],
                                                   dtype=np.float32)
    np.savez(os.path.join(RESULTS_DIR, "e1_fresh_maddpg.npz"), **npz_dict)
    with open(os.path.join(RESULTS_DIR, "e1_fresh_maddpg.json"),
              "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"  MADDPG 完成: E={out['MADDPG']['mean']['energy']:.4f} "
          f"T={out['MADDPG']['mean']['latency']:.3f} "
          f"suc={out['MADDPG']['mean']['success_rate']:.2%} "
          f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()

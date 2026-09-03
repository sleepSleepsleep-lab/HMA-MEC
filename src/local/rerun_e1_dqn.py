# -*- coding: utf-8 -*-
"""E1 新增 DQN 基线: 每 seed 训练 500 ep (同 SAC/DDPG 规模) + 评估 50×200.
进程级 spawn 并行, 更新 e1_comparison (新增 DQN 键)。"""
import os, sys, json
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_REPO_ROOT)
_REPO_ROOT = os.path.dirname(_REPO_ROOT)


HERE = _REPO_ROOT
sys.path.insert(0, os.path.join(HERE, "src"))
sys.path.insert(0, os.path.join(HERE, "src/local"))

import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from config import NUM_USERS, NUM_EDGE_SERVERS, RESULTS_DIR, SEED

K, M = NUM_USERS, NUM_EDGE_SERVERS
N_SEEDS, N_EPISODES, N_STEPS, DQN_EP = 5, 50, 200, 500
OUTPUT_NPZ = os.path.join(RESULTS_DIR, "e1_comparison.npz")
OUTPUT_JSON = os.path.join(RESULTS_DIR, "e1_comparison.json")


def run_seed(sd):
    from environment import MECEnvironment
    from baseline_dqn import DQNAgent
    ep_metrics = []
    env = MECEnvironment(num_users=K, num_servers=M, seed=SEED + sd)
    agent = DQNAgent(env)
    agent.train(env, episodes=DQN_EP, verbose=False)
    for ep in range(N_EPISODES):
        env = MECEnvironment(num_users=K, num_servers=M, seed=SEED + sd + ep)
        e, t, s, sl = [], [], [], []
        for _ in range(N_STEPS):
            a = agent.predict(env._get_state(), env)
            ns, _, done, info = env.step(a)
            e.append(info['energy']); t.append(info['latency'])
            s.append(info['success_rate']); sl.append(info['priority_sla'])
            if done: break
        ep_metrics.append(dict(energy=float(np.mean(e)), latency=float(np.mean(t)),
                               success_rate=float(np.mean(s)), priority_sla=float(np.mean(sl))))
    print(f"  [seed {sd}] DQN 评估完成", flush=True)
    return sd, ep_metrics


if __name__ == "__main__":
    from local.experiment_common import save_npz, load_npz
    print(f"  E1 DQN 基线: seeds={N_SEEDS}, 每 seed 训练 {DQN_EP} ep")
    collected = {}
    ctx = mp.get_context('spawn')
    with ProcessPoolExecutor(max_workers=N_SEEDS, mp_context=ctx) as ex:
        for sd, em in ex.map(run_seed, range(N_SEEDS)):
            collected[sd] = em
    per_seed = [r for sd in range(N_SEEDS) for r in collected[sd]]
    agg = {k: [r[k] for r in per_seed]
           for k in ('energy', 'latency', 'success_rate', 'priority_sla')}
    dqn = {'per_seed': per_seed,
           'mean': {k: float(np.mean(v)) for k, v in agg.items()},
           'std':  {k: float(np.std(v)) for k, v in agg.items()}}
    m = dqn['mean']
    print(f"  [DQL] DQN  E={m['energy']:.4f} T={m['latency']:.3f} "
          f"suc={m['success_rate']:.2%} sla={m['priority_sla']:.2%}")

    old = load_npz(OUTPUT_NPZ)
    old['DQN'] = dqn
    save_npz(OUTPUT_NPZ, old)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump({x: {'mean': y['mean'], 'std': y['std'],
                       'n_samples': len(y['per_seed'])}
                   for x, y in old.items()}, f, ensure_ascii=False, indent=2)
    print(f"  保存 {len(old)} 方法 -> {OUTPUT_NPZ} / {OUTPUT_JSON}")

# -*- coding: utf-8 -*-
"""公平重跑 E1: HMA-Distill/Hybrid (公平 Refiner) + MPC 基线, 论文规模.
进程级 spawn 并行 (5 seeds × 3 方法), 更新 e1_comparison (新增 MPC)。"""
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
from local.experiment_common import save_npz, load_npz, compose_action

K, M = NUM_USERS, NUM_EDGE_SERVERS
N_SEEDS, N_EPISODES, N_STEPS = 5, 50, 200
OUTPUT_NPZ = os.path.join(RESULTS_DIR, "e1_comparison.npz")
OUTPUT_JSON = os.path.join(RESULTS_DIR, "e1_comparison.json")
METHODS = ("HMA-Distill", "HMA-Hybrid", "MPC")


def run_seed(sd):
    from environment import MECEnvironment
    from agent_runner import HMAAgentRunner
    from local.baseline_mpc import MPCBaseline
    res = {}
    for mname in METHODS:
        ep_metrics = []
        for ep in range(N_EPISODES):
            env = MECEnvironment(num_users=K, num_servers=M, seed=SEED + sd + ep)
            if mname == "MPC":
                method = MPCBaseline(omega=(0.5, 0.5))
                e, t, s, sl = [], [], [], []
                for _ in range(N_STEPS):
                    a = method.predict(env._get_state(), env)
                    ns, _, done, info = env.step(a)
                    e.append(info['energy']); t.append(info['latency'])
                    s.append(info['success_rate']); sl.append(info['priority_sla'])
                    if done: break
            else:
                mode = mname.split("-")[1]
                runner = HMAAgentRunner(env=env, mode=mode)   # 公平 Refiner (无 QoS 偏好)
                e, t, s, sl = [], [], [], []
                for _ in range(N_STEPS):
                    st = env._get_state()
                    out = runner.run_step(state=st, agents_reuse=True)
                    a = compose_action(out['plan'], K, M)
                    ns, _, done, info = env.step(a)
                    e.append(info['energy']); t.append(info['latency'])
                    s.append(info['success_rate']); sl.append(info['priority_sla'])
                    if done: break
            ep_metrics.append(dict(
                energy=float(np.mean(e)), latency=float(np.mean(t)),
                success_rate=float(np.mean(s)), priority_sla=float(np.mean(sl))))
        res[mname] = ep_metrics
        print(f"  [seed {sd}] {mname} 完成", flush=True)
    return sd, res


if __name__ == "__main__":
    print(f"  公平 E1 重跑: {METHODS} × {N_SEEDS} seeds × {N_EPISODES} ep")
    collected = {}
    ctx = mp.get_context('spawn')
    with ProcessPoolExecutor(max_workers=N_SEEDS, mp_context=ctx) as ex:
        for sd, res in ex.map(run_seed, range(N_SEEDS)):
            collected[sd] = res

    new_results = {}
    for mname in METHODS:
        per_seed = [r for sd in range(N_SEEDS) for r in collected[sd][mname]]
        agg = {k: [r[k] for r in per_seed]
               for k in ('energy', 'latency', 'success_rate', 'priority_sla')}
        new_results[mname] = {
            'per_seed': per_seed,
            'mean': {k: float(np.mean(v)) for k, v in agg.items()},
            'std':  {k: float(np.std(v)) for k, v in agg.items()},
        }
        m = new_results[mname]['mean']
        print(f"  [HMA|MPC] {mname:12s} E={m['energy']:.4f} T={m['latency']:.3f} "
              f"suc={m['success_rate']:.2%} sla={m['priority_sla']:.2%}")

    old = load_npz(OUTPUT_NPZ)
    for mname in METHODS:
        old[mname] = new_results[mname]
    save_npz(OUTPUT_NPZ, old)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump({m: {'mean': d['mean'], 'std': d['std'],
                       'n_samples': len(d['per_seed'])}
                   for m, d in old.items()}, f, ensure_ascii=False, indent=2)
    print(f"  保存 {len(old)} 方法 -> {OUTPUT_NPZ} / {OUTPUT_JSON}")
    print("\n  公平 E1 对比 (统一目标 J=0.5E+0.5T):")
    for m in ('Greedy', 'AllEdge', 'GA', 'MPC', 'SAC', 'DDPG', 'HMA-Distill', 'HMA-Hybrid'):
        if m in old:
            d = old[m]['mean']
            print(f"  {m:14s} E={d['energy']:.4f} T={d['latency']:.3f} "
                  f"suc={d['success_rate']:.2%} sla={d['priority_sla']:.2%}")

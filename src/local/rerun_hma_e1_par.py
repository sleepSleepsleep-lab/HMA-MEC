# -*- coding: utf-8 -*-
"""进程级 seed 并行重跑 E1 HMA-Distill/Hybrid (QoS 强化 Refiner, 论文规模).
用 ProcessPoolExecutor(5) 把 5 个 seed 分派到独立子进程, 利用多核加速."""
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


def run_seed(sd):
    """单个 seed 上跑 2 个 HMA 方法, 返回 (seed, {method: [ep_metrics]})。"""
    from environment import MECEnvironment
    from agent_runner import HMAAgentRunner
    res = {}
    for mname in ("HMA-Distill", "HMA-Hybrid"):
        mode = mname.split("-")[1]
        ep_metrics = []
        for ep in range(N_EPISODES):
            env = MECEnvironment(num_users=K, num_servers=M, seed=SEED + sd + ep)
            runner = HMAAgentRunner(env=env, mode=mode)   # 默认 QoS 强化 Refiner
            e, t, s, sl = [], [], [], []
            for _ in range(N_STEPS):
                st = env._get_state()
                out = runner.run_step(state=st, agents_reuse=True)
                a = compose_action(out['plan'], K, M)
                ns, _, done, info = env.step(a)
                e.append(info['energy']); t.append(info['latency'])
                s.append(info['success_rate']); sl.append(info['priority_sla'])
                if done:
                    break
            ep_metrics.append(dict(
                energy=float(np.mean(e)), latency=float(np.mean(t)),
                success_rate=float(np.mean(s)), priority_sla=float(np.mean(sl))))
        res[mname] = ep_metrics
        print(f"  [seed {sd}] {mname} 完成 ({len(ep_metrics)} ep)", flush=True)
    return sd, res


if __name__ == "__main__":
    print(f"  E1 HMA 并行重跑: {N_SEEDS} seeds × {N_EPISODES} ep × {N_STEPS} 步")
    collected = {}
    ctx = mp.get_context('spawn')     # fork 子进程无法重初始化 CUDA, 用 spawn
    with ProcessPoolExecutor(max_workers=N_SEEDS, mp_context=ctx) as ex:
        for sd, res in ex.map(run_seed, range(N_SEEDS)):
            collected[sd] = res

    # 组装 results dict (与 run_multi_episodes 同结构)
    hma_results = {}
    for mname in ("HMA-Distill", "HMA-Hybrid"):
        per_seed = [r for sd in range(N_SEEDS) for r in collected[sd][mname]]
        agg = {k: [r[k] for r in per_seed]
               for k in ('energy', 'latency', 'success_rate', 'priority_sla')}
        hma_results[mname] = {
            'per_seed': per_seed,
            'mean': {k: float(np.mean(v)) for k, v in agg.items()},
            'std':  {k: float(np.std(v)) for k, v in agg.items()},
        }
        m = hma_results[mname]['mean']
        print(f"  [HMA {mname}] E={m['energy']:.4f} T={m['latency']:.3f} "
              f"suc={m['success_rate']:.2%} sla={m['priority_sla']:.2%}")

    old = load_npz(OUTPUT_NPZ)
    for mname in ("HMA-Distill", "HMA-Hybrid"):
        old[mname] = hma_results[mname]
    save_npz(OUTPUT_NPZ, old)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump({m: {'mean': d['mean'], 'std': d['std'],
                       'n_samples': len(d['per_seed'])}
                   for m, d in old.items()}, f, ensure_ascii=False, indent=2)
    print(f"  已保存 {len(old)} 方法 -> {OUTPUT_NPZ} / {OUTPUT_JSON}")
    print("\n  E1 更新后对比:")
    for m in ('Greedy', 'AllEdge', 'GA', 'B8-SingleLLM', 'HMA-Distill', 'HMA-Hybrid'):
        if m in old:
            d = old[m]['mean']
            print(f"  {m:14s} E={d['energy']:.4f} T={d['latency']:.3f} "
                  f"suc={d['success_rate']:.2%} sla={d['priority_sla']:.2%}")

# -*- coding: utf-8 -*-
"""
E1 主对比重跑 (rerun_e1_fresh.py, 2026-09-01)
==============================================
动机: 原 E1 评估种子 S+s+e 存在碰撞 (n=250 中独立环境仅 54), 本脚本以互不碰撞
的种子结构重跑全部 14 个方法:
  - 评估 episode 种子: SEED + s*100 + e   (s in [0,5), e in [0,50) -> 250 个独立环境)
  - DRL/LeDRL 训练种子: SEED + s*100      (每 seed 一个 agent, 训练预算与原 E1 一致:
      SAC/DDPG/DQN/MADDPG 各 500 ep, LeDRL 300 ep)
方法口径与原 E1 完全一致 (run_e1_main.py / rerun_hma_e1.py 的 spec 配置)。

用法:
  python3 local/rerun_e1_fresh.py --methods Greedy,AllLocal,MPC --shard basic
  python3 local/rerun_e1_fresh.py --methods GA --shard ga_s3 --seeds 3
  python3 local/rerun_e1_fresh.py --methods SAC --shard sac --device cuda
输出: results/e1_fresh_{shard}.npz/.json (与 e1_comparison_v2.npz 同格式)
"""
import os, sys, json, time, argparse, copy
from concurrent.futures import ThreadPoolExecutor
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)
sys.path.insert(0, HERE)

from config import (NUM_USERS, NUM_EDGE_SERVERS, SEED, RESULTS_DIR,
                    MAX_STEPS, CHECKPOINT_DIR)
from environment import MECEnvironment
from local.experiment_common import (run_episode, _instantiate_method,
                                     compose_action, save_npz)

N_SEEDS = 5
N_EPISODES = 50
N_STEPS = 200


def eval_seed(method, seed_i, K, M, n_ep):
    """在 n_ep 个互不碰撞的独立环境上评估一个 seed 的 agent."""
    method_obj = _instantiate_method(
        {'name': method, 'epochs': 500}, K, M,
        seed=SEED + seed_i * 100, policy_path=None)
    # LLM 方法需要 spec 里的 llm_backend / epochs 300
    if method in ('B7-LeDRL', 'B8-SingleLLM'):
        method_obj = _instantiate_method(
            {'name': method, 'epochs': 300, 'llm_backend': 'local_vllm'},
            K, M, seed=SEED + seed_i * 100, policy_path=None)
    if method in ('B7-LeDRL', 'B8-SingleLLM'):
        # LLM 方法: SingleLLM 无状态共享实例可并发; LeDRL 含 lambda 衰减等可变
        # 状态, 采用串行评估 + 每 episode 刷新 LLM 先验缓存 (与原 E1 串行评估
        # 口径一致, 且避免 deepcopy 大对象/线程锁问题)
        if method == 'B8-SingleLLM':
            def _run_one(e):
                env = MECEnvironment(num_users=K, num_servers=M,
                                     seed=SEED + seed_i * 100 + e)
                return run_episode(method_obj, env, n_steps=N_STEPS)
            with ThreadPoolExecutor(max_workers=16) as ex:
                ep_results = list(ex.map(_run_one, range(n_ep)))
        else:
            ep_results = []
            for e in range(n_ep):
                env = MECEnvironment(num_users=K, num_servers=M,
                                     seed=SEED + seed_i * 100 + e)
                method_obj._prior_cache = None   # 每 episode 刷新 LLM 先验
                method_obj._prior_failed = False
                r = run_episode(method_obj, env, n_steps=N_STEPS)
                ep_results.append(r)
        print(f"    [{method}] s{seed_i} 评估完成: "
              f"E={np.mean([r['energy'] for r in ep_results]):.4f} "
              f"T={np.mean([r['latency'] for r in ep_results]):.3f} "
              f"suc={np.mean([r['success_rate'] for r in ep_results]):.2%}",
              flush=True)
        return ep_results

    ep_results = []
    for e in range(n_ep):
        env = MECEnvironment(num_users=K, num_servers=M,
                             seed=SEED + seed_i * 100 + e)
        r = run_episode(method_obj, env, n_steps=N_STEPS)
        ep_results.append(r)
        if (e + 1) % 10 == 0:
            print(f"    [{method}] s{seed_i} ep{e+1}/50 "
                  f"E={r['energy']:.4f} T={r['latency']:.3f} "
                  f"suc={r['success_rate']:.2%}", flush=True)
    return ep_results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--methods', required=True)
    ap.add_argument('--shard', required=True)
    ap.add_argument('--seeds', type=int, default=N_SEEDS)
    ap.add_argument('--episodes', type=int, default=N_EPISODES)
    args = ap.parse_args()

    methods = [m.strip() for m in args.methods.split(',')]
    n_ep = args.episodes
    out = {}
    for method in methods:
        t0 = time.time()
        all_eps = []
        for s in range(args.seeds):
            all_eps.extend(eval_seed(method, s, NUM_USERS, NUM_EDGE_SERVERS,
                                     n_ep))
        agg = {k: [r[k] for r in all_eps]
               for k in ('energy', 'latency', 'success_rate', 'priority_sla')}
        out[method] = {
            'mean': {k: float(np.mean(v)) for k, v in agg.items()},
            'std': {k: float(np.std(v)) for k, v in agg.items()},
            'n_samples': len(all_eps),
            'per_episode': agg,
        }
        print(f"  [{method}] 完成: E={out[method]['mean']['energy']:.4f} "
              f"T={out[method]['mean']['latency']:.3f} "
              f"suc={out[method]['mean']['success_rate']:.2%} "
              f"({time.time()-t0:.0f}s)", flush=True)

    # 保存 (npz: {method}__{metric}__{vals/mean/std})
    npz_dict = {}
    for m, d in out.items():
        for met, v in d['per_episode'].items():
            npz_dict[f"{m}__{met}__vals"] = np.array(v, dtype=np.float32)
            npz_dict[f"{m}__{met}__mean"] = np.array([d['mean'][met]],
                                                     dtype=np.float32)
            npz_dict[f"{m}__{met}__std"] = np.array([d['std'][met]],
                                                    dtype=np.float32)
    npz_path = os.path.join(RESULTS_DIR, f"e1_fresh_{args.shard}.npz")
    np.savez(npz_path, **npz_dict)
    json_path = os.path.join(RESULTS_DIR, f"e1_fresh_{args.shard}.json")
    json_out = {m: {'mean': d['mean'], 'std': d['std'],
                    'n_samples': d['n_samples']} for m, d in out.items()}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_out, f, ensure_ascii=False, indent=1)
    print(f"  已保存 -> {npz_path} / {json_path}")


if __name__ == "__main__":
    main()

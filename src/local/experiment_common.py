# -*- coding: utf-8 -*-
"""
================================================================
实验公共框架 (local/experiment_common.py)
================================================================
本文件提供六类实验 E1-E6 共用的工具与流程，避免重复代码:

  1) run_episode(method, env, n_steps)  → 评估并返回每 episode 指标均值
  2) run_multi_episodes(methods, env, n_seeds, n_episodes, n_steps) → dict{method:{seeds results}}
  3) save_npz(path, results_dict) / load_npz
  4) compose_action(plan, K, M)        → 把 HMA plan 转为 env 可吃的 action 向量

实验脚本只需:
  - 从 baselines 与 agent_runner 取到 method 实现
  - 调用 run_episode / run_multi_episodes 即可
================================================================
"""

import os
import sys
import json
import numpy as np
from typing import Dict, List, Callable, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)
sys.path.insert(0, HERE)  # Add local/ to path for baselines import

from config import (
    NUM_USERS, NUM_EDGE_SERVERS, MAX_STEPS, MAX_EPISODES,
    RESULTS_DIR, SEED,
)
from environment import MECEnvironment
from agent_define import make_agents
from agent_runner import HMAAgentRunner
from baselines import (
    GreedyBaseline, AllLocalBaseline, AllEdgeBaseline, RandomBaseline,
    SACAgent, DDPGAgent, evaluate as eval_baseline,
)


# ============================================================
# 辅助: HMA plan → environment 期望的 action 向量
# ============================================================
def compose_action(plan, K, M):
    """plan {'alpha':(K,), 'server':(K,), 'cloud':(K,) optional} → action vector."""
    has_cloud = 'cloud' in plan and plan['cloud'] is not None
    n_cols = 3 if has_cloud else 2
    action = np.zeros(n_cols * K, dtype=np.float32)
    action[0::n_cols] = np.clip(plan['alpha'], 0.01, 1.0).astype(np.float32)
    action[1::n_cols] = (np.clip(plan['server'], 0, M - 1).astype(np.float32) +
                    0.5) / M
    if has_cloud:
        action[2::n_cols] = np.asarray(plan['cloud'], dtype=np.float32)
    return action


# ============================================================
# 单一个 episode 评估 (通用)
# ============================================================
def run_episode(method, env, n_steps=MAX_STEPS,
                reuse_agents: bool = True,
                method_name: Optional[str] = None,
                policy_path: Optional[str] = None) -> Dict:
    """跑一个 episode, 返回每 step 指标均值。

    参数:
        method:        对象, 有 .name 与 .predict() (.name == 'HMA-MEC' 走 agent_runner)
        env:           MECEnvironment
        n_steps:       每 episode 内部步数
        reuse_agents:  HMA 模式下是否复用 Agent 实例
        policy_path:   仅 HMA-Distill/Hybrid 模式需要

    返回: dict {'energy','latency','success_rate','priority_sla','token_count'(仅 HMA)}
    """
    s = env.reset()
    energy, lat, suc, sla = [], [], [], []
    hma_runner = None
    if method_name is None and hasattr(method, 'name'):
        method_name = method.name

    # 若是 HMA 字符串, 构造一个 HMAAgentRunner
    if isinstance(method, str) and method.startswith("HMA"):
        mode = method.split("-")[1] if "-" in method else "Distill"
        # 仅当 policy_path 非空时才传入, 否则让 HMAAgentRunner 使用默认路径
        hma_kwargs = {'env': env, 'mode': mode}
        if policy_path:
            hma_kwargs['policy_path'] = policy_path
        hma_runner = HMAAgentRunner(**hma_kwargs)

    for _ in range(n_steps):
        if hma_runner is not None:
            out = hma_runner.run_step(state=s, agents_reuse=reuse_agents)
            a = compose_action(out['plan'], env.K, env.M)
        else:
            a = method.predict(s, env)
        ns, _, d, info = env.step(a)
        energy.append(info['energy']); lat.append(info['latency'])
        suc.append(info['success_rate']); sla.append(info['priority_sla'])
        s = ns
        if d: break
    return {
        'energy': float(np.mean(energy)),
        'latency': float(np.mean(lat)),
        'success_rate': float(np.mean(suc)),
        'priority_sla': float(np.mean(sla)),
    }


# ============================================================
# 多 seed × 多 episode 评估
# ============================================================
def run_multi_episodes(method_specs: List[Dict],
                       K: int = NUM_USERS, M: int = NUM_EDGE_SERVERS,
                       n_seeds: int = 3, n_episodes: int = 5,
                       n_steps: int = MAX_STEPS,
                       policy_path: Optional[str] = None,
                       verbose: bool = False) -> Dict:
    """对若干方法在多种子下做评估。

    参数:
        method_specs: [{'name':'Greedy', 'class':GreedyBaseline, ...},
                       {'name':'HMA-MEC', 'hma_mode':'Distill'}, ...]
        K, M:           系统规模
        n_seeds, n_episodes, n_steps: 重复次数
        policy_path:    仅 HMA-Distill/Hybrid 模式需要
    返回:
        {method_name: {
            'per_seed': [[ep1,ep2,...] × n_seeds],
            'mean':     {'energy','latency',...},
            'std':      {...}
        }}
    """
    out = {}
    for spec in method_specs:
        name = spec['name']
        out[name] = {'per_seed': [], 'mean': None, 'std': None}
        for seed_i in range(n_seeds):
            seed = SEED + seed_i
            method = _instantiate_method(spec, K, M, seed, policy_path)
            for ep in range(n_episodes):
                env = MECEnvironment(num_users=K, num_servers=M, seed=seed+ep)
                r = run_episode(method, env, n_steps=n_steps)
                out[name]['per_seed'].append(r)
                if verbose:
                    print(f"    [{name}] seed={seed} ep={ep} "
                          f"E={r['energy']:.5f} T={r['latency']:.3f} "
                          f"suc={r['success_rate']:.2%} "
                          f"sla={r['priority_sla']:.2%}")
        agg = {k: [r[k] for r in out[name]['per_seed']]
               for k in ('energy','latency','success_rate','priority_sla')}
        out[name]['mean'] = {k: float(np.mean(v)) for k, v in agg.items()}
        out[name]['std']  = {k: float(np.std(v))  for k, v in agg.items()}
    return out


def _instantiate_method(spec, K, M, seed, policy_path):
    """根据 spec 实例化一个 method 对象或 HMA 字符串."""
    if spec['name'] == 'Greedy':
        return GreedyBaseline(env=None)
    if spec['name'] == 'AllLocal':
        return AllLocalBaseline(env=None)
    if spec['name'] == 'AllEdge':
        return AllEdgeBaseline(env=None)
    if spec['name'] == 'Random':
        return RandomBaseline(env=None)
    if spec['name'] == 'SAC':
        env = MECEnvironment(num_users=K, num_servers=M, seed=seed)
        sac = SACAgent(env)
        sac.train(env, episodes=spec.get("epochs", 50), verbose=False)
        return sac
    if spec['name'] == 'DDPG':
        env = MECEnvironment(num_users=K, num_servers=M, seed=seed)
        ddpg = DDPGAgent(env)
        ddpg.train(env, episodes=spec.get("epochs", 50), verbose=False)
        return ddpg
    if spec['name'].startswith('HMA'):
        return spec["name"]   # 让 run_episode 内构造 HMAAgentRunner
    raise ValueError(f"未知 method: {spec['name']}")


# ============================================================
# npz 保存/加载
# ============================================================
def save_npz(path: str, results: Dict):
    """把 run_multi_episodes 返回的字典扁平化保存为 npz (便于绘图)."""
    flat = {}
    for m, d in results.items():
        for k in ('energy','latency','success_rate','priority_sla'):
            arr = np.array([r[k] for r in d['per_seed']], dtype=np.float32)
            flat[f"{m}__{k}__vals"]    = arr
            flat[f"{m}__{k}__mean"]   = np.array([d['mean'][k]], dtype=np.float32)
            flat[f"{m}__{k}__std"]    = np.array([d['std'][k]] , dtype=np.float32)
        for k in d.get('extra', {}):
            flat[f"{m}__extra__{k}"] = np.array(d['extra'][k], dtype=np.float32)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **flat)
    print(f"  npz 保存: {path}  ({len(flat)} keys)")
    return path


def load_npz(path: str) -> Dict:
    """加载 save_npz 的 npz, 还原结构."""
    raw = np.load(path, allow_pickle=False)
    methods = set()
    for k in raw.files:
        m, *rest = k.split("__")
        methods.add(m)
    out = {}
    for m in methods:
        per_seed_metrics = []
        per_seed_vals = {k: raw[f"{m}__{k}__vals"]
                          for k in ('energy','latency','success_rate','priority_sla')
                          if f"{m}__{k}__vals" in raw.files}
        # 重新组装 per_seed list
        if per_seed_vals:
            n = len(next(iter(per_seed_vals.values())))
            for i in range(n):
                per_seed_metrics.append({k: float(per_seed_vals[k][i])
                                          for k in per_seed_vals})
        out[m] = {
            'per_seed': per_seed_metrics,
            'mean': {k: float(raw[f"{m}__{k}__mean"][0])
                      for k in ('energy','latency','success_rate','priority_sla')
                      if f"{m}__{k}__mean" in raw.files},
            'std':  {k: float(raw[f"{m}__{k}__std"][0])
                      for k in ('energy','latency','success_rate','priority_sla')
                      if f"{m}__{k}__std"  in raw.files},
        }
    return out
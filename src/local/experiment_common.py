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
from concurrent.futures import ThreadPoolExecutor
from local.results_store import Recorder

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)
sys.path.insert(0, HERE)  # Add local/ to path for baselines import

from config import (
    NUM_USERS, NUM_EDGE_SERVERS, MAX_STEPS, MAX_EPISODES,
    RESULTS_DIR, SEED, GA_POP_SIZE, GA_GENERATIONS,
)
from environment import MECEnvironment
from agent_define import make_agents
from agent_runner import HMAAgentRunner
from baselines import (
    GreedyBaseline, AllLocalBaseline, AllEdgeBaseline, RandomBaseline,
    SACAgent, DDPGAgent, evaluate as eval_baseline,
)
from baseline_dqn import DQNAgent


# ============================================================
# 辅助: HMA plan → environment 期望的 action 向量
# ============================================================
def compose_action(plan, K, M):
    """plan {'alpha':(K,), 'server':(K,)} → (2*K,) float32."""
    action = np.zeros(2 * K, dtype=np.float32)
    action[0::2] = np.clip(plan['alpha'], 0.01, 1.0).astype(np.float32)
    action[1::2] = (np.clip(plan['server'], 0, M - 1).astype(np.float32) +
                    0.5) / M
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
                       verbose: bool = False,
                       save_callback: Optional[Callable] = None,
                       record_experiment: Optional[str] = None) -> Dict:
    """对若干方法在多种子下做评估。

    参数:
        method_specs: [{'name':'Greedy', 'class':GreedyBaseline, ...},
                       {'name':'HMA-MEC', 'hma_mode':'Distill'}, ...]
        K, M:           系统规模
        n_seeds, n_episodes, n_steps: 重复次数
        policy_path:    仅 HMA-Distill/Hybrid 模式需要
        save_callback:  每完成一个方法后回调(部分结果 dict), 用于长跑增量保存
    返回:
        {method_name: {
            'per_seed': [[ep1,ep2,...] × n_seeds],
            'mean':     {'energy','latency',...},
            'std':      {...}
        }}

    性能说明 (2026-08 整改): B7-LeDRL / B8-SingleLLM 每个 seed 相互独立
    (各自独立的 SAC 训练/LLM 调用), 且 LLM 调用是主要耗时 (vLLM 并发批处理
    已实测 5 并发无额外时延)。因此对这两个方法按 seed 并行执行, 总时长
    约缩短为 1/n_seeds。其余方法维持串行语义不变。
    """
    # B7/B8 每 seed 独立且以 LLM 调用为主, 并行执行利用 vLLM 并发批处理
    LLM_PARALLEL_NAMES = ('B7-LeDRL', 'B8-SingleLLM')

    rec = (Recorder(record_experiment,
                    config={"K": K, "M": M, "n_seeds": n_seeds,
                            "n_episodes": n_episodes, "n_steps": n_steps})
           if record_experiment else None)
    out = {}
    for spec in method_specs:
        name = spec['name']
        out[name] = {'per_seed': [], 'mean': None, 'std': None}
        parallel = name in LLM_PARALLEL_NAMES and n_seeds > 1

        def _run_one_seed(seed_i):
            seed = SEED + seed_i
            method = _instantiate_method(spec, K, M, seed, policy_path)
            ep_results = []
            for ep in range(n_episodes):
                env = MECEnvironment(num_users=K, num_servers=M, seed=seed+ep)
                # 2026-08 整改: 透传 spec 里的 policy_path, 否则 HMA-Distill/
                # Hybrid 永远使用默认权重 (E9 多后端按模型权重评估曾因此失效)
                r = run_episode(method, env, n_steps=n_steps,
                                policy_path=spec.get('policy_path'))
                ep_results.append(r)
                if rec is not None:
                    rec.add(method=name, seed=seed, episode=ep, metrics=r)
                if verbose:
                    print(f"    [{name}] seed={seed} ep={ep} "
                          f"E={r['energy']:.5f} T={r['latency']:.3f} "
                          f"suc={r['success_rate']:.2%} "
                          f"sla={r['priority_sla']:.2%}")
            return ep_results

        if parallel:
            print(f"  [{name}] seed 级并行执行 (n_seeds={n_seeds}, "
                  f"vLLM 并发批处理)...")
            with ThreadPoolExecutor(max_workers=n_seeds) as pool:
                per_seed_lists = list(
                    pool.map(_run_one_seed, range(n_seeds)))
            # 保持 per_seed 平铺语义 ([ep1..epN] × n_seeds), 与串行版一致
            out[name]['per_seed'] = [
                r for seed_eps in per_seed_lists for r in seed_eps]
        else:
            for seed_i in range(n_seeds):
                out[name]['per_seed'].extend(_run_one_seed(seed_i))

        agg = {k: [r[k] for r in out[name]['per_seed']]
               for k in ('energy','latency','success_rate','priority_sla')}
        out[name]['mean'] = {k: float(np.mean(v)) for k, v in agg.items()}
        out[name]['std']  = {k: float(np.std(v))  for k, v in agg.items()}
        print(f"  [{name}] 完成: E={out[name]['mean']['energy']:.5f} "
              f"T={out[name]['mean']['latency']:.3f} "
              f"suc={out[name]['mean']['success_rate']:.2%}")
        if save_callback:
            save_callback(out)
    if rec is not None:
        rec.close()
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
    # ---- 2026-08 整改: MPC 滚动时域 / DQN 深增强基线 ----
    if spec['name'] == 'DQN':
        env = MECEnvironment(num_users=K, num_servers=M, seed=seed)
        dqn = DQNAgent(env)
        dqn.train(env, episodes=spec.get("epochs", 50), verbose=False)
        return dqn
    if spec['name'] == 'MPC':
        from local.baseline_mpc import MPCBaseline
        return MPCBaseline(omega=(0.5, 0.5))
    # ---- C6 整改: 注册新基线 GA / B7-LeDRL / B8-SingleLLM ----
    if spec['name'] == 'GA':
        from local.ga_baseline import GAOffloadBaseline
        return GAOffloadBaseline(
            pop_size=spec.get("pop", GA_POP_SIZE),
            generations=spec.get("gens", GA_GENERATIONS),
            omega=spec.get("omega", (0.5, 0.5)))
    if spec['name'] in ('B8-SingleLLM', 'B7-LeDRL'):
        from local.baselines_llm import (SingleLLMBaseline, LeDRLBaseline,
                                          LLMPromptBuilder, FakeLLMClient)
        backend = spec.get('llm_backend')
        if backend == 'heuristic_proxy':      # --smoke: 假 LLM 验证管线
            llm = FakeLLMClient(K=K, M=M)
        else:
            from llm_client import get_llm_client
            llm = get_llm_client(backend=backend)
        if spec['name'] == 'B8-SingleLLM':
            return SingleLLMBaseline(llm_client=llm,
                                     prompt_builder=LLMPromptBuilder())
        env = MECEnvironment(num_users=K, num_servers=M, seed=seed)
        ledrl = LeDRLBaseline(env, llm_client=llm,
                              prompt_builder=LLMPromptBuilder())
        ledrl.train(env, episodes=spec.get("epochs", 300), verbose=False)
        return ledrl
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
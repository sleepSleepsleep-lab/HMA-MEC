# -*- coding: utf-8 -*-
"""
================================================================
E21 (2026-08): 验证器精化基线扩展 —— 大规模版本 (E20 的正式化)
================================================================
把 E20 的 n=10 机制演示升级为与 E1 完全同口径的大规模实验:
  - 4 个 DRL/MARL 基线 (SAC/DDPG/DQN/MADDPG) × 5 seeds 训练
    (500 episode 预算, 与 E1 相同)
  - 每 seed 在其对应 50 个评估 episode 上同时记录两种口径:
      raw     : 策略原始输出 (E1 口径)
      refined : 策略输出 → 同一验证器逐用户精化 → 执行
  - 汇总 n=250/变体 的 mean/std; HMA-Distill 对照取 E1 同结构
    (env seed = SEED+sd+ep) 的 n=250 结果 (90.40%)
目的: 以与主实验相同的统计功效确认"精化后性能由验证器主导、
任意学习策略+验证器与 HMA 相当"的边界, 以及"加装精化不改变
相对结论"的陈述。
================================================================
"""
import os, sys, json, time, functools
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)
sys.path.insert(0, os.path.join(SRC, "local"))

from config import NUM_USERS, NUM_EDGE_SERVERS, RESULTS_DIR, SEED, MAX_STEPS
from environment import MECEnvironment
from local.baselines import SACAgent, DDPGAgent
from local.baseline_dqn import DQNAgent
from local.baseline_maddpg import MADDPGAgent
from local.plan_refiner import PlanRefiner
from local.results_store import Recorder

K, M = NUM_USERS, NUM_EDGE_SERVERS
EP_TRAIN = 500
N_SEEDS = 5
EP_EVAL = 50
OUT_JSON = os.path.join(RESULTS_DIR, "e21_refined_baselines_full.json")


def act_to_plan(a, K, M):
    alpha = np.clip(np.asarray(a[0::2], dtype=float), 0.01, 1.0)
    server = np.clip(np.floor(np.asarray(a[1::2], dtype=float) * M),
                     0, M - 1).astype(int)
    return alpha, server


def build_agent(name, env):
    if name == "SAC":
        return SACAgent(env)
    if name == "DDPG":
        return DDPGAgent(env)
    if name == "DQN":
        return DQNAgent(env)
    return MADDPGAgent(env)


def run_seed(name, sd):
    """单 seed: 训练 (500ep 预算) → 50 评估 episode × (raw/refined)."""
    _rec = Recorder("e21", config={"name": name, "sd": sd})
    train_env = MECEnvironment(num_users=K, num_servers=M, seed=SEED + sd)
    agent = build_agent(name, train_env)
    t1 = time.time()
    ret = agent.train(train_env, episodes=EP_TRAIN, verbose=False)
    n_ep = len(ret) if hasattr(ret, '__len__') and not hasattr(ret, 'predict') \
        else EP_TRAIN
    refiner = PlanRefiner(omega=(0.5, 0.5))
    raw_all, ref_all = [], []
    for ep in range(EP_EVAL):
        e = MECEnvironment(num_users=K, num_servers=M, seed=SEED + sd + ep)
        r_raw = _run_episode(e, agent, refiner, use_refine=False)
        e2 = MECEnvironment(num_users=K, num_servers=M, seed=SEED + sd + ep)
        r_ref = _run_episode(e2, agent, refiner, use_refine=True)
        raw_all.append(r_raw)
        ref_all.append(r_ref)
        _rec.add(method=name, seed=sd, episode=ep, metrics=r_raw, tag="raw")
        _rec.add(method=name, seed=sd, episode=ep, metrics=r_ref, tag="refined")
    print(f"  [seed {sd}] {name}: 训练 {n_ep}ep ({time.time()-t1:.0f}s) "
          f"raw={np.mean([r['suc'] for r in raw_all]):.1%} "
          f"refined={np.mean([r['suc'] for r in ref_all]):.1%}", flush=True)
    _rec.close()
    return raw_all, ref_all


def _run_episode(env, agent, refiner, use_refine):
    E, T, S, SL = [], [], [], []
    st = env.reset()
    for _ in range(MAX_STEPS):
        a = agent.predict(st, env)
        alpha, server = act_to_plan(a, env.K, env.M)
        if use_refine:
            alpha, server = refiner.refine(env, alpha, server)
        act = np.zeros(2 * env.K, np.float32)
        act[0::2] = np.clip(alpha, 0.01, 1.0)
        act[1::2] = (np.clip(server, 0, env.M - 1) + 0.5) / env.M
        ns, _, d, info = env.step(act)
        E.append(info['energy']); T.append(info['latency'])
        S.append(info['success_rate']); SL.append(info['priority_sla'])
        st = ns
        if d:
            break
    return dict(E=float(np.mean(E)), T=float(np.mean(T)),
                suc=float(np.mean(S)), sla=float(np.mean(SL)))


def agg(recs):
    n = len(recs)
    return {
        "mean": {k: float(np.mean([r[k] for r in recs])) for k in recs[0]},
        "std": {k: float(np.std([r[k] for r in recs])) for k in recs[0]},
        "n": n,
    }


def main():
    t0 = time.time()
    print("=" * 62)
    print(f"  E21: 验证器精化基线扩展 (大规模, {N_SEEDS} seeds × "
          f"{EP_EVAL} ep = {N_SEEDS*EP_EVAL}/变体)")
    print("=" * 62)
    ctx = __import__("multiprocessing").get_context("spawn")
    out = {}
    for name in ["SAC", "DDPG", "DQN", "MADDPG"]:
        with ProcessPoolExecutor(max_workers=N_SEEDS, mp_context=ctx) as ex:
            results = list(ex.map(functools.partial(run_seed, name),
                                  range(N_SEEDS)))
        raw_all = [r for rs in results for r in rs[0]]
        ref_all = [r for rs in results for r in rs[1]]
        out[name] = {"raw": agg(raw_all), "refined": agg(ref_all),
                     "per_episode": {"raw": raw_all, "refined": ref_all}}
        m = out[name]
        print(f"  {name:8s} raw={m['raw']['mean']['suc']:.2%}±{m['raw']['std']['suc']:.2%} "
              f"refined={m['refined']['mean']['suc']:.2%}±{m['refined']['std']['suc']:.2%} "
              f"(+{(m['refined']['mean']['suc']-m['raw']['mean']['suc'])*100:.1f}pp)",
              flush=True)
    json.dump(out, open(OUT_JSON, "w"), indent=2, ensure_ascii=False)
    print(f"  保存 -> {OUT_JSON} (总耗时 {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
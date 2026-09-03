# -*- coding: utf-8 -*-
"""
================================================================
E20 (2026-08): 验证器精化的基线扩展对照 —— "加装精化不改变相对结论"
================================================================
将反事实验证器精化 (PlanRefiner) 以相同方式接入四个已训练的
单体/多智能体 DRL 基线 (SAC/DDPG/DQN/MADDPG, 各 1 seed × 500ep,
与 E1 同预算), 比较:
  raw     : 策略原始输出直接执行 (E1 口径)
  refined : 策略输出 → 同一验证器逐用户精化 → 执行
目的: 把 E3 HMA-RandomSeed 的间接证据升级为直接证据——验证器作为
框架内部组件可接入任意策略; 精化提升所有策略, 但策略间相对结论
(DRL<<HMA) 不变。
================================================================
"""
import os, sys, json, time
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
EP_EVAL = 10
OUT_JSON = os.path.join(RESULTS_DIR, "e20_refined_baselines.json")


def act_to_plan(a, K, M):
    alpha = np.clip(np.asarray(a[0::2], dtype=float), 0.01, 1.0)
    server = np.clip(np.floor(np.asarray(a[1::2], dtype=float) * M),
                     0, M - 1).astype(int)
    return alpha, server


def run_env(env, refiner, raw_or_refined, use_refine, n_steps):
    E, T, S, SL = [], [], [], []
    st = env.reset()
    for _ in range(n_steps):
        a = raw_or_refined.predict(st, env)
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


def build_agent(name, env):
    if name == "SAC":
        return SACAgent(env)
    if name == "DDPG":
        return DDPGAgent(env)
    if name == "DQN":
        return DQNAgent(env)
    return MADDPGAgent(env)


def main():
    t0 = time.time()
    print("=" * 62)
    print("  E20: 验证器精化的 DRL 基线扩展对照")
    print("=" * 62)
    out = {}
    _rec = Recorder("e20")
    for name in ["SAC", "DDPG", "DQN", "MADDPG"]:
        env = MECEnvironment(num_users=K, num_servers=M, seed=SEED)
        agent = build_agent(name, env)
        t1 = time.time()
        ret = agent.train(env, episodes=EP_TRAIN, verbose=False)
        n_ep = len(ret) if hasattr(ret, '__len__') and not hasattr(ret, 'predict') \
            else EP_TRAIN
        print(f"  [{name}] 训练 {n_ep}ep ({time.time()-t1:.0f}s)", flush=True)
        refiner = PlanRefiner(omega=(0.5, 0.5))
        recs = {"raw": [], "refined": []}
        for ep in range(EP_EVAL):
            for tag, use_refine in [("raw", False), ("refined", True)]:
                e = MECEnvironment(num_users=K, num_servers=M,
                                   seed=SEED + 100 + ep)
                r = run_env(e, refiner, agent, use_refine, MAX_STEPS)
                recs[tag].append(r)
                _rec.add(method=name, seed=sd, episode=ep, metrics=r, tag=tag)
        m_raw = {k: float(np.mean([r[k] for r in recs["raw"]]))
                 for k in recs["raw"][0]}
        m_ref = {k: float(np.mean([r[k] for r in recs["refined"]]))
                 for k in recs["refined"][0]}
        out[name] = {"raw": m_raw, "refined": m_ref,
                     "suc_gain_pp": (m_ref["suc"] - m_raw["suc"]) * 100,
                     "epochs_used": n_ep}
        print(f"  {name:8s} raw  : E={m_raw['E']:.3f} T={m_raw['T']:.3f} "
              f"suc={m_raw['suc']:.1%}")
        print(f"  {name:8s} refin: E={m_ref['E']:.3f} T={m_ref['T']:.3f} "
              f"suc={m_ref['suc']:.1%} (+{(m_ref['suc']-m_raw['suc'])*100:.1f}pp)",
              flush=True)
    _rec.close()
    json.dump(out, open(OUT_JSON, "w"), indent=2, ensure_ascii=False)
    print(f"  保存 -> {OUT_JSON} (总耗时 {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
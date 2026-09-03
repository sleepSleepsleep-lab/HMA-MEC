# -*- coding: utf-8 -*-
"""
E20(i) 验证器精化基线扩展重跑 (rerun_e20_fresh.py, 2026-09-02)
==============================================================
- 种子: 训练 SEED + s*100 (500ep 预算, 与 E1 重跑一致), 评估 SEED + s*100 + e
  (50 ep/seed, 250 个真独立 episode)
- 每 episode 配对: raw (策略原始输出) / refined (策略输出 -> 同一
  PlanRefiner(ω=(0.5,0.5)) 逐用户精化), 两口径用同种子独立 env (任务/信道序列相同)
- 用法: python3 local/rerun_e20_fresh.py --method SAC --seed-index 0
  输出: results/e20_fresh_{method}_s{seed}.npz (逐 episode vals)
"""
import os, sys, time, argparse
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

K, M = NUM_USERS, NUM_EDGE_SERVERS
EP_TRAIN = 500
EP_EVAL = 50


def build_agent(name, env):
    if name == "SAC":
        return SACAgent(env)
    if name == "DDPG":
        return DDPGAgent(env)
    if name == "DQN":
        return DQNAgent(env)
    return MADDPGAgent(env)


def act_to_plan(a, K, M):
    alpha = np.clip(np.asarray(a[0::2], dtype=float), 0.01, 1.0)
    server = np.clip(np.floor(np.asarray(a[1::2], dtype=float) * M),
                     0, M - 1).astype(int)
    return alpha, server


def run_episode(env, agent, refiner, use_refine):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', required=True)
    ap.add_argument('--seed-index', type=int, required=True)
    args = ap.parse_args()
    name, sd = args.method, args.seed_index

    train_env = MECEnvironment(num_users=K, num_servers=M,
                               seed=SEED + sd * 100)
    agent = build_agent(name, train_env)
    t1 = time.time()
    agent.train(train_env, episodes=EP_TRAIN, verbose=False)
    print(f"  [{name} s{sd}] 训练完成 ({time.time()-t1:.0f}s)", flush=True)

    refiner = PlanRefiner(omega=(0.5, 0.5))
    raw_all, ref_all = [], []
    for ep in range(EP_EVAL):
        seed = SEED + sd * 100 + ep
        r_raw = run_episode(MECEnvironment(num_users=K, num_servers=M,
                                           seed=seed), agent, refiner, False)
        r_ref = run_episode(MECEnvironment(num_users=K, num_servers=M,
                                           seed=seed), agent, refiner, True)
        raw_all.append(r_raw)
        ref_all.append(r_ref)
        if (ep + 1) % 10 == 0:
            print(f"    [{name} s{sd}] ep{ep+1}/50 "
                  f"raw={np.mean([r['suc'] for r in raw_all]):.1%} "
                  f"ref={np.mean([r['suc'] for r in ref_all]):.1%}",
                  flush=True)

    out = {}
    for tag, recs in (("raw", raw_all), ("refined", ref_all)):
        for met, key in (('E', 'energy'), ('T', 'latency'),
                         ('suc', 'success_rate'), ('sla', 'priority_sla')):
            v = np.array([r[met] for r in recs], dtype=np.float32)
            out[f"{name}__{key}__{tag}__vals"] = v
            out[f"{name}__{key}__{tag}__mean"] = np.array([v.mean()],
                                                          dtype=np.float32)
            out[f"{name}__{key}__{tag}__std"] = np.array([v.std()],
                                                         dtype=np.float32)
    npz_path = os.path.join(RESULTS_DIR, f"e20_fresh_{name}_s{sd}.npz")
    np.savez(npz_path, **out)
    print(f"  [{name} s{sd}] 完成: raw suc="
          f"{np.mean([r['suc'] for r in raw_all]):.2%} refined suc="
          f"{np.mean([r['suc'] for r in ref_all]):.2%} "
          f"({time.time()-t1:.0f}s) -> {npz_path}", flush=True)


if __name__ == "__main__":
    main()

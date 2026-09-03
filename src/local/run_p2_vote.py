# -*- coding: utf-8 -*-
"""
P2-6 (2026-08): 投票基线 —— 回应 "Debate or Vote" (NeurIPS 2025) 质疑
================================================================
在同一验证器精化器下, 增加"投票"种子: 对每个状态做 5 次独立单次 LLM
采样, 服务器选择按多数表决、卸载比例取均值, 与 E15 的单次 LLM 种子对照
(normal + dishonest 场景, 10 种子 x 40 步)。辩论 (65 调用/步) vs
投票 (5 调用/步) vs 单次 (1 调用/步) 的成本-收益对比。
输出: results/p2_vote.json
================================================================
"""
import os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)
sys.path.insert(0, os.path.join(SRC, "local"))

from config import NUM_USERS, NUM_EDGE_SERVERS, SEED, RESULTS_DIR
from environment import MECEnvironment
from local.plan_refiner import PlanRefiner
from local.results_store import Recorder

K, M = NUM_USERS, NUM_EDGE_SERVERS
N_SEEDS = 10
N_STEPS = 40
OUT_JSON = os.path.join(RESULTS_DIR, "p2_vote.json")

SYSTEM_PLAN = (
    "You are an MEC task-offloading orchestrator. Given the current system "
    "state, decide for each user k the fraction alpha_k (0=all offload to "
    "edge, 1=all local) and the server m_k in 0..%d. "
    "Favor low latency for tight deadlines. " % (M - 1) +
    'Output ONLY JSON: {"alpha": [.. 8 floats ..], "server": [.. 8 ints ..]}')


def make_env(sd, scenario):
    env = MECEnvironment(num_users=K, num_servers=M, seed=SEED + sd)
    env.reset()
    if scenario == "dishonest":
        env.tasks[0]['priority'] = 3
        env.tasks[1]['C'] *= 3.0
    return env


def _single_vote(env, llm, st):
    txt = env.state_to_text(state=st)
    r = llm.chat_json(SYSTEM_PLAN, txt)
    a = np.asarray(r.get('alpha', []), dtype=float)
    s = np.asarray(r.get('server', []), dtype=int)
    if len(a) != K or len(s) != K:
        a = np.full(K, 0.5); s = np.abs(env.channels).argmax(1)
    a = np.clip(a, 0.01, 1.0)
    s = np.clip(s, 0, M - 1).astype(int)
    return a, s


def seed_vote(env, llm, st, n_votes=5):
    """5 次独立单次采样 (并发): server 多数表决, alpha 取均值."""
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
        futs = [ex.submit(_single_vote, env, llm, st) for _ in range(n_votes)]
        res = [f.result() for f in futs]
    alphas = [r[0] for r in res]
    servers = [r[1] for r in res]
    servers = np.asarray(servers)          # (n_votes, K)
    alphas = np.asarray(alphas)            # (n_votes, K)
    # 多数表决 (平票取第一个出现的)
    voted = np.array([np.bincount(servers[:, k], minlength=M).argmax()
                      for k in range(K)])
    alpha_m = alphas.mean(0)
    return alpha_m, voted


def run_ep(env, llm, refiner):
    E, T, S, SL = [], [], [], []
    for _ in range(N_STEPS):
        st = env._get_state()
        a, s = seed_vote(env, llm, st)
        ar, sr = refiner.refine(env, a.copy(), s.copy())
        act = np.zeros(2 * K, np.float32)
        act[0::2] = np.clip(ar, 0.01, 1.0)
        act[1::2] = (np.clip(sr, 0, M - 1) + 0.5) / M
        ns, _, d, info = env.step(act)
        E.append(info['energy']); T.append(info['latency'])
        S.append(info['success_rate']); SL.append(info['priority_sla'])
        if d:
            break
    return dict(E=float(np.mean(E)), T=float(np.mean(T)),
                suc=float(np.mean(S)), sla=float(np.mean(SL)))


def main():
    from llm_client import get_llm_client
    llm = get_llm_client()
    llm._respect_rate_limit = lambda: None
    import threading
    print("=" * 60)
    print("  P2-6: 投票基线 (5 采样多数表决, n=%d x %d 步, 并发)"
          % (N_SEEDS, N_STEPS))
    print("=" * 60)
    _rec = Recorder("p2_vote")
    _lock = threading.Lock()
    out = {}
    import concurrent.futures
    for sc in ["normal", "dishonest"]:
        rows = []
        def _one(sd):
            env = make_env(sd, sc)
            refiner = PlanRefiner(omega=(0.5, 0.5))
            r = run_ep(env, llm, refiner)
            r['calls_per_step'] = 5.0
            with _lock:
                _rec.add(method="Vote", seed=sd, episode=sd, metrics=r,
                         scenario=sc)
            return r
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            for r in ex.map(_one, range(N_SEEDS)):
                rows.append(r)
        mean = {k: float(np.mean([r[k] for r in rows]))
                for k in ('E', 'T', 'suc', 'sla')}
        mean['calls_per_step'] = 5.0
        out[sc] = mean
        print(f"  {sc:10s} E={mean['E']:.4f} T={mean['T']:.4f} "
              f"suc={mean['suc']:.1%} sla={mean['sla']:.1%}")
    _rec.close()
    json.dump(out, open(OUT_JSON, "w"), indent=2, ensure_ascii=False)
    print(f"  保存 -> {OUT_JSON}")


if __name__ == "__main__":
    main()

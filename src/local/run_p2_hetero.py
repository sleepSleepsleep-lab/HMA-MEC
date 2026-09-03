# -*- coding: utf-8 -*-
"""
P2-3 (2026-08): 异构服务器变体 —— 服务器算力差异化 F_m ∈ {5,10,20,40} GHz
================================================================
响应"服务器同构"的建模质疑: 边缘服务器算力异构, 验证器闭环与语义先验
在不重训前提下的适配性。对比: HMA-Distill / MPC / GA / Greedy, n=10。
输出: results/p2_hetero.json
================================================================
"""
import os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)

from config import (NUM_USERS, NUM_EDGE_SERVERS, SEED, RESULTS_DIR,
                    CHECKPOINT_DIR)
from environment import MECEnvironment
from agent_runner import HMAAgentRunner
from local.plan_refiner import PlanRefiner
from local.experiment_common import compose_action
from local.results_store import Recorder

K, M = NUM_USERS, NUM_EDGE_SERVERS
N_EPISODES = 10
N_STEPS = 100
HETERO_F = [5.0, 10.0, 20.0, 40.0]   # GHz (异构)
POLICY_PATH = os.path.join(CHECKPOINT_DIR, "distilled_policy.pth")
OUT_JSON = os.path.join(RESULTS_DIR, "p2_hetero.json")


class HeteroEnv(MECEnvironment):
    """异构服务器算力变体."""

    def __init__(self, num_users=K, num_servers=M, seed=0):
        super().__init__(num_users=num_users, num_servers=num_servers, seed=seed)
        self.f_edge = np.array(HETERO_F[:M], dtype=np.float64) * 1e9


class _Greedy:
    def predict(self, st, e):
        act = np.zeros(2 * e.K, np.float32)
        act[0::2] = 0.5
        act[1::2] = (np.abs(e.channels).argmax(1) + 0.5) / e.M
        return act


def run_env(env, kind, obj):
    E, T, S, SL = [], [], [], []
    for _ in range(N_STEPS):
        st = env._get_state()
        if kind == "HMA-Distill":
            out = obj.run_step(state=st, agents_reuse=True)
            act = compose_action(out['plan'], env.K, env.M)
        elif kind == "MPC":
            a = np.full(env.K, 0.5)
            s = np.abs(env.channels).argmax(1).astype(int)
            ar, sr = obj.refine(env, a.copy(), s.copy())
            act = np.zeros(2 * env.K, np.float32)
            act[0::2] = np.clip(ar, 0.01, 1.0)
            act[1::2] = (np.clip(sr, 0, env.M - 1) + 0.5) / env.M
        else:
            act = obj.predict(st, env)
        ns, _, d, info = env.step(act)
        E.append(info['energy']); T.append(info['latency'])
        S.append(info['success_rate']); SL.append(info['priority_sla'])
        if d:
            break
    return dict(E=float(np.mean(E)), T=float(np.mean(T)),
                suc=float(np.mean(S)), sla=float(np.mean(SL)))


def main():
    t0 = time.time()
    print("=" * 60)
    print("  P2-3: 异构服务器变体 F_m =", HETERO_F, "GHz (n=%d)" % N_EPISODES)
    print("=" * 60)
    _rec = Recorder("p2_hetero", config={"f_edge": HETERO_F})
    out = {}
    for kind in ["HMA-Distill", "MPC", "GA", "Greedy"]:
        rows = []
        for sd in range(N_EPISODES):
            env = HeteroEnv(num_users=K, num_servers=M, seed=SEED + sd)
            env.reset()
            if kind == "HMA-Distill":
                obj = HMAAgentRunner(env=env, mode="Distill",
                                     policy_path=POLICY_PATH, agents=None)
            elif kind == "MPC":
                obj = type('MPC', (), {'refine': PlanRefiner(omega=(0.5, 0.5)).refine})()
            elif kind == "GA":
                from local.ga_baseline import GAOffloadBaseline as GABaseline
                obj = GABaseline()
            else:
                obj = _Greedy()
            r = run_env(env, kind, obj)
            rows.append(r)
            _rec.add(method=kind, seed=sd, episode=sd, metrics=r)
        mean = {k: float(np.mean([r[k] for r in rows]))
                for k in ('E', 'T', 'suc', 'sla')}
        out[kind] = mean
        print(f"  {kind:12s} E={mean['E']:.4f} T={mean['T']:.4f} "
              f"suc={mean['suc']:.1%} sla={mean['sla']:.1%}")
    _rec.close()
    json.dump(out, open(OUT_JSON, "w"), indent=2, ensure_ascii=False)
    print(f"  保存 -> {OUT_JSON} (耗时 {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()

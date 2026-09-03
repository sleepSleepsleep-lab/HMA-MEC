# -*- coding: utf-8 -*-
"""
P2-4 (2026-08): 服务器负载均衡方差指标
================================================================
在基准场景下记录各方法运行期每步的服务器负载 (server_load, CPU 周期),
报告跨服务器负载的方差 (负载均衡程度) 与均值。对比: HMA-Distill / MPC / GA。
输出: results/p2_loadvar.json
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
N_EPISODES = 5
N_STEPS = 100
POLICY_PATH = os.path.join(CHECKPOINT_DIR, "distilled_policy.pth")
OUT_JSON = os.path.join(RESULTS_DIR, "p2_loadvar.json")


class _Greedy:
    def predict(self, st, e):
        act = np.zeros(2 * e.K, np.float32)
        act[0::2] = 0.5
        act[1::2] = (np.abs(e.channels).argmax(1) + 0.5) / e.M
        return act


def run_env(env, kind, obj):
    """返回指标 + 逐步服务器负载序列 (用于方差统计)."""
    E, T, S, SL = [], [], [], []
    loads = []   # 每步 (M,) 服务器负载 (CPU 周期)
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
        loads.append(env.server_load.copy() if hasattr(env, 'server_load')
                     else np.zeros(M))
        if d:
            break
    loads = np.asarray(loads)                       # (steps, M)
    norm = np.maximum(loads.sum(1, keepdims=True), 1e-9)
    frac = loads / norm                              # 每步服务器负载占比
    var_per_step = frac.var(1)                       # 每步跨服务器方差
    return dict(E=float(np.mean(E)), T=float(np.mean(T)),
                suc=float(np.mean(S)), sla=float(np.mean(SL)),
                load_var=float(np.mean(var_per_step)),   # 负载方差均值 (占比口径)
                load_std=float(np.mean(np.sqrt(var_per_step))))


def main():
    t0 = time.time()
    print("=" * 60)
    print("  P2-4: 服务器负载均衡方差 (n=%d x %d 步)" % (N_EPISODES, N_STEPS))
    print("=" * 60)
    _rec = Recorder("p2_loadvar")
    out = {}
    for kind in ["HMA-Distill", "MPC", "GA"]:
        rows = []
        for sd in range(N_EPISODES):
            env = MECEnvironment(num_users=K, num_servers=M, seed=SEED + sd)
            env.reset()
            if kind == "HMA-Distill":
                obj = HMAAgentRunner(env=env, mode="Distill",
                                     policy_path=POLICY_PATH, agents=None)
            elif kind == "MPC":
                obj = type('MPC', (), {'refine': PlanRefiner(omega=(0.5, 0.5)).refine})()
            else:
                from local.ga_baseline import GAOffloadBaseline as GABaseline
                obj = GABaseline()
            r = run_env(env, kind, obj)
            rows.append(r)
            _rec.add(method=kind, seed=sd, episode=sd,
                     metrics={k: v for k, v in r.items()})
        mean = {k: float(np.mean([r[k] for r in rows]))
                for k in ('E', 'T', 'suc', 'sla', 'load_var', 'load_std')}
        out[kind] = mean
        print(f"  {kind:12s} E={mean['E']:.4f} T={mean['T']:.4f} "
              f"suc={mean['suc']:.1%} | 负载方差={mean['load_var']:.4f} "
              f"标准差={mean['load_std']:.4f}")
    _rec.close()
    json.dump(out, open(OUT_JSON, "w"), indent=2, ensure_ascii=False)
    print(f"  保存 -> {OUT_JSON} (耗时 {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()

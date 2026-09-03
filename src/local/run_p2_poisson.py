# -*- coding: utf-8 -*-
"""
P2-1 (2026-08): 泊松到达/动态任务变体 —— 回应"为何不考虑动态到达"
================================================================
在环境 step 后, 每个用户以概率 p_arr 刷新任务参数 (模拟泊松式动态到达,
替换已完成的旧任务), 其余机制不变 (不重训, 验证器闭环兜底)。
对比: HMA-Distill / MPC / GA / Greedy, n=10 episode x 100 步。
输出: results/p2_poisson.json
================================================================
"""
import os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)

from config import (NUM_USERS, NUM_EDGE_SERVERS, SEED, RESULTS_DIR,
                    CHECKPOINT_DIR, TASK_DATA_MIN, TASK_DATA_MAX,
                    TASK_CYCLES_PER_BIT, TASK_DEADLINE_MIN,
                    TASK_DEADLINE_MAX, TASK_PRIORITY_PROB)
from environment import MECEnvironment
from agent_runner import HMAAgentRunner
from local.plan_refiner import PlanRefiner
from local.experiment_common import compose_action
from local.results_store import Recorder

K, M = NUM_USERS, NUM_EDGE_SERVERS
N_EPISODES = 10
N_STEPS = 100
P_ARR = 0.15          # 每步每用户任务刷新概率 (泊松到达近似)
POLICY_PATH = os.path.join(CHECKPOINT_DIR, "distilled_policy.pth")
OUT_JSON = os.path.join(RESULTS_DIR, "p2_poisson.json")


class PoissonEnv(MECEnvironment):
    """动态到达变体: 每步以 P_ARR 概率刷新用户任务 (泊松式动态负载)."""

    def __init__(self, num_users=K, num_servers=M, seed=0):
        super().__init__(num_users=num_users, num_servers=num_servers, seed=seed)
        self.p_arr = P_ARR

    def step(self, action, intrinsic_reward_fn=None):
        out = super().step(action, intrinsic_reward_fn=intrinsic_reward_fn)
        for k in range(self.K):
            if self._rng.random() < self.p_arr:
                D = float(self._rng.uniform(TASK_DATA_MIN, TASK_DATA_MAX))
                C = D * float(self._rng.uniform(*TASK_CYCLES_PER_BIT))
                tau = float(self._rng.uniform(TASK_DEADLINE_MIN, TASK_DEADLINE_MAX))
                p = int(self._rng.choice([1, 2, 3], p=TASK_PRIORITY_PROB))
                self.tasks[k] = {'D': D, 'C': C, 'tau': tau, 'priority': p}
        return out


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
    print("  P2-1: 泊松到达/动态任务变体 (p_arr=%.2f, n=%d)" % (P_ARR, N_EPISODES))
    print("=" * 60)
    _rec = Recorder("p2_poisson", config={"p_arr": P_ARR, "n_steps": N_STEPS})
    out = {}
    for kind in ["HMA-Distill", "MPC", "GA", "Greedy"]:
        rows = []
        for sd in range(N_EPISODES):
            env = PoissonEnv(num_users=K, num_servers=M, seed=SEED + sd)
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
            r['seed'] = sd
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

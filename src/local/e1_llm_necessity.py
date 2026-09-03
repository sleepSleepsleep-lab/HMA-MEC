# -*- coding: utf-8 -*-
"""LLM 必要性专项: 同一搜索器(PlanRefiner)下, 不同种子来源的性能边际.
场景: 普通 + 困难(链路故障/信道退化/不诚实UA).
对比: LLM种子(HMA-Distill) vs 启发式种子(MPC) vs 随机种子(Random+Refiner)."""
import os, sys
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_REPO_ROOT)
_REPO_ROOT = os.path.dirname(_REPO_ROOT)

sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src", "local"))
import numpy as np
from config import NUM_USERS, NUM_EDGE_SERVERS, SEED
from environment import MECEnvironment
from agent_runner import HMAAgentRunner
from local.baseline_mpc import MPCBaseline
from local.plan_refiner import PlanRefiner
from local.experiment_common import compose_action

K, M = NUM_USERS, NUM_EDGE_SERVERS
RES = os.path.join(_REPO_ROOT, "results")


def make_env(sd, scenario, ep=0):
    env = MECEnvironment(num_users=K, num_servers=M, seed=SEED + sd + ep)
    if scenario == "link_fail":
        env.set_link_error(3, 1, 0.9); env.set_link_error(5, 2, 0.9)
    elif scenario == "channel_degrade":
        for k in range(K):
            env.channels[k, :] *= np.where(np.arange(M) == 3, 0.15, 1.0)  # server3 信道恶化
    elif scenario == "dishonest":
        # 用户0 谎报高优先级(实际低) + 用户1 谎报任务量
        env.tasks[0]['priority'] = 3
        env.tasks[1]['C'] *= 3.0
    return env


class RandomSeedRefiner:
    """随机种子 + 同一 Refiner 搜索 (LLM 种子对照下界)。"""
    def __init__(self):
        self.refiner = PlanRefiner(omega=(0.5, 0.5))
    def predict(self, st, env):
        alpha = np.random.uniform(0.05, 0.95, K)
        server = np.random.randint(0, M, K)
        a, s = self.refiner.refine(env, alpha, server)
        act = np.zeros(2 * K, np.float32)
        act[0::2] = np.clip(a, 0.01, 1.0); act[1::2] = (np.clip(s, 0, M - 1) + 0.5) / M
        return act


def run(predictor, env, n_steps=120, runner=None):
    E, T, S, SL = [], [], [], []
    for _ in range(n_steps):
        st = env._get_state()
        if runner is not None:
            out = runner.run_step(state=st, agents_reuse=True)
            a = compose_action(out['plan'], K, M)
        else:
            a = predictor.predict(st, env)
        ns, _, d, info = env.step(a)
        E.append(info['energy']); T.append(info['latency'])
        S.append(info['success_rate']); SL.append(info['priority_sla'])
        if d: break
    return dict(E=float(np.mean(E)), T=float(np.mean(T)),
                suc=float(np.mean(S)), sla=float(np.mean(SL)))


scenarios = ["normal", "link_fail", "channel_degrade", "dishonest"]
builders = {
    "LLM种子(HMA)": lambda env: (HMAAgentRunner(env=env, mode="Distill"), "runner"),
    "启发式种子(MPC)": lambda env: (MPCBaseline(), "pred"),
    "随机种子+Refiner": lambda env: (RandomSeedRefiner(), "pred"),
}
summary = {}
for sc in scenarios:
    print(f"\n===== 场景 {sc} =====")
    summary[sc] = {}
    for label, mk in builders.items():
        rows = []
        for sd in range(3):
            env = make_env(sd, sc)
            obj, kind = mk(env)
            r = run(obj, env, runner=(obj if kind == "runner" else None)) \
                if kind == "pred" else run(None, env, runner=obj)
            rows.append(r)
        mean = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        summary[sc][label] = mean
        print(f"  {label:24s} E={mean['E']:.3f} T={mean['T']:.3f} "
              f"suc={mean['suc']:.1%} sla={mean['sla']:.1%}")

import json
os.makedirs(RES, exist_ok=True)
json.dump(summary, open(f"{RES}/e1_llm_necessity.json", "w"),
          indent=2, ensure_ascii=False)
print(f"\n  保存 -> {RES}/e1_llm_necessity.json")

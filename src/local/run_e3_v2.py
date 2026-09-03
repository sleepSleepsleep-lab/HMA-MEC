# -*- coding: utf-8 -*-
"""
================================================================
E3 消融 (v2026-08 重构) — HMA-Distill 方法组件消融
================================================================
针对当前主线 (蒸馏策略 + 验证器精化闭环), 量化各组件贡献:
  HMA-Full       : 蒸馏策略 + PlanRefiner (完整)              = 论文 HMA-Distill
  HMA-NoRefiner  : 仅蒸馏策略前向, 关闭验证器精化            → 精化闭环贡献
  MPC            : 启发式种子 + 同一 PlanRefiner (去 LLM)     → LLM(教师) 先验贡献
  HMA-RandomSeed : 随机种子 + 同一 PlanRefiner               → 种子质量下界
参照: GA (离线最优), SAC (深 RL)。
公平: 所有基于 Refiner 的变体使用同一目标 J=0.5E+0.5T+硬罚。
================================================================
"""
import os, sys, json
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_REPO_ROOT)
_REPO_ROOT = os.path.dirname(_REPO_ROOT)

sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src", "local"))
import numpy as np
from config import NUM_USERS, NUM_EDGE_SERVERS, SEED, RESULTS_DIR
from environment import MECEnvironment
from agent_runner import HMAAgentRunner
from local.baseline_mpc import MPCBaseline
from local.plan_refiner import PlanRefiner
from local.experiment_common import compose_action
from local.results_store import Recorder

K, M = NUM_USERS, NUM_EDGE_SERVERS
# 种子结构 2026-09-03 修正: SEED+sd*100+ep (与 E1 重跑一致, 原 sd+ep 存在碰撞)
N_SEEDS, N_EPISODES, N_STEPS = (int(os.environ.get("E3V2_SEEDS", 5)),
                                     int(os.environ.get("E3V2_EPS", 20)),
                                     int(os.environ.get("E3V2_STEPS", 200)))
OUT = f"{RESULTS_DIR}/e3_component_ablation.json"


class RandomSeedRefiner:
    def __init__(self):
        self.refiner = PlanRefiner(omega=(0.5, 0.5))
    def predict(self, st, env):
        alpha = np.random.uniform(0.05, 0.95, K)
        server = np.random.randint(0, M, K)
        a, s = self.refiner.refine(env, alpha, server)
        act = np.zeros(2 * K, np.float32)
        act[0::2] = np.clip(a, 0.01, 1.0); act[1::2] = (np.clip(s, 0, M - 1) + 0.5) / M
        return act


def run_ep(method, env, kind):
    e, t, s, sl = [], [], [], []
    for _ in range(N_STEPS):
        st = env._get_state()
        if kind == "runner":
            out = method.run_step(state=st, agents_reuse=True)
            a = compose_action(out['plan'], K, M)
        else:
            a = method.predict(st, env)
        ns, _, done, info = env.step(a)
        e.append(info['energy']); t.append(info['latency'])
        s.append(info['success_rate']); sl.append(info['priority_sla'])
        if done: break
    return dict(energy=float(np.mean(e)), latency=float(np.mean(t)),
                success_rate=float(np.mean(s)), priority_sla=float(np.mean(sl)))


def build(name, env):
    if name == "HMA-Full":
        return HMAAgentRunner(env=env, mode="Distill"), "runner"
    if name == "HMA-NoRefiner":
        return HMAAgentRunner(env=env, mode="Distill", use_refiner=False), "runner"
    if name == "MPC":
        return MPCBaseline(), "pred"
    if name == "HMA-RandomSeed":
        return RandomSeedRefiner(), "pred"


METHODS = ["HMA-Full", "HMA-NoRefiner", "MPC", "HMA-RandomSeed"]
ONLY = os.environ.get("E3V2_ONLY")
if ONLY:
    METHODS = [m for m in METHODS if m == ONLY]
    OUT = OUT.replace(".json", f"_{ONLY}.json")
results = {}
_rec = Recorder("e3", config={"n_seeds": N_SEEDS, "n_episodes": N_EPISODES})
for name in METHODS:
    rows = []
    for sd in range(N_SEEDS):
        for ep in range(N_EPISODES):
            env = MECEnvironment(num_users=K, num_servers=M, seed=SEED + sd * 100 + ep)  # E1 同构非碰撞种子
            obj, kind = build(name, env)
            r = run_ep(obj, env, kind)
            rows.append(r)
            _rec.add(method=name, seed=sd, episode=ep, metrics=r)
    mean = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    std = {k: float(np.std([r[k] for r in rows])) for k in rows[0]}
    results[name] = {"mean": mean, "std": std}
    print(f"  {name:16s} E={mean['energy']:.4f} T={mean['latency']:.3f} "
          f"suc={mean['success_rate']:.2%} sla={mean['priority_sla']:.2%}")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
_rec.close()
json.dump(results, open(OUT, "w"), indent=2, ensure_ascii=False)
print(f"\n  保存 -> {OUT}")

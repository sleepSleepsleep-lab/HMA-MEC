# -*- coding: utf-8 -*-
"""HMA/PlanRefiner 参数微调扫描 (公平目标 J=0.5E+0.5T 下提升搜索质量).
评估: 3 seeds × 8 ep × 200 步, 各配置对比 E/T/suc/sla."""
import os, sys
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_REPO_ROOT)
_REPO_ROOT = os.path.dirname(_REPO_ROOT)

sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src", "local"))
import numpy as np
from config import NUM_USERS, NUM_EDGE_SERVERS, SEED
from environment import MECEnvironment
from local.plan_refiner import PlanRefiner
from local.experiment_common import compose_action
from agent_runner import HMAAgentRunner

K, M = NUM_USERS, NUM_EDGE_SERVERS
GRID_FINE = (0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95)
GRID_BASE = (0.05, 0.2, 0.4, 0.6, 0.8, 0.95)


class MultiSeedRefiner:
    """从 多个种子 (LLM蒸馏 + 信道最优启发式) 各自精化, 按同一 fitness 取最优 plan."""
    def __init__(self, policy_runner=None, rounds=3, grid=GRID_BASE,
                 server_set="cur+chan", use_llm_seed=True, use_heu_seed=True):
        self.pr = policy_runner
        self.refiner = PlanRefiner(omega=(0.5, 0.5), rounds=rounds)
        self.grid = grid
        self.server_set = server_set
        self.use_llm_seed = use_llm_seed
        self.use_heu_seed = use_heu_seed

    def _seeds(self, env):
        seeds = []
        if self.use_llm_seed and self.pr is not None:
            out = self.pr.infer(env._get_state(), deterministic=False)
            seeds.append((np.asarray(out['plan']['alpha'], float).copy(),
                          np.asarray(out['plan']['server'], int).copy()))
        if self.use_heu_seed:
            K_ = env.K
            alpha = np.full(K_, 0.5)
            server = np.asarray(env.channels).argmax(axis=1)
            seeds.append((alpha.copy(), server.copy()))
            # 补充: 低能耗(小 f_edge) 服务器定向种子
            lowF = int(np.argsort(np.asarray(env.f_edge))[0])
            server2 = np.full(K_, lowF)
            seeds.append((np.full(K_, 0.3), server2))
        if not seeds:
            K_ = env.K
            seeds.append((np.random.uniform(0.05, 0.95, K_),
                          np.random.randint(0, M, K_)))
        return seeds

    def plan(self, env):
        best = None; bestfit = -1e18
        for alpha0, server0 in self._seeds(env):
            a, s = self.refiner.refine(env, alpha0, server0,
                                       alpha_grid=self.grid)
            f = self.refiner.fitness(env, {'alpha': a, 'server': s})
            if f > bestfit:
                bestfit, best = f, (a, s)
        return {'alpha': best[0], 'server': best[1]}


def run(method, env, n_steps=150, msr=False, runner=False):
    e, t, s, sl = [], [], [], []
    for _ in range(n_steps):
        st = env._get_state()
        if msr:
            plan = method.plan(env)
            a = compose_action(plan, K, M)
        elif runner:
            out = method.run_step(state=st, agents_reuse=True)
            a = compose_action(out['plan'], K, M)
        else:
            a = method.predict(st, env)
        ns, _, done, info = env.step(a)
        e.append(info['energy']); t.append(info['latency'])
        s.append(info['success_rate']); sl.append(info['priority_sla'])
        if done: break
    return dict(E=float(np.mean(e)), T=float(np.mean(t)),
                suc=float(np.mean(s)), sla=float(np.mean(sl)))


configs = {
    "A_基线(rounds3,grid6)":  dict(rounds=3, grid=GRID_BASE, use_llm_seed=True, use_heu_seed=False),
    "B_rounds4":              dict(rounds=4, grid=GRID_BASE, use_llm_seed=True, use_heu_seed=False),
    "C_grid10(加密)":         dict(rounds=3, grid=GRID_FINE, use_llm_seed=True, use_heu_seed=False),
    "D_多种子(LLM+启发式)":    dict(rounds=3, grid=GRID_BASE, use_llm_seed=True, use_heu_seed=True),
    "E_rounds4+多种子":       dict(rounds=4, grid=GRID_BASE, use_llm_seed=True, use_heu_seed=True),
    "F_rounds3+grid10+多种子": dict(rounds=3, grid=GRID_FINE, use_llm_seed=True, use_heu_seed=True),
}

# 构造共用一个 policy_runner 的变体 (用蒸馏权重)
from distill_agent import PolicyAgentRunner
policy_path = os.path.join(_REPO_ROOT, "results", "checkpoints", "distilled_policy.pth")
pr = PolicyAgentRunner(model_path=policy_path, K=K, M=M)

# 参照: 当前 HMA-Distill (runner 完整链路)
for name, cfg in configs.items():
    rows = []
    for sd in range(3):
        for ep in range(5):
            env = MECEnvironment(num_users=K, num_servers=M, seed=SEED + sd + ep)
            msr = MultiSeedRefiner(policy_runner=(pr if cfg['use_llm_seed'] else None),
                                   rounds=cfg['rounds'], grid=cfg['grid'],
                                   use_llm_seed=cfg['use_llm_seed'],
                                   use_heu_seed=cfg['use_heu_seed'])
            rows.append(run(msr, env, msr=True))
    mean = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    print(f"  {name:26s} E={mean['E']:.4f} T={mean['T']:.3f} "
          f"suc={mean['suc']:.2%} sla={mean['sla']:.2%}")

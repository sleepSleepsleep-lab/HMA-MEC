# -*- coding: utf-8 -*-
"""
================================================================
遗传算法基线 (local/ga_baseline.py)
================================================================
GA 基线：离线最优参考（对照 CORE-LEO 下层遗传调度器）。
每步用遗传算法搜索 (alpha, server) 方案，适应度 = 目标函数
（env.simulate 评估，不污染真实状态）。

统一接口与 baselines.py 一致：
    predict(state, env) -> action np.array(2K,)

性能估算（K=8/M=4）：30 × 50 = 1500 次 simulate/步，
每次 simulate ~50μs → 每步 ~0.1s，200 步 ~20s/episode；
E1 论文规模（5 seed × 50 ep）≈ 1.4 小时，可接受。

用法:
    python local/ga_baseline.py --smoke
================================================================
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)

from config import (
    GA_POP_SIZE, GA_GENERATIONS, GA_MUTATION_RATE, GA_ELITE, GA_SEED_MIX,
)


class GAOffloadBaseline:
    """遗传算法基线：离线最优参考。每步用 GA 搜索 (alpha, server) 方案。"""

    name = "GA"

    def __init__(self, pop_size=GA_POP_SIZE, generations=GA_GENERATIONS,
                 mutation_rate=GA_MUTATION_RATE, elite=GA_ELITE,
                 seed_mix=GA_SEED_MIX, omega=(0.5, 0.5),
                 fitness_cache_size=4096):
        """
        参数:
            pop_size:     种群大小
            generations:  迭代代数
            mutation_rate: 变异概率
            elite:        精英保留数
            seed_mix:     初始种群中启发式种子（Greedy/AllEdge/AllLocal）比例
            omega:        能耗-时延偏好权重 (ω_e, ω_l)
            fitness_cache_size: 相同 (state, plan) 适应度缓存上限
        """
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite = min(elite, pop_size)
        self.seed_mix = seed_mix
        self.omega = tuple(omega)
        self._cache = {}
        self._cache_size = fitness_cache_size
        self._n_sim = 0          # 累计 simulate 次数（用于统计）

    # ---------------- 适应度 ----------------

    def _fitness(self, plan, env):
        """适应度 = -(ω_e·E + ω_l·T) - 底线惩罚；用 env.simulate，不污染状态。"""
        key = (env.step_count,
               tuple(np.round(plan['alpha'], 4)),
               tuple(int(x) for x in plan['server']))
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        sim = env.simulate(plan)
        self._n_sim += 1
        cost = (self.omega[0] * sim['energy']
                + self.omega[1] * sim['latency'])
        if sim['success_rate'] < 0.5 or sim['priority_sla'] < 0.5:
            cost += 10.0                      # 底线惩罚（对齐 VA 接受规则）
        fit = -cost
        if len(self._cache) >= self._cache_size:
            self._cache.clear()
        self._cache[key] = fit
        return fit

    # ---------------- 遗传操作 ----------------

    def _init_population(self, env):
        """初始化种群：启发式种子 + 随机个体。"""
        K, M = env.K, env.M
        n_seed = int(self.pop_size * self.seed_mix)
        pop = []
        best_channel = env.channels.argmax(axis=1)
        for i in range(self.pop_size):
            if i < n_seed:
                # 三种启发式种子交替：Greedy / AllEdge / AllLocal
                kind = i % 3
                if kind == 0:      # Greedy: 最优信道服务器, alpha=0.5
                    alpha = np.full(K, 0.5)
                    server = best_channel.copy()
                elif kind == 1:    # AllEdge: 最优信道服务器, 几乎全卸载
                    alpha = np.full(K, 0.01)
                    server = best_channel.copy()
                else:              # AllLocal: 几乎全本地
                    alpha = np.full(K, 0.99)
                    server = np.zeros(K, dtype=int)
            else:
                alpha = np.random.uniform(0.01, 1.0, K)
                server = np.random.randint(0, M, K)
            pop.append({'alpha': alpha, 'server': server.astype(int)})
        return pop

    def _crossover(self, p1, p2, K):
        """按维概率交换父代基因（alpha 与 server 独立交换）。"""
        mask_a = np.random.rand(K) < 0.5
        mask_s = np.random.rand(K) < 0.5
        child = {
            'alpha': np.where(mask_a, p1['alpha'], p2['alpha']),
            'server': np.where(mask_s, p1['server'], p2['server']),
        }
        return child

    def _mutate(self, ind, env):
        """变异：alpha 高斯扰动 / server 重采样（按 mutation_rate）。"""
        K, M = env.K, env.M
        a = ind['alpha'].copy()
        s = ind['server'].copy()
        for k in range(K):
            if np.random.rand() < self.mutation_rate:
                a[k] = float(np.clip(a[k] + np.random.normal(0, 0.1), 0.01, 1.0))
            if np.random.rand() < self.mutation_rate:
                s[k] = int(np.random.randint(0, M))
        return {'alpha': a, 'server': s}

    def _tournament_select(self, pop, fits, k=3):
        idx = np.random.choice(len(pop), size=k, replace=False)
        best_i = idx[int(np.argmax([fits[i] for i in idx]))]
        return pop[best_i]

    # ---------------- 主入口 ----------------

    def predict(self, state, env):
        """GA 搜索当前状态下的最优卸载方案。"""
        K, M = env.K, env.M
        pop = self._init_population(env)
        fits = [self._fitness(p, env) for p in pop]
        best_idx = int(np.argmax(fits))

        for _ in range(self.generations):
            # 精英保留
            order = np.argsort(fits)[::-1]
            new_pop = [pop[i] for i in order[:self.elite]]
            # 锦标赛选择 + 交叉 + 变异 生成其余个体
            while len(new_pop) < self.pop_size:
                p1 = self._tournament_select(pop, fits)
                p2 = self._tournament_select(pop, fits)
                child = self._crossover(p1, p2, K)
                child = self._mutate(child, env)
                new_pop.append(child)
            pop = new_pop
            fits = [self._fitness(p, env) for p in pop]
            bi = int(np.argmax(fits))
            if fits[bi] > fits[best_idx]:
                best_idx = bi

        best = pop[best_idx]
        action = np.zeros(env.action_dim, dtype=np.float32)
        for k in range(K):
            action[k * 2] = float(np.clip(best['alpha'][k], 0.01, 1.0))
            action[k * 2 + 1] = (float(int(best['server'][k])) + 0.5) / M
        return action


# ============================================================
# smoke 自检
# ============================================================
if __name__ == "__main__":
    from environment import MECEnvironment

    print("=" * 60)
    print("  GA 基线 smoke 自检 (2 ep × 10 步)")
    print("=" * 60)
    env = MECEnvironment(num_users=8, num_servers=4, seed=0)
    ga = GAOffloadBaseline(pop_size=20, generations=15)
    env.reset()
    e, t, suc, sla = [], [], [], []
    for s in range(10):
        a = ga.predict(env._get_state(), env)
        _, _, d, info = env.step(a)
        e.append(info['energy']); t.append(info['latency'])
        suc.append(info['success_rate']); sla.append(info['priority_sla'])
    print(f"  E={np.mean(e):.5f}  T={np.mean(t):.3f}  "
          f"suc={np.mean(suc):.2%}  sla={np.mean(sla):.2%}  "
          f"sims={ga._n_sim}")
    print("=" * 60)

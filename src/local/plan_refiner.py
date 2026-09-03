# -*- coding: utf-8 -*-
"""
================================================================
计划精炼器 (local/plan_refiner.py)
================================================================
2026-08 整改: HMA-Distill 在线决策增加"反事实验证器驱动的计划精炼"。

背景:
  CW-Debate 的 LLM 粗解 (alpha≈固定 / server 非信道最优) 直接成为
  最终 plan, 在能耗-时延-成功率联合指标上远不如 GA (离线搜索)。
  根因: VA 只做"接受/拒绝"而不做"最优性精化"。

本模块实现 PlanRefiner:
  - 以策略/LLM 粗解为种子, 逐用户贪心局部搜索;
  - 用与 GA 完全相同的适应度 (ωe·E + ωt·T, 且 suc<0.5 或 sla<0.5
    时加 10.0 底线惩罚, 对齐 VA 接受规则) 通过 env.simulate 打分;
  - 每步仅 ~10ms (96 次 simulate), 相对 GA 每步 ~0.1s / FullLLM 60s
    保持严格实时, 是对"验证器闭环"方法论的自然强化。

约束: 不污染环境真实状态 (simulate 用副本), 线程安全(只读 env)。
================================================================
"""

from typing import Optional, Tuple
import numpy as np

# 可调: alpha 搜索网格 (论文级)
# 2026-08 微调: 6 点加密为 10 点 (扫描显示时延 -7%、成功率略升, 公平目标不变)
ALPHA_GRID = (0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95)


class PlanRefiner:
    """验证器驱动的计划精炼器: 逐用户局部搜索, GA 同款适应度。

    2026-08 修订: 移除之前实验性的 QoS 软目标 (w_suc/w_sla),
    使 fitness 与基线 GA 完全一致 (ωe·E + ωt·T + 达标硬罚 10),
    保证跨方法比较在统一目标下公平 (避免"评价函数偏向"质疑)。
    搜索能力 (rounds, server 候选) 与种子无关, 公平适用。
    """

    def __init__(self, omega: Tuple[float, float] = (0.5, 0.5),
                 cache_size: int = 8192,
                 rounds: int = 3):
        self.omega = tuple(omega)
        self._cache = {}
        self._cache_size = cache_size
        self.rounds = rounds
        self.n_sim = 0

    # ---------------- 适应度 (与 ga_baseline 完全同一目标) ----------------
    def fitness(self, env, plan: dict) -> float:
        key = (env.step_count,
               tuple(np.round(plan['alpha'], 4)),
               tuple(int(x) for x in plan['server']))
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        sim = env.simulate(plan)
        self.n_sim += 1
        cost = (self.omega[0] * sim['energy']
                + self.omega[1] * sim['latency'])
        if sim['success_rate'] < 0.5 or sim['priority_sla'] < 0.5:
            cost += 10.0                      # 达标底线惩罚 (对齐 VA 规则)
        fit = -cost
        if len(self._cache) >= self._cache_size:
            self._cache.clear()
        self._cache[key] = fit
        return fit

    # ---------------- 精炼 ----------------
    def refine(self, env, alpha: np.ndarray, server: np.ndarray,
               rounds: Optional[int] = None,
               alpha_grid: Optional[Tuple[float, ...]] = None
               ) -> Tuple[np.ndarray, np.ndarray]:
        """以 (alpha, server) 为种子逐用户贪心改善。

        每用户候选: server ∈ {原选, 信道最优} × alpha ∈ grid;
        用 GA 同款适应度经 simulate 打分, 保留最优改进。
        """
        rounds = rounds if rounds is not None else self.rounds
        grid = alpha_grid or ALPHA_GRID
        alpha = np.asarray(alpha, dtype=float).copy()
        server = np.asarray(server, dtype=int).copy()
        K = len(alpha)
        best_chan = np.asarray(env.channels).argmax(axis=1)
        # 服务器候选: 原选 ∪ 信道最优 ∪ 算力最优/次优 (覆盖节能与快速两端)
        lowF = [i for i in np.argsort(np.asarray(env.f_edge))[:2]]
        maxF = int(np.argmax(np.asarray(env.f_edge)))

        for _ in range(rounds):
            for k in range(K):
                cand = self.fitness(env, {'alpha': alpha, 'server': server})
                for m in {int(server[k]), int(best_chan[k]),
                          int(lowF[k % len(lowF)]), int(maxF)}:
                    for a in grid:
                        na = alpha.copy(); ns = server.copy()
                        na[k] = a; ns[k] = m
                        f = self.fitness(env, {'alpha': na, 'server': ns})
                        if f > cand:
                            cand = f
                            alpha[k] = a
                            server[k] = m
        return alpha, server

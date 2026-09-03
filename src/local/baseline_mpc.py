# -*- coding: utf-8 -*-
"""
================================================================
MPC / 滚动时域规划基线 (local/baseline_mpc.py)
================================================================
2026-08 新增: 作为 HMA 的"去掉 LLM"最小对照 (需用于 LLM 必要性量化)。

设计:
  - 与 PlanRefiner 使用完全相同的搜索器 (同一 fitness ωe·E+ωt·T+达标硬罚、
    相同 rounds、相同 alpha/server 候选), 唯一区别是**种子来源**:
      HMA-Distill+Refiner : 种子 = LLM 蒸馏策略输出 (语义粗解)
      MPC (本基线)        : 种子 = 纯启发式 (α=0.5, server=每用户信道最优)
  - 因此 MPC 与 HMA 在性能上的差异 = "LLM/教师先验"在同一搜索器下的边际贡献,
    是量化 LLM 必要性的最小对照。
  - 每步与 Refiner 同量级开销 (~10ms), 属"在线短视规划/模型预测式方案",
    符合仿真器可用场景下经典 MPC/RHC 的定位。
================================================================
"""

import numpy as np
from typing import Optional, Tuple

from local.plan_refiner import PlanRefiner


class MPCBaseline:
    """滚动时域规划: 无 LLM 种子, 启发式种子 + 验证器搜索。"""

    def __init__(self, env=None, omega: Tuple[float, float] = (0.5, 0.5),
                 rounds: int = 3):
        self.env = env
        self.omega = tuple(omega)
        self.name = "MPC"
        # 与 HMA-Distill 的 Refiner 完全相同的搜索器
        self.refiner = PlanRefiner(omega=omega, rounds=rounds)
        self.n_calls = 0

    def _seed(self, env) -> Tuple[np.ndarray, np.ndarray]:
        """启发式种子: α=0.5 (ES 量级本地), server=每用户信道最优。"""
        K, M = env.K, env.M
        alpha = np.full(K, 0.5, dtype=np.float64)
        server = np.asarray(env.channels).argmax(axis=1)
        return alpha, server

    def predict(self, state, env):
        """滚动时域: 启发式种子 → 同一 Refiner 精化 → action。"""
        K, M = env.K, env.M
        self.env = env
        alpha, server = self._seed(env)
        alpha, server = self.refiner.refine(env, alpha, server)
        action = np.zeros(2 * K, dtype=np.float32)
        action[0::2] = np.clip(alpha, 0.01, 1.0).astype(np.float32)
        action[1::2] = (np.clip(server, 0, M - 1).astype(np.float32) + 0.5) / M
        self.n_calls += 1
        return action

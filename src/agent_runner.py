# -*- coding: utf-8 -*-
"""
================================================================
HMA-MEC 在线推理运行器 (agent_runner.py)
================================================================
本文件把 CW-Debate 在线辩论 与 Distill-Agent 单次前向 整合为统一的
``按模式分派'' 的在线推理入口。三种模式由 mode 参数选择:

  "FullLLM"  : 仅 CW-Debate (在线 LLM 调用, 用于离线蒸馏阶段)
  "Distill"  : 仅 PolicyAgentNet 前向 (零 LLM 调用, 实时性最高)
  "Hybrid"   : 默认走 Distill, 若 conf_min < tau_low 触发 FullLLM 兜底

对外提供:
  run_step(env, mode, ...)  给定环境与 mode, 一键完成单步决策 (返回 plan + meta)

需要 GPU / 大规模场景时, 把 distillation model 的大批量训练放到 GPU 服务器;
本运行器本身只在 CPU 上做单步推理, 适合在线部署。
================================================================
"""

import os
import time
import logging
from typing import Dict, Optional

import numpy as np

from config import (
    HYBRID_CONFIDENCE_LOW, NUM_USERS, NUM_EDGE_SERVERS,
    CHECKPOINT_DIR,
)
from cw_debate import cw_debate
from agent_define import make_agents
from distill_agent import PolicyAgentRunner, PolicyAgentNet

logger = logging.getLogger(__name__)


# 默认的蒸馏策略网权重路径 (训练完成后由 DistillAgentTrainer 写入)
DEFAULT_POLICY_PATH = os.path.join(CHECKPOINT_DIR, "distilled_policy.pth")


# ============================================================
# 单步推理运行器
# ============================================================
class HMAAgentRunner:
    """单步决策的主入口, 三个 mode 共享状态对象。"""

    def __init__(self, env, mode: str = "Distill",
                 policy_path: str = DEFAULT_POLICY_PATH,
                 llm=None, agents: Optional[Dict] = None,
                 tau_low: float = HYBRID_CONFIDENCE_LOW,
                 preference: Optional[np.ndarray] = None,
                 verbose: bool = False):
        """
        参数:
            env:         MECEnvironment 实例
            mode:        "FullLLM" | "Distill" | "Hybrid"
            policy_path: Distill/Hybrid 模式下使用的策略网权重
            llm:         仅 FullLLM/Hybrid 触发兜底时使用
            agents:      已构造的多智能体集合; None 时由 make_agents 创建
            tau_low:     Hybrid 模式下, 蒸馏网 conf_min 低于该值即触发兜底
            preference:  可选, 固定 OA 偏好权重 [ω_e, ω_l], 仅 FullLLM 模式生效
        """
        self.env = env
        self.mode = mode
        self.tau_low = tau_low
        self.preference = preference
        self.verbose = verbose
        self.llm = llm

        if agents is None:
            self.agents = make_agents(env, with_va=True)
        else:
            self.agents = agents

        # Distill 网仅在 Distill/Hybrid 模式下加载
        if mode in {"Distill", "Hybrid"}:
            try:
                self.policy_runner = PolicyAgentRunner(
                    model_path=policy_path,
                    K=env.K, M=env.M)
            except Exception as e:
                logger.warning(f"AgentRunner 无可用蒸馏权重, "
                               f"将以 ``Random'' 兜底: {e}")
                self.policy_runner = None
        else:
            self.policy_runner = None

        # 在线统计
        self.stats = {
            'n_distill':   0,
            'n_hybrid_llm_trigger': 0,
            'n_fullllm':   0,
            'distill_latency_ms': 0.0,
            'fullllm_latency_ms': 0.0,
            'conf_min_history': [],
        }

    # ----------------- 单步对外接口 -----------------
    def run_step(self, state: Optional[np.ndarray] = None,
                 agents_reuse: bool = True) -> Dict:
        """根据 self.mode 完成单步决策。

        参数:
            state:        可选, 直接传入状态向量; 默认调用 env._get_state()
            agents_reuse: 是否复用 self.agents; False 时每步 make_agents(env)

        返回:
            dict {
                'plan':        {'alpha': (K,), 'server': (K,)},
                'mode_used':   str,
                'conf_min':    float (Distill 模式才有意义),
                'fallback_triggered': bool (Hybrid 模式专属),
                'meta':        统计信息
            }
        """
        if state is None:
            state = self.env._get_state()

        if not agents_reuse or self.agents is None:
            self.agents = make_agents(self.env, with_va=True)

        if self.mode == "Distill":
            return self._run_distill(state)
        elif self.mode == "FullLLM":
            return self._run_full_llm(state)
        elif self.mode == "Hybrid":
            return self._run_hybrid(state)
        else:
            raise ValueError(f"未知 mode: {self.mode}")

    # ----------------- Distill -----------------
    def _run_distill(self, state) -> Dict:
        cloud = np.zeros(self.env.K, dtype=bool)
        if self.policy_runner is None:
            alpha = np.random.uniform(0.01, 0.99, self.env.K).astype(np.float32)
            server = np.random.randint(0, self.env.M, self.env.K)
            conf_min = 0.0
        else:
            t0 = time.time()
            out = self.policy_runner.infer(state, deterministic=False)
            t1 = time.time()
            self.stats['distill_latency_ms'] += (t1 - t0) * 1000
            self.stats['n_distill'] += 1
            alpha = out['plan']['alpha']
            server = out['plan']['server']
            cloud = out['plan'].get('cloud', np.zeros(self.env.K, dtype=bool))
            conf_min = out['conf_min']
        self.stats['conf_min_history'].append(conf_min)
        return {
            'plan':                 {'alpha': alpha, 'server': server, 'cloud': cloud},
            'mode_used':            'Distill',
            'conf_min':             conf_min,
            'fallback_triggered':  False,
        }

    # ----------------- FullLLM (CW-Debate) -----------------
    def _run_full_llm(self, state) -> Dict:
        t0 = time.time()
        # 无显式 LLM 时, 退化为 CW-Debate 启发式 mode (Distill 启发式)
        cw_mode = "FullLLM" if self.llm is not None else "Distill"
        out = cw_debate(self.env, self.agents, mode=cw_mode,
                        llm=self.llm, verbose=self.verbose,
                        fixed_preference=self.preference)
        t1 = time.time()
        self.stats['fullllm_latency_ms'] += (t1 - t0) * 1000
        self.stats['n_fullllm'] += 1
        if self.verbose:
            print(f"[AgentRunner] FullLLM step: "
                  f"rounds={out['rounds_used']}, fallback={out['fallback_count']}")
        return {
            'plan':                 out['plan'],
            'mode_used':            'FullLLM',
            'conf_min':             float(np.min(
                out['confidence_history'][-1]) if
                out['confidence_history'] else 0.0),
            'fallback_triggered':   False,
            'rounds_used':         out['rounds_used'],
            'va_accept':            out['va_result']['accept'],
        }

    # ----------------- Hybrid -----------------
    def _run_hybrid(self, state) -> Dict:
        if self.policy_runner is None:
            # 无蒸馏权重且无 LLM 客户端时退化为随机动作，避免无 LLM 时 Hybrid 失败
            if self.llm is None:
                alpha = np.random.uniform(0.01, 0.99, self.env.K).astype(np.float32)
                server = np.random.randint(0, self.env.M, self.env.K)
                cloud = np.zeros(self.env.K, dtype=bool)
                conf_min = 0.0
                self.stats['conf_min_history'].append(conf_min)
                return {'plan': {'alpha': alpha, 'server': server, 'cloud': cloud},
                        'mode_used': 'Hybrid-Random',
                        'conf_min': conf_min,
                        'fallback_triggered': False}
            return self._run_full_llm(state)

        t0 = time.time()
        out = self.policy_runner.infer(state, deterministic=False)
        t1 = time.time()
        self.stats['distill_latency_ms'] += (t1 - t0) * 1000
        alpha = out['plan']['alpha']
        server = out['plan']['server']
        cloud = out['plan'].get('cloud', np.zeros(self.env.K, dtype=bool))
        conf_min = out['conf_min']
        self.stats['conf_min_history'].append(conf_min)
        self.stats['n_distill'] += 1

        if conf_min < self.tau_low:
            # 困难状态 -> 触发在线辩论
            t0 = time.time()
            if self.llm is None:
                # 无 LLM 客户端：用 CW-Debate 启发式模式（mode=Distill）充当兜底，
                # 仍走完整五轮交互但不调用任何外部 LLM
                debate_out = cw_debate(self.env, self.agents, mode="Distill",
                                        llm=None, verbose=self.verbose,
                                        fixed_preference=self.preference)
            else:
                debate_out = cw_debate(self.env, self.agents, mode="FullLLM",
                                        llm=self.llm, verbose=self.verbose,
                                        fixed_preference=self.preference)
            t1 = time.time()
            self.stats['fullllm_latency_ms'] += (t1 - t0) * 1000
            self.stats['n_fullllm'] += 1
            self.stats['n_hybrid_llm_trigger'] += 1
            if self.verbose:
                print(f"[AgentRunner] Hybrid fallback: "
                      f"conf_min={conf_min:.3f} < tau_low={self.tau_low:.2f}")
            return {
                'plan':                debate_out['plan'],
                'mode_used':           'Hybrid-LLM' if self.llm is not None else 'Hybrid-Heuristic',
                'conf_min':            conf_min,
                'fallback_triggered': True,
                'rounds_used':         debate_out['rounds_used'],
                'va_accept':          debate_out['va_result']['accept'],
            }
        return {
            'plan':                {'alpha': alpha, 'server': server, 'cloud': cloud},
            'mode_used':           'Hybrid-Distill',
            'conf_min':            conf_min,
            'fallback_triggered': False,
        }

    # ----------------- 统计接口 -----------------
    def stats_summary(self) -> Dict:
        per_distill_avg = (self.stats['distill_latency_ms'] /
                           max(self.stats['n_distill'], 1))
        per_fullllm_avg = (self.stats['fullllm_latency_ms'] /
                           max(self.stats['n_fullllm'], 1))
        n_total = self.stats['n_distill'] + self.stats['n_fullllm']
        hybrid_trigger_rate = (self.stats['n_hybrid_llm_trigger'] /
                               max(n_total, 1))
        conf_hist = np.array(self.stats['conf_min_history'])
        return {
            'n_distill_steps':      self.stats['n_distill'],
            'n_fullllm_steps':      self.stats['n_fullllm'],
            'hybrid_trigger_rate':  hybrid_trigger_rate,
            'avg_distill_latency_ms': per_distill_avg,
            'avg_fullllm_latency_ms': per_fullllm_avg,
            'conf_min_mean':        (float(conf_hist.mean()) if len(conf_hist) else None),
            'conf_min_min':         (float(conf_hist.min()) if len(conf_hist) else None),
        }


# ============================================================
# 一键接口 run_step
# ============================================================
def run_step(env, mode: str = "Distill",
             policy_path: str = DEFAULT_POLICY_PATH,
             state: Optional[np.ndarray] = None) -> Dict:
    """便利接口: 不再保留 HMAAgentRunner 实例时一次性调用。"""
    runner = HMAAgentRunner(env=env, mode=mode, policy_path=policy_path)
    return runner.run_step(state)


# ============================================================
# 自检
# ============================================================
if __name__ == "__main__":
    from environment import MECEnvironment

    print("=" * 60)
    print("  HMAAgentRunner 三模式自检 (mode=Distill, CPU)")
    print("=" * 60)
    env = MECEnvironment(num_users=8, num_servers=4, seed=42)
    env.reset()
    runner = HMAAgentRunner(env=env, mode="Distill",
                            policy_path=os.path.join(CHECKPOINT_DIR,
                                                    "smoke_policy.pth"))
    # 不显式 make_agents 也是 ok 的 (内部 make_agents)
    n_steps = 10
    for i in range(n_steps):
        out = runner.run_step()
        plan = out['plan']
        if i == 0:
            print(f"  step 1: mode_used={out['mode_used']}, "
                  f"conf_min={out['conf_min']:.3f}, "
                  f"alpha[:3]={plan['alpha'][:3]}, "
                  f"server[:3]={plan['server'][:3]}")
    stats = runner.stats_summary()
    print(f"  跑完 {n_steps} 步后:\n  {stats}")
    print("=" * 60)
    print("  AgentRunner 三模式骨架通了; Hybrid / FullLLM 验证将随实验展开.")
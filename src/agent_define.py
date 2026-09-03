# -*- coding: utf-8 -*-
"""
================================================================
A1: Agent 定义模型 (agent_define.py)
================================================================
本文件实现「场景—功能—边界」三元组形式的 Agent 定义模型（S-F Model），
并据此把 MEC 卸载问题中的决策实体形式化为四类 Agent：
    UA (User Agent)        × K 个
    EA (Edge Agent)        × M 个
    OA (Orchestrator Agent) × 1 个
    VA (Verifier Agent)     × 1 个

每类 Agent 拥有：
  - 场景集 S_a：可感知与作用的物理范围
  - 功能集 F_a：可执行的动作算子集合
  - 边界集 B_a：不可越权的约束

本文件不涉及具体辩论协议（cw_debate.py）或蒸馏推理（distill_agent.py），
仅给出 Agent 的「身份信息」与「最小可调度接口」，便于后续模块解耦设计。
================================================================
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import numpy as np

from config import (
    NUM_USERS, NUM_EDGE_SERVERS, STATE_DIM,
    CONFIDENCE_THRESHOLD, CONSENSUS_EPSILON,
    DEBATE_MAX_ROUNDS,
)


# ============================================================
# Agent 角色类型枚举
# ============================================================
ROLE_UA = "UA"   # 用户智能体
ROLE_EA = "EA"   # 边缘智能体
ROLE_OA = "OA"   # 编排智能体
ROLE_VA = "VA"   # 验证智能体


# Agent 可执行的功能算子枚举（用于「功能集 F_a」的形式化）
FUNCTIONS = {
    # UA 功能
    "UA_PROFILE":     "任务画像与本地资源估算",
    "UA_PROPOSE":     "生成本地候选卸载方案",
    "UA_REPORT_CONF": "申报候选方案的置信度",
    "UA_CRITIQUE":    "对他人的相同服务器选择提出冲突提示",
    # EA 功能
    "EA_CAPACITY":    "容量与负载上限申报",
    "EA_ADMIT":       "接受 / 拒绝本地适配批评估",
    "EA_PREDICT":     "本地负载短时预测",
    # OA 功能
    "OA_ARBITRATE":   "冲突消解与共识调度",
    "OA_PREFERENCE":  "多目标偏好推断",
    "OA_TOM":         "Theory-of-Mind 反事实预测",
    "OA_CONSENSUS":   "终止判断与最终方案输出",
    "OA_INVOKE_VA":   "调用验证智能体",
    # VA 功能
    "VA_SIMULATE":    "反事实仿真",
    "VA_REJECT":      "对 reward hacking 方案拒绝采样",
}


# ============================================================
# Agent 基类
# ============================================================
@dataclass
class Agent:
    """所有 Agent 的公共基类，承载身份信息与共享接口。"""
    agent_id: str                 # 全局唯一标识
    role: str                      # 角色（UA/EA/OA/VA）
    scenarios: Set[str]           # 场景集 S_a
    functions: Set[str]           # 功能集 F_a
    boundaries: Set[str]          # 边界集 B_a
    state_cache: Optional[Dict] = field(default=None)

    def observe(self, state, env_text):
        """把当前系统状态与自然语言描述存入缓存。"""
        self.state_cache = {
            'state': np.asarray(state, dtype=np.float32),
            'text':  env_text,
        }

    def clear_cache(self):
        self.state_cache = None


# ============================================================
# UA: 用户智能体
# ============================================================
class UserAgent(Agent):
    """对应单个用户 k 的智能体。

    场景：仅关注用户 k 自身的任务、本地算力、与各服务器的信道。
    功能：画像 / 候选方案生成 / 置信度自评 / 跨用户冲突批判
    边界：只负责自身决策，不能越权调度他人任务
    """
    def __init__(self, k: int, env=None):
        sid = f"UA-{k}"
        super().__init__(
            agent_id=sid,
            role=ROLE_UA,
            scenarios={f"user_{k}"},
            functions={"UA_PROFILE", "UA_PROPOSE",
                       "UA_REPORT_CONF", "UA_CRITIQUE"},
            boundaries={"ONLY_OWN_TASK"},
        )
        self.k = k
        self.env = env  # 弱引用，仅读取信息，不调用 step

    def local_profile(self):
        """返回该用户任务的画像 dict。"""
        assert self.state_cache is not None, "UA.observe() 必须先被调用"
        t = self.env.tasks[self.k]
        return {
            'D': t['D'], 'C': t['C'],
            'tau': t['tau'], 'priority': t['priority'],
            'f_local': float(self.env.f_local[self.k]),
        }

    def propose(self):
        """生成本地候选卸载方案。

        本方法仅返回占位结构（决策由各后端填充）：
          {'alpha': float, 'server': int, 'confidence': float, 'reason': str}
        """
        raise NotImplementedError("由 cw_debate.py 中的 LLM 驱动实现")

    def critique(self, other_proposal):
        """对其他 UA 的提议给出冲突提示（若选择同一服务器）。"""
        raise NotImplementedError("由 cw_debate.py 中的 LLM 驱动实现")


# ============================================================
# EA: 边缘智能体
# ============================================================
class EdgeAgent(Agent):
    """对应单个边缘服务器 m 的智能体。

    场景：仅关注服务器 m 的容量、负载、能耗
    功能：容量申报 / 适配批评估 / 负载预测
    边界：不能跨服务器重分配
    """
    def __init__(self, m: int, env=None):
        sid = f"EA-{m}"
        super().__init__(
            agent_id=sid,
            role=ROLE_EA,
            scenarios={f"server_{m}"},
            functions={"EA_CAPACITY", "EA_ADMIT", "EA_PREDICT"},
            boundaries={"NO_CROSS_REALLOCATE"},
        )
        self.m = m
        self.env = env

    def capacity(self):
        """当前剩余容量估计。"""
        # 简单形式化：f_edge - 现有负载
        f = float(self.env.f_edge[self.m])
        load = float(self.env.server_load[self.m])
        return max(0.0, f - load)

    def admit(self, proposals_for_m):
        """是否接纳提交到本服务器的所有卸载任务。"""
        raise NotImplementedError("由 cw_debate.py 中的 LLM 驱动实现")

    def predict_load(self, steps=1):
        """服务器负载短时预测（最简形式：保持当前值）。"""
        return float(self.env.server_load[self.m])


# ============================================================
# OA: 编排智能体
# ============================================================
class OrchestratorAgent(Agent):
    """全局编排智能体，单一实例。

    场景：所有用户、所有服务器、所有 Agent 提案
    功能：偏好仲裁 / 共识调度 / Theory-of-Mind / 调用 VA / 终止判断
    边界：不直接改写物理动作，仅输出 {alpha_k, m_k}
    """
    def __init__(self, env=None):
        super().__init__(
            agent_id="OA-0",
            role=ROLE_OA,
            scenarios={"global"},
            functions={"OA_ARBITRATE", "OA_PREFERENCE",
                       "OA_TOM", "OA_CONSENSUS", "OA_INVOKE_VA"},
            boundaries={"NO_DIRECT_PHYSICAL_ACTION"},
        )
        self.env = env
        # 偏好权重向量 omega，可由后续模块基于状态推断
        self.preference = np.array([0.5, 0.5], dtype=np.float32)
        # 记录每个 episode 的辩论历史，供理论分析使用
        self.debate_history: List[Dict] = []

    def arbitrate(self, proposals, critiques):
        """对一批提案与冲突进行仲裁，返回修正后的提案集合。"""
        raise NotImplementedError("由 orchestrator.py 实现")

    def consensus(self, prev_conf, cur_conf, round_idx):
        """终止判定：置信度变化是否已收敛。"""
        if round_idx + 1 >= DEBATE_MAX_ROUNDS:
            return True
        if prev_conf is None:
            return False
        delta = float(np.max(np.abs(np.asarray(cur_conf) -
                                     np.asarray(prev_conf))))
        return delta < CONSENSUS_EPSILON


# ============================================================
# VA: 验证智能体
# ============================================================
class VerifierAgent(Agent):
    """独立第三方验证智能体，调用环境反事实仿真。"""
    def __init__(self, env=None):
        super().__init__(
            agent_id="VA-0",
            role=ROLE_VA,
            scenarios={"global"},
            functions={"VA_SIMULATE", "VA_REJECT"},
            boundaries={"NO_MODIFY_REAL_ENV"},
        )
        self.env = env

    def verify(self, plan, oa_prediction=None):
        """对方案做反事实仿真。

        参数：
            plan:           dict {'alpha':(K,), 'server':(K,)}
            oa_prediction:  OA 自评估的能耗 / 时延估计（可选）
        返回：dict {
            'accept': bool,
            'sim_result': dict,
            'reason': str
        }
        """
        from verifier import evaluate_plan  # 延迟导入避免循环依赖
        result = self.env.simulate(plan)
        flag = True
        reason = ""
        if oa_prediction is not None:
            for key in ('energy', 'latency'):
                pred = float(oa_prediction.get(key, result[key]))
                actual = float(result[key])
                if pred > 1e-6:
                    gap = abs(actual - pred) / pred
                    if gap > 0.15:
                        flag = False
                        reason += f"{key} 偏差 {gap:.2%}; "
        if not result['priority_sla'] >= 0.5:
            flag = False
            reason += f"高优先级 SLA 偏低 ({result['priority_sla']:.2%}); "
        return {'accept': flag, 'sim_result': result, 'reason': reason}


# ============================================================
# Agent 生成器：依据系统规模实例化
# ============================================================
def make_agents(env, with_va=True):
    """根据 MEC 环境实例生成全套 Agent。

    参数：
        env:       MECEnvironment 实例
        with_va:   是否同时生成 Verifier（消融实验可关闭）
    返回：dict {
        'UA': List[UserAgent],
        'EA': List[EdgeAgent],
        'OA': OrchestratorAgent,
        'VA': Optional[VerifierAgent],
    }
    """
    K, M = env.K, env.M
    uas = [UserAgent(k, env=env) for k in range(K)]
    eas = [EdgeAgent(m, env=env) for m in range(M)]
    oa = OrchestratorAgent(env=env)
    va = VerifierAgent(env=env) if with_va else None
    return {'UA': uas, 'EA': eas, 'OA': oa, 'VA': va, 'env': env}


def agent_topology_summary(agents):
    """返回 Agent 系统的拓扑摘要，便于日志与论文表格描述。"""
    n_ua = len(agents['UA'])
    n_ea = len(agents['EA'])
    n_oa = 1 if agents.get('OA') is not None else 0
    n_va = 1 if agents.get('VA') is not None else 0
    return {
        'K': n_ua, 'M': n_ea,
        'total_agents': n_ua + n_ea + n_oa + n_va,
        'has_verifier': n_va == 1,
        'roles': [a.role for a in agents['UA']] +
                 [a.role for a in agents['EA']] +
                 ([agents['OA'].role] if agents.get('OA') else []) +
                 ([agents['VA'].role] if agents.get('VA') else []),
    }


# ============================================================
# 简易自检：被直接运行时打印 Agent 拓扑示例
# ============================================================
if __name__ == "__main__":
    from environment import MECEnvironment
    env = MECEnvironment(num_users=NUM_USERS, num_servers=NUM_EDGE_SERVERS,
                         seed=42)
    state = env.reset()
    agents = make_agents(env)
    for a in agents['UA']:
        a.observe(state, env.state_to_text())
    for a in agents['EA']:
        a.observe(state, env.state_to_text())
    agents['OA'].observe(state, env.state_to_text())
    if agents['VA'] is not None:
        agents['VA'].observe(state, env.state_to_text())
    print("Agent 拓扑摘要：", agent_topology_summary(agents))
    print("UA[0] 画像：", agents['UA'][0].local_profile())
    print("EA[0] 剩余容量：", agents['EA'][0].capacity())
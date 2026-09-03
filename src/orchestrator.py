# -*- coding: utf-8 -*-
"""
================================================================
编排智能体 OA 完整实现 (orchestrator.py)
================================================================
本模块实现 HMA-MEC 中编排智能体（Orchestrator Agent, OA）的全部逻辑：
  1. 偏好推断   infer_preference(state) -> omega [omega_e, omega_l]
  2. 冲突仲裁   arbitrate(proposals, critiques, preference) -> revised
  3. Theory-of-Mind 预测  predict_confidence_impact(revised) -> confidences
  4. 共识终止判据 consensus(prev_conf, cur_conf, round_idx) -> bool
  5. 调用验证智能体 invoke_va(plan, va_runner, oa_prediction) -> verify_out
  6. 输出最终方案 finalize_plan(revised) -> plan (含 alpha 与 server)

OA 不直接改写物理动作，输出 (alpha_k, m_k) 复合决策供环境推进。
所有 LLM 调用都通过 llm_client.LLMClient 完成，便于后端切换。
================================================================
"""

from typing import Dict, List, Optional, Tuple

import json
import logging
import numpy as np

from config import (
    CONSENSUS_EPSILON, DEBATE_MAX_ROUNDS,
    NUM_USERS, NUM_EDGE_SERVERS,
)
from llm_client import get_llm_client, parse_json_response

logger = logging.getLogger(__name__)


# ============================================================
# Prompt 模板
# ============================================================
PREFERENCE_SYSTEM_PROMPT = (
    "You are an optimization advisor for a mobile edge computing (MEC) system. "
    "Your job is to determine the current trade-off priority between energy "
    "saving and latency reduction. Respond in JSON only."
)

PREFERENCE_USER_TEMPLATE = (
    "Current MEC system state:\n{state_text}\n\n"
    "Analyze the system and determine the priority trade-off between energy "
    "saving and low latency. Consider these factors:\n"
    "  - If many devices are battery-constrained and tasks have loose "
    "deadlines: prioritize energy (energy_weight high).\n"
    "  - If many high-priority tasks have tight deadlines: prioritize "
    "latency (latency_weight high).\n"
    "  - If the system is under heavy load: balance both.\n"
    "Respond in JSON only with the following format:\n"
    '{{\n'
    '  "energy_weight": <float between 0.0 and 1.0>,\n'
    '  "latency_weight": <float between 0.0 and 1.0>,\n'
    '  "reasoning": "<one sentence reasoning>"\n'
    '}}\n'
    "Note: energy_weight + latency_weight should approximately equal 1.0."
)

ARBITRATE_SYSTEM_PROMPT = (
    "You are the orchestrator agent for a mobile edge computing (MEC) system. "
    "Your job is to arbitrate conflicts among user agents who chose the same "
    "edge server and produced a load that exceeds capacity. You must revise "
    "the proposals while respecting each user's task deadline and priority."
)

ARBITRATE_USER_TEMPLATE = (
    "Current system state (text):\n{state_text}\n\n"
    "User proposals (round r):\n{proposals_json}\n\n"
    "Edge server capacity / current load:\n{edge_info_json}\n\n"
    "Conflict critiques:\n{critiques_json}\n\n"
    "Predicted OA preference: energy_weight={omega_e:.3f}, "
    "latency_weight={omega_l:.3f}\n"
    "Note: if energy preference dominates, you may increase the local "
    "execution ratio alpha to save transmission/server energy; if latency "
    "preference dominates, you may decrease alpha to offload more.\n\n"
    "Please arbitrate the conflicts (only adjust users whose server is "
    "overcrowded). Output JSON only with format:\n"
    '{{\n'
    '  "revised_proposals": [\n'
    '    {{"user": <int>, "alpha": <float in [0.01,1.0]>, '
    '"server": <int in [0,M-1]>, "reason": "<short>"}},\n'
    '    ...\n'
    '  ],\n'
    '  "estimated_energy_kJ": <float>,\n'
    '  "estimated_latency_s": <float>,\n'
    '  "reasoning": "<one sentence>"\n'
    '}}'
)

TOM_SYSTEM_PROMPT = (
    "You are a Theory-of-Mind predictor for the user agents in a MEC "
    "system. Given a revised set of proposals, predict each user's "
    "confidence (in [0,1]) on accepting their assigned proposal. "
    "Higher confidence means the user is satisfied with the assignment."
)

TOM_USER_TEMPLATE = (
    "Current system state (text):\n{state_text}\n\n"
    "Revised proposals:\n{proposals_json}\n\n"
    "Predict per-user confidence in [0,1]. Output JSON only with format:\n"
    '{{\n'
    '  "confidences": [<float>, ..., <float>] (length = K)\n'
    '}}'
)


# ============================================================
# 编排智能体算法层
# ============================================================
class OrchestratorAgentRunner:
    """编排智能体 OA 的算法层实现。

    与 agent_define.OrchestratorAgent（身份载体）解耦：
      - OrchestratorAgent（定义层）持有身份 / 边界信息
      - OrchestratorAgentRunner（本类）封装所有计算逻辑
    """

    def __init__(self, env, llm=None, with_llm: bool = True,
                 preference: Optional[np.ndarray] = None):
        """
        参数：
            env:        MECEnvironment 实例
            llm:        LLMClient 实例；若 None 且 with_llm=True 则自动获取
            with_llm:   是否使用 LLM 调用；False 时退化为启发式实现（用于
                        CPU 自检 / 蒸馏数据生成中的轻量回退）
            preference: 初始偏好向量 [omega_e, omega_l]，None 则等权重
        """
        self.env = env
        self.with_llm = with_llm
        if with_llm:
            self.llm = llm or get_llm_client()
        else:
            self.llm = None
        self.preference = (preference.copy() if preference is not None
                           else np.array([0.5, 0.5], dtype=np.float32))
        self._fixed_preference = preference is not None
        self.last_oa_prediction: Optional[Dict] = None

    # -------------------- (1) 偏好推断 --------------------
    def infer_preference(self, state, state_text) -> np.ndarray:
        if self._fixed_preference:
            return self.preference
        """从状态推断能耗 vs 时延偏好权重 omega。

        参数：
            state:      np.ndarray (state_dim,)
            state_text: 自然语言状态描述
        返回：
            np.ndarray (2,) -- [omega_e, omega_l]，sum=1
        """
        if not self.with_llm or self.llm is None:
            # 启发式退化：依据高优先级任务比例与平均时延约束决定偏好
            return self._heuristic_preference(state)

        try:
            user = PREFERENCE_USER_TEMPLATE.format(state_text=state_text)
            resp = self.llm.chat_json(
                PREFERENCE_SYSTEM_PROMPT, user,
                temperature=0.0, max_tokens=300)
            e = float(resp.get("energy_weight", 0.5))
            l = float(resp.get("latency_weight", 0.5))
            vec = np.array([e, l], dtype=np.float32)
            s = vec.sum()
            if s > 0:
                vec = vec / s
            else:
                vec = np.array([0.5, 0.5], dtype=np.float32)
            self.preference = np.clip(vec, 0.0, 1.0).astype(np.float32)
        except Exception as e:
            logger.warning(f"OA 偏好推断失败，回退启发式: {e}")
            self.preference = self._heuristic_preference(state)
        return self.preference

    def _heuristic_preference(self, state) -> np.ndarray:
        """不依赖 LLM 的启发式偏好推断。
        
        根据系统状态中的时延紧迫度与优先级分布，输出能耗-时延偏好权重。
        时延约束越紧、高优先级任务越多 → ω_l 增大（偏好低时延）。
        """
        # 状态前 4K 维：每用户 (D, C, tau, p)
        K = self.env.K
        per_user = state[:4 * K].reshape(K, 4)
        # 紧迫度 = 平均时延约束倒数（tau 小则紧）
        tightness = float(np.mean(1.0 / (per_user[:, 2] + 1e-3)))
        # 优先级偏置：高优先级越多偏向时延
        prio_w = float(np.mean(per_user[:, 3] * 3.0))  # 后向兼容，但 E_obs 已被归一化
        # 简单加权：tightness 高 + 高优先级多 -> omega_l 高
        omega_l = 0.5 + 0.25 * np.tanh((tightness - 0.5) * 2) \
                     + 0.10 * (prio_w - 1.5)
        omega_l = float(np.clip(omega_l, 0.1, 0.9))
        omega_e = 1.0 - omega_l
        return np.array([omega_e, omega_l], dtype=np.float32)

    # -------------------- (2) 冲突仲裁 --------------------
    def arbitrate(self, proposals: Dict, critiques: List[Dict],
                  state, state_text, flags: Optional[Dict] = None,
                  fallback_count: int = 0) -> Dict:
        """基于偏好和冲突批判，仲裁修正的候选方案。

        参数：
            proposals: dict {'user_k': {'alpha','server','confidence','reason'}}
            critiques:  list of critique dicts
            state:      np.ndarray 状态向量
            state_text: 自然语言状态描述
            flags:     消融 flags
            fallback_count: 当前回退轮次, 用于在回退时产生不同方案
        返回：
            dict {
                'revised':  [{'user','alpha','server','reason'}, ...],
                'estimated_energy_kJ': float,
                'estimated_latency_s': float,
                'reasoning':  str,
            }
        """
        flags = flags or {}
        # 整理输入
        edge_info = []
        for m in range(self.env.M):
            edge_info.append({
                'server': m,
                'capacity_cycles': float(self.env.f_edge[m]),
                'current_load': float(self.env.server_load[m]),
            })
        proposals_json = json.dumps(proposals, ensure_ascii=False, indent=2)
        edge_info_json = json.dumps(edge_info, ensure_ascii=False, indent=2)
        critiques_json = json.dumps(critiques, ensure_ascii=False, indent=2)

        if not self.with_llm or self.llm is None:
            return self._heuristic_arbitrate(proposals, edge_info,
                                              critiques=critiques, flags=flags,
                                              fallback_count=fallback_count)

        omega_e, omega_l = float(self.preference[0]), float(self.preference[1])
        user = ARBITRATE_USER_TEMPLATE.format(
            state_text=state_text,
            proposals_json=proposals_json,
            edge_info_json=edge_info_json,
            critiques_json=critiques_json,
            omega_e=omega_e, omega_l=omega_l,
        )
        resp = self.llm.chat_json(
            ARBITRATE_SYSTEM_PROMPT, user,
            temperature=0.0, max_tokens=700)
        if not resp:
            return self._heuristic_arbitrate(proposals, edge_info)
        self.last_oa_prediction = {
            'energy': float(resp.get('estimated_energy_kJ', 0.0)),
            'latency': float(resp.get('estimated_latency_s', 0.0)),
        }
        return {
            'revised': resp.get('revised_proposals', []),
            'estimated_energy_kJ': self.last_oa_prediction['energy'],
            'estimated_latency_s': self.last_oa_prediction['latency'],
            'reasoning': resp.get('reasoning', ''),
        }

    def _heuristic_arbitrate(self, proposals, edge_info,
                             critiques=None, flags=None,
                             fallback_count=0) -> Dict:
        """启发式仲裁：把过多堆叠在同一服务器的用户均匀打散到负载较轻的服务器。

        参数：
            proposals: dict {'user_k': {'alpha','server','confidence',...}}
            edge_info: list of server info dicts
            critiques: 批判列表 (由 UAs/EAs 产生)
            flags:     消融实验 flags
            fallback_count: 当前回退轮次, 用于在回退时产生不同方案
        """
        flags = flags or {}
        critiques = critiques or []
        K = self.env.K
        M = self.env.M
        # 提取 (user, server, alpha)
        revised = []
        server_users = {m: [] for m in range(M)}
        for kk, v in proposals.items():
            u = int(v.get('user', int(kk.split('_')[-1]) if '_' in kk else int(kk)))
            a = float(np.clip(v.get('alpha', 0.5), 0.01, 1.0))
            s = int(np.clip(v.get('server', 0), 0, M - 1))
            revised.append({'user': u, 'alpha': a, 'server': s, 'reason': ''})
            server_users[s].append(u)

        # 如果 disable_boundary，允许跨服务器任意分配（打破边界约束）
        if flags.get("disable_boundary"):
            # 把用户均匀分散到所有服务器
            for idx, r in enumerate(revised):
                r['server'] = idx % M
                r['reason'] = 'boundary-free: 均匀分散'
            server_users = {m: [r['user'] for r in revised if r['server'] == m]
                            for m in range(M)}

        # ---- ω 引导 α 修正（E5 Pareto 展开的关键机制，2026-08 整改）----
        # ω_e 高 → 多本地执行（α↑，省传输与服务器能耗）
        # ω_l 高 → 多卸载（α↓，低时延）
        # 连续可调，默认最大 ±0.15；E1 中 preference 默认等权重时 alpha_shift=0，
        # 行为与旧版完全一致（向后兼容）。
        pref_e, pref_l = float(self.preference[0]), float(self.preference[1])
        alpha_shift = (pref_e - pref_l) * 0.15
        if not flags.get("disable_alpha_shift"):
            for r in revised:
                r['alpha'] = float(np.clip(r['alpha'] + alpha_shift, 0.01, 1.0))
                r['reason'] = (r.get('reason', '')
                               + f" | ω引导α{alpha_shift:+.2f}")

        # 处理过载：若服务器接入用户数 > floor(K/M) + 1，把多余者迁到负载较轻的服务器
        avg_load = max(1, K // M + 1)
        for m in range(M):
            users_on_m = server_users[m]
            if len(users_on_m) > avg_load:
                # 构建迁移候选集
                if flags.get("disable_priority"):
                    # 无优先级信息：按用户 index 升序（不加权）
                    users_on_m_sorted = sorted(users_on_m)
                else:
                    # 按 priority 升序排（优先级低者更易被迁移）
                    users_on_m_sorted = sorted(
                        users_on_m,
                        key=lambda k_: self.env.tasks[k_]['priority'])
                to_move = users_on_m_sorted[avg_load:]

                # 选择目标服务器：考虑偏好权重
                free_servers = [mm for mm in range(M)
                                if len(server_users[mm]) < avg_load]
                pref_e, pref_l = float(self.preference[0]), float(self.preference[1])
                # 偏好时延时 + 无偏好时：选负载最低的服务器
                # 偏好能耗时：选信道最好的服务器（降低传输能耗）
                for k_ in to_move:
                    if not free_servers:
                        break
                    if fallback_count > 0:
                        # 回退轮次: 用不同策略, 将用户迁移到不同的服务器
                        shuffled = sorted(free_servers,
                                          key=lambda x: hash((k_, fallback_count, x)))
                        best_m = shuffled[0]
                    elif pref_e >= pref_l + 0.2:
                        # 偏好能耗：选信道最好的可用服务器
                        best_m = max(free_servers,
                                     key=lambda mm: self.env.channels[k_, mm])
                    else:
                        # 偏好时延或均衡：选当前负载最低的服务器
                        best_m = min(free_servers,
                                     key=lambda mm: len(server_users[mm]))
                    for r in revised:
                        if r['user'] == k_:
                            r['server'] = best_m
                            r['reason'] = (f"arbitrate(ω_e={pref_e:.2f}): "
                                           f"过载迁至 server_{best_m}")
                            server_users[m].remove(k_)
                            server_users[best_m].append(k_)
                            free_servers.remove(best_m)
                            break

        # 计算 OA 自评估（启发式估算能耗和时延）
        try:
            plan = self.finalize_plan(revised, K, M)
            sim = self.env.simulate(plan)
            # OA 自评估直接使用仿真结果（无额外噪声）
            self.last_oa_prediction = {
                'energy': float(sim['energy']),
                'latency': float(sim['latency']),
            }
            est_e = float(sim['energy'])
            est_t = float(sim['latency'])
        except Exception:
            self.last_oa_prediction = None
            est_e, est_t = 0.0, 0.0

        return {
            'revised': revised,
            'estimated_energy_kJ': est_e,
            'estimated_latency_s': est_t,
            'reasoning': 'heuristic-arbitrate',
        }

    # -------------------- (3) Theory-of-Mind 预测 --------------------
    def predict_confidence_impact(self, revised: List[Dict],
                                  state, state_text,
                                  prev_conf: Optional[np.ndarray] = None
                                  ) -> np.ndarray:
        """预测每用户对修正方案的接受置信度。

        参数：
            revised:    OA 仲裁后的 [{'user','alpha','server'}, ...]
            state:      当前状态向量
            state_text: 状态自然语言描述
            prev_conf:  前一轮置信度（若为 None，则完全不依赖）
        返回：
            np.ndarray (K,) —— 每用户置信度 in [0, 1]
        """
        K = self.env.K
        if not self.with_llm or self.llm is None:
            return self._heuristic_confidence(revised, prev_conf)
        try:
            user = TOM_USER_TEMPLATE.format(
                state_text=state_text,
                proposals_json=json.dumps(revised, ensure_ascii=False))
            resp = self.llm.chat_json(
                TOM_SYSTEM_PROMPT, user,
                temperature=0.0, max_tokens=400)
            confs = resp.get('confidences', [])
            if not isinstance(confs, list) or len(confs) != K:
                return self._heuristic_confidence(revised, prev_conf)
            return np.clip(np.asarray(confs, dtype=np.float32), 0.0, 1.0)
        except Exception as e:
            logger.warning(f"OA ToM 预测失败，回退启发式: {e}")
            return self._heuristic_confidence(revised, prev_conf)

    def _heuristic_confidence(self, revised: List[Dict],
                              prev_conf: Optional[np.ndarray] = None
                              ) -> np.ndarray:
        """不依赖 LLM 的 ToM 启发式：根据 MEC 物理量预测置信度。

        置信度公式（三维 MEC 物理量感知版）：
          c_k = sigmoid(β1·Δ_t + β2·SINR + β3·(1-ρ))

        其中 Δ_t = (τ_max - T_k)/τ_max 为时延余量；
        SINR 由信道增益 h 归一化近似；
        ρ = 服务器累计负载 / 容量，反映算力竞争程度。

        2026.07 修改：从一维 sigmoid(β·Δ_t) 扩展为三维。
        """
        from config import CONFIDENCE_BETA_TAU, CONFIDENCE_BETA_SINR, CONFIDENCE_BETA_LOAD
        K = self.env.K
        M = self.env.M
        confs = np.zeros(K, dtype=np.float32)
        # 先计算各服务器负载率
        server_loads = np.zeros(M)
        for r in revised:
            s = int(np.clip(r.get('server', 0), 0, M - 1))
            u = int(r['user'])
            if 0 <= u < K:
                off_cycles = (1 - float(np.clip(r.get('alpha', 0.5), 0.01, 1.0))) * self.env.tasks[u]['C']
                server_loads[s] += off_cycles
        for r in revised:
            u = int(r['user'])
            if u < 0 or u >= K:
                continue
            tau = self.env.tasks[u]['tau']
            # 粗估时延：本地与边缘二者 max
            a = float(np.clip(r['alpha'], 0.01, 1.0))
            loc_T = a * self.env.tasks[u]['C'] / self.env.f_local[u]
            s = int(np.clip(r['server'], 0, self.env.M - 1))
            h = self.env.channels[u, s]
            from environment import BANDWIDTH, NOISE_POWER, TX_POWER_USER
            rate = BANDWIDTH * np.log2(1 + TX_POWER_USER * h / NOISE_POWER)
            tx_T = (1 - a) * self.env.tasks[u]['D'] / max(rate, 1e-9)
            comp_T = (1 - a) * self.env.tasks[u]['C'] / self.env.f_edge[s]
            off_T = tx_T + comp_T
            T = max(loc_T, off_T)

            # 三维置信度：
            # Δ_t: 时延余量分量
            delta_t = (tau - T) / max(tau, 1e-3)
            # SINR 分量：信道增益归一化
            sinr_norm = float(np.clip(h / 1e-5, 0.0, 1.0))
            # 负载分量：服务器充裕度 1-ρ
            rho = float(server_loads[s] / max(self.env.f_edge[s], 1e-9)) if self.env.f_edge[s] > 0 else 0.0
            confs[u] = float(1.0 / (1.0 + np.exp(-(
                CONFIDENCE_BETA_TAU * delta_t
                + CONFIDENCE_BETA_SINR * sinr_norm
                + CONFIDENCE_BETA_LOAD * (1.0 - min(rho, 1.0))
            ))))
        # 若 prev_conf 提供且相邻用户置信度变化极小，平滑一下
        if prev_conf is not None:
            confs = 0.7 * confs + 0.3 * prev_conf[:K]
        return confs

    # -------------------- (4) 共识终止判据 --------------------
    def consensus(self, prev_conf: Optional[np.ndarray],
                  cur_conf: np.ndarray, round_idx: int) -> bool:
        """终止判据：置信度变化是否已收敛或达到最大轮次。

        条件：
          - 已达到 DEBATE_MAX_ROUNDS
          - 或 prev_conf != None 且 max|cur - prev| < CONSENSUS_EPSILON
        """
        if round_idx + 1 >= DEBATE_MAX_ROUNDS:
            return True
        if prev_conf is None:
            return False
        delta = float(np.max(np.abs(np.asarray(cur_conf) -
                                     np.asarray(prev_conf))))
        return delta < CONSENSUS_EPSILON

    # -------------------- (5) 调用验证智能体 --------------------
    def invoke_va(self, plan: Dict, va_runner,
                  oa_prediction: Optional[Dict] = None) -> Dict:
        """调用 VerifierAgentRunner 对方案做反事实验证。"""
        if oa_prediction is None:
            oa_prediction = self.last_oa_prediction
        return va_runner.verify(plan, oa_prediction=oa_prediction)

    # -------------------- (6) 输出最终方案 --------------------
    @staticmethod
    def finalize_plan(revised: List[Dict], K: int, M: int) -> Dict:
        """把仲裁后的方案转换为环境可执行的 plan。

        参数：
            revised: [{'user','alpha','server'}, ...]
            K, M:    系统规模
        返回：
            dict {'alpha': float32 (K,), 'server': int (K,)}
        """
        alpha = np.full(K, 0.5, dtype=np.float32)
        server = np.zeros(K, dtype=int)
        for r in revised:
            u = int(r['user'])
            if 0 <= u < K:
                alpha[u] = float(np.clip(r.get('alpha', 0.5), 0.01, 1.0))
                server[u] = int(np.clip(r.get('server', 0), 0, M - 1))
        return {'alpha': alpha, 'server': server}


# ============================================================
# 自检
# ============================================================
if __name__ == "__main__":
    from environment import MECEnvironment
    from verifier import VerifierAgentRunner

    print("=" * 60)
    print("  OrchestratorAgentRunner 自检 (with_llm=False, 启发式)")
    print("=" * 60)
    env = MECEnvironment(num_users=8, num_servers=4, seed=42)
    state = env.reset()
    text = env.state_to_text()

    oa = OrchestratorAgentRunner(env, with_llm=False)
    # 偏好推断
    omega = oa.infer_preference(state, text)
    print(f"  omega = {omega}")
    # 假提案：全选 server 0
    proposals = {f"user_{k}": {'user': k, 'alpha': 0.5,
                               'server': 0, 'confidence': 0.4, 'reason': ''}
                 for k in range(env.K)}
    arb = oa.arbitrate(proposals, [], state, text)
    print(f"  arbritrated revised count = {len(arb['revised'])}")
    # ToM 预测
    confs = oa.predict_confidence_impact(arb['revised'], state, text)
    print(f"  ToM confidences = {confs}")
    # finalize + VA
    plan = oa.finalize_plan(arb['revised'], env.K, env.M)
    va = VerifierAgentRunner(env)
    out = oa.invoke_va(plan, va)
    print(f"  VA verify: accept={out['accept']}, reason={out['reason']}")
    sim = out['sim_result']
    print(f"  sim: energy={sim['energy']:.4f} kJ, "
          f"latency={sim['latency']:.3f} s, "
          f"success={sim['success_rate']:.2%}, "
          f"sla={sim['priority_sla']:.2%}")
    print("=" * 60)
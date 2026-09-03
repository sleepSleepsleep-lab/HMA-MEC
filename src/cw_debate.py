# -*- coding: utf-8 -*-
"""
================================================================
A2: 置信度加权多轮辩论协议 (cw_debate.py)
================================================================
本模块实现 HMA-MEC 的核心交互协议 CW-Debate：
按以下五轮完成单步决策：
  Round 0  状态广播            由 OA 分发当前状态给所有 UA / EA
  Round 1  局部提议 (Propose)   每 UA 提交 (alpha_k, m_k, confidence, reason)
                               每 EA 提交容量申报
  Round 2  交叉批判 (Critique)  UA 间冲突提示 + EA 校验是否容纳
  Round 3  OA 仲裁 (Arbitrate)  OA 基于偏好 + 批判修正提案，做 ToM 预测
  Round 4  验证   (Verify)      VA 反事实仿真 + 拒绝采样 + 必要回退
  Round 5  共识终止 (Consensus) 当置信度变化 < epsilon 或达到最大轮次

支持三种运行模式：
  - "FullLLM"    在线调用 LLM，最高质量 (高 token 成本，用于离线蒸馏)
  - "Distill"    不调用 LLM，纯启发式 (用于实时推理；A3 完成后由蒸馏网替代)
  - "Hybrid"     启发式为基础，仅在低置信度时触发 LLM 调用

提供 cw_debate(env, agents, oa_runner, va_runner, mode) 统一入口。
================================================================
"""

import time
import json
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import (
    NUM_USERS, NUM_EDGE_SERVERS,
    CONFIDENCE_THRESHOLD, CONSENSUS_EPSILON, DEBATE_MAX_ROUNDS,
    VERIFY_MAX_FALLBACK,
)
from llm_client import get_llm_client, parse_json_response
from verifier import VerifierAgentRunner
from orchestrator import OrchestratorAgentRunner

logger = logging.getLogger(__name__)


# ============================================================
# Prompt 模板（UA / EA）
# ============================================================
UA_PROPOSE_SYSTEM = (
    "You are a user agent (UA) in a mobile edge computing (MEC) system. "
    "Your task is to propose a local offloading decision for one specific "
    "user. You must output JSON only."
)

UA_PROPOSE_USER_TEMPLATE = (
    "Current MEC system state:\n{state_text}\n\n"
    "Your assigned user is user_{k} (0-indexed).\n"
    "Task profile: data D={D:.1f} bit, required cycles C={C:.1f}, "
    "deadline tau={tau:.3f} s, priority p={p}.\n"
    "Local CPU frequency: {f_loc:.2e} cycles/s.\n"
    "Edge servers (capacity vs current load):\n{edge_info}\n\n"
    "Choose:\n"
    "  - alpha in [0.01, 1.0]: local execution ratio (alpha = 1 means all local)\n"
    "  - server: index in [0, M-1]\n"
    "  - confidence in [0, 1]: how confident you are that this decision satisfies your deadline\n"
    "Respond in JSON only:\n"
    '{{\n'
    '  "alpha": <float>,\n'
    '  "server": <int>,\n'
    '  "confidence": <float>,\n'
    '  "reason": "<short sentence>"\n'
    '}}'
)

UA_CRITIQUE_SYSTEM = (
    "You are a user agent (UA) for user_{k} in a MEC system. Other UAs have "
    "proposed their offloading decisions; some of them may overload a server "
    "you also chose. Point out potential conflicts for the orchestrator."
)

UA_CRITIQUE_USER_TEMPLATE = (
    "Your proposal: server={my_server}, alpha={my_alpha:.2f}.\n"
    "Other UAs' proposals that chose the SAME server as you:\n{same_server}\n\n"
    "If you think the server is overloaded, output a critique. "
    "Respond in JSON only:\n"
    '{{\n'
    '  "critique": "<one short sentence>",\n'
    '  "severity": "<low|medium|high>"\n'
    '}}'
)

EA_CAPACITY_SYSTEM = (
    "You are an edge agent (EA) for server_{m} in a MEC system. Report your "
    "remaining capacity and assess whether the proposed incoming load is "
    "acceptable. Respond in JSON only."
)

EA_CAPACITY_USER_TEMPLATE = (
    "Current system state:\n{state_text}\n\n"
    "Server_{m} capacity: F_m = {F:.2e} cycles/s; "
    "current load: L_m = {L:.2e} cycles/s.\n"
    "Incoming proposals to server_{m}:\n{proposals}\n\n"
    "Output JSON only:\n"
    '{{\n'
    '  "accept": <boolean>,\n'
    '  "remaining_after": <float cycles>,\n'
    '  "reason": "<one short sentence>"\n'
    '}}'
)


# ============================================================
# UA 行为：提议 / 批判
# ============================================================
class UserAgentRunner:
    """用户智能体 UA 的算法层（与 agent_define.UserAgent 解耦）。"""

    def __init__(self, ua, llm=None, with_llm: bool = True):
        self.ua = ua
        self.with_llm = with_llm
        self.llm = llm

    # -------------------- 局部提议 --------------------
    def propose(self, state, state_text) -> Dict:
        """生成本地候选卸载方案 (alpha, server, confidence, reason)。"""
        env = self.ua.env
        k = self.ua.k
        t = env.tasks[k]
        edge_info_lines = []
        for m in range(env.M):
            h = env.channels[k, m]
            rate = env._get_state()  # no-op; just keep signature simple
            edge_info_lines.append(
                f"  server_{m}: F={env.f_edge[m]:.2e}, "
                f"current_load={env.server_load[m]:.2e}, h_k_m={h:.2e}")
        edge_info = "\n".join(edge_info_lines)

        if not self.with_llm or self.llm is None:
            return self._heuristic_propose(env, k, t)

        user_prompt = UA_PROPOSE_USER_TEMPLATE.format(
            state_text=state_text, k=k,
            D=t['D'], C=t['C'], tau=t['tau'], p=t['priority'],
            f_loc=env.f_local[k], edge_info=edge_info)
        try:
            resp = self.llm.chat_json(
                UA_PROPOSE_SYSTEM, user_prompt,
                temperature=0.0, max_tokens=300)
            alpha = float(np.clip(resp.get('alpha', 0.5), 0.01, 1.0))
            server = int(np.clip(resp.get('server', 0), 0, env.M - 1))
            conf = float(np.clip(resp.get('confidence', 0.5), 0.0, 1.0))
            reason = resp.get('reason', '')
        except Exception as e:
            logger.warning(f"UA-{k} 提议 LLM 调用失败，回退启发式: {e}")
            return self._heuristic_propose(env, k, t)
        return {'user': k, 'alpha': alpha, 'server': server,
                'confidence': conf, 'reason': str(reason)}

    def _heuristic_propose(self, env, k, t) -> Dict:
        """启发式提议：基于信道增益与时延约束选择服务器与本地比例。"""
        # 计算 alpha 使 max(local_time, off_time) <= tau
        # 简单策略：在每服务器中选择 (1-alpha) 使 off_time 满足约束
        best = None  # (T, alpha, server)
        for m in range(env.M):
            h = env.channels[k, m]
            from environment import BANDWIDTH, NOISE_POWER, TX_POWER_USER
            rate = BANDWIDTH * np.log2(1 + TX_POWER_USER * h / NOISE_POWER)
            tx_T_const = t['D'] / max(rate, 1e-9)
            comp_T_const = t['C'] / env.f_edge[m]
            # 假设 alpha -> ?, T = max(alpha*C/f_loc, (1-alpha)*(tx_T_const+comp_T_const))
            # find alpha 使 T <= tau
            # 数值搜索：alpha = 0..1
            best_local = None
            for trial_alpha in np.linspace(0.01, 0.99, 25):
                loc_T = trial_alpha * t['C'] / env.f_local[k]
                off_T = (1 - trial_alpha) * (tx_T_const + comp_T_const)
                T = max(loc_T, off_T)
                if T <= t['tau']:
                    if (best_local is None) or (best_local[0] > T):
                        best_local = (T, trial_alpha, m)
            # 若没有满足约束的 alpha，取最小 T 的
            if best_local is None:
                trial_alpha = 0.5
                loc_T = trial_alpha * t['C'] / env.f_local[k]
                off_T = (1 - trial_alpha) * (tx_T_const + comp_T_const)
                T = max(loc_T, off_T)
                best_local = (T, trial_alpha, m)
            if (best is None) or (best[0] > best_local[0]):
                best = best_local
        T, alpha, server = best
        # 置信度计算（三维 MEC 物理量感知版）：
        #   c_k = sigmoid(β1·Δ_t + β2·SINR + β3·(1-ρ))
        # 其中 Δ_t = (τ_max - T)/τ_max 为时延余量，
        # SINR = h·P/(N0·B) 近似为信道增益 h 的归一化值，
        # ρ = server_load / capacity 为服务器负载率。
        #
        # 2026.07 修改：从一维 sigmoid(β·Δ_t) 扩展为三维，
        # 使置信度反映 MEC 特有的信道竞争与算力耦合特征。
        from config import CONFIDENCE_BETA_TAU, CONFIDENCE_BETA_SINR, CONFIDENCE_BETA_LOAD
        delta_t = (t['tau'] - T) / max(t['tau'], 1e-3)
        # 信道质量分量：使用用户 k 到所选服务器 m 的信道增益 h，
        # 归一化到 [0,1] 区间
        h = env.channels[k, server] if hasattr(env, 'channels') else 0.5
        sinr_norm = float(np.clip(h / 1e-5, 0.0, 1.0))
        # 服务器负载分量：基于当前该服务器上的累计任务量
        rho = float(env.server_load[server] / max(env.f_edge[server], 1e-9)) if hasattr(env, 'server_load') else 0.0
        conf = float(1.0 / (1.0 + np.exp(-(
            CONFIDENCE_BETA_TAU * delta_t
            + CONFIDENCE_BETA_SINR * sinr_norm
            + CONFIDENCE_BETA_LOAD * (1.0 - min(rho, 1.0))
        ))))
        return {'user': k, 'alpha': float(alpha), 'server': int(server),
                'confidence': conf, 'reason': 'heuristic-propose'}

    # -------------------- 批判 --------------------
    def critique(self, my_proposal: Dict, other_proposals: List[Dict]
                 ) -> Optional[Dict]:
        """对分配到同一服务器的其他 UA 提议给出冲突批判。"""
        my_s = my_proposal['server']
        same = [p for p in other_proposals
                if p['user'] != my_proposal['user'] and p['server'] == my_s]
        if not same:
            return None  # 无冲突
        if not self.with_llm or self.llm is None:
            return self._heuristic_critique(my_proposal, same)
        env = self.ua.env
        user_prompt = UA_CRITIQUE_USER_TEMPLATE.format(
            my_server=my_s, my_alpha=my_proposal['alpha'],
            same_server=json.dumps(same, ensure_ascii=False, indent=2),
            k=self.ua.k)
        try:
            resp = self.llm.chat_json(
                UA_CRITIQUE_SYSTEM.format(k=self.ua.k), user_prompt,
                temperature=0.0, max_tokens=200)
            return {
                'from_user': self.ua.k,
                'server': my_s,
                'critique': resp.get('critique', default=''),
                'severity': resp.get('severity', 'low'),
            }
        except Exception as e:
            logger.warning(f"UA-{self.ua.k} 批判 LLM 调用失败，回退启发式: {e}")
            return self._heuristic_critique(my_proposal, same)

    def _heuristic_critique(self, my_proposal, same) -> Dict:
        """启发式批判：估算服务器聚合负载是否超过容量。"""
        env = self.ua.env
        s = my_proposal['server']
        total_cycles = 0.0
        for p in [my_proposal] + same:
            u = p['user']
            total_cycles += (1 - p['alpha']) * env.tasks[u]['C']
        capacity = env.f_edge[s]
        severity = 'low' if total_cycles < capacity * 0.6 else (
                   'medium' if total_cycles < capacity * 1.0 else 'high')
        return {
            'from_user': self.ua.k,
            'server': s,
            'critique': f"server_{s} 累计负载 {total_cycles:.2e} "
                        f"(capacity {capacity:.2e})",
            'severity': severity,
        }


# ============================================================
# EA 行为：容量申报 / 接纳校验
# ============================================================
class EdgeAgentRunner:
    """边缘智能体 EA 的算法层。"""

    def __init__(self, ea, llm=None, with_llm: bool = True):
        self.ea = ea
        self.with_llm = with_llm
        self.llm = llm

    def report_capacity(self, state, state_text, proposals_for_m: List[Dict]
                        ) -> Dict:
        """对该服务器接收到的所有提议做容量申报与接纳判定。"""
        if not self.with_llm or self.llm is None:
            return self._heuristic_capacity(state, state_text, proposals_for_m)
        env = self.ea.env
        m = self.ea.m
        user_prompt = EA_CAPACITY_USER_TEMPLATE.format(
            state_text=state_text, m=m,
            F=env.f_edge[m], L=env.server_load[m],
            proposals=json.dumps(proposals_for_m, ensure_ascii=False, indent=2))
        try:
            resp = self.llm.chat_json(
                EA_CAPACITY_SYSTEM, user_prompt,
                temperature=0.0, max_tokens=300)
            accept = bool(resp.get('accept', True))
            remain = float(resp.get('remaining_after', 0.0))
            reason = resp.get('reason', '')
        except Exception as e:
            logger.warning(f"EA-{m} 容量申报 LLM 调用失败，回退启发式: {e}")
            return self._heuristic_capacity(state, state_text, proposals_for_m)
        return {'server': m, 'accept': accept,
                'remaining': remain, 'reason': str(reason)}

    def _heuristic_capacity(self, state, state_text, proposals_for_m) -> Dict:
        env = self.ea.env
        m = self.ea.m
        total_cycles = sum((1 - p['alpha']) * env.tasks[p['user']]['C']
                            for p in proposals_for_m)
        capacity = env.f_edge[m]
        remain = max(0.0, capacity - total_cycles)
        accept = total_cycles <= capacity
        return {'server': m, 'accept': accept, 'remaining': remain,
                'reason': f"累载 {total_cycles:.2e}, 容量 {capacity:.2e}"}


# ============================================================
# CW-Debate 五轮协议主循环
# ============================================================
def cw_debate(env, agents: Dict,
              mode: str = "Distill",
              llm=None,
              verbose: bool = False,
              fixed_preference: Optional[np.ndarray] = None,
              flags: Optional[Dict] = None,
              process_feedback: bool = False) -> Dict:
    """执行一轮完整的 CW-Debate 协议，输出最终卸载方案。

    参数：
        env:              MECEnvironment 实例
        agents:           make_agents() 返回的 dict {'UA', 'EA', 'OA', 'VA'}
        mode:             运行模式: "FullLLM" | "Distill" | "Hybrid"
        llm:              可选的 LLMClient 实例（若为 None 则按 mode 决定）
        verbose:          是否打印每轮日志（调试用）
        fixed_preference: 可选, 固定 OA 偏好权重 [ω_e, ω_l],
                          若为 None 则由 OA 自动推断
        flags:            消融实验用开关字典, 键为下列之一:
                            disable_va, disable_conf_gating, disable_tom,
                            disable_boundary, disable_priority, disable_pref,
                            disable_propose, disable_critique, disable_arbitrate,
                            disable_consensus
        process_feedback: 可选, 开启后把每一轮合并方案的反事实仿真结果
                          (能耗/时延/成功率/SLA 与 VA 判定) 追加到下一轮
                          LLM 文本上下文, 使辩论基于客观仿真证据继续协商。
                          默认 False, 保持原协议行为不变。

    返回：
        dict {
            'plan':            {'alpha': (K,), 'server': (K,)},
            'final_proposals': [{...}, ...],
            'confidence_history': [[...], [...], ...],
            'rounds_used':     int,
            'va_result':        dict,
            'fallback_count':   int,
            'estimated_energy_kJ': float,
            'estimated_latency_s': float,
        }
    """
    K, M = env.K, env.M
    flags = flags or {}

    # 若需要关闭优先级通道，提前修改状态向量（将 priority 分量置零）
    state = env._get_state()
    if flags.get("disable_priority"):
        state = state.copy()
        for k in range(K):
            state[k * 4 + 3] = 0.0
    state_text = env.state_to_text()

    # 决定是否启用 LLM
    if mode == "FullLLM":
        with_llm = True
        if llm is None:
            llm = get_llm_client()
    elif mode == "Distill":
        with_llm = False  # 蒸馏模式/启发式兜底
        llm = None
    elif mode == "Hybrid":
        with_llm = True
        if llm is None:
            llm = get_llm_client()
    else:
        raise ValueError(f"未知 mode: {mode}")

    # 构造各 Agent 的算法层 runner
    ua_runners = [UserAgentRunner(ua, llm=llm, with_llm=with_llm)
                  for ua in agents['UA']]
    ea_runners = [EdgeAgentRunner(ea, llm=llm, with_llm=with_llm)
                  for ea in agents['EA']]
    oa_runner = OrchestratorAgentRunner(env, llm=llm, with_llm=with_llm,
                                         preference=fixed_preference)
    va_runner = VerifierAgentRunner(env)

    # ---- Round 0: 状态广播 ----
    # 让所有 Agent observe 当前状态
    for ua in agents['UA']:
        ua.observe(state, state_text)
    for ea in agents['EA']:
        ea.observe(state, state_text)
    agents['OA'].observe(state, state_text)
    if agents.get('VA') is not None:
        agents['VA'].observe(state, state_text)

    if verbose:
        print("[CW-Debate] Round 0: 状态广播完成")

    # OA 偏好推断（若 disable_pref 则固定 ω=(0.5,0.5)）
    if flags.get("disable_pref"):
        omega = np.array([0.5, 0.5], dtype=np.float32)
    else:
        omega = oa_runner.infer_preference(state, state_text)

    prev_conf = None
    cur_conf = None
    confidence_history: List[List[float]] = []
    fallback_count = 0
    rounds_used = 0
    revised = []           # OA 输出
    final_plan = None
    va_result = None

    # 初始化 arb 默认值，避免 disable_arbitrate 时引用未定义变量
    arb = {'revised': [], 'estimated_energy_kJ': 0.0, 'estimated_latency_s': 0.0}

    rounds_used = DEBATE_MAX_ROUNDS  # 默认设到上限，若提前终止会更新

    # ---- 仿真反馈注入（process_feedback=True 时启用） ----
    # 2026.08 新增：把每一轮经 VA 反事实仿真的合并方案结果追加到下一轮
    # LLM 文本上下文，使辩论主体基于客观仿真证据继续协商。
    fb_lines: List[str] = []

    def _push_fb(r: int) -> None:
        s = va_out.get('sim_result') or {}
        if not s:
            return
        fb_lines.append(
            f"Round {r + 1} merged-plan simulation: "
            f"energy={s.get('energy', float('nan')):.3f} kJ, "
            f"latency={s.get('latency', float('nan')):.3f} s, "
            f"success_rate={s.get('success_rate', float('nan')):.1%}, "
            f"SLA={s.get('priority_sla', float('nan')):.1%}; "
            f"verdict={va_out.get('reason', '')}")

    for r in range(DEBATE_MAX_ROUNDS):
        rounds_used = r + 1

        # 本轮 LLM 文本上下文：开启 feedback 时追加既往仿真证据
        round_state_text = state_text
        if process_feedback and fb_lines:
            round_state_text = (
                state_text +
                "\n\n[Simulation feedback from previous negotiation rounds]\n" +
                "\n".join(fb_lines) +
                "\n[End of simulation feedback]")

        # ---- Round 1: 局部提议 ----
        proposals = {}
        for k, ur in enumerate(ua_runners):
            if flags.get("disable_propose"):
                # 随机提案替代 UAPropose
                proposals[f"user_{k}"] = {
                    'user': k,
                    'alpha': float(np.random.uniform(0.01, 0.99)),
                    'server': int(np.random.randint(0, M)),
                    'confidence': 0.5,
                    'reason': 'random-propose (ablation)',
                }
            else:
                p = ur.propose(state, round_state_text)
                # 消融: disable_conf_gating —— 压低置信度使所有 UA 进入批判
                if flags.get("disable_conf_gating") and not with_llm:
                    p['confidence'] = 0.4  # 低于默认阈值 0.6, 确保通过门控
                proposals[f"user_{k}"] = p

        # ---- Round 2: 交叉批判 ----
        critiques: List[Dict] = []
        propos_list = [proposals[f"user_{k}"] for k in range(K)]

        if not flags.get("disable_critique"):
            # 置信度门控: disable_conf_gating 时全部参与批判
            gate_threshold = 1.0 if flags.get("disable_conf_gating") else CONFIDENCE_THRESHOLD
            
            # ---- 子图修剪：负载门控 ----
            # 2026.07 新增：仅在服务器负载率较高时触发交叉批判。
            # 原理：当服务器负载充裕（ρ << 1）时，用户同时选择该服务器
            # 不会产生严重的资源竞争，交叉批判的信息增益有限，可安全跳过
            # 以降低通信复杂度。仅当服务器负载率 > LOAD_GATE 时才在该
            # 服务器的用户子图内触发全连接批判。
            # 这一设计对应论文中批判子图修剪策略，使通信拓扑从完全图
            # 退化为稀疏子图，降低大规模 K 场景下的 token 开销。
            LOAD_GATE = 0.8  # 负载门控阈值：仅 ρ > 0.8 时触发批判
            
            # 计算每服务器的负载率
            server_load_ratios = {}
            for m in range(M):
                users_on_m = [p for p in propos_list if p['server'] == m]
                total_off_cycles = sum(
                    (1 - p['alpha']) * env.tasks[p['user']]['C']
                    for p in users_on_m
                )
                server_load_ratios[m] = total_off_cycles / max(env.f_edge[m], 1e-9)
            
            for k, ur in enumerate(ua_runners):
                my_p = propos_list[k]
                my_s = my_p['server']
                # 子图修剪：仅在服务器负载率高时触发批判
                if my_p['confidence'] < gate_threshold and server_load_ratios.get(my_s, 0) > LOAD_GATE:
                    crit = ur.critique(my_p, propos_list)
                    if crit is not None:
                        critiques.append(crit)
            # EA 校验
            ea_capacity = []
            for m, er in enumerate(ea_runners):
                proposals_for_m = [p for p in propos_list if p['server'] == m]
                cap = er.report_capacity(state, round_state_text, proposals_for_m)
                ea_capacity.append(cap)
                if not cap['accept']:
                    critiques.append({
                        'from_user': -1,
                        'server': m,
                        'critique': f"EA-{m}: {cap['reason']}",
                        'severity': 'high'})

        # ---- Round 3: OA 仲裁 + Theory-of-Mind ----
        if flags.get("disable_arbitrate"):
            # 跳过仲裁，直接使用 Round 1 提案
            revised = [proposals[f"user_{k}"] for k in range(K)]
            # 跳过 ToM 预测
            cur_conf = np.full(K, 0.5, dtype=np.float32)
        else:
            arb = oa_runner.arbitrate(proposals, critiques, state, round_state_text,
                                       flags=flags,
                                       fallback_count=fallback_count)
            revised = arb['revised']
            # ToM 预测（disable_tom 时用 uniform 置信度）
            if flags.get("disable_tom"):
                cur_conf = np.full(K, 0.5, dtype=np.float32)
            else:
                cur_conf = oa_runner.predict_confidence_impact(
                    revised, state, round_state_text, prev_conf=prev_conf)
        confidence_history.append(list(cur_conf))

        # ---- Round 4: 验证 (反事实仿真 + 拒绝采样) ----
        plan = oa_runner.finalize_plan(revised, K, M)
        if flags.get("disable_va") or flags.get("disable_verify"):
            # 跳过 VA 验证，始终接受
            va_out = {'accept': True, 'sim_result': {},
                       'reason': 'skip (ablation)'}
        else:
            if arb.get('estimated_energy_kJ', 0.0) > 0:
                oa_pred = {
                    'energy':   arb['estimated_energy_kJ'],
                    'latency':  arb['estimated_latency_s'],
                }
            else:
                oa_pred = None
            va_out = oa_runner.invoke_va(plan, va_runner, oa_prediction=oa_pred)
        va_result = va_out

        # ---- 触发回退 ----
        if (not va_out['accept']) and fallback_count < VERIFY_MAX_FALLBACK:
            fallback_count += 1
            reject_reason = (
                f"{va_out['reason']}; "
                f"sim_energy={va_out['sim_result']['energy']:.4f} kJ; "
                f"sim_latency={va_out['sim_result']['latency']:.3f} s")
            if verbose:
                print(f"[CW-Debate] Round {r+1} #{fallback_count}: "
                      f"{reject_reason}")
            prev_conf = cur_conf.copy()
            if process_feedback:
                _push_fb(r)
            continue

        # ---- Round 5: 共识终止 ----
        if va_out['accept']:
            if flags.get("disable_consensus"):
                # 关闭共识: 强制跑满 R_max 轮
                prev_conf = cur_conf
                if process_feedback:
                    _push_fb(r)
                continue
            if oa_runner.consensus(prev_conf, cur_conf, r):
                final_plan = plan
                break
            else:
                prev_conf = cur_conf
                if process_feedback:
                    _push_fb(r)
                continue
        else:
            final_plan = plan
            break

    if final_plan is None:
        final_plan = plan
        if verbose:
            print("[CW-Debate] 达到最大轮次后强制输出")

    # 注：消融实验的模块移除效果已在主循环中通过 flags 控制
    # （如 disable_va、disable_tom 等均在对应轮次中跳过处理），
    # 不再通过后处理模拟的方式修改最终方案。

    return {
        'plan':              final_plan,
        'final_proposals':   revised,
        'confidence_history': confidence_history,
        'rounds_used':       rounds_used,
        'va_result':         va_result,
        'fallback_count':    fallback_count,
        'estimated_energy_kJ': (arb['estimated_energy_kJ']
                                if 'arb' in locals() else None),
        'estimated_latency_s': (arb['estimated_latency_s']
                                if 'arb' in locals() else None),
    }


# ============================================================
# 自检
# ============================================================
if __name__ == "__main__":
    from environment import MECEnvironment
    from agent_define import make_agents

    print("=" * 60)
    print("  CW-Debate 自检 (mode=Distill, 不调用 LLM)")
    print("=" * 60)
    env = MECEnvironment(num_users=8, num_servers=4, seed=42)
    env.reset()
    agents = make_agents(env, with_va=True)

    out = cw_debate(env, agents, mode="Distill", llm=None, verbose=True)
    plan = out['plan']
    sim = env.simulate(plan)
    print(f"  最终方案 alpha = {plan['alpha']}")
    print(f"  最终方案 server = {plan['server']}")
    print(f"  辩论轮数 = {out['rounds_used']}")
    print(f"  回退次数 = {out['fallback_count']}")
    print(f"  置信度历史 (最后轮) = {out['confidence_history'][-1] if out['confidence_history'] else None}")
    print(f"  VA 接受 = {out['va_result']['accept']}, reason={out['va_result']['reason']}")
    print(f"  反事实仿真: energy={sim['energy']:.4f} kJ, "
          f"latency={sim['latency']:.3f} s, "
          f"success={sim['success_rate']:.2%}, "
          f"sla={sim['priority_sla']:.2%}")
    print("=" * 60)
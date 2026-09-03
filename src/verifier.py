# -*- coding: utf-8 -*-
"""
================================================================
A4: 验证智能体模块 (verifier.py)
================================================================
本模块完整实现 VerifierAgent 对候选卸载方案的「反事实仿真 + 拒绝采样」逻辑，
是 HMA-MEC「提出 -- 批判 -- 验证」闭环中验证环节的核心。

主要功能：
  1. evaluate_plan(env, plan)         —— 反事实仿真接口的便捷包装
  2. VerifierAgent.verify(plan, ...)  —— 对单一方案作接受/拒绝判定
  3. VerifierAgent.select_best(...)   —— 从一批候选方案中拒绝采样选出最优
  4. VerifierAgent.reject_reason(...) —— 拒绝理由文本，便于 OA 重仲裁

所谓「反事实仿真」是指：在当前真实环境的快照上预演候选方案，
但不修改真实环境状态（由 MECEnvironment.simulate() 保证），
从而获得客观的能耗 / 时延 / SLA 评估。LLM 推理可能给出物理上
不可行、或在奖励信号上作弊的方案 (reward hacking)，VA 用仿真
结果与 OA 自评估之间的偏差作为拒绝依据，构成独立第三方验证。
================================================================
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from config import (
    VERIFY_GAP_TOLERANCE,        # OA 自评估与仿真结果之间允许的相对偏差阈值
    VERIFY_MAX_FALLBACK,         # 最多回退次数
)


# ============================================================
# 反事实仿真接口
# ============================================================
def evaluate_plan(env, plan):
    """反事实仿真接口的便捷包装。

    参数：
        env:  MECEnvironment 实例（必须提供 simulate() 方法）
        plan: dict {'alpha': (K,), 'server': (K,)}
    返回：
        dict: 至少包含 {'energy', 'latency', 'success_rate', 'priority_sla'}
    """
    return env.simulate(plan)


# ============================================================
# 默认拒绝 / 接受判据
# ============================================================
def _gap(actual: float, predicted: float, eps: float = 1e-6) -> float:
    """相对偏差：|actual - predicted| / max(|predicted|, eps)。"""
    return abs(actual - predicted) / max(abs(predicted), eps)


def default_accept_rule(sim_result: Dict,
                        oa_prediction: Optional[Dict] = None,
                        gap_tol: float = VERIFY_GAP_TOLERANCE,
                        sla_floor: float = 0.5,
                        success_floor: float = 0.5) -> Tuple[bool, str]:
    """默认接受/拒绝规则。

    参数：
        sim_result:   反事实仿真结果 dict
        oa_prediction: OA 自评估 dict（可选）；若为 None 则只做绝对底线校验
        gap_tol:       OA 自评估与仿真之间允许的相对偏差
        sla_floor:     高优先级 SLA 达成率的下限
        success_floor: 总体任务成功率的下限
    返回：
        (accept: bool, reason: str)
    """
    reasons = []

    # (a) 绝对底线校验：SLA 与成功率必须达到下限
    if sim_result['priority_sla'] < sla_floor:
        reasons.append(
            f"高优先级 SLA 偏低 ({sim_result['priority_sla']:.2%} < {sla_floor:.0%})")
    if sim_result['success_rate'] < success_floor:
        reasons.append(
            f"任务成功率偏低 ({sim_result['success_rate']:.2%} < {success_floor:.0%})")

    # (b) 相对偏差校验：OA 自评估与仿真之间的偏差
    if oa_prediction is not None:
        for key, label in [('energy', '能耗'), ('latency', '时延')]:
            if key in oa_prediction and key in sim_result:
                g = _gap(float(sim_result[key]), float(oa_prediction[key]))
                if g > gap_tol:
                    reasons.append(
                        f"{label}与 OA 预测偏差 {g:.2%} (超过 {gap_tol:.0%})")

    accept = (len(reasons) == 0)
    return accept, "; ".join(reasons) if reasons else ""


# ============================================================
# 验证智能体（与 agent_define.py 中 VerifierAgent 解耦的算法层）
# ============================================================
class VerifierAgentRunner:
    """验证智能体算法层。

    与 agent_define.py 中 VerifierAgent（身份载体）解耦：
      - VerifierAgent（定义层）持有身份 / 边界信息，提供 verify() 接口
      - VerifierAgentRunner（算法层）封装拒绝采样、最优选择、回退控制等
    """

    def __init__(self,
                 env,
                 gap_tol: float = VERIFY_GAP_TOLERANCE,
                 max_fallback: int = VERIFY_MAX_FALLBACK,
                 sla_floor: float = 0.5,
                 success_floor: float = 0.5):
        """
        参数：
            env:           MECEnvironment 实例
            gap_tol:       OA 自评估与仿真间允许的相对偏差
            max_fallback:  最多回退次数
            sla_floor:     高优先级 SLA 下限
            success_floor: 总体成功率下限
        """
        self.env = env
        self.gap_tol = gap_tol
        self.max_fallback = max_fallback
        self.sla_floor = sla_floor
        self.success_floor = success_floor

    # -------------------- 单方案验证 --------------------
    def verify(self, plan: Dict,
               oa_prediction: Optional[Dict] = None) -> Dict:
        """对单一方案做反事实仿真并判定是否接受。

        参数：
            plan:          dict {'alpha': (K,), 'server': (K,)}
            oa_prediction: 可选，OA 自评估 dict
        返回：
            dict {
                'accept':     bool,
                'sim_result': dict,        # 反事实仿真结果
                'reason':     str,         # 拒绝/接受理由
                'fallback_remaining': int  # 剩余可回退次数
            }
        """
        sim = evaluate_plan(self.env, plan)
        accept, reason = default_accept_rule(
            sim, oa_prediction,
            gap_tol=self.gap_tol,
            sla_floor=self.sla_floor,
            success_floor=self.success_floor)
        return {
            'accept': accept,
            'sim_result': sim,
            'reason': reason,
            'fallback_remaining': self.max_fallback,
        }

    # -------------------- 多候选拒绝采样 --------------------
    def select_best(self, plans: List[Dict],
                    oa_predictions: Optional[List[Dict]] = None,
                    preference: Optional[np.ndarray] = None
                    ) -> Tuple[Dict, Dict, int]:
        """从一批候选方案中选择最佳方案。

        流程：
          1. 对每个候选方案做反事实仿真
          2. 拒绝任一底线校验失败的方案
          3. 在剩余方案中按 preference 加权代价选最优
          4. 若全部方案均被拒绝，退而选择 ``代价最小'' 的方案
             并在 reason 中提示 ``未通过底线校验''

        参数：
            plans:          候选方案列表 [{'alpha','server'}, ...]
            oa_predictions: 每个方案对应的 OA 自评估（可选）
            preference:    长度 2 的偏好向量 [omega_e, omega_l]，加权代价
                           cost = omega_e * energy + omega_l * latency
                           若为 None，则等权重 [0.5, 0.5]

        返回：
            (best_plan, best_sim_result, best_index)
        """
        assert len(plans) > 0, "候选方案列表不能为空"
        if oa_predictions is None:
            oa_predictions = [None] * len(plans)
        if preference is None:
            preference = np.array([0.5, 0.5], dtype=np.float32)
        preference = np.asarray(preference, dtype=np.float32)
        # 归一化偏好，避免和不为 1 时影响量纲
        if preference.sum() > 0:
            preference = preference / preference.sum()

        results = []
        for i, (plan, pred) in enumerate(zip(plans, oa_predictions)):
            sim = evaluate_plan(self.env, plan)
            accept, reason = default_accept_rule(
                sim, pred,
                gap_tol=self.gap_tol,
                sla_floor=self.sla_floor,
                success_floor=self.success_floor)
            cost = float(preference[0] * sim['energy'] +
                         preference[1] * sim['latency'])
            results.append({
                'plan': plan, 'sim': sim, 'accept': accept,
                'reason': reason, 'cost': cost, 'index': i,
            })

        # 优先在已通过底线校验的方案中选最优
        accepted = [r for r in results if r['accept']]
        if accepted:
            best = min(accepted, key=lambda r: r['cost'])
            return best['plan'], best['sim'], best['index']

        # 全部拒绝，则退而选择代价最小者（提示底线未通过）
        best = min(results, key=lambda r: r['cost'])
        best['reason'] += " (所有候选均未通过底线校验，回退到代价最小方案)"
        return best['plan'], best['sim'], best['index']

    # -------------------- 回退控制 --------------------
    def should_fallback(self, verify_result: Dict,
                        fallback_count: int) -> bool:
        """是否应当触发回退（让 OA 重新仲裁）。

        条件：方案被拒绝 AND 还有剩余回退次数。
        """
        if verify_result['accept']:
            return False
        remaining = self.max_fallback - fallback_count
        return remaining > 0

    # -------------------- 拒绝理由文本 --------------------
    @staticmethod
    def reject_reason(verify_result: Dict) -> str:
        """生成给 OA 重仲裁参考的拒绝理由文本。"""
        if verify_result['accept']:
            return ""
        sim = verify_result.get('sim_result', {})
        reason = verify_result.get('reason', "")
        return (f"VA 拒绝：{reason}；"
                f"实测量: energy={sim.get('energy', 0):.4f} kJ, "
                f"latency={sim.get('latency', 0):.3f} s, "
                f"success={sim.get('success_rate', 0):.2%}, "
                f"priority_sla={sim.get('priority_sla', 0):.2%}.")


# ============================================================
# 兼容 agent_define.VerifierAgent.verify() 的 thin-wrapper
# ============================================================
def verify_with_runner(va, plan, oa_prediction=None):
    """给 agent_define.VerifierAgent.verify() 使用的便捷函数。

    va 即 VerifierAgent（agent_define.py 中定义）。
    本函数返回与 agent_define.py 中 VerifierAgent.verify() 兼容的格式，
    但拒绝判定使用本模块的默认规则。
    """
    env = va.env
    runner = VerifierAgentRunner(env)
    return runner.verify(plan, oa_prediction)


# ============================================================
# 自检
# ============================================================
if __name__ == "__main__":
    from environment import MECEnvironment

    print("=" * 60)
    print("  VerifierAgentRunner 自检")
    print("=" * 60)

    env = MECEnvironment(num_users=8, num_servers=4, seed=42)
    env.reset()

    runner = VerifierAgentRunner(env)

    # candidate 1: 合理方案（局部估算）
    plan1 = {
        'alpha':  np.full(8, 0.5, dtype=np.float32),
        'server': np.array([0, 1, 2, 3, 0, 1, 2, 3]),
    }
    # candidate 2: 全本地（在低延迟场景下大概率不通过底线）
    plan2 = {
        'alpha':  np.full(8, 0.99, dtype=np.float32),
        'server': np.zeros(8, dtype=int),
    }
    # candidate 3: 全卸载到同一服务器（高冲突，大概率过载）
    plan3 = {
        'alpha':  np.full(8, 0.01, dtype=np.float32),
        'server': np.zeros(8, dtype=int),
    }

    plans = [plan1, plan2, plan3]
    preds = [
        {'energy': 0.011, 'latency': 0.9},
        {'energy': 0.020, 'latency': 0.8},
        {'energy': 0.005, 'latency': 1.4},
    ]
    pref = np.array([0.5, 0.5], dtype=np.float32)
    best_plan, best_sim, idx = runner.select_best(plans, preds, pref)
    print(f"  最优方案 index={idx}: energy={best_sim['energy']:.4f} kJ, "
          f"latency={best_sim['latency']:.3f} s, "
          f"success={best_sim['success_rate']:.2%}, "
          f"priority_sla={best_sim['priority_sla']:.2%}")

    # 单方案验证示例
    out = runner.verify(plan1, preds[0])
    print(f"  plan1 verify: accept={out['accept']}, reason={out['reason']}")
    print("=" * 60)
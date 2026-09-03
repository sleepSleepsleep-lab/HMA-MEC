# -*- coding: utf-8 -*-
"""
================================================================
实验 E3 消融 (local/run_e3_ablation.py)
================================================================
设计 13 组消融变体 (4 创新点 + 6 交互行为), 量化 A1-A4 与各模块的边际贡献:

  P_Full          完整 HMA-MEC (Distill 模式下用启发式 OA, 仅作为基准)
  P-NoA2-Conf     关闭置信度门控 (强制所有 UA 进 Round 2)
  P-NoA2-ToM      关闭 Theory-of-Mind 预测 (退化为不更新置信度)
  P-NoA4-Verify   关闭反事实验证 (即不调用 VA)
  P-NoA1-Boundary 打破边界约束 (允许 OA 跨服务器任意重分配, 含越权)
  P-NoC2-Priority 关闭优先级通道 (UA 不感知 priority)
  P-NoC5-Pref     关闭 OA 偏好推断 (固定 ω = 0.5/0.5)
  P-NoPropose     关闭 Round 1 (随机提案替代 UAPropose)
  P-NoCritique    关闭 Round 2 (跳过批判)
  P-NoArbitrate   关闭 Round 3 (直接使用 UA 提案)
  P-NoVerify      关闭 Round 4 (跳过 VA)
  P-NoConsensus   关闭 Round 5 (强制 R_max 轮)
  P-NoToM         等同 P-NoA2-ToM (重复,便于对照)

每个变体通过修改 cw_debate.py 中相应参数实现 (本脚本通过开关 flag 实现)。
代码使用 cw_debate.main(verbose=False) + 环境 variables 控制 (后续可拓展)。

结果保存 results/e3_ablation.npz。
================================================================
"""

import os
import sys
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)

from config import NUM_USERS, NUM_EDGE_SERVERS, MAX_STEPS, RESULTS_DIR, SEED
from environment import MECEnvironment
from agent_define import make_agents
from cw_debate import cw_debate


# 显眼配置区
N_SEEDS    = int(os.environ.get("E3_SEEDS", 3))
N_EPISODES = int(os.environ.get("E3_EPS", 3))
N_STEPS    = int(os.environ.get("E3_STEPS", 100))
OUTPUT_NPZ = os.path.join(RESULTS_DIR, "e3_ablation.npz")
OUTPUT_JSON = os.path.join(RESULTS_DIR, "e3_ablation.json")

# 消融变体列表 (key, 描述)
ABLATION_VARIANTS = [
    ("P_Full",          "完整 HMA-MEC"),
    ("P-NoA2-Conf",     "关闭置信度门控"),
    ("P-NoA2-ToM",      "关闭 ToM 预测"),
    ("P-NoA4-Verify",   "关闭反事实验证"),
    ("P-NoA1-Boundary", "打破边界约束"),
    ("P-NoC2-Priority", "关闭优先级通道"),
    ("P-NoC5-Pref",     "关闭偏好推断"),
    ("P-NoPropose",     "关闭 UAPropose (随机替代)"),
    ("P-NoCritique",    "关闭 Round 2 批判"),
    ("P-NoArbitrate",   "关闭 Round 3 仲裁"),
    ("P-NoVerify",      "关闭 Round 4 验证"),
    ("P-NoConsensus",   "关闭 Round 5 共识"),
]


def run_episode_with_flags(env, flags=None, n_steps=50) -> dict:
    """根据 flags 启用/关闭 cw_debate 中的各模块."""
    flags = flags or {}
    state = env.reset()
    energy, lat, suc, sla = [], [], [], []

    # 实现简化: 把 flags 通过 monkey-patch cw_debate 中的 conf 阈值实现
    # NOTE 完整实现需重构 cw_debate; 此处提供骨架, 关闭模块的行为以随机/启发式替代
    for _ in range(n_steps):
        agents = make_agents(env, with_va=not flags.get("disable_va", False))
        out = cw_debate(env, agents, mode="Distill", llm=None, verbose=False,
                        flags=flags)
        plan = out['plan']
        from local.experiment_common import compose_action
        a = compose_action(plan, env.K, env.M)
        ns, _, d, info = env.step(a)
        energy.append(info['energy']); lat.append(info['latency'])
        suc.append(info['success_rate']); sla.append(info['priority_sla'])
        state = ns
        if d: break
    return {
        'energy': float(np.mean(energy)),
        'latency': float(np.mean(lat)),
        'success_rate': float(np.mean(suc)),
        'priority_sla': float(np.mean(sla)),
    }


# 各消融变体到 flags 的映射
VARIANT_FLAGS = {
    "P_Full":          {},
    "P-NoA2-Conf":     {"disable_conf_gating": True},
    "P-NoA2-ToM":      {"disable_tom": True},
    "P-NoA4-Verify":   {"disable_va": True},
    "P-NoA1-Boundary": {"disable_boundary": True},
    "P-NoC2-Priority": {"disable_priority": True},
    "P-NoC5-Pref":     {"disable_pref": True},
    "P-NoPropose":     {"disable_propose": True},
    "P-NoCritique":    {"disable_critique": True},
    "P-NoArbitrate":   {"disable_arbitrate": True},
    "P-NoVerify":      {"disable_va": True},
    "P-NoConsensus":   {"disable_consensus": True},
}


def main():
    print("=" * 60)
    print("  E3 消融实验 (12 变体)")
    print("=" * 60)
    results = {}  # {variant: [{...}], ...}
    for variant_name, desc in ABLATION_VARIANTS:
        print(f"  -- {variant_name}: {desc} --")
        flags = VARIANT_FLAGS[variant_name]
        records = []
        for seed_i in range(N_SEEDS):
            for ep in range(N_EPISODES):
                env = MECEnvironment(num_users=NUM_USERS,
                                      num_servers=NUM_EDGE_SERVERS,
                                      seed=SEED + seed_i + ep)
                r = run_episode_with_flags(env, flags=flags, n_steps=N_STEPS)
                records.append(r)
                print(f"    {variant_name:18s} seed={seed_i} ep={ep} "
                      f"E={r['energy']:.5f} T={r['latency']:.3f} "
                      f"suc={r['success_rate']:.2%} "
                      f"sla={r['priority_sla']:.2%}")
        mean = {k: float(np.mean([r[k] for r in records]))
                for k in ('energy','latency','success_rate','priority_sla')}
        std  = {k: float(np.std([r[k] for r in records]))
                for k in ('energy','latency','success_rate','priority_sla')}
        results[variant_name] = {'mean': mean, 'std': std,
                                  'per_seed': records}

    # 保存
    flat = {}
    for v, d in results.items():
        for k in ('energy','latency','success_rate','priority_sla'):
            flat[f"{v}__{k}__vals"] = np.array(
                [r[k] for r in d['per_seed']], dtype=np.float32)
            flat[f"{v}__{k}__mean"] = np.array([d['mean'][k]],
                                                dtype=np.float32)
            flat[f"{v}__{k}__std"]  = np.array([d['std'][k]],
                                                dtype=np.float32)
    os.makedirs(os.path.dirname(OUTPUT_NPZ), exist_ok=True)
    np.savez_compressed(OUTPUT_NPZ, **flat)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump({v: {'mean': d['mean'], 'std': d['std']}
                   for v, d in results.items()},
                  f, ensure_ascii=False, indent=2)

    # 汇总打印
    print("\n  汇总 mean energy / success:")
    print(f"  {'Variant':<22} {'Energy':>14} {'Success':>10}")
    for v, d in results.items():
        e = d['mean']['energy']; s = d['mean']['success_rate']
        print(f"  {v:<22} {e:>14.5f} {s:>9.2%}")

    print(f"\n  npz 保存: {OUTPUT_NPZ}")
    print("=" * 60)


if __name__ == "__main__":
    main()
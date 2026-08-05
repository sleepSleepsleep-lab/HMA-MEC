# -*- coding: utf-8 -*-
"""
================================================================
实验 E1 主对比 (local/run_e1_main.py)
================================================================
本脚本在基准 8 用户 4 服务器场景下，对 B1-B4 + SAC + DDPG + HMA(Distill/Hybrid)
七至八种方法在多种子重复实验下评估能耗 / 时延 / 成功率 / 高优先级 SLA，
保存 npz 与 json 摘要供 fig_e1_main.py 绘图与论文写入。

特性:
  - 可在 CPU 上完成 (SAC/DDPG 训练 epoch 显眼配置)
  - 结果自动保存到 results/e1_comparison.npz, 可重用

运行:
    python local/run_e1_main.py
================================================================
"""

import os
import sys
import json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)

from config import (
    NUM_USERS, NUM_EDGE_SERVERS, MAX_STEPS,
    RESULTS_DIR, SEED,
)
from local.experiment_common import run_multi_episodes, save_npz
from local.baselines import SAC_EPOCHS, DDPG_EPOCHS


# ============================================================
# 显眼配置区
# ============================================================
N_SEEDS    = 2             # 种子数; 论文/期刊建议 5
N_EPISODES = 3             # 每种子下 episode 数; 期刊建议 50
N_STEPS    = 100           # 每 episode 步数 (缩短到 100 验证用; 论文用 200)
SAC_EP    = 30             # SAC 训练 epoch; 论文建议 500
DDPG_EP   = 30             # DDPG 训练 epoch
# USE_HMA: (Distill, Hybrid) 任选启用; FullLLM 仅在用户允许调用 LLM 时启用
USE_HMA   = (True, True)

OUTPUT_NPZ  = os.path.join(RESULTS_DIR, "e1_comparison.npz")
OUTPUT_JSON = os.path.join(RESULTS_DIR, "e1_comparison.json")


# ============================================================
# 主入口
# ============================================================
def main():
    print("=" * 60)
    print("  E1 主对比实验 (7-8 个方法 × 多种子)")
    print("=" * 60)
    print(f"  K={NUM_USERS}  M={NUM_EDGE_SERVERS}  "
          f"seeds={N_SEEDS}  ep/seed={N_EPISODES}  "
          f"steps/ep={N_STEPS}")
    method_specs = [
        {'name': 'Greedy',    },
        {'name': 'AllLocal', },
        {'name': 'AllEdge',  },
        {'name': 'Random',   },
        {'name': 'SAC',       "epochs": SAC_EP},
        {'name': 'DDPG',      "epochs": DDPG_EP},
    ]
    if USE_HMA[0]: method_specs.append({'name': 'HMA-Distill'})
    if USE_HMA[1]: method_specs.append({'name': 'HMA-Hybrid'})
    # 同时增加 B7 (COMLLM-lite) 启发式 LLM 单决策者对照
    # 实现位于 local.baselines. 单一 LLM 决策者直接以启发式 AllEdge 替代
    # (不调用 LLM 时, COMLLM-lite 的"语义推理"等价于 AllEdge 启发式)
    # method_specs.append({'name': 'B7-COMLLM-lite'})  # TODO: 启用开关

    results = run_multi_episodes(
        method_specs,
        K=NUM_USERS, M=NUM_EDGE_SERVERS,
        n_seeds=N_SEEDS, n_episodes=N_EPISODES, n_steps=N_STEPS,
        verbose=False,
    )
    save_npz(OUTPUT_NPZ, results)

    # 打印汇总表
    print("\n  汇总:")
    print(f"  {'Method':<14} {'Energy(kJ)':>14} "
          f"{'Latency(s)':>14} {'SuccessRate':>14} "
          f"{'SLA':>14}")
    print("  " + "-" * 70)
    for m, d in results.items():
        e = d['mean']['energy']; la = d['mean']['latency']
        s = d['mean']['success_rate']; sl = d['mean']['priority_sla']
        es = d['std']['energy']; las = d['std']['latency']
        ss = d['std']['success_rate']; sls = d['std']['priority_sla']
        print(f"  {m:<14} {f'{e:.5f}±{es:.5f}':>14} "
              f"{f'{la:.3f}±{las:.3f}':>14} "
              f"{f'{s:.2%}±{ss:.2%}':>14} "
              f"{f'{sl:.2%}±{sls:.2%}':>14}")

    # 写 json 摘要
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump({
            m: {'mean': d['mean'], 'std': d['std'],
                'n_samples': len(d['per_seed'])}
            for m, d in results.items()
        }, f, ensure_ascii=False, indent=2)
    print(f"\n  摘要保存: {OUTPUT_JSON}")
    print("=" * 60)


if __name__ == "__main__":
    main()
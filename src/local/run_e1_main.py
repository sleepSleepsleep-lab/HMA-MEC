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
    RESULTS_DIR, SEED, GA_POP_SIZE, GA_GENERATIONS,
)
from local.experiment_common import run_multi_episodes, save_npz
from local.baselines import SAC_EPOCHS, DDPG_EPOCHS


# ============================================================
# 显眼配置区
# ============================================================
N_SEEDS    = 5             # 种子数 (D3 整改: 原 2, 论文规模 5)
N_EPISODES = 50            # 每种子下 episode 数 (D3 整改: 原 3, 论文规模 50)
N_STEPS    = 200           # 每 episode 步数 (D3 整改: 原 100, 论文规模 200)
SAC_EP    = 500            # SAC 训练 epoch (D3/M3 整改: 原 30, 论文规模 500)
DDPG_EP   = 500            # DDPG 训练 epoch (原 30)
# USE_HMA: (Distill, Hybrid) 任选启用; FullLLM 仅在用户允许调用 LLM 时启用
USE_HMA   = (True, True)

OUTPUT_NPZ  = os.path.join(RESULTS_DIR, "e1_comparison.npz")
OUTPUT_JSON = os.path.join(RESULTS_DIR, "e1_comparison.json")


def _save_callback(partial):
    """长跑增量保存: 每完成一个方法立即落盘, 防止中途失败丢失已算结果。"""
    import numpy as np
    from local.experiment_common import save_npz
    save_npz(OUTPUT_NPZ, partial)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump({
            m: {'mean': d['mean'], 'std': d['std'],
                'n_samples': len(d['per_seed'])}
            for m, d in partial.items()
        }, f, ensure_ascii=False, indent=2)
    print(f"  [增量保存] 已保存 {len(partial)} 个方法 -> {OUTPUT_NPZ} / {OUTPUT_JSON}")


# ============================================================
# 主入口
# ============================================================
def main(smoke=False, llm_backend=None):
    global N_SEEDS, N_EPISODES, N_STEPS, SAC_EP, DDPG_EP
    if smoke:
        # --smoke: 临时小参数验证管线（不产生论文级数据）
        N_SEEDS, N_EPISODES, N_STEPS = 1, 1, 10
        SAC_EP, DDPG_EP = 5, 5
    backend = llm_backend or ('heuristic_proxy' if smoke else 'local_vllm')
    b7_epochs = 5 if smoke else 300

    print("=" * 60)
    print("  E1 主对比实验 (7-8 个方法 × 多种子)")
    print("=" * 60)
    print(f"  K={NUM_USERS}  M={NUM_EDGE_SERVERS}  "
          f"seeds={N_SEEDS}  ep/seed={N_EPISODES}  "
          f"steps/ep={N_STEPS}")
    print(f"  B7/B8 LLM 后端: {backend}")
    method_specs = [
        {'name': 'Greedy',    },
        {'name': 'AllLocal', },
        {'name': 'AllEdge',  },
        {'name': 'Random',   },
        {'name': 'SAC',       "epochs": SAC_EP},
        {'name': 'DDPG',      "epochs": DDPG_EP},
        # C5 整改: GA 离线最优参考 (Q5) + B7/B8 真实现 (01 文档, 替换原 B7-COMLLM-lite TODO)
        {'name': 'GA',        "pop": GA_POP_SIZE, "gens": GA_GENERATIONS},
        {'name': 'B7-LeDRL',  'epochs': b7_epochs, 'llm_backend': backend},
        {'name': 'B8-SingleLLM', 'llm_backend': backend},
    ]
    if USE_HMA[0]: method_specs.append({'name': 'HMA-Distill'})
    if USE_HMA[1]: method_specs.append({'name': 'HMA-Hybrid'})

    results = run_multi_episodes(
        method_specs,
        K=NUM_USERS, M=NUM_EDGE_SERVERS,
        n_seeds=N_SEEDS, n_episodes=N_EPISODES, n_steps=N_STEPS,
        verbose=False,
        save_callback=_save_callback,   # 2026-08: 长跑增量保存
        record_experiment="e1",          # results_store: 运行中逐 episode 记录
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="小规模冒烟(1 seed×1 ep×10 步, B7/B8 用假 LLM)")
    parser.add_argument("--llm-backend", default=None,
                        help="B7/B8 的 LLM 后端: local_vllm/deepseek/openai/qwen/"
                             "local_transformers, 或 heuristic_proxy(假 LLM)")
    args = parser.parse_args()
    main(smoke=args.smoke, llm_backend=args.llm_backend)
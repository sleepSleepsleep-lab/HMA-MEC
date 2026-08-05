# -*- coding: utf-8 -*-
"""
================================================================
实验 E2 可扩展性 (local/run_e2_scalability.py)
================================================================
固定 M = 4, 改变 K ∈ {4, 8, 12, 16, 24, 32}, 对各方法在不同规模下重复评估,
观察:
  - 性能指标随 K 退化程度
  - HMA 的 token 数与单步推理时延随 K 是否线性增长 (验证稀疏通信必要性, 仅 Hybrid)
  - Agent 拓扑生成时延

结果保存 results/e2_scalability.npz; 由 fig_e2_scalability.py 绘图。
================================================================
"""

import os
import sys
import time
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)

from config import NUM_EDGE_SERVERS, MAX_STEPS, RESULTS_DIR, SEED
from local.experiment_common import run_multi_episodes, save_npz
from local.baselines import SAC_EPOCHS, DDPG_EPOCHS


# 显眼配置区
K_LIST = [4, 8, 12, 16, 24, 32]
M_FIX  = 4
N_SEEDS    = 2
N_EPISODES = 3
N_STEPS    = 100
SAC_EP    = 30
DDPG_EP   = 30
USE_SAC   = False   # 设为 False 跳过 SAC 训练以加速；论文用 True
USE_DDPG  = False
USE_HMA_DISTILL = True
USE_HMA_HYBRID  = True

OUTPUT_NPZ = os.path.join(RESULTS_DIR, "e2_scalability.npz")
OUTPUT_JSON = os.path.join(RESULTS_DIR, "e2_scalability.json")


def main():
    print("=" * 60)
    print("  E2 可扩展性实验  K∈", K_LIST)
    print("=" * 60)
    all_results = {}     # {K -> {method -> {mean,std,per_seed}}}

    for K in K_LIST:
        print(f"\n  -- K = {K}  M = {M_FIX} --")
        method_specs = [
            {'name': 'Greedy'},
            {'name': 'AllLocal'},
            {'name': 'AllEdge'},
            {'name': 'Random'},
        ]
        if USE_SAC: method_specs.append({'name': 'SAC', 'epochs': SAC_EP})
        if USE_HMA_DISTILL:
            method_specs.append({'name': 'HMA-Distill'})
        if USE_HMA_HYBRID:
            method_specs.append({'name': 'HMA-Hybrid'})

        results = run_multi_episodes(method_specs, K=K, M=M_FIX,
                                       n_seeds=N_SEEDS,
                                       n_episodes=N_EPISODES,
                                       n_steps=N_STEPS, verbose=False)
        all_results[K] = results

    # 扁平保存
    flat = {}
    for K, results in all_results.items():
        for m, d in results.items():
            for k in ('energy','latency','success_rate','priority_sla'):
                key = f"K{K}__{m}__{k}__mean"
                flat[key] = np.array([d['mean'][k]], dtype=np.float32)
            # 完整 per_seed 值列表 (用于箱线图)
            for k in ('energy','latency','success_rate','priority_sla'):
                key2 = f"K{K}__{m}__{k}__vals"
                flat[key2] = np.array([r[k] for r in d['per_seed']],
                                       dtype=np.float32)
    os.makedirs(os.path.dirname(OUTPUT_NPZ), exist_ok=True)
    np.savez_compressed(OUTPUT_NPZ, **flat)

    # 打印精简汇总
    print("\n  汇总 mean energy:")
    print(f"  {'K':<6}", *[f"{m:<14}" for m in
                        ['Greedy','AllLocal','AllEdge','Random',
                         'SAC', 'HMA-Distill']])
    for K, results in all_results.items():
        line = f"  {K:<6}"
        for m in ['Greedy','AllLocal','AllEdge','Random','SAC','HMA-Distill']:
            e = results.get(m, {}).get('mean', {}).get('energy', None)
            line += f"  {e:>14.5f}" if e is not None else f"  {'--':>14}"
        print(line)

    print(f"\n  npz 保存: {OUTPUT_NPZ}")
    print("=" * 60)


if __name__ == "__main__":
    main()
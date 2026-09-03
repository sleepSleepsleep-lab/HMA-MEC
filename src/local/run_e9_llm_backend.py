# -*- coding: utf-8 -*-
"""
================================================================
E9 多 LLM 后端对比 (local/run_e9_llm_backend.py)
================================================================
检验 HMA-MEC 框架在不同 LLM 后端下的辩论数据质量，进而影响蒸馏策略
的最终性能。候选后端：

  1. deepseek-v4-flash   (论文默认, DeepSeek)
  2. Qwen3.7-plus        (阿里云百炼)
  3. Qwen3.7-max         (阿里云百炼)

实验流程：
  对每个后端 b:
    1) 在 GPU 服务器上用 src/gpu/gen_distill_dataset.py 采集辩论样本,
       保存 results/debate_dataset_{b}.jsonl
    2) 用 DistillAgentTrainer (按统一 E9_EPOCHS=100, batch=128) 训练
       得到 results/checkpoints/distill_{b}.pth
    3) 在基准 (K, M) 下评估 HMA-Distill 与 (可选) Hybrid 模式指标

本脚本不直接采集 LLM 样本（保持与 GPU 服务器解耦）；它仅执行
*训练 + 评估*两步——所有数据采集需通过 cy gen_distill_dataset.py 完成。

若 checkpoint 不存在，则跳过该后端并打印警告。

输出 results/e9_llm_backend.npz + .json。
================================================================
"""

import os
import sys
import json
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)

from config import (
    NUM_USERS, NUM_EDGE_SERVERS, RESULTS_DIR, SEED,
    CHECKPOINT_DIR, POLICY_NET_EPOCHS, POLICY_NET_BATCH,
    LLM_MODEL,
)
from environment import MECEnvironment
from distill_agent import (DistillAgentTrainer, load_debate_dataset)
from agent_runner import HMAAgentRunner
from local.experiment_common import compose_action


# ============================================================
# 显眼配置区
# ============================================================
# 后端到对应辩论数据集与（可选）预训练 checkpoint 的映射。
# 用户在 GPU 服务器完成 LLM 采集并保存为对应的 jsonl 文件后,
# 本脚本即可自动训练和评估。
BACKEND_CONFIG = {
    "Qwen2.5-7B":   os.path.join(RESULTS_DIR, "debate_dataset_qwen2_5_7b.jsonl"),
    "Llama-3.1-8B": os.path.join(RESULTS_DIR, "debate_dataset_llama_3_1_8b.jsonl"),
    "Mistral-7B":   os.path.join(RESULTS_DIR, "debate_dataset_mistral_7b.jsonl"),
    "DeepSeek":     os.path.join(RESULTS_DIR, "debate_dataset_deepseek.jsonl"),
    "Qwen3.7-Plus": os.path.join(RESULTS_DIR, "debate_dataset_qwen3.7-plus.jsonl"),
    "Qwen3.7-Max":  os.path.join(RESULTS_DIR, "debate_dataset_qwen3.7-max.jsonl"),
    "Qwen3.6-Plus": os.path.join(RESULTS_DIR, "debate_dataset_qwen3.6-plus.jsonl"),
    "OpenAI-4o":    os.path.join(RESULTS_DIR, "debate_dataset_openai.jsonl"),
    "Qwen3-8B":     os.path.join(RESULTS_DIR, "debate_dataset_qwen3_8b.jsonl"),
    "Llama-3-8B":   os.path.join(RESULTS_DIR, "debate_dataset_llama3_8b.jsonl"),
}
EPOCHS  = int(os.environ.get("E9_EPOCHS", 30))
BATCH   = int(os.environ.get("E9_BATCH",   128))
ALIGN_N = int(os.environ.get("E9_ALIGN_N", 479))  # P1-1: 三后端统一样本量对齐
N_SEEDS    = 2
N_EPISODES = 3
N_STEPS    = 100
OUTPUT_NPZ  = os.path.join(RESULTS_DIR, "e9_llm_backend.npz")
OUTPUT_JSON = os.path.join(RESULTS_DIR, "e9_llm_backend.json")
OUT_CKPT_PREFIX = os.path.join(CHECKPOINT_DIR, "distill_backend_{name}.pth")


def train_eval_backend(name, dataset_path):
    print(f"  - backend={name}, dataset={dataset_path}")
    if not os.path.exists(dataset_path):
        print(f"    [skip] 数据集不存在 (请在 GPU 服务器采集后重跑)")
        return None
    states, alphas, servers, confs = load_debate_dataset(
        dataset_path, max_samples=ALIGN_N,
        target_K=NUM_USERS, target_M=NUM_EDGE_SERVERS)
    if states is None or len(states) < 100:
        print(f"    [skip] 样本量不足 (n={len(states) if states is not None else 0})")
        return None
    K_data = alphas.shape[1]
    M_data = (states.shape[1] - 4 * K_data) // 2
    if 4 * K_data + 2 * M_data != states.shape[1]:
        print(f"    [skip] state 维度不匹配")
        return None
    if K_data != NUM_USERS or M_data != NUM_EDGE_SERVERS:
        print(f"    [skip] K={K_data},M={M_data} 与基准 ({NUM_USERS},{NUM_EDGE_SERVERS}) 不匹配")
        return None
    save_path = OUT_CKPT_PREFIX.format(name=name)
    trainer = DistillAgentTrainer(
        K=K_data, M=M_data, state_dim=states.shape[1],
        epochs=EPOCHS, batch=BATCH, save_path=save_path)
    t0 = time.time()
    trainer.train(states, alphas, servers, confidences=confs, val_ratio=0.1)
    train_dt = time.time() - t0
    print(f"    train done in {train_dt:.1f}s")

    energy, lat, suc, sla = [], [], [], []
    for sd in range(N_SEEDS):
        for ep in range(N_EPISODES):
            env = MECEnvironment(num_users=K_data, num_servers=M_data,
                                  seed=SEED + sd + ep)
            env.reset()
            runner = HMAAgentRunner(env=env, mode='Distill',
                                     policy_path=save_path, agents=None)
            for _ in range(N_STEPS):
                state = env._get_state()
                out = runner.run_step(state=state, agents_reuse=True)
                a = compose_action(out['plan'], env.K, env.M)
                ns, _, d, info = env.step(a)
                energy.append(info['energy'])
                lat.append(info['latency'])
                suc.append(info['success_rate'])
                sla.append(info['priority_sla'])
                if d: break
    return {
        'energy':   float(np.mean(energy)),
        'latency':  float(np.mean(lat)),
        'success':  float(np.mean(suc)),
        'sla':      float(np.mean(sla)),
        'train_dt_s': float(train_dt),
        'K': int(K_data), 'M': int(M_data),
        'n_samples': int(len(states)),
    }


def main():
    print("=" * 60)
    print(f"  E9 多 LLM 后端对比 (epochs={EPOCHS}, batch={BATCH})")
    print("=" * 60)
    results = {'backend': [], 'energy': [], 'latency': [],
               'success': [], 'sla': [], 'train_dt': [],
               'K': [], 'M': [], 'n_samples': []}
    for name, path in BACKEND_CONFIG.items():
        print(f"\n  -- {name} --")
        m = train_eval_backend(name, path)
        if m is None:
            continue
        results['backend'].append(name)
        results['energy'].append(m['energy'])
        results['latency'].append(m['latency'])
        results['success'].append(m['success'])
        results['sla'].append(m['sla'])
        results['train_dt'].append(m['train_dt_s'])
        results['K'].append(m['K']); results['M'].append(m['M'])
        results['n_samples'].append(m['n_samples'])
        print(f"    E={m['energy']:.5f} T={m['latency']:.3f} "
              f"suc={m['success']:.2%} sla={m['sla']:.2%} "
              f"train_dt={m['train_dt_s']:.1f}s")

    flat = {k: np.array(v, dtype=np.float32) for k, v in results.items()
            if k != 'backend'}
    if results['backend']:
        # encode backend names as object array
        flat['backends'] = np.array(results['backend'], dtype=object)
    os.makedirs(os.path.dirname(OUTPUT_NPZ), exist_ok=True)
    np.savez_compressed(OUTPUT_NPZ, **flat)
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  npz 保存: {OUTPUT_NPZ}")
    print("=" * 60)


if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-
"""
================================================================
E8 蒸馏数据规模消融 (local/run_e8_distill_size.py)
================================================================
将蒸馏数据集规模 |D_debate| 视为自变量：{1k, 2k, 5k, 10k, 20k}
对每个规模从 debate_dataset.jsonl 均匀抽样指定数量样本，
重复训练 Distill 策略网络 (同一架构、同样 lr 与 epochs)，然后在
基准 (K=8, M=4) 场景下评估 Distill 模式能耗/时延/成功率/SLA。

若 GPU 不可用，本脚本可在 CPU 上小规模训练 (EPOCHS=20, BATCH=64);
若需训练 5 个网络 (5 种 |D|) 且 epochs=100, 建议在 GPU 服务器上
运行。脚本通过读取环境变量 E8_EPOCHS / E8_BATCH 控制规模。

输出:
  results/e8_distill_size.npz   -- 各 |D| 下的指标均值
  results/checkpoints/distill_d{N}.pth  -- 各规模下训练得到的策略网络
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
    NUM_USERS, NUM_EDGE_SERVERS, MAX_STEPS, RESULTS_DIR, SEED,
    CHECKPOINT_DIR, POLICY_NET_EPOCHS, POLICY_NET_BATCH,
    DISTILL_DATASET_SIZE, POLICY_NOISE_STD,
)
from environment import MECEnvironment
from distill_agent import (
    DistillAgentTrainer, PolicyAgentRunner, PolicyAgentNet,
    load_debate_dataset,
)
from agent_runner import HMAAgentRunner
from local.experiment_common import compose_action


# ============================================================
# 显眼配置区 (CPU/GPU 自适应)
# ============================================================
D_SIZES = [1000, 2000, 5000, 10000, 20000]   # 数据规模扫描点
EPOCHS  = int(os.environ.get("E8_EPOCHS", 30))     # 训练 epoch 数 (GPU 上建议调 100)
BATCH   = int(os.environ.get("E8_BATCH",   128))
VAL_RATIO = 0.1

N_SEEDS    = 2
N_EPISODES = 3
N_STEPS    = 100

DEBATE_DATASET = os.path.join(RESULTS_DIR, "debate_dataset.jsonl")
OUTPUT_NPZ  = os.path.join(RESULTS_DIR, "e8_distill_size.npz")
OUTPUT_JSON = os.path.join(RESULTS_DIR, "e8_distill_size.json")
OUT_CKPT_PREFIX = os.path.join(CHECKPOINT_DIR, "distill_d{N}.pth")


def train_and_eval_size(N, n_seeds=N_SEEDS, n_episodes=N_EPISODES,
                         n_steps=N_STEPS):
    """对规模 N 训练一个新 distill 模型, 评估并返回指标。

    数据集 K, M 自适应：从数据形状推导 K, M, 并在评估时使用同一组规模。
    """
    print(f"  - loading up to {N} samples from {DEBATE_DATASET}")
    if not os.path.exists(DEBATE_DATASET):
        print(f"  [warn] debate_dataset.jsonl 不存在, 跳过")
        return None, None
    states, alphas, servers, clouds, confs = load_debate_dataset(
        DEBATE_DATASET, max_samples=N,
        target_K=NUM_USERS, target_M=NUM_EDGE_SERVERS)
    if states is None or len(states) < N // 5:
        print(f"  [warn] 样本数不足 ({states.shape if states is not None else 0})")
        return None, None
    n_avail = len(states)
    K_data = alphas.shape[1]
    M_data = (states.shape[1] - 4 * K_data) // 2
    print(f"  - 实际数据 K={K_data}, M={M_data}, state_dim={states.shape[1]}")
    if K_data != NUM_USERS or M_data != NUM_EDGE_SERVERS:
        print(f"  [warn] K={K_data} 或 M={M_data} 与目标 ({NUM_USERS},{NUM_EDGE_SERVERS}) 不匹配, 跳过")
        return None, None
    if (4 * K_data + 2 * M_data) != states.shape[1]:
        print(f"  [warn] state 形状与 (K,M) 不匹配, 跳过该尺度")
        return None, None
    # 均匀采到 N (若可)
    if n_avail > N:
        idx = np.linspace(0, n_avail - 1, N).astype(int)
        states = states[idx]; alphas = alphas[idx]
        servers = servers[idx]
        if confs is not None:
            confs = confs[idx]
        if clouds is not None:
            clouds = clouds[idx]
    print(f"  - actual samples used: {len(states)}")
    save_path = OUT_CKPT_PREFIX.format(N=N)
    trainer = DistillAgentTrainer(
        K=K_data, M=M_data,
        state_dim=states.shape[1],
        lr=1e-3, epochs=EPOCHS, batch=BATCH,
        save_path=save_path)
    t0 = time.time()
    trainer.train(states, alphas, servers, clouds=clouds, confidences=confs,
                  val_ratio=VAL_RATIO)
    train_dt = time.time() - t0
    print(f"  - train done in {train_dt:.1f}s")
    # 评估（沿用与该数据规模匹配的 env）
    energy, lat, suc, sla = [], [], [], []
    for sd in range(n_seeds):
        for ep in range(n_episodes):
            env = MECEnvironment(num_users=K_data,
                                  num_servers=M_data,
                                  seed=SEED + sd + ep)
            env.reset()
            runner = HMAAgentRunner(env=env, mode='Distill',
                                     policy_path=save_path, agents=None)
            for _ in range(n_steps):
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
        'actual_samples': int(n_avail),
        'K': int(K_data),
        'M': int(M_data),
    }, save_path


def main():
    print("=" * 60)
    print(f"  E8 蒸馏数据规模消融 (sizes={D_SIZES}, epochs={EPOCHS}, batch={BATCH})")
    print("=" * 60)
    results = {'size': [], 'energy': [], 'latency': [],
               'success': [], 'sla': [], 'train_dt': [],
               'actual': [], 'ckpt': [],
               'K': [], 'M': []}
    for N in D_SIZES:
        print(f"\n  -- |D| = {N} --")
        m, ckpt = train_and_eval_size(N)
        if m is None:
            continue
        results['size'].append(N)
        results['energy'].append(m['energy'])
        results['latency'].append(m['latency'])
        results['success'].append(m['success'])
        results['sla'].append(m['sla'])
        results['train_dt'].append(m['train_dt_s'])
        results['actual'].append(m['actual_samples'])
        results['K'].append(m.get('K', NUM_USERS))
        results['M'].append(m.get('M', NUM_EDGE_SERVERS))
        results['ckpt'].append(ckpt)
        print(f"    E={m['energy']:.5f}  T={m['latency']:.3f}  "
              f"suc={m['success']:.2%}  sla={m['sla']:.2%}  "
              f"train_dt={m['train_dt_s']:.1f}s (K={m.get('K', NUM_USERS)}, M={m.get('M', NUM_EDGE_SERVERS)})")

    flat = {k: np.array(v, dtype=np.float32) for k, v in results.items()
            if k not in ('ckpt',)}
    flat['ckpts'] = np.array(results['ckpt'], dtype=object)
    os.makedirs(os.path.dirname(OUTPUT_NPZ), exist_ok=True)
    np.savez_compressed(OUTPUT_NPZ, **flat)
    # JSON dump for human inspection
    out_json = {k: v for k, v in results.items() if k != 'ckpt'}
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(out_json, f, ensure_ascii=False, indent=2)
    print(f"\n  npz 保存: {OUTPUT_NPZ}")
    print("=" * 60)


if __name__ == "__main__":
    main()
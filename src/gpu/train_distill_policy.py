# -*- coding: utf-8 -*-
"""
================================================================
A3 蒸馏策略网络训练 (gpu/train_distill_policy.py)
================================================================
本脚本在 GPU 服务器（或 CPU 兼容）上读取 src/results/debate_dataset.jsonl
中累积的 (state, alpha, server, confidence) 辩论数据，按
Laplace NLL + CrossEntropy + MSE 训练策略网络，权重保存为
distilled_policy.pth，供 agent_runner.py 在线推理。

特性:
   1. 自动断点续训: 读取已有 distilled_policy.pth, 在其上继续优化
   2. GPU 优先: 自动选择 CUDA, 无 GPU 则 CPU
   3. 显眼配置区: 集中训练超参数

运行:
    python gpu/train_distill_policy.py [--epochs 100] [--batch 128]
================================================================
"""

import os
import sys
import json
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)

from config import (
    NUM_USERS, NUM_EDGE_SERVERS, STATE_DIM,
    POLICY_NET_EPOCHS, POLICY_NET_BATCH,
    RESULTS_DIR, CHECKPOINT_DIR, SEED,
)
from distill_agent import (
    PolicyAgentNet, DistillAgentTrainer, count_debate_records,
)


# ============================================================
# 显眼配置区
# ============================================================
DATASET_PATH = os.path.join(RESULTS_DIR, "debate_dataset.jsonl")
SAVE_PATH    = os.path.join(CHECKPOINT_DIR,   "distilled_policy.pth")


def load_dataset(path, max_samples=None):
    """从 jsonl 读取数据集, 过滤为基准规模 (K=8, M=4), 返回 ndarray."""
    K = NUM_USERS; M = NUM_EDGE_SERVERS
    expected_sd = K * 4 + M * 2
    states, alphas, servers, clouds, confs = [], [], [], [], []
    n_skipped = 0

    if not os.path.exists(path):
        raise FileNotFoundError(f"数据集不存在: {path}; 请先运行 gen_distill_dataset.py")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            s = np.asarray(r["state"], dtype=np.float32)
            a = np.asarray(r["alpha"], dtype=np.float32)
            m = np.asarray(r["server"], dtype=int)
            # 只保留基准规模 (K=8, M=4) 的样本
            if len(s) != expected_sd or len(a) != K or len(m) != K:
                n_skipped += 1
                continue
            states.append(s)
            alphas.append(a)
            servers.append(m)
            if r.get("cloud") is not None:
                clouds.append(np.asarray(r["cloud"], dtype=int))
            if r.get("confidence") is not None:
                confs.append(np.asarray(r["confidence"], dtype=np.float32))

    if not states:
        raise ValueError(f"数据集在基准规模 K={K}, M={M} 下无可用样本 (跳过了 {n_skipped} 条其他规模)")

    print(f"  过滤: 基准规模样本 {len(states)} 条, 跳过 {n_skipped} 条其他规模")
    states  = np.stack(states, axis=0)
    alphas  = np.stack(alphas, axis=0)
    servers = np.stack(servers, axis=0)
    clouds  = np.stack(clouds, axis=0) if clouds else None
    confs   = np.stack(confs, axis=0) if confs else None

    if max_samples is not None and len(states) > max_samples:
        idx = np.random.permutation(len(states))[:max_samples]
        states, alphas, servers = states[idx], alphas[idx], servers[idx]
        if clouds is not None: clouds = clouds[idx]
        if confs is not None: confs = confs[idx]
    return states, alphas, servers, clouds, confs


# ============================================================
# 主入口
# ============================================================
def main(epochs=POLICY_NET_EPOCHS, batch=POLICY_NET_BATCH,
         max_samples=None, dataset_path=None, save_path=None):
    import logging
    logging.basicConfig(level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger("train_distill")

    ds_path = dataset_path or DATASET_PATH
    sv_path = save_path or SAVE_PATH

    print("=" * 60)
    print("  蒸馏策略网训练 (DistillAgentTrainer)")
    print(f"  数据集: {ds_path}")
    print(f"  保存至: {sv_path}")
    print("=" * 60)

    n_records = count_debate_records(ds_path)
    print(f"  数据集已存在 {n_records} 条")

    print(f"  加载 {ds_path} ...")
    states, alphas, servers, clouds, confs = load_dataset(ds_path,
                                                          max_samples=max_samples)
    print(f"  基准规模样本: N={len(states)}, "
          f"state_dim={states.shape[1]}, K={alphas.shape[1]}")
    if clouds is not None:
        print(f"  云端标志监督可用: shape {clouds.shape}")
    if confs is not None:
        print(f"  置信度监督可用: shape {confs.shape}")

    # 确保 K, M 与数据匹配
    K, M = NUM_USERS, NUM_EDGE_SERVERS

    trainer = DistillAgentTrainer(
        K=K, M=M,
        state_dim=STATE_DIM,
        epochs=epochs, batch=batch,
        save_path=sv_path,
    )
    model, history = trainer.train(states, alphas, servers, clouds=clouds, confidences=confs)

    # 保存训练历史到 npy 文件（论文训练曲线图用）
    history_path = os.path.splitext(sv_path)[0] + "_history.npy"
    np.save(history_path, np.array(history, dtype=np.float32))
    print(f"  训练历史保存至: {history_path}")

    print("=" * 60)
    print(f"  完成. 最佳 epoch={trainer.best_epoch}, "
          f"best_val={trainer.best_val:.4f}")
    print(f"  权重保存至: {trainer.save_path}")
    print(f"  训练历史保存至: {history_path}")
    print("=" * 60)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=POLICY_NET_EPOCHS)
    ap.add_argument("--batch",  type=int, default=POLICY_NET_BATCH)
    ap.add_argument("--max_samples", type=int, default=None,
                    help="最多使用多少样本训练 (调试可用)")
    ap.add_argument("--dataset", type=str, default=None,
                    help='数据集路径, 默认 results/debate_dataset.jsonl')
    ap.add_argument("--save", type=str, default=None,
                    help='保存路径, 默认 results/checkpoints/distilled_policy.pth')
    args = ap.parse_args()
    main(epochs=args.epochs, batch=args.batch,
         max_samples=args.max_samples,
         dataset_path=args.dataset, save_path=args.save)
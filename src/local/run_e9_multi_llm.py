# -*- coding: utf-8 -*-
"""
===============================================================
E9 多 LLM 后端对比实验 (local/run_e9_multi_llm.py)
===============================================================
本脚本编排 E9 完整流程：对 4 个 LLM（Qwen3.6-27B / Qwen3.5-9B /
Llama-3.1-8B / Mistral-7B）分别执行：

  1) 辩论数据生成（调用 gpu/gen_distill_dataset.py 或直接调用其 generate 接口）
  2) 蒸馏策略网络训练（调用 gpu/train_distill_policy.py）
  3) 策略评估（调用 local/experiment_common.py 的 evaluate 接口）
  4) JSON 解析成功率收集
  5) 随机动作基线对照

运行方式（分阶段）:
  # Phase 1: 数据生成 (GPU + vLLM, 每模型 ~1-2 小时)
  python local/run_e9_multi_llm.py --phase gen

  # Phase 2: 蒸馏训练 (GPU, 每模型 ~30 分钟)
  python local/run_e9_multi_llm.py --phase train

  # Phase 3: 评估 (CPU, 每模型 ~10 分钟)
  python local/run_e9_multi_llm.py --phase eval

  # Phase 4: 全流程 (谨慎使用, 可能需要 10+ 小时)
  python local/run_e9_multi_llm.py --phase all

结果：
  results/e9_multi_llm_results.json  — 汇总表
  results/debate_dataset_{model}.jsonl  — 各模型辩论数据
  results/checkpoints/policy_{model}.pth  — 各模型蒸馏策略
===============================================================
"""

import os
import sys
import json
import time
import argparse
import subprocess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
GPU_DIR = os.path.join(SRC, "gpu")
sys.path.insert(0, SRC)

from config import (
    RESULTS_DIR, CHECKPOINT_DIR, NUM_USERS, NUM_EDGE_SERVERS,
    LLM_MODEL_REGISTRY,
)

# ============================================================
# >>>>>>>>>>>>>>>>>>  显眼配置区  <<<<<<<<<<<<<<<<<<
# ============================================================

# 每模型预设的对比样本数（主力模型 5000，其余 1000）
N_SAMPLES = {
    "Qwen3.6-27B":  5000,
    "Qwen3.5-9B":   1000,
    "Llama-3.1-8B": 1000,
    "Mistral-7B":   1000,
}

# 蒸馏训练参数 (每模型)
TRAIN_EPOCHS = 100
TRAIN_BATCH = 128

# 评估参数
EVAL_SEEDS = 3
EVAL_EPISODES = 5
EVAL_STEPS = 100

# 随机基线样本数 (用于训练随机动作蒸馏策略)
RANDOM_N_SAMPLES = 1000

# ============================================================
# 辅助路径
# ============================================================

def debate_data_path(model_name):
    """返回模型对应辩论数据路径"""
    key = model_name.lower().replace(".", "_").replace("-", "_")
    return os.path.join(RESULTS_DIR, f"debate_dataset_{key}.jsonl")

def checkpoint_path(model_name):
    """返回模型对应蒸馏策略权重路径"""
    key = model_name.lower().replace(".", "_").replace("-", "_")
    return os.path.join(CHECKPOINT_DIR, f"policy_{key}.pth")

def history_path(model_name):
    """返回训练历史路径"""
    ckpt = checkpoint_path(model_name)
    return os.path.splitext(ckpt)[0] + "_history.npy"


# ============================================================
# Phase 1: 辩论数据生成
# ============================================================

def phase_gen(skip_existing=True):
    """对注册表中每个模型, 调用 gen_distill_dataset 生成辩论数据。"""
    gen_script = os.path.join(GPU_DIR, "gen_distill_dataset.py")
    if not os.path.exists(gen_script):
        print(f"[E9] 错误: 未找到 {gen_script}")
        return

    for model_name, info in LLM_MODEL_REGISTRY.items():
        port = info["port"]
        out_path = debate_data_path(model_name)
        n_target = N_SAMPLES.get(model_name, 1000)

        # 检查已有数据量
        if skip_existing and os.path.exists(out_path):
            n_existing = sum(1 for _ in open(out_path, encoding="utf-8"))
            if n_existing >= n_target:
                print(f"[E9] 跳过 {model_name}: 已有 {n_existing} >= {n_target} 条")
                continue
            print(f"[E9] {model_name}: 已有 {n_existing} / {n_target} 条, 继续补充")

        print(f"\n{'='*60}")
        print(f"  Phase 1: 生成 {model_name} (端口 {port}) 辩论数据")
        print(f"  目标: {n_target} 条 -> {out_path}")
        print(f"{'='*60}")

        cmd = [
            sys.executable, gen_script,
            "--model", model_name,
            "--port", str(port),
            "--target", str(n_target),
            "--output", out_path,
            "--workers", "20",
        ]
        t0 = time.time()
        result = subprocess.run(cmd, cwd=SRC)
        dt = time.time() - t0
        if result.returncode == 0:
            print(f"  [E9] {model_name} 数据生成完成, 耗时 {dt/3600:.2f}h")
        else:
            print(f"  [E9] ⚠ {model_name} 数据生成失败 (returncode={result.returncode})")


def _gen_random_data(path, n_samples):
    """生成随机动作的 '伪辩论' 数据, 用于随机基线。"""
    import random
    rng = np.random.RandomState(42)
    records = []
    for i in range(n_samples):
        K = NUM_USERS
        M = NUM_EDGE_SERVERS
        state = np.concatenate([
            rng.uniform(0, 1, K * 4).astype(np.float32),
            rng.uniform(0, 1, M * 2).astype(np.float32),
        ])
        alpha = rng.uniform(0.01, 0.99, K).astype(np.float32)
        server = rng.randint(0, M, K).astype(int)
        confidence = rng.uniform(0.0, 1.0, K).astype(np.float32)
        records.append({
            "state": state.tolist(),
            "alpha": alpha.tolist(),
            "server": server.tolist(),
            "confidence": confidence.tolist(),
            "fingerprint": f"random-{i}",
        })
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  [E9] 随机基线数据: {len(records)} 条 -> {path}")


# ============================================================
# Phase 2: 蒸馏训练
# ============================================================

def phase_train():
    """对每个模型的辩论数据, 训练蒸馏策略网络。"""
    train_script = os.path.join(GPU_DIR, "train_distill_policy.py")
    if not os.path.exists(train_script):
        print(f"[E9] 错误: 未找到 {train_script}")
        return

    # 包含随机基线
    all_models = list(LLM_MODEL_REGISTRY.keys()) + ["Random"]

    for model_name in all_models:
        if model_name in LLM_MODEL_REGISTRY:
            ds_path = debate_data_path(model_name)
            save_path = checkpoint_path(model_name)
            label = model_name
        else:
            # 随机基线: 检查是否有数据, 无则生成
            ds_path = os.path.join(RESULTS_DIR, "debate_dataset_random.jsonl")
            save_path = os.path.join(CHECKPOINT_DIR, "policy_random.pth")
            label = "Random"
            if not os.path.exists(ds_path):
                print(f"  [E9] 生成随机基线数据...")
                _gen_random_data(ds_path, RANDOM_N_SAMPLES)

        if not os.path.exists(ds_path):
            print(f"  [E9] 跳过 {label}: 数据不存在 ({ds_path})")
            continue

        n_records = sum(1 for _ in open(ds_path, encoding="utf-8"))
        print(f"\n{'='*60}")
        print(f"  Phase 2: 训练 {label} 蒸馏策略")
        print(f"  数据: {ds_path} ({n_records} 条)")
        print(f"  保存: {save_path}")
        print(f"{'='*60}")

        cmd = [
            sys.executable, train_script,
            "--dataset", ds_path,
            "--save", save_path,
            "--epochs", str(TRAIN_EPOCHS),
            "--batch", str(TRAIN_BATCH),
        ]
        t0 = time.time()
        result = subprocess.run(cmd, cwd=SRC)
        dt = time.time() - t0
        if result.returncode == 0:
            print(f"  [E9] {label} 训练完成, 耗时 {dt/60:.1f} 分钟")
        else:
            print(f"  [E9] ⚠ {label} 训练失败 (returncode={result.returncode})")


# ============================================================
# Phase 3: 策略评估
# ============================================================

def phase_eval():
    """评估所有蒸馏策略 + 基线的性能。"""
    from environment import MECEnvironment
    from agent_runner import HMAAgentRunner
    from local.baselines import (
        GreedyBaseline, AllLocalBaseline, AllEdgeBaseline,
        RandomBaseline, SACAgent, DDPGAgent,
        evaluate as eval_baseline,
    )
    from local.experiment_common import run_multi_episodes

    K, M = NUM_USERS, NUM_EDGE_SERVERS
    results = {}

    # ----- 评估每个模型的蒸馏策略 -----
    all_models = list(LLM_MODEL_REGISTRY.keys()) + ["Random"]
    for model_name in all_models:
        if model_name in LLM_MODEL_REGISTRY:
            ckpt = checkpoint_path(model_name)
            label = f"HMA-Distill ({model_name})"
        else:
            ckpt = os.path.join(CHECKPOINT_DIR, "policy_random.pth")
            label = "HMA-Distill (Random)"

        if not os.path.exists(ckpt):
            print(f"  [E9] 跳过 {label}: 权重不存在 ({ckpt})")
            continue

        print(f"  评估 {label} ...")
        try:
            method_specs = [{"name": "HMA-Distill", "policy_path": ckpt}]
            out = run_multi_episodes(
                method_specs, K=K, M=M,
                n_seeds=EVAL_SEEDS, n_episodes=EVAL_EPISODES,
                n_steps=EVAL_STEPS)
            results[label] = {
                "mean": out.get("HMA-Distill", {}).get("mean", {}),
                "std":  out.get("HMA-Distill", {}).get("std", {}),
            }
            print(f"    energy={results[label]['mean'].get('energy', '?'):.4f}  "
                  f"latency={results[label]['mean'].get('latency', '?'):.3f}  "
                  f"success={results[label]['mean'].get('success_rate', '?'):.2%}")
        except Exception as e:
            print(f"    ⚠ 评估失败: {e}")

    # ----- 启发式基线 -----
    print("  评估启发式基线 ...")
    for method_class, method_name in [
        (GreedyBaseline, "Greedy"),
        (AllLocalBaseline, "AllLocal"),
        (AllEdgeBaseline, "AllEdge"),
        (RandomBaseline, "Random"),
    ]:
        try:
            env = MECEnvironment(num_users=K, num_servers=M)
            method = method_class(env)
            r = eval_baseline(method, env, n_steps=EVAL_STEPS)
            results[f"Baseline-{method_name}"] = {
                "mean": r, "std": {}
            }
            print(f"    {method_name}: energy={r['energy']:.4f}")
        except Exception as e:
            print(f"    ⚠ {method_name}: {e}")

    # ----- SAC / DDPG (简短训练) -----
    if SACAgent is not None:
        print("  评估 SAC (简短训练 30 ep) ...")
        try:
            env = MECEnvironment(num_users=K, num_servers=M)
            sac = SACAgent(env)
            sac.train(env, episodes=30, verbose=False)
            r = eval_baseline(sac, env, n_steps=EVAL_STEPS)
            results["Baseline-SAC"] = {"mean": r, "std": {}}
            print(f"    SAC: energy={r['energy']:.4f}")
        except Exception as e:
            print(f"    ⚠ SAC: {e}")

    # ----- 保存结果 -----
    out_path = os.path.join(RESULTS_DIR, "e9_multi_llm_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[E9] 结果保存至: {out_path}")

    # ----- 打印汇总表 -----
    print("\n" + "=" * 70)
    print("  E9 多 LLM 后端对比 — 汇总")
    print("=" * 70)
    print(f"  {'模型':<25s} {'能耗(kJ)':<12s} {'时延(s)':<12s} {'成功率':<10s} {'SLA':<10s}")
    print("  " + "-" * 70)
    for label, data in sorted(results.items()):
        m = data.get("mean", {})
        print(f"  {label:<25s} {m.get('energy', 0):<12.4f} "
              f"{m.get('latency', 0):<12.3f} "
              f"{m.get('success_rate', 0):<10.2%} "
              f"{m.get('priority_sla', 0):<10.2%}")
    print("=" * 70)


# ============================================================
# 全流程
# ============================================================

def phase_all():
    """依次执行 gen → train → eval。"""
    print("=" * 60)
    print("  E9 全流程: gen → train → eval")
    print("  注意: 这可能耗时 10+ 小时, 建议分阶段运行")
    print("=" * 60)
    phase_gen()
    _gen_random_data(
        os.path.join(RESULTS_DIR, "debate_dataset_random.jsonl"),
        RANDOM_N_SAMPLES)
    phase_train()
    phase_eval()


# ============================================================
# CLI 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="E9 多 LLM 后端对比实验")
    parser.add_argument("--phase", type=str,
                        choices=["gen", "train", "eval", "all"],
                        default="all",
                        help="执行阶段: gen=数据生成, train=蒸馏训练, "
                             "eval=评估, all=全流程")
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="数据生成时跳过已完成的文件")
    args = parser.parse_args()

    phases = {
        "gen":   lambda: phase_gen(skip_existing=args.skip_existing),
        "train": phase_train,
        "eval":  phase_eval,
        "all":   phase_all,
    }
    phases[args.phase]()


if __name__ == "__main__":
    main()

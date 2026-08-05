# -*- coding: utf-8 -*-
"""
================================================================
A3 蒸馏数据集生成 - 并行版 (gpu/gen_distill_dataset.py)
================================================================
本脚本在 GPU 服务器（含本地 CPU + 远程 LLM API）上运行，
通过反复调用 FullLLM 模式的 \textsc{CW-Debate}，收集
$(s_t, a^{\star}_t, \hat{\mathbf{c}}^{\star})$ 对，构成
$\mathcal{D}_{debate}$（jsonl 格式保存），供后续在 GPU 上训练
$\pi^\text{agent}_\theta$。

特性：
  1. 断点续存 —— 已生成的样本追加到 JSONL 文件；中断后再运行会自动
     跳过已完成的样本，避免重复调用 API
  2. 多种子采样 —— 用不同 $(K, M, \text{seed})$ 组合覆盖规模变化
  3. 错误重试与速率控制 —— 由 llm_client.LLMClient 自动处理
  4. 显眼配置区 —— 所有可调参数集中在文件顶部
  5. 多线程并行 —— ThreadPoolExecutor,并行调用 LLM API
     注意: 每个 worker 独立持有 LLMClient + Agent 实例;不共享 torch 张量

运行：
    python gpu/gen_distill_dataset.py
可后台长跑:
    nohup python gpu/gen_distill_dataset.py > logs/gen_distill.log 2>&1 &

典型用时：
    deepseek-v4-flash API 单次 CW-Debate (~6 轮 × ~40 calls) 约 30--60 秒，
    串行 50000 条总耗时 ~12--24 小时;
    并行 N_WORKERS=20 时单任务平均时延被掩盖, 总耗时≈1/20,约 1--2 小时。
================================================================
"""

import os
import sys
import time
import json
import logging
import threading
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# 把 src 根目录加入 sys.path
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)

from config import (
    NUM_USERS, NUM_EDGE_SERVERS, MAX_STEPS,
    DISTILL_DATASET_SIZE, RESULTS_DIR, SEED,
)
from environment import MECEnvironment
from agent_define import make_agents
from cw_debate import cw_debate
from distill_agent import save_debate_record, count_debate_records
from llm_client import get_llm_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(RESULTS_DIR, "gen_distill.log"),
                            encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ============================================================
# >>>>>>>>>>>>>>>>>>  显眼配置区  <<<<<<<<<<<<<<<<<<
# ============================================================
# 这里集中所有可能需要修改的参数,特别是训练时长。
# ============================================================

# 目标样本总数（断点续存会跳过已生成的部分）
TARGET_SAMPLES = DISTILL_DATASET_SIZE  # 由 config 决定, 默认 50000

# 场景 ``(K, M)'' 列表：按一定比例随机抽取以覆盖不同系统规模
SCALES = [
    (4, 2),     # 小: 4 用户 2 服务器
    (8, 4),     # 基准
    (12, 4),    # 中等
    (16, 4),    # 较大
]

# 每个场景的采样权重（SCALES 的概率，自动归一化）
SCALE_PROB = [0.15, 0.5, 0.2, 0.15]

# 每个种子下生成的 episode 数上限 (避免单一种子下样本高度相似)
SEEDS = list(range(0, 50))

# 数据输出文件（JSONL 格式）
OUTPUT_PATH = os.path.join(RESULTS_DIR, "debate_dataset.jsonl")

# 每完成 N 条样本打印一次进度
PRINT_EVERY = 100

# 每 N 条样本写入状态文件（便于前端查询进度）
PROGRESS_DUMP_EVERY = 500

# ---- 新增: 并行配置 ----
# local_transformers 后端共享单模型 + 锁，实际为串行推理，
# N_WORKERS 主要作用是线程池管理，不影响实际并发。
# 建议 N_WORKERS=8~32，模型推理由全局锁串行化。
N_WORKERS = 32
WRITE_LOCK = threading.Lock()   # 写入 jsonl 时的互斥锁

# 模型名 (可选覆盖 config.LLM_MODEL)
# 设为 None 则使用 config 中的值。
MODEL_NAME = None

# vLLM 服务端口 (可选覆盖 config.LLM_LOCAL_PORT)
# 当 LLM_BACKEND 为 local_vllm 时，各模型在不同端口提供服务
VLLM_PORT = None


# ============================================================
# 线程局部存储: 每个 worker 独立持有 LLM 客户端
# ============================================================
_thread_local = threading.local()

def get_thread_llm(model_override=None, port_override=None):
    """每个线程独立创建 LLMClient 实例,避免 OpenAI client 状态串扰。

    参数：
        model_override: 可选, 覆盖 config.LLM_MODEL
        port_override:  可选, 覆盖 config.LLM_LOCAL_PORT (用于 E9 多模型)
    """
    key = (model_override, port_override)
    attr = f"llm_{model_override or ''}_{port_override or ''}"
    if not hasattr(_thread_local, attr) or getattr(_thread_local, attr) is None:
        try:
            if port_override is not None:
                # E9 多模型模式: 连接指定端口的 vLLM 服务
                # 直接构造 _OpenAICompatibleClient 而非调用 get_llm_client
                from llm_client import _OpenAICompatibleClient
                effective_model = model_override or "default"
                client = _OpenAICompatibleClient(
                    model=effective_model,
                    api_key="EMPTY",
                    base_url=f"http://localhost:{port_override}/v1",
                    backend="local_vllm")
            else:
                client = get_llm_client(model=model_override)
            setattr(_thread_local, attr, client)
        except Exception as e:
            logger.error(f"创建 LLM 客户端失败: {e}")
            setattr(_thread_local, attr, None)
    return getattr(_thread_local, attr)


# ============================================================
# 单条样本生成：随机抽样 → 单步 CW-Debate → 记录
# ============================================================
def select_scale(rng):
    """按 SCALE_PROB 抽样 (K, M)."""
    return SCALES[int(rng.choice(len(SCALES), p=SCALE_PROB))]


def generate_one_sample(task_idx, fp_log_path, model_override=None,
                        port_override=None):
    """随机构造状态 → 在线 CW-Debate → 追加到 jsonl.

    参数:
        task_idx:      int, 任务编号 (用于种子)
        fp_log_path:   str, 输出文件路径 (线程安全写入用 WRITE_LOCK)
        model_override: str or None, 覆盖 config.LLM_MODEL
        port_override:  int or None, 覆盖 config.LLM_LOCAL_PORT
    返回:
        (success: bool, fingerprint: str)
    """
    # 每个 task 自己的 numpy 随机状态 (task_idx 偏移避免重复)
    rng = np.random.RandomState(SEED + 1 + task_idx)
    K, M = select_scale(rng)
    seed = int(rng.choice(SEEDS))
    fp_prefix = f"K{K}-M{M}-seed{seed}"
    fingerprint = f"{fp_prefix}-t{task_idx}"

    env = MECEnvironment(num_users=K, num_servers=M, seed=seed)
    np_rng = np.random.RandomState(seed + task_idx)
    n_warmup = int(np_rng.randint(0, 30 + 1))
    s = env.reset()
    for _ in range(n_warmup):
        a = np_rng.uniform(0, 1, env.action_dim).astype(np.float32)
        s, _, done, _ = env.step(a)
        if done: s = env.reset()

    llm = get_thread_llm(model_override=model_override,
                         port_override=port_override)
    if llm is None:
        return False, fingerprint

    agents = make_agents(env, with_va=True)
    try:
        out = cw_debate(env, agents, mode="FullLLM", llm=llm, verbose=False)
        alpha = out['plan']['alpha']; server = out['plan']['server']
        cloud = out['plan'].get('cloud', np.zeros(K, dtype=bool))
        confidences = (np.asarray(out['confidence_history'][-1])
                       if out['confidence_history'] else
                       np.zeros(K, dtype=np.float32))
        record = {
            'state':      np.asarray(env._get_state(),
                                     dtype=np.float32).tolist(),
            'alpha':      np.asarray(alpha, dtype=np.float32).tolist(),
            'server':     np.asarray(server, dtype=int).tolist(),
            'cloud':      np.asarray(cloud, dtype=int).tolist(),
            'confidence': (np.asarray(confidences,
                                     dtype=np.float32).tolist()),
            'fingerprint': fingerprint,
        }
        # 线程安全写入
        with WRITE_LOCK:
            with open(fp_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return True, fingerprint
    except Exception as e:
        logger.warning(f"样本生成失败 ({fingerprint}): {e}")
        return False, fingerprint


# ============================================================
# 主流程 (并行版)
# ============================================================
def main():
    n_done = count_debate_records(OUTPUT_PATH)
    effective_model = MODEL_NAME or 'config.LLM_MODEL'
    logger.info(f"启动并行生成 (N_WORKERS={N_WORKERS}, model={effective_model}): "
                f"已完成 {n_done} / 目标 {TARGET_SAMPLES}")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    t0 = time.time()
    n_failed = 0
    n_success = 0
    next_task_idx = n_done  # 任务编号从已完成数起,便于断点恢复

    while n_done < TARGET_SAMPLES:
        # 一次性提交一个 batch 的任务 (不超过剩余数与 N_WORKERS×2)
        batch_size = min(N_WORKERS * 2, TARGET_SAMPLES - n_done)
        batch_tasks = [(next_task_idx + i, OUTPUT_PATH, MODEL_NAME, VLLM_PORT)
                       for i in range(batch_size)]
        next_task_idx += batch_size

        with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
            futures = {pool.submit(generate_one_sample, *t): t
                       for t in batch_tasks}
            for fut in as_completed(futures):
                try:
                    success, fp = fut.result()
                except Exception as e:
                    success, fp = False, str(e)
                if success:
                    n_success += 1
                    n_done += 1
                else:
                    n_failed += 1
                # 进度报告
                if n_done % PRINT_EVERY == 0:
                    dt = time.time() - t0
                    rate = n_done / max(dt, 1)
                    eta = (TARGET_SAMPLES - n_done) / max(rate, 1e-3)
                    logger.info(
                        f"  progress {n_done}/{TARGET_SAMPLES}  "
                        f"rate={rate:.3f} samples/s  "
                        f"ETA={eta/3600:.2f}h  "
                        f"failed={n_failed}")
                if n_done % PROGRESS_DUMP_EVERY == 0:
                    _dump_progress(n_done, n_failed, time.time()-t0)
        # 检测停止文件
        if os.path.exists(os.path.join(RESULTS_DIR, "STOP_GEN")):
            logger.info("检测到 STOP_GEN, 退出.")
            break

    dt = time.time() - t0
    logger.info(f"完成: success {n_success}, failed {n_failed}, "
                 f"total {n_success+n_failed}, duration {dt/3600:.2f}h")
    _dump_progress(n_done, n_failed, dt)


def _dump_progress(n_done, n_failed, elapsed):
    """将进度写入文件, 便于外部监控."""
    path = os.path.join(RESULTS_DIR, "gen_distill_progress.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "done": n_done,
            "failed": n_failed,
            "elapsed_sec": elapsed,
            "n_workers": N_WORKERS,
        }, f, ensure_ascii=False)


def smoke_test():
    """生成 3 个样本, 验证流水线可用."""
    global TARGET_SAMPLES, PRINT_EVERY, PROGRESS_DUMP_EVERY, N_WORKERS
    orig = (TARGET_SAMPLES, PRINT_EVERY, PROGRESS_DUMP_EVERY, N_WORKERS)
    base_done = count_debate_records(OUTPUT_PATH)
    TARGET_SAMPLES = base_done + 3
    PRINT_EVERY = 1
    PROGRESS_DUMP_EVERY = 1
    N_WORKERS = 4  # smoke 用小 worker 数
    print("=" * 60)
    print("  小规模试跑 3 个样本, 验证并行流水线")
    print("=" * 60)
    main()
    TARGET_SAMPLES, PRINT_EVERY, PROGRESS_DUMP_EVERY, N_WORKERS = orig


def run():
    """CLI 入口: 解析参数后启动生成."""
    import argparse
    m = globals()
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="小规模试跑 3 个样本")
    parser.add_argument("--workers", type=int, default=None,
                        help="覆盖 N_WORKERS 设置")
    parser.add_argument("--model", type=str, default=None,
                        help='覆盖 LLM 模型名, 如 "deepseek-v4-flash", "deepseek-v4-pro"')
    parser.add_argument("--target", type=int, default=None,
                        help="覆盖目标样本总数 TARGET_SAMPLES")
    parser.add_argument("--port", type=int, default=None,
                        help="vLLM 服务端口 (覆盖 config.LLM_LOCAL_PORT, 用于 E9)")
    parser.add_argument("--output", type=str, default=None,
                        help="输出 JSONL 文件路径 (覆盖 OUTPUT_PATH)")
    args = parser.parse_args()
    if args.workers is not None:
        m["N_WORKERS"] = args.workers
    if args.model is not None:
        m["MODEL_NAME"] = args.model
        print(f"  模型名覆盖: {args.model}")
    if args.port is not None:
        m["VLLM_PORT"] = args.port
        print(f"  vLLM 端口覆盖: {args.port}")
    if args.output is not None:
        m["OUTPUT_PATH"] = args.output
        print(f"  输出路径覆盖: {args.output}")
    else:
        print(f"  输出路径: {m['OUTPUT_PATH']}")
    if args.target is not None:
        m["TARGET_SAMPLES"] = args.target
        print(f"  目标样本数覆盖: {args.target}")
    if args.smoke:
        smoke_test()
    else:
        main()


if __name__ == "__main__":
    run()
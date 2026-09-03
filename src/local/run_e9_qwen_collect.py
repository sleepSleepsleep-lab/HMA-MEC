"""
E9 Qwen 后端: 小规模辩论数据采集脚本
====================================
在阿里云百炼免费 100 万 token 额度内, 采集 Qwen3.7-plus / Qwen3.7-max
的辩论数据, 用于 E9 多后端对比实验.

Token 预算分析:
  DeepSeek 生产数据:  468,212,015 tokens / 10,000 样本 ≈ 46,800 tokens/样本
  按 K=8, M=4, R_max=5 计算, 每条样本约 50 次 LLM 调用

  1,000,000 tokens 可支撑:
  - 方案 A (K=4, M=2, R_max=3, 500 条):  ~240K tokens  ← 推荐
  - 方案 B (K=8, M=4, R_max=3, 100 条):  ~280K tokens
  - 方案 C (K=8, M=4, R_max=5, 20 条):   ~936K tokens  ← 不推荐, 样本太少

  推荐方案 A:
    模型     样本数  每次辩论约需 tokens  小计
    qwen3.7-plus  300    480                 144K
    qwen3.7-max   200    480                  96K
    预留重试       -      -                   60K
    总计          500    -                   300K  (远低于 1M 限额)

使用方法:
  1. 在 config.py 中设置 LLM_BACKEND = "qwen" (或运行时传参)
  2. 运行: python local/run_e9_qwen_collect.py
  3. 完成后运行: python local/run_e9_llm_backend.py  (训练+评估)
"""

import os, sys, json, time, re, threading, logging
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)

from config import RESULTS_DIR, SEED
from environment import MECEnvironment
from agent_define import make_agents
from cw_debate import cw_debate
from distill_agent import save_debate_record, count_debate_records
from llm_client import get_llm_client

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s")

# ============================================================
# 显眼配置区
# ============================================================
# 后端与模型列表 (每个生成一个独立数据集文件)
BACKENDS = [
    {"name": "qwen3.6-plus", "backend": "qwen", "model": "qwen3.6-plus", "target": 300}]

# 场景配置 (只用小规模, 节省 tokens)
SCALES = [(4, 2), (8, 4)]
SCALE_PROB = [0.4, 0.6]
SEEDS = list(range(0, 30))
N_WORKERS = 10
WRITE_LOCK = threading.Lock()

_thread_local = threading.local()

def get_thread_llm(backend, model):
    if not hasattr(_thread_local, "llm") or _thread_local.llm is None:
        _thread_local.llm = get_llm_client(backend=backend, model=model)
    return _thread_local.llm


def generate_one_sample(task_idx, backend_cfg, output_path):
    rng = np.random.RandomState(SEED + 1 + task_idx)
    scale_idx = int(rng.choice(len(SCALES), p=SCALE_PROB))
    K, M = SCALES[scale_idx]
    seed = int(rng.choice(SEEDS))
    fingerprint = f"K{K}-M{M}-seed{seed}-{backend_cfg['name']}-t{task_idx}"

    env = MECEnvironment(num_users=K, num_servers=M, seed=seed)
    np_rng = np.random.RandomState(seed + task_idx)
    n_warmup = int(np_rng.randint(0, 15 + 1))
    s = env.reset()
    for _ in range(n_warmup):
        a = np_rng.uniform(0, 1, env.action_dim).astype(np.float32)
        s, _, done, _ = env.step(a)
        if done:
            s = env.reset()

    llm = get_thread_llm(backend_cfg["backend"], backend_cfg["model"])
    if llm is None:
        return False, fingerprint

    agents = make_agents(env, with_va=True)
    try:
        out = cw_debate(env, agents, mode="FullLLM", llm=llm, verbose=False)
        alpha = out['plan']['alpha']
        server = out['plan']['server']
        confidences = (np.asarray(out['confidence_history'][-1])
                       if out['confidence_history'] else np.zeros(K, dtype=np.float32))
        save_debate_record(output_path, env._get_state(), alpha, server, confidences, fingerprint)
        return True, fingerprint
    except Exception as e:
        logging.warning(f"样本生成失败 ({fingerprint}): {e}")
        return False, fingerprint


def collect_for_backend(cfg):
    name = cfg["name"]
    target = cfg["target"]
    output_path = os.path.join(RESULTS_DIR, f"debate_dataset_{name}.jsonl")

    n_done = count_debate_records(output_path)
    logging.info(f"[{name}] 已有 {n_done}/{target} 条, 开始采集...")

    t0 = time.time()
    n_fail = 0
    next_idx = n_done

    while n_done < target:
        batch = min(N_WORKERS * 2, target - n_done)
        tasks = [(next_idx + i, cfg, output_path) for i in range(batch)]
        next_idx += batch

        with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
            futures = {pool.submit(generate_one_sample, *t): t for t in tasks}
            for fut in as_completed(futures):
                ok, fp = fut.result()
                if ok:
                    n_done += 1
                else:
                    n_fail += 1
                if n_done % 20 == 0:
                    rate = n_done / max(time.time() - t0, 1)
                    eta = (target - n_done) / max(rate, 1e-3)
                    logging.info(f"  [{name}] {n_done}/{target}  rate={rate:.2f}/s  ETA={eta/60:.1f}min  fail={n_fail}")

    dt = time.time() - t0
    logging.info(f"[{name}] 完成: {n_done} 条, 失败 {n_fail}, 耗时 {dt/60:.1f} 分钟")


def main():
    print("=" * 60)
    print("  E9 Qwen 后端辩论数据采集")
    print(f"  额度预算: 每模型约 500K tokens (远低于 1M 免费限额)")
    print("=" * 60)
    for cfg in BACKENDS:
        print(f"\n  开始采集: {cfg['name']} (target={cfg['target']})")
        collect_for_backend(cfg)
    print("\n" + "=" * 60)
    print("  采集完成! 现在运行 python local/run_e9_llm_backend.py 进行训练+评估")
    print("=" * 60)


if __name__ == "__main__":
    main()

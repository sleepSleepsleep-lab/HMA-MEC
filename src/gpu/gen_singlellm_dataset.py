# -*- coding: utf-8 -*-
"""
单次 LLM 教师蒸馏数据采样 (gen_singlellm_dataset.py, 2026-09-02)
================================================================
评审意见 P0-2: 缺"辩论教师 vs 单次 LLM 教师"的蒸馏对比。
本脚本用单次 LLM 调用 (与 SingleLLM 相同的 prompt/解析, 1 call/state)
在 K=8, M=4 的状态分布上采样 5010 条教师数据 (与辩论数据集同规模),
记录格式与 debate_dataset.jsonl 兼容 (confidence 填 0.5 中立值,
单次调用无语义置信度), 供 train_distill_policy.py 加载训练。
"""
import os, sys, json, time, threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)
sys.path.insert(0, os.path.join(SRC, "local"))

from config import (NUM_USERS, NUM_EDGE_SERVERS, RESULTS_DIR, SEED)
from environment import MECEnvironment
from llm_client import get_llm_client
from local.baselines_llm import LLMPromptBuilder, parse_plan

K, M = NUM_USERS, NUM_EDGE_SERVERS
TARGET = 5010
N_WORKERS = 16
OUT_PATH = os.path.join(RESULTS_DIR, "debate_dataset_singlellm.jsonl")
WRITE_LOCK = threading.Lock()

_thread_local = threading.local()


def get_llm():
    if not hasattr(_thread_local, 'llm'):
        _thread_local.llm = get_llm_client(backend='local_vllm')
        _thread_local.pb = LLMPromptBuilder()
    return _thread_local.llm, _thread_local.pb


def gen_one(i):
    rng = np.random.RandomState(SEED + 1000 + i)
    seed = int(rng.randint(0, 10000))
    env = MECEnvironment(num_users=K, num_servers=M, seed=seed)
    n_warmup = int(rng.randint(0, 30 + 1))
    s = env.reset()
    for _ in range(n_warmup):
        a = rng.uniform(0, 1, env.action_dim).astype(np.float32)
        s, _, done, _ = env.step(a)
        if done:
            s = env.reset()
    llm, pb = get_llm()
    system, user = pb.build(env)
    resp = llm.chat(system, user, temperature=0.0, max_tokens=256)
    plan = parse_plan(resp, K, M)
    if plan is None:
        return False
    record = {
        'state': np.asarray(env._get_state(), dtype=np.float32).tolist(),
        'alpha': np.asarray(plan['alpha'], dtype=np.float32).tolist(),
        'server': np.asarray(plan['server'], dtype=int).tolist(),
        'confidence': [0.5] * K,
        'fingerprint': f"K{K}-M{M}-seed{seed}-singlellm-t{i}",
    }
    with WRITE_LOCK:
        with open(OUT_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return True


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    if not os.path.exists(OUT_PATH):
        open(OUT_PATH, "w").close()
    n_done = 0
    for line in open(OUT_PATH, encoding='utf-8'):
        if line.strip():
            n_done += 1
    print(f"已有 {n_done}/{TARGET} 条, 继续采样", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        idx = n_done
        while n_done < TARGET:
            batch = list(ex.map(gen_one, range(idx, idx + N_WORKERS * 4)))
            ok = sum(1 for b in batch if b)
            n_done += ok
            idx += N_WORKERS * 4
            print(f"  {n_done}/{TARGET} 完成 ({time.time()-t0:.0f}s, "
                  f"{ok}/{len(batch)} ok)", flush=True)
    print(f"完成 -> {OUT_PATH} (总耗时 {time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()

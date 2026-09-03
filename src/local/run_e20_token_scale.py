# -*- coding: utf-8 -*-
"""
================================================================
E20b (2026-08): LLM token 成本随用户数 K 的规模曲线
================================================================
统计完整 CW-Debate (FullLLM) 单步决策在 K ∈ {4, 8, 12, 16} (M=4)
下的 LLM 调用次数与 token 消耗 (prompt/completion), 验证协议
通信复杂度 O(K+M+1) 的线性增长, 支撑稀疏通信/距离的必要性论证。
================================================================
"""
import os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)
sys.path.insert(0, os.path.join(SRC, "local"))

from config import RESULTS_DIR, SEED
from environment import MECEnvironment
from agent_define import make_agents
from cw_debate import cw_debate

OUT_JSON = os.path.join(RESULTS_DIR, "e20_token_scale.json")
KS = [4, 8, 12, 16]
M = 4


def main():
    from llm_client import get_llm_client
    llm = get_llm_client()
    print("=" * 62)
    print("  E20b: LLM token 成本随 K 增长 (M=4, FullLLM CW-Debate 单步)")
    print("=" * 62)
    rows = []
    for K in KS:
        env = MECEnvironment(num_users=K, num_servers=M, seed=SEED)
        env.reset()
        agents = make_agents(env, with_va=True)
        llm.usage_stats = {'prompt_tokens': 0, 'completion_tokens': 0,
                           'calls': 0}
        t0 = time.time()
        out = cw_debate(env, agents, mode="FullLLM", llm=llm, verbose=False)
        dt = time.time() - t0
        u = llm.usage_stats
        r = dict(K=K, M=M,
                 calls=int(u['calls']),
                 prompt_tokens=int(u['prompt_tokens']),
                 completion_tokens=int(u['completion_tokens']),
                 total_tokens=int(u['prompt_tokens'] + u['completion_tokens']),
                 rounds_used=int(out['rounds_used']),
                 wall_s=float(dt))
        rows.append(r)
        print(f"  K={K:3d} 调用={r['calls']:3d} prompt={r['prompt_tokens']:6d} "
              f"completion={r['completion_tokens']:5d} 总token={r['total_tokens']:7d} "
              f"轮数={r['rounds_used']} 耗时={dt:.0f}s", flush=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    json.dump({"rows": rows}, open(OUT_JSON, "w"), indent=2,
              ensure_ascii=False)
    print(f"  保存 -> {OUT_JSON}")


if __name__ == "__main__":
    main()
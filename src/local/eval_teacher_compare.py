# -*- coding: utf-8 -*-
"""
教师蒸馏对比评估 (eval_teacher_compare.py, 2026-09-02)
======================================================
评审 P0-2: 辩论教师 vs 单次 LLM 教师的蒸馏质量对比。
对两个教师蒸馏网络 (辩论: distilled_policy.pth / 单次: distill_singlellm.pth)
在同一批真独立 episode (SEED + s*100 + e, s in [0,5), e in [0,10), n=50) 上,
分别评估 raw (蒸馏网络直接输出) 与 refined (同一 PlanRefiner 精化) 指标。
"""
import os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)
sys.path.insert(0, os.path.join(SRC, "local"))

from config import NUM_USERS, NUM_EDGE_SERVERS, RESULTS_DIR, CHECKPOINT_DIR, SEED
from environment import MECEnvironment
from distill_agent import PolicyAgentRunner
from local.plan_refiner import PlanRefiner

K, M = NUM_USERS, NUM_EDGE_SERVERS
N_SEEDS, N_EPISODES, N_STEPS = 5, 10, 200
TEACHERS = {
    "Debate(辩论教师)": os.path.join(CHECKPOINT_DIR, "distilled_policy.pth"),
    "SingleLLM(单次教师)": os.path.join(CHECKPOINT_DIR, "distill_singlellm.pth"),
}
OUT_JSON = os.path.join(RESULTS_DIR, "teacher_compare.json")


def run_episode(runner, env, refiner, use_refine):
    E, T, S, SL = [], [], [], []
    st = env.reset()
    for _ in range(N_STEPS):
        out = runner.infer(st, deterministic=False)
        alpha, server = out['plan']['alpha'], out['plan']['server']
        if use_refine:
            alpha, server = refiner.refine(env, alpha, server)
        act = np.zeros(2 * env.K, np.float32)
        act[0::2] = np.clip(alpha, 0.01, 1.0)
        act[1::2] = (np.clip(server, 0, env.M - 1) + 0.5) / env.M
        ns, _, d, info = env.step(act)
        E.append(info['energy']); T.append(info['latency'])
        S.append(info['success_rate']); SL.append(info['priority_sla'])
        st = ns
        if d:
            break
    return dict(E=float(np.mean(E)), T=float(np.mean(T)),
                suc=float(np.mean(S)), sla=float(np.mean(SL)))


def main():
    refiner = PlanRefiner(omega=(0.5, 0.5))
    out = {}
    for name, path in TEACHERS.items():
        runner = PolicyAgentRunner(model_path=path)
        raw_all, ref_all = [], []
        for s in range(N_SEEDS):
            for e in range(N_EPISODES):
                seed = SEED + s * 100 + e
                r_raw = run_episode(runner,
                                    MECEnvironment(num_users=K, num_servers=M,
                                                   seed=seed), refiner, False)
                r_ref = run_episode(runner,
                                    MECEnvironment(num_users=K, num_servers=M,
                                                   seed=seed), refiner, True)
                raw_all.append(r_raw); ref_all.append(r_ref)
        def agg(rows):
            return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        out[name] = {"raw": agg(raw_all), "refined": agg(ref_all),
                     "n": len(raw_all),
                     "per_episode": {"raw": raw_all, "refined": ref_all}}
        print(f"  {name:22s} raw: suc={out[name]['raw']['suc']:.1%} "
              f"T={out[name]['raw']['T']:.3f} | refined: "
              f"suc={out[name]['refined']['suc']:.1%} "
              f"T={out[name]['refined']['T']:.3f} "
              f"SLA={out[name]['refined']['sla']:.1%} "
              f"E={out[name]['refined']['E']:.3f}", flush=True)
    json.dump(out, open(OUT_JSON, "w"), indent=1, ensure_ascii=False)
    print(f"已保存 -> {OUT_JSON}")


if __name__ == "__main__":
    main()

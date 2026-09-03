# -*- coding: utf-8 -*-
"""
E19 高动态 trace 变体 (rerun_e19_highdyn.py, 2026-09-02)
=========================================================
评审意见: E19 的 trace 波动温和 (λ std=0.34), 未构成压力测试。
本脚本构造高动态负载序列 λ' = clip(1 + 2*(λ-1), 0.3, 3.0) 并重归一化
(均值仍为 1, 波动幅度约为原 trace 的 2 倍), 复用 E19 的 TraceEnv/
make_method 在相同 8 起始相位上重跑 trace 口径; uniform 对照直接取原
e19_trace.json 的 per_run.uniform (同一批均匀环境), 计算保持率。
"""
import os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)
sys.path.insert(0, HERE)

from config import SEED, RESULTS_DIR
from local import run_e19_trace as E19

OUT_JSON = os.path.join(RESULTS_DIR, "e19_trace_highdyn.json")
OLD_JSON = os.path.join(RESULTS_DIR, "e19_trace.json")


def main():
    # 构造高动态 λ (std ~ 2x), 保持 mean=1
    lam_old = E19.LAM
    lam2 = np.clip(1.0 + 2.0 * (lam_old - 1.0), E19.LAM_MIN, E19.LAM_MAX)
    lam2 = lam2 / max(lam2.mean(), 1e-9)
    E19.LAM = lam2.astype(np.float32)   # patch 模块全局, TraceEnv 引用生效

    old = json.load(open(OLD_JSON))
    n_steps = len(lam2)
    phases = [int(x) for x in np.linspace(0, n_steps, E19.N_PHASES,
                                          endpoint=False)]
    print(f"λ': min={lam2.min():.2f} max={lam2.max():.2f} "
          f"mean={lam2.mean():.2f} std={lam2.std():.2f} "
          f"(原 std={lam_old.std():.2f})", flush=True)

    out = {"n_steps": n_steps, "lam_min": float(lam2.min()),
           "lam_max": float(lam2.max()), "lam_std": float(lam2.std()),
           "phases": phases, "amplify": 2.0}
    t0 = time.time()
    for name in ["HMA-Distill", "MPC", "GA", "Greedy"]:
        recs_t = []
        for ph in phases:
            sd = SEED + ph * 137
            env = E19.TraceEnv(use_trace=True, seed=sd, start_phase=ph)
            obj, kind = E19.make_method(env, name)
            r = E19.run_env(env, kind, obj, n_steps)
            recs_t.append(r)
        recs_u = old[name]["per_run"]["uniform"]   # 原均匀对照 (同种子)
        mu = {k: float(np.mean([r[k] for r in recs_u])) for k in recs_u[0]}
        mt = {k: float(np.mean([r[k] for r in recs_t])) for k in recs_t[0]}
        keep = mt['suc'] / max(mu['suc'], 1e-9)
        out[name] = {"uniform": mu, "trace_highdyn": mt,
                     "suc_keep_rate": keep, "n": len(recs_t),
                     "per_run": {"uniform": recs_u,
                                 "trace_highdyn": recs_t}}
        print(f"  {name:14s} uniform suc={mu['suc']:.1%} | "
              f"highdyn suc={mt['suc']:.1%} | 保持率={keep:.1%} "
              f"({time.time()-t0:.0f}s)", flush=True)
    json.dump(out, open(OUT_JSON, "w"), indent=1, ensure_ascii=False)
    print(f"已保存 -> {OUT_JSON}")


if __name__ == "__main__":
    main()

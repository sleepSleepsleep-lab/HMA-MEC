# -*- coding: utf-8 -*-
"""
E11 扩样重跑 (rerun_e11_n20.py, 2026-09-02)
============================================
评审意见: E11 每场景 n=10 seeds 偏小。本脚本将每场景扩样至 20 个独立种子
(seed = SEED + sd, sd in [0,20), 与旧 10 个种子结构一致且互不碰撞),
重跑全部 7 场景 x 3 方法 (LLM种子=蒸馏策略+精化 / MPC启发式 / 随机种子+精化),
保存逐种子数据供 n=20 配对 Wilcoxon 与 Holm 重算。
"""
import os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)
sys.path.insert(0, HERE)

from config import SEED, RESULTS_DIR
from local import e1_llm_necessity_v2 as E11
from environment import MECEnvironment

N_SEEDS_NEW = 20
OUT_JSON = os.path.join(RESULTS_DIR, "e11_necessity_n20.json")


def main():
    from scipy.stats import wilcoxon
    builders = {
        "LLM种子(HMA)": lambda env: (E11.HMAAgentRunner(env=env, mode="Distill"), "runner"),
        "MPC(启发式)": lambda env: (E11.MPCBaseline(), "pred"),
        "Random+Refiner": lambda env: (E11.RandomSeedRefiner(), "pred"),
    }
    summary = {}
    t0 = time.time()
    for sc in E11.SCENARIOS:
        print(f"\n===== 场景 {sc} (n={N_SEEDS_NEW}) =====", flush=True)
        summary[sc] = {}
        per_method = {label: [] for label in builders}
        for sd in range(N_SEEDS_NEW):
            env = E11.make_env(sd, sc)
            for label, mk in builders.items():
                obj, kind = mk(env)
                if kind == "runner":
                    r = E11.run(None, env, runner=obj, scenario=sc)
                else:
                    r = E11.run(obj, env, scenario=sc)
                per_method[label].append(r)
        for label, rows in per_method.items():
            mean = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
            std = {k: float(np.std([r[k] for r in rows])) for k in rows[0]}
            summary[sc][label] = {"mean": mean, "std": std,
                                  "per_seed": rows}
            print(f"  {label:16s} E={mean['E']:.3f} T={mean['T']:.3f}±{std['T']:.3f} "
                  f"suc={mean['suc']:.1%} sla={mean['sla']:.1%}", flush=True)
        aT = [r['T'] for r in per_method["LLM种子(HMA)"]]
        bT = [r['T'] for r in per_method["MPC(启发式)"]]
        aS = [r['suc'] for r in per_method["LLM种子(HMA)"]]
        bS = [r['suc'] for r in per_method["MPC(启发式)"]]
        def _w(x, y):
            if len(set(np.round(x, 6))) < 2 or len(set(np.round(y, 6))) < 2:
                return float('nan'), 1.0
            try:
                w, p = wilcoxon(x, y)
                return float(w), float(p)
            except Exception:
                return float('nan'), 1.0
        wt, pt = _w(aT, bT)
        ws, ps = _w(aS, bS)
        summary[sc]["_stats"] = {
            "llm_vs_mpc_latency_gain_pct": float(np.mean(aT) / np.mean(bT) - 1.0),
            "llm_vs_mpc_suc_gain_pp": float((np.mean(aS) - np.mean(bS)) * 100),
            "wilcoxon_latency_p": float(pt),
            "wilcoxon_success_p": float(ps),
            "n_seeds": N_SEEDS_NEW,
        }
        print(f"  LLM vs MPC: 时延增益="
              f"{summary[sc]['_stats']['llm_vs_mpc_latency_gain_pct']*100:.2f}% "
              f"p={pt:.4f} | 成功率 {summary[sc]['_stats']['llm_vs_mpc_suc_gain_pp']:+.2f}pp "
              f"p={ps:.4f}", flush=True)
    json.dump(summary, open(OUT_JSON, "w"), indent=1, ensure_ascii=False)
    print(f"\n已保存 -> {OUT_JSON} (总耗时 {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()

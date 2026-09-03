# -*- coding: utf-8 -*-
"""
E9 多 LLM 后端对比 —— 权威重跑 (rerun_e9_llm_backend_v2.py, 2026-09-02)
=======================================================================
背景: 论文表 tab:e9 声称 Qwen2.5-7B/Llama-3.1-8B/Mistral-7B 三后端成功率
92.7/91.9/93.6% (15 episode), 而归档 results/e9_llm_backend.npz (同日生成,
脚本 2 种子 x 3 episode、种子结构 SEED+sd+ep 含碰撞) 只有 91.44/91.88/91.94%,
两批矛盾。本脚本用三后端已训练权重 (results/checkpoints/distill_backend_*.pth,
2026-08-25) 以 E1 式互不碰撞种子结构重跑权威评估:

  - n = 30 / 后端 = 5 种子 x 6 episode, 每 episode 200 步 (与环境 done 对齐);
  - episode 环境种子: SEED + s*100 + e  (s in [0,5), e in [0,6)), 30 个全独立环境;
  - 协议与 run_e9_llm_backend.py / run_episode 完全一致: mode='Distill'
    (PolicyAgentNet 前向, stochastic alpha 采样, deterministic=False) + 默认
    PlanRefiner (反事实验证器精化, 纯 numpy/env.simulate, 无任何 LLM 调用);
  - 指标 = 每 episode 内 per-step 均值 (info['energy'] kJ / latency s /
    success_rate / priority_sla), done 截断 (恒 200 步);
  - 可复现: 每 episode 前 torch.manual_seed(SEED+s*100+e) 重播种
    (env 已用自身 RandomState(seed), 策略采样用全局 torch RNG)。
  - CPU-only: 启动时须 CUDA_VISIBLE_DEVICES="" (勿扰动 vLLM GPU 服务)。

并行: 任务粒度 = (后端, s, e) 单 episode; multiprocessing spawn, 默认 6 worker
(机器同时跑 4 个 E3 CPU 重跑, 128 核, 中等并行度避免抢满)。

输出: results/e9_llm_backend_v2.npz + .json (per-episode vals / mean / std /
SE / n / 种子结构说明 / 口径字段), 键风格与 e1_fresh_*.npz 兼容
({backend}__{metric}__vals|mean|std)。
"""
import os
import sys
import json
import time
import argparse
import multiprocessing as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)
sys.path.insert(0, HERE)

from config import SEED, NUM_USERS, NUM_EDGE_SERVERS, RESULTS_DIR

BACKEND_CKPTS = {
    "Qwen2.5-7B":   "distill_backend_Qwen2.5-7B.pth",
    "Llama-3.1-8B": "distill_backend_Llama-3.1-8B.pth",
    "Mistral-7B":   "distill_backend_Mistral-7B.pth",
}
N_SEEDS = 5
N_EPISODES = 6
N_STEPS = 200          # 与 E1 episode 全长度一致 (env done = 200)
TORCH_THREADS = 4

OUT_NPZ = os.path.join(RESULTS_DIR, "e9_llm_backend_v2.npz")
OUT_JSON = os.path.join(RESULTS_DIR, "e9_llm_backend_v2.json")


def eval_one_episode(backend, ckpt_path, s, e, n_steps=N_STEPS):
    """跑 (backend, seed_i=s, episode=e) 一个 episode, 返回该 episode 各指标均值."""
    import numpy as np
    import torch

    torch.set_num_threads(TORCH_THREADS)
    episode_seed = SEED + s * 100 + e
    torch.manual_seed(episode_seed)          # 可复现: 策略采样 RNG 重播种
    np.random.seed(episode_seed)

    from environment import MECEnvironment
    from agent_runner import HMAAgentRunner
    from local.experiment_common import compose_action

    env = MECEnvironment(num_users=NUM_USERS, num_servers=NUM_EDGE_SERVERS,
                         seed=episode_seed)
    env.reset()
    runner = HMAAgentRunner(env=env, mode="Distill", policy_path=ckpt_path)
    energy, lat, suc, sla = [], [], [], []
    for _ in range(n_steps):
        state = env._get_state()
        out = runner.run_step(state=state, agents_reuse=True)
        a = compose_action(out["plan"], env.K, env.M)
        _, _, d, info = env.step(a)
        energy.append(info["energy"])
        lat.append(info["latency"])
        suc.append(info["success_rate"])
        sla.append(info["priority_sla"])
        if d:
            break
    return {
        "backend": backend, "s": s, "e": e,
        "seed": episode_seed, "n_steps_done": env.step_count,
        "energy": float(np.mean(energy)),
        "latency": float(np.mean(lat)),
        "success_rate": float(np.mean(suc)),
        "priority_sla": float(np.mean(sla)),
    }


def run_task(args):
    backend, ckpt_path, s, e = args
    t0 = time.time()
    r = eval_one_episode(backend, ckpt_path, s, e)
    r["wall_s"] = time.time() - t0
    print(f"    [{backend}] s{s} e{e} (seed={r['seed']}) "
          f"E={r['energy']:.4f} T={r['latency']:.3f} "
          f"suc={r['success_rate']:.2%} sla={r['priority_sla']:.2%} "
          f"steps={r['n_steps_done']} ({r['wall_s']:.1f}s)", flush=True)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6,
                    help="并行 worker 数 (机器同时有 E3 重跑, 用中等并行度)")
    ap.add_argument("--backends", type=str, default=None,
                    help="逗号分隔的后端子集 (默认全部)")
    ap.add_argument("--smoke", action="store_true",
                    help="每后端只跑 1 个 episode 测速")
    args = ap.parse_args()

    ckpt_dir = os.path.join(RESULTS_DIR, "checkpoints")
    backends = list(BACKEND_CKPTS)
    if args.backends:
        backends = [b for b in args.backends.split(",") if b in BACKEND_CKPTS]
    tasks = []
    for b in backends:
        p = os.path.join(ckpt_dir, BACKEND_CKPTS[b])
        if not os.path.exists(p):
            print(f"[skip] 缺失权重 {p}")
            continue
        if args.smoke:
            tasks.append((b, p, 0, 0))
        else:
            for s in range(N_SEEDS):
                for e in range(N_EPISODES):
                    tasks.append((b, p, s, e))
    if not tasks:
        print("无任务"); return

    print(f"E9 v2 权威重跑: backends={backends}, n={len(tasks)} episode, "
          f"workers={args.workers}, 种子结构 SEED+s*100+e, {N_STEPS} 步/episode")
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "":
        print("[警告] 检测到 CUDA_VISIBLE_DEVICES 未置空, 建议 "
              "CUDA_VISIBLE_DEVICES='' 运行以避免扰动 vLLM")

    t_all = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers) as pool:
        results = list(pool.imap_unordered(run_task, tasks))
    results.sort(key=lambda r: (r["backend"], r["s"], r["e"]))
    print(f"全部完成: {len(results)} episode, "
          f"{time.time()-t_all:.0f}s total")

    # ---- 聚合 ----
    metrics = ("energy", "latency", "success_rate", "priority_sla")
    out = {}
    npz_dict = {}
    for b in backends:
        eps = [r for r in results if r["backend"] == b]
        agg = {m: np.array([r[m] for r in eps], dtype=np.float64)
               for m in metrics}
        out[b] = {
            "mean": {m: float(agg[m].mean()) for m in metrics},
            "std": {m: float(agg[m].std()) for m in metrics},        # ddof=0
            "std_ddof1": {m: float(agg[m].std(ddof=1)) for m in metrics},
            "se": {m: float(agg[m].std(ddof=1) / np.sqrt(len(eps)))
                   for m in metrics},
            "n": len(eps),
            "n_seeds": N_SEEDS, "n_episodes_per_seed": N_EPISODES,
            "n_steps": N_STEPS,
            "episodes": [{"s": r["s"], "e": r["e"], "seed": r["seed"],
                          "n_steps_done": r["n_steps_done"],
                          **{m: r[m] for m in metrics}}
                         for r in eps],
        }
        for m in metrics:
            vals = np.array([r[m] for r in eps], dtype=np.float32)
            npz_dict[f"{b}__{m}__vals"] = vals
            npz_dict[f"{b}__{m}__mean"] = np.array([out[b]["mean"][m]],
                                                   dtype=np.float32)
            npz_dict[f"{b}__{m}__std"] = np.array([out[b]["std"][m]],
                                                  dtype=np.float32)

    meta = {
        "experiment": "E9 LLM backend comparison (authoritative rerun v2)",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "weights": {b: os.path.join(ckpt_dir, BACKEND_CKPTS[b])
                    for b in backends},
        "n_total_episodes": len(results),
        "seed_structure": "SEED + s*100 + e, s in [0,5), e in [0,6) "
                          "(E1 式互不碰撞, SEED=42)",
        "n_steps_per_episode": N_STEPS,
        "mode": "Distill policy forward (stochastic alpha) + PlanRefiner, "
                "no LLM calls",
        "metric_semantics": "per-episode mean of per-step info: energy kJ, "
                            "latency s, success_rate, priority_sla",
        "rng": "per-episode torch.manual_seed(episode_seed); env uses "
               "own RandomState(seed)",
        "evaluator": os.path.abspath(__file__),
        "se_note": "SE = std(ddof=1)/sqrt(n)",
    }
    payload = {"meta": meta, "backends": out}
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    npz_dict["meta_seed_structure"] = np.array(
        [meta["seed_structure"]], dtype=object)
    np.savez_compressed(OUT_NPZ, **npz_dict)
    print(f"已保存: {OUT_NPZ} / {OUT_JSON}")

    # 汇总打印
    for b in backends:
        o = out[b]
        print(f"\n  [{b}] n={o['n']}: "
              f"E={o['mean']['energy']:.4f}±{o['se']['energy']:.4f} "
              f"T={o['mean']['latency']:.4f}±{o['se']['latency']:.4f} "
              f"suc={o['mean']['success_rate']:.4f}±{o['se']['success_rate']:.4f} "
              f"sla={o['mean']['priority_sla']:.4f}±{o['se']['priority_sla']:.4f}")


if __name__ == "__main__":
    main()

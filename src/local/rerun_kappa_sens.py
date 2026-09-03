# -*- coding: utf-8 -*-
"""
κ_e 敏感性评估 (rerun_kappa_sens.py, 2026-09-02)
=================================================
评审意见: κ_e = 1e-27 (与终端同量级) 是结论敏感假设, 缺敏感性分析。
本脚本在 κ_e ∈ {1e-28, 1e-27, 1e-26} 下重放代表方法的评估
(蒸馏策略/MPC/GA 经验证器精化后的端到端指标, 真独立种子结构 S+s*100+e,
每组合 5 seeds × 10 episodes × 200 步), 检验相对结论的稳定性。

用法: python3 local/rerun_kappa_sens.py --kappa 1e-28 --method HMA-Distill --shard k1e28_hmad
输出: results/kappa_sens_{shard}.npz
"""
import os, sys, time, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)
sys.path.insert(0, HERE)

# ---- 关键: 在 import environment 之前覆写 κ_e ----
import config
KAPPA = float(sys.argv[sys.argv.index('--kappa') + 1])
config.KAPPA_EDGE = KAPPA

from config import NUM_USERS, NUM_EDGE_SERVERS, RESULTS_DIR, SEED, MAX_STEPS
from environment import MECEnvironment
from local.plan_refiner import PlanRefiner

N_SEEDS, N_EPISODES, N_STEPS = 5, 10, 200


def run_episode(method_kind, name, env, refiner):
    E, T, S, SL = [], [], [], []
    st = env.reset()
    for _ in range(N_STEPS):
        if name == 'HMA-Distill':
            out = method_kind.infer(st, deterministic=False)
            alpha, server = out['plan']['alpha'], out['plan']['server']
        elif name == 'MPC':
            alpha, server = np.full(env.K, 0.5), np.argmax(env.channels, axis=1)
        elif name == 'GA':
            act = method_kind.predict(env._get_state(), env)
            alpha = np.clip(np.asarray(act[0::2], dtype=float), 0.01, 1.0)
            server = np.clip(np.floor(np.asarray(act[1::2], dtype=float) * env.M),
                             0, env.M - 1).astype(int)
        if refiner is not None:
            # 仅 HMA-Distill 需要外部验证器精化; MPC/GA 的 predict 内部已含
            # 同一 PlanRefiner 搜索 (与 E1 评估口径一致)
            alpha, server = refiner.refine(env, alpha, server)
        else:
            act = method_kind.predict(env._get_state(), env)
            alpha = np.clip(np.asarray(act[0::2], dtype=float), 0.01, 1.0)
            server = np.clip(np.floor(np.asarray(act[1::2], dtype=float) * env.M),
                             0, env.M - 1).astype(int)
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
    ap = argparse.ArgumentParser()
    ap.add_argument('--kappa', type=float, required=True)
    ap.add_argument('--method', required=True)
    ap.add_argument('--shard', required=True)
    args = ap.parse_args()

    from distill_agent import PolicyAgentRunner
    from local.ga_baseline import GAOffloadBaseline
    from local.baseline_mpc import MPCBaseline

    K, M = NUM_USERS, NUM_EDGE_SERVERS
    refiner = PlanRefiner(omega=(0.5, 0.5)) if args.method == 'HMA-Distill' else None
    if args.method == 'HMA-Distill':
        agent = PolicyAgentRunner(model_path=os.path.join(
            RESULTS_DIR, 'checkpoints', 'distilled_policy.pth'))
    elif args.method == 'MPC':
        agent = MPCBaseline()
    elif args.method == 'GA':
        agent = GAOffloadBaseline()
    else:
        raise ValueError(args.method)

    print(f"κ_e = {args.kappa:g}, 方法 = {args.method}", flush=True)
    recs = []
    for s in range(N_SEEDS):
        for e in range(N_EPISODES):
            env = MECEnvironment(num_users=K, num_servers=M,
                                 seed=SEED + s * 100 + e)
            r = run_episode(agent, args.method, env, refiner)
            recs.append(r)
        print(f"  s{s}: suc={np.mean([r['suc'] for r in recs]):.1%} "
              f"T={np.mean([r['T'] for r in recs]):.3f}", flush=True)

    out = {k: np.array([r[k] for r in recs], dtype=np.float32)
           for k in ('E', 'T', 'suc', 'sla')}
    np.savez(os.path.join(RESULTS_DIR, f"kappa_sens_{args.shard}.npz"),
             kappa=KAPPA, method=args.method, **out)
    print(f"  保存 kappa_sens_{args.shard}.npz: "
          f"E={out['E'].mean():.4f} T={out['T'].mean():.3f} "
          f"suc={out['suc'].mean():.1%} sla={out['sla'].mean():.1%}", flush=True)


if __name__ == "__main__":
    main()

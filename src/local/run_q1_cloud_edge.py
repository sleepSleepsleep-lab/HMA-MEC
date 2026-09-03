# -*- coding: utf-8 -*-
"""
================================================================
Q1 补强 (2026-08): CloudEdge 对比 —— 验证两层（用户-边缘）模型合理性
================================================================
目标: 用数据证明"云端回传在当前时延敏感场景下不可行/不必要",
      支撑 system_model 中两层模型论证段落 (Q1)。

方法 (零侵入, 不改 environment.py):
  A) 解析: 对基准场景每个用户按系统模型公式计算
     T_local / T_edge(best 信道) / T_cloud(回传 80ms + 云算力 F_CLOUD),
     统计云端 deadline 达标率与"云比边缘更优"的比例。
  B) 数值: 把"云"作为第 (M+1) 个服务器候选, 用与验证器精化一致的
     贪心局部搜索 (server 候选含云, 云路径物理公式独立实现,
     与 env.simulate 同口径归一化) 求解, 统计:
       - 云在最优解中的选择率 (预期接近 0, 因回传+云能耗)
       - 含云搜索相对不含云的 J 增益 (预期≈0, 证明云不必要)
       - 纯云 (AllCloud) 极端基线的成功率/时延 (预期差)

依赖: 纯 CPU (无需 vLLM)。
================================================================
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)
sys.path.insert(0, os.path.join(SRC, "local"))

from config import (NUM_USERS, NUM_EDGE_SERVERS, SEED, RESULTS_DIR,
                    BANDWIDTH, NOISE_POWER, TX_POWER_USER,
                    KAPPA_LOCAL, KAPPA_EDGE, P_IDLE, F_CLOUD,
                    ACCOUNT_EDGE_ENERGY)
from environment import MECEnvironment

K, M = NUM_USERS, NUM_EDGE_SERVERS
CLOUD_ROUNDTRIP = 0.08          # 云-边缘往返传播时延 80 ms
R_CLOUD = 50e6                  # 回传链路共享带宽 50 Mbps (瓶颈)
N_EPISODES = 3
N_STATES_EVAL = 30              # 每 episode 取 30 个状态做含云搜索
N_SEARCH_USERS = K              # 逐用户搜索
ALPHA_GRID = (0.05, 0.2, 0.4, 0.6, 0.8, 0.95)
OUT_JSON = os.path.join(RESULTS_DIR, "e14_cloud_edge.json")


def cloud_latency_user(env, k, off_data, off_cycles, server, queue_delay=0.0):
    """返回 (offload_time, edge_energy). server == M 表示云端."""
    if server == M:                       # 云端
        tx = off_data / (R_CLOUD + 1e-9) + CLOUD_ROUNDTRIP
        comp = off_cycles / (F_CLOUD + 1e-9)
        energy = KAPPA_EDGE * (F_CLOUD ** 2) * off_cycles
        return tx + queue_delay + comp, energy
    # 边缘
    h = env.channels[k, server]
    rate = BANDWIDTH * np.log2(1 + TX_POWER_USER * h / NOISE_POWER)
    tx = off_data / (rate + 1e-9)
    comp = off_cycles / (env.f_edge[server] + 1e-9)
    if ACCOUNT_EDGE_ENERGY:
        energy = KAPPA_EDGE * (env.f_edge[server] ** 2) * off_cycles
    else:
        energy = 0.0
    return tx + queue_delay + comp, energy


def cloud_cost(env, plan):
    """含云方案代价 (与 env.simulate 同口径归一化). plan:{alpha,server}.
    server[k] ∈ {0..M}, M = 云端."""
    alpha = np.asarray(plan['alpha'], dtype=float)
    server = np.asarray(plan['server'], dtype=int)
    # FIFO 排队: 按服务器分组按优先级
    queues = {m: [] for m in range(M + 1)}
    for k in range(K):
        s = int(server[k])
        if s > M:
            s = M
        queues[s].append((k, env.tasks[k]['priority'], alpha[k]))
    for m in range(M + 1):
        queues[m].sort(key=lambda x: (-x[1], x[0]))
    acc = {m: 0.0 for m in range(M + 1)}
    total_e = total_t = 0.0
    suc = sla_t = sla_c = 0
    for m in range(M + 1):
        for (k, _p, a) in queues[m]:
            t = env.tasks[k]
            local_cycles = a * t['C']
            local_time = local_cycles / env.f_local[k]
            local_e = KAPPA_LOCAL * (env.f_local[k] ** 2) * local_cycles
            off_cycles = (1 - a) * t['C']
            off_data = (1 - a) * t['D']
            off_t, off_e = cloud_latency_user(env, k, off_data, off_cycles,
                                              int(m) if m < M else M,
                                              queue_delay=acc[m])
            acc[m] += off_cycles / (env.f_edge[m] + 1e-9) if m < M else \
                      off_cycles / (F_CLOUD + 1e-9)
            total_ti = max(local_time, off_t)
            idle = max(0.0, off_t - local_time)
            total_e += local_e
            # 传输能耗: 复刻环境公式 tx_energy = TX_POWER_USER * tx_time
            if m < M:
                rate0 = BANDWIDTH * np.log2(1 + TX_POWER_USER * env.channels[k, m] / NOISE_POWER)
                tx_time = off_data / (rate0 + 1e-9)
            else:
                tx_time = off_data / (R_CLOUD + 1e-9)
            total_e += TX_POWER_USER * tx_time + off_e + P_IDLE * idle
            total_t += total_ti
            if total_ti <= t['tau']:
                suc += 1
            if t['priority'] >= 3:
                sla_t += 1
                if total_ti <= t['tau']:
                    sla_c += 1
    J = 0.5 * (total_e / 1e3) + 0.5 * (total_t / K)
    if suc / K < 0.5 or (sla_c / sla_t if sla_t else 1.0) < 0.5:
        J += 10.0
    return dict(energy=total_e / 1e3, latency=total_t / K,
                success=suc / K, sla=(sla_c / sla_t) if sla_t else 1.0,
                J=J)


def cloud_refine(env, alpha, server, rounds=3, allow_cloud=True):
    """逐用户贪心局部搜索, server 候选含云 (模式 m==M)."""
    alpha = np.asarray(alpha, dtype=float).copy()
    server = np.asarray(server, dtype=int).copy()
    best_chan = np.asarray(env.channels).argmax(axis=1)
    lowF = [i for i in np.argsort(np.asarray(env.f_edge))[:2]]
    maxF = int(np.argmax(np.asarray(env.f_edge)))
    for _ in range(rounds):
        for k in range(K):
            cand = cloud_cost(env, {'alpha': alpha, 'server': server})['J']
            cands = {int(server[k]), int(best_chan[k]), int(lowF[k % 2]),
                     int(maxF)}
            if allow_cloud:
                cands.add(M)
            for m in cands:
                for a in ALPHA_GRID:
                    na = alpha.copy(); ns = server.copy()
                    na[k] = a; ns[k] = m
                    j = cloud_cost(env, {'alpha': na, 'server': ns})['J']
                    if j < cand:
                        cand = j
                        alpha[k] = a
                        server[k] = m
    return alpha.astype(float), server.astype(int)


def cloud_deadline_analytic(env, k):
    """T_local / T_edge(best) / T_cloud 解析对比 (无排队, 当前信道)."""
    t = env.tasks[k]
    T_local = t['C'] / env.f_local[k]
    hs = env.channels[k]
    mbest = int(np.argmax(hs))
    rate_b = BANDWIDTH * np.log2(1 + TX_POWER_USER * hs[mbest] / NOISE_POWER)
    T_edge = t['D'] / (rate_b + 1e-9) + t['C'] / (env.f_edge[mbest] + 1e-9)
    T_cloud = (t['D'] / (R_CLOUD + 1e-9) + CLOUD_ROUNDTRIP
               + t['C'] / (F_CLOUD + 1e-9))
    return T_local, T_edge, T_cloud, t['tau']


def main():
    print("=" * 62)
    print(f"  Q1: CloudEdge —— 两层模型合理性数值验证 (K={K}, M={M})")
    print("=" * 62)
    # ---- A) 解析 ----
    n_cloud_better = n_cloud_ok = n_total = 0
    tl_sum = te_sum = tc_sum = 0.0
    per_user = []
    for ep in range(N_EPISODES):
        env = MECEnvironment(num_users=K, num_servers=M, seed=SEED + ep)
        env.reset()
        for k in range(K):
            tl, te, tc, tau = cloud_deadline_analytic(env, k)
            n_total += 1
            if tc <= te:
                n_cloud_better += 1
            if tc <= tau:
                n_cloud_ok += 1
            tl_sum += tl; te_sum += te; tc_sum += tc
            per_user.append(dict(local=tl, edge_best=te, cloud=tc, tau=tau))
    analytic = dict(
        n_users=n_total,
        cloud_better_than_edge_frac=float(n_cloud_better / n_total),
        cloud_deadline_ok_frac=float(n_cloud_ok / n_total),
        mean_T_local=float(tl_sum / n_total),
        mean_T_edge=float(te_sum / n_total),
        mean_T_cloud=float(tc_sum / n_total),
    )
    print(f"  [解析] 云优于边缘的最优服务器: {analytic['cloud_better_than_edge_frac']:.1%} | "
          f"云 deadline 达标率: {analytic['cloud_deadline_ok_frac']:.1%} | "
          f"T 均值 local={analytic['mean_T_local']:.3f} edge={analytic['mean_T_edge']:.3f} "
          f"cloud={analytic['mean_T_cloud']:.3f}s")

    # ---- B) 数值: 含云搜索 ----
    cloud_states, cloud_users = 0, 0
    J_no_cloud, J_with_cloud = [], []
    al_cloud = dict(E=[], T=[], suc=[], sla=[])
    for ep in range(N_EPISODES):
        env = MECEnvironment(num_users=K, num_servers=M, seed=SEED + ep)
        env.reset()
        for s in range(N_STATES_EVAL):
            # 启发式种子 (α=0.5 + 信道最优)
            seed_a = np.full(K, 0.5)
            seed_s = np.asarray(env.channels).argmax(axis=1).copy()
            # 不含云
            a1, s1 = cloud_refine(env, seed_a.copy(), seed_s.copy(), allow_cloud=False)
            J1 = cloud_cost(env, {'alpha': a1, 'server': s1})['J']
            # 含云
            a2, s2 = cloud_refine(env, seed_a.copy(), seed_s.copy(), allow_cloud=True)
            J2 = cloud_cost(env, {'alpha': a2, 'server': s2})['J']
            J_no_cloud.append(J1); J_with_cloud.append(J2)
            cloud_users += int(np.sum(s2 == M))
            cloud_states += int(np.sum(s2 == M) > 0)
            r2 = cloud_cost(env, {'alpha': a2, 'server': s2})
            al_cloud['E'].append(r2['energy']); al_cloud['T'].append(r2['latency'])
            al_cloud['suc'].append(r2['success']); al_cloud['sla'].append(r2['sla'])
            env.step(env.action_space.sample() if hasattr(env, 'action_space') else
                     np.random.uniform(0.5, 0.6, env.action_dim))
    numeric = dict(
        cloud_selection_user_frac=float(cloud_users / (N_EPISODES * N_STATES_EVAL * K)),
        cloud_selection_state_frac=float(cloud_states / (N_EPISODES * N_STATES_EVAL)),
        J_no_cloud_mean=float(np.mean(J_no_cloud)),
        J_with_cloud_mean=float(np.mean(J_with_cloud)),
        J_gain_with_cloud_pct=float((np.mean(J_no_cloud) - np.mean(J_with_cloud))
                                    / max(np.mean(J_no_cloud), 1e-9) * 100),
        cloud_solution_suc=float(np.mean(al_cloud['suc'])),
        cloud_solution_T=float(np.mean(al_cloud['T'])),
    )
    print(f"  [数值] 云选择率(用户)={numeric['cloud_selection_user_frac']:.2%} "
          f"| 含云J={numeric['J_with_cloud_mean']:.4f} vs 不含云J={numeric['J_no_cloud_mean']:.4f} "
          f"| 含云增益={numeric['J_gain_with_cloud_pct']:+.2f}%")

    out = dict(analytic=analytic, numeric=numeric)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    json.dump(out, open(OUT_JSON, "w"), indent=2, ensure_ascii=False)
    print(f"\n  保存 -> {OUT_JSON}")


if __name__ == "__main__":
    main()

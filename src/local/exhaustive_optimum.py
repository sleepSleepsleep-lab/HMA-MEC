# -*- coding: utf-8 -*-
"""
穷举最优上界 (exhaustive_optimum.py, 2026-09-01)
=================================================
对每个状态在全部 M^K 服务器组合上搜索统一目标
J = 0.5·E(kJ) + 0.5·T(s) + 1.0·ΣI[T_k>τ_k]（与 E1 评估口径一致）
的最优近似解：α 在 ALPHA_GRID 上做两轮坐标上升（优先级正向 + 反向），
使高优先级用户能内部化其对低优先级用户的排队外部性。物理公式与
environment.py 的 step() 逐项一致（Shannon 速率、优先级 FIFO 排队、
本地/传输/边缘/闲置能耗、ACCOUNT_EDGE_ENERGY=True）。

性能: 排队影响的增量更新（每用户候选只重算受影响用户），单状态约 1s。

输出:
  results/exhaustive_optimum.json  每 episode 的最优近似 (J*, E, T, suc, sla)
                                    与同一 episode 集合上 HMA-Distill 的对应
                                    指标, 用于计算最优性差距
"""
import os, sys, json, time, itertools
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)
sys.path.insert(0, os.path.join(SRC, "local"))

from config import (NUM_USERS, NUM_EDGE_SERVERS, SEED, RESULTS_DIR,
                    CHECKPOINT_DIR, BANDWIDTH, NOISE_POWER, TX_POWER_USER,
                    KAPPA_LOCAL, KAPPA_EDGE, P_IDLE)
from environment import MECEnvironment
from distill_agent import PolicyAgentRunner
from local.plan_refiner import PlanRefiner
from local.experiment_common import compose_action

ALPHA_GRID = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55,
                       0.65, 0.75, 0.85, 0.95], dtype=float)
PENALTY = 1.0          # E1 评估口径: 每失败用户 +1.0
OMEGA = (0.5, 0.5)
N_SEEDS, N_EPISODES, N_STEPS = 5, 2, 100
POLICY_PATH = os.path.join(CHECKPOINT_DIR, "distilled_policy.pth")


def build_assignments(K, M):
    return np.array(list(itertools.product(range(M), repeat=K)),
                    dtype=np.int64)


class StateSolver:
    """对单个状态做穷举式最优近似求解 (向量化于全部 M^K 组合)."""

    def __init__(self, assignments):
        self.A = assignments          # (A_count, K)
        self.A_count = assignments.shape[0]

    def solve(self, env):
        K = env.K
        A = self.A
        tasks = env.tasks
        D = np.array([t['D'] for t in tasks])
        C = np.array([t['C'] for t in tasks])
        tau = np.array([t['tau'] for t in tasks])
        p = np.array([t['priority'] for t in tasks])
        f_loc = np.asarray(env.f_local, dtype=float)
        f_edge = np.asarray(env.f_edge, dtype=float)
        h = np.asarray(env.channels)

        rate = BANDWIDTH * np.log2(1 + TX_POWER_USER * h / NOISE_POWER)  # (K,M)
        rate_sel = rate[np.arange(K)[None, :], A]       # (A_count, K)
        f_sel = f_edge[A]                                # (A_count, K)

        # 优先级顺序 (高优先在前, 同优先按编号升序) 与"在 k 之后"的掩码
        order = np.lexsort((np.arange(K), -p))
        behind = np.zeros((K, K), dtype=bool)   # behind[j, k] = j 排在 k 之后
        for i in range(K):
            for j in range(K):
                if (p[j] < p[i]) or (p[j] == p[i] and j > i):
                    behind[i, j] = True
        # 受用户 k 的 α 影响的用户: k 自身 + 与其同服务器且排在 k 之后的用户
        # mask_aff[k][j] 为 (A_count,) 布尔 (j 是否受影响), 逐状态预计算
        aff_idx = [[k] + [j for j in range(K) if behind[k, j]]
                   for k in range(K)]
        aff_mask = []   # 列表 of dict {j: (A_count,) bool}
        for k in range(K):
            d = {}
            for j in aff_idx[k]:
                if j == k:
                    d[j] = np.ones(self.A_count, dtype=bool)
                else:
                    d[j] = (A[:, j] == A[:, k])
            aff_mask.append(d)

        # ---- 当前方案逐用户量 (随坐标上升更新) ----
        alpha = np.full((self.A_count, K), 0.5)
        off_c = (1.0 - alpha) * C[None, :]
        off_d = (1.0 - alpha) * D[None, :]
        cur = {
            'loc_t': alpha * C[None, :] / f_loc[None, :],
            'tx': off_d / (rate_sel + 1e-9),
            'comp': off_c / f_sel,
            'loc_e': KAPPA_LOCAL * f_loc[None, :] ** 2 * alpha * C[None, :],
            'tx_e': TX_POWER_USER * off_d / (rate_sel + 1e-9),
            'edge_e': KAPPA_EDGE * f_sel ** 2 * off_c,
        }
        # 排队时延 q (A_count, K): Σ_{j 在前同服务器} off_c_j / F
        q = np.zeros((self.A_count, K))
        for i in range(K):
            acc = np.zeros(self.A_count)
            for j in range(K):
                if behind[j, i]:      # j 排在 i 之前
                    acc += off_c[:, j] * (A[:, j] == A[:, i])
            q[:, i] = acc / f_sel[:, i]

        def costs(total, energy, ok):
            return (OMEGA[0] * energy / 1e3 + OMEGA[1] * total / K
                    + PENALTY * (1.0 - ok))

        def evaluate():
            total = np.maximum(cur['loc_t'], cur['tx'] + q + cur['comp'])
            energy = (cur['loc_e'] + cur['tx_e'] + cur['edge_e']
                      + P_IDLE * np.maximum(0.0, total - cur['loc_t']))
            ok = total <= tau[None, :]
            cost = costs(total, energy, ok)
            return cost, total, energy, ok

        cost, total, energy, ok = evaluate()
        J = cost.sum(1)

        # ---- 两轮坐标上升 (正向 + 反向) ----
        for direction in (order, order[::-1]):
            for k in direction:
                a_old = alpha[:, k].copy()
                G = ALPHA_GRID[None, :]                     # (1, nG)
                d_alpha = a_old[:, None] - G                # (A, nG)
                aff = aff_idx[k]
                # 受影响用户的 q 随候选 α 变化 (A, |aff|, nG)
                q_aff = np.zeros((self.A_count, len(aff), len(ALPHA_GRID)))
                for ii, j in enumerate(aff):
                    if j == k:
                        q_aff[:, ii, :] = q[:, k:k + 1]
                    else:
                        q_aff[:, ii, :] = (q[:, j][:, None]
                                           + d_alpha * C[k] / f_sel[:, j:j + 1]
                                           * aff_mask[k][j][:, None])
                # 受影响用户的 α 相关量随候选变化
                a3 = np.zeros((self.A_count, len(aff), len(ALPHA_GRID)))
                for ii, j in enumerate(aff):
                    if j == k:
                        a3[:, ii, :] = G
                    else:
                        a3[:, ii, :] = alpha[:, j:j + 1]
                off_c3 = (1.0 - a3) * C[None, :, None][:, aff, :]
                off_d3 = (1.0 - a3) * D[None, :, None][:, aff, :]
                loc_t3 = a3 * C[None, :, None][:, aff, :] / f_loc[None, aff, None]
                tx3 = off_d3 / (rate_sel[:, aff, None] + 1e-9)
                comp3 = off_c3 / f_sel[:, aff, None]
                loc_e3 = (KAPPA_LOCAL * f_loc[None, aff, None] ** 2
                          * a3 * C[None, :, None][:, aff, :])
                tx_e3 = TX_POWER_USER * tx3
                edge_e3 = KAPPA_EDGE * f_sel[:, aff, None] ** 2 * off_c3
                total3 = np.maximum(loc_t3, tx3 + q_aff + comp3)
                energy3 = (loc_e3 + tx_e3 + edge_e3
                           + P_IDLE * np.maximum(0.0, total3 - loc_t3))
                ok3 = total3 <= tau[None, aff, None]
                cost3 = costs(total3, energy3, ok3)          # (A, |aff|, nG)
                # 旧成本 (受影响用户当前值)
                cost_old = np.stack([cost[:, j] for j in aff], axis=1)  # (A,|aff|)
                J_cand = (J[:, None] - cost_old.sum(1)[:, None]
                          + cost3.sum(1))                    # (A, nG)
                best_c = np.argmin(J_cand, axis=1)           # (A,)
                a_new = ALPHA_GRID[best_c]
                changed = a_new != a_old
                if not changed.any():
                    continue
                # 提交: 更新 alpha / q / cur / cost / J
                alpha[:, k] = a_new
                d_alpha_f = (a_old - a_new) * C[k]
                for ii, j in enumerate(aff):
                    if j == k:
                        continue
                    upd = changed & aff_mask[k][j]
                    q[upd, j] += (d_alpha_f[upd] / f_sel[upd, j])
                off_c[:, k] = (1.0 - a_new) * C[k]
                off_d[:, k] = (1.0 - a_new) * D[k]
                cur['loc_t'][:, k] = a_new * C[k] / f_loc[k]
                cur['tx'][:, k] = off_d[:, k] / (rate_sel[:, k] + 1e-9)
                cur['comp'][:, k] = off_c[:, k] / f_sel[:, k]
                cur['loc_e'][:, k] = (KAPPA_LOCAL * f_loc[k] ** 2
                                      * a_new * C[k])
                cur['tx_e'][:, k] = TX_POWER_USER * cur['tx'][:, k]
                cur['edge_e'][:, k] = (KAPPA_EDGE * f_sel[:, k] ** 2
                                       * off_c[:, k])
                # 受影响用户的成本重算 (用新 q)
                for j in aff:
                    tot = np.maximum(cur['loc_t'][:, j],
                                     cur['tx'][:, j] + q[:, j]
                                     + cur['comp'][:, j])
                    ene = (cur['loc_e'][:, j] + cur['tx_e'][:, j]
                           + cur['edge_e'][:, j]
                           + P_IDLE * np.maximum(0.0, tot - cur['loc_t'][:, j]))
                    okj = tot <= tau[j]
                    cost[:, j] = costs(tot, ene, okj)
                J = cost.sum(1)

        cost, total, energy, ok = evaluate()
        best = int(np.argmin(J))
        sla_mask = p == 3
        sla = float(ok[best][sla_mask].mean()) if sla_mask.any() else float('nan')
        return {
            'J': float(J[best]),
            'energy': float(energy[best].sum() / 1e3),
            'latency': float(total[best].mean()),
            'success_rate': float(ok[best].mean()),
            'sla': sla,
        }


def run_hma_episode(env, runner, refiner, n_steps):
    e_sum = t_sum = suc_sum = sla_sum = 0.0
    for _ in range(n_steps):
        state = env._get_state()
        out_d = runner.infer(state, deterministic=False)
        a_r, s_r = refiner.refine(env, out_d['plan']['alpha'],
                                  out_d['plan']['server'])
        _, _, _, info = env.step(compose_action({'alpha': a_r,
                                                 'server': s_r}, env.K, env.M))
        e_sum += info['energy']; t_sum += info['latency']
        suc_sum += info['success_rate']; sla_sum += info['priority_sla']
    return {'energy': e_sum / n_steps, 'latency': t_sum / n_steps,
            'success_rate': suc_sum / n_steps, 'priority_sla': sla_sum / n_steps}


def main():
    K, M = NUM_USERS, NUM_EDGE_SERVERS
    solver = StateSolver(build_assignments(K, M))
    print(f"组合数 M^K = {solver.A_count}, 网格 {len(ALPHA_GRID)} 档 α, "
          f"{N_SEEDS} seeds × {N_EPISODES} ep × {N_STEPS} 步")

    env = MECEnvironment(num_users=K, num_servers=M, seed=SEED)
    env.reset()
    t0 = time.time()
    r0 = solver.solve(env)
    t1 = time.time()
    print(f"单状态耗时 {t1-t0:.2f}s  (J*={r0['J']:.4f}, "
          f"suc={r0['success_rate']*100:.0f}%)")

    runner = PolicyAgentRunner(model_path=POLICY_PATH)
    refiner = PlanRefiner(omega=(0.5, 0.5))
    rows_ex, rows_hm = [], []
    t_all = time.time()
    for s in range(N_SEEDS):
        for e in range(N_EPISODES):
            env = MECEnvironment(num_users=K, num_servers=M, seed=SEED + s + e)
            env.reset()
            ep_ex = {'J': [], 'energy': [], 'latency': [], 'suc': [], 'sla': []}
            for _ in range(N_STEPS):
                r = solver.solve(env)
                ep_ex['J'].append(r['J'])
                ep_ex['energy'].append(r['energy'])
                ep_ex['latency'].append(r['latency'])
                ep_ex['suc'].append(r['success_rate'])
                ep_ex['sla'].append(r['sla'])
                env.step(env.sample_action())
            env2 = MECEnvironment(num_users=K, num_servers=M, seed=SEED + s + e)
            env2.reset()
            hm = run_hma_episode(env2, runner, refiner, N_STEPS)
            hm['J'] = (0.5 * hm['energy'] + 0.5 * hm['latency']
                       + 1.0 * (1.0 - hm['success_rate']))
            rows_ex.append({kk: float(np.mean(v)) for kk, v in ep_ex.items()})
            rows_hm.append(hm)
            gap = (rows_hm[-1]['J'] - rows_ex[-1]['J']) / rows_ex[-1]['J']
            print(f"  s{s}e{e}: J*={rows_ex[-1]['J']:.4f} "
                  f"(suc {rows_ex[-1]['suc']*100:.0f}%) | "
                  f"HMA J={rows_hm[-1]['J']:.4f} "
                  f"(suc {hm['success_rate']*100:.0f}%) "
                  f"| 差距 {gap*100:+.1f}%  ({time.time()-t_all:.0f}s)")

    def agg(rows, names):
        return {n: {'mean': float(np.mean([r[n] for r in rows])),
                    'std': float(np.std([r[n] for r in rows])),
                    'vals': [r[n] for r in rows]} for n in names}

    names_ex = ['J', 'energy', 'latency', 'suc', 'sla']
    names_hm = ['J', 'energy', 'latency', 'success_rate', 'priority_sla']
    out = {'exhaustive': agg(rows_ex, names_ex),
           'hma_distill': agg(rows_hm, names_hm),
           'config': {'K': K, 'M': M, 'n_episodes': N_SEEDS * N_EPISODES,
                      'n_steps': N_STEPS, 'penalty': PENALTY,
                      'alpha_grid': list(ALPHA_GRID)}}
    with open(os.path.join(RESULTS_DIR, "exhaustive_optimum.json"),
              "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n已保存 -> results/exhaustive_optimum.json")
    print(f"穷举最优: J* = {out['exhaustive']['J']['mean']:.4f}, "
          f"suc = {out['exhaustive']['suc']['mean']*100:.1f}%, "
          f"E = {out['exhaustive']['energy']['mean']:.4f} kJ, "
          f"T = {out['exhaustive']['latency']['mean']:.3f} s")
    print(f"HMA-Distill(同集): J = {out['hma_distill']['J']['mean']:.4f}, "
          f"suc = {out['hma_distill']['success_rate']['mean']*100:.1f}%")
    gap = ((out['hma_distill']['J']['mean'] - out['exhaustive']['J']['mean'])
           / out['exhaustive']['J']['mean'])
    print(f"J 最优性差距 = {gap*100:.1f}%")


if __name__ == "__main__":
    main()

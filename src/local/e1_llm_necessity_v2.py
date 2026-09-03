# -*- coding: utf-8 -*-
"""
================================================================
LLM 必要性专项 v2 (2026-08 强化)
================================================================
同一验证器精化器 (PlanRefiner) 下, 不同种子来源的性能边际:
  LLM种子(HMA-Distill) vs 启发式种子(MPC) vs 随机种子+Refiner

对比场景 (普通 + 6 类逆境):
  normal / link_fail / channel_degrade / dishonest / high_load /
  tight_deadline / multi_fault

新场景设计 (让启发式先验 FAIL, 暴露 LLM 语义先验价值):
  high_load      : 任务量/数据量 ×2 -> 服务器与本地均过载
  tight_deadline : deadline ×0.4 -> 时延约束收紧
  multi_fault    : t=60 叠加 server_fail + link_fail + channel_degrade
统计: 6 seeds × 120 步, 逐方法; 对 LLM vs MPC 的每种子配对做
      Wilcoxon 符号秩检验 (时延 / 成功率).
================================================================
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)
sys.path.insert(0, os.path.join(SRC, "local"))

from config import NUM_USERS, NUM_EDGE_SERVERS, SEED
from environment import MECEnvironment
from agent_runner import HMAAgentRunner
from local.baseline_mpc import MPCBaseline
from local.results_store import Recorder
from local.plan_refiner import PlanRefiner
from local.experiment_common import compose_action

K, M = NUM_USERS, NUM_EDGE_SERVERS
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results")
RES = os.path.normpath(RES)
N_SEEDS = 10
N_STEPS = 120

SCENARIOS = ["normal", "link_fail", "channel_degrade", "dishonest",
             "high_load", "tight_deadline", "multi_fault"]


def make_env(sd, scenario, ep=0):
    env = MECEnvironment(num_users=K, num_servers=M, seed=SEED + sd + ep)
    if scenario == "link_fail":
        env.set_link_error(3, 1, 0.9); env.set_link_error(5, 2, 0.9)
    elif scenario == "channel_degrade":
        env.channels[:, 3] *= 0.15
    elif scenario == "dishonest":
        env.tasks[0]['priority'] = 3
        env.tasks[1]['C'] *= 3.0
    elif scenario == "high_load":
        for t in env.tasks:
            t['C'] *= 2.0
            t['D'] *= 2.0
    elif scenario == "tight_deadline":
        for t in env.tasks:
            t['tau'] *= 0.4
    return env


def apply_mid_perturb(env, i):
    """multi_fault 即发扰动: t=60 叠加服务器宕机+链路故障+信道劣化."""
    if i == 60:
        env.f_edge[0] = 1.0                      # server0 宕机
        env.set_link_error(3, 1, 0.9)
        env.set_link_error(5, 2, 0.9)
        env.channels[:, 3] *= 0.2                # server3 信道劣化


class RandomSeedRefiner:
    def __init__(self):
        self.refiner = PlanRefiner(omega=(0.5, 0.5))
    def predict(self, st, env):
        alpha = np.random.uniform(0.05, 0.95, K)
        server = np.random.randint(0, M, K)
        a, s = self.refiner.refine(env, alpha, server)
        act = np.zeros(2 * K, np.float32)
        act[0::2] = np.clip(a, 0.01, 1.0); act[1::2] = (np.clip(s, 0, M - 1) + 0.5) / M
        return act


def run(predictor, env, n_steps=N_STEPS, runner=None, scenario="normal"):
    E, T, S, SL = [], [], [], []
    for i in range(n_steps):
        if scenario == "multi_fault":
            apply_mid_perturb(env, i)
        st = env._get_state()
        if runner is not None:
            out = runner.run_step(state=st, agents_reuse=True)
            a = compose_action(out['plan'], K, M)
        else:
            a = predictor.predict(st, env)
        ns, _, d, info = env.step(a)
        E.append(info['energy']); T.append(info['latency'])
        S.append(info['success_rate']); SL.append(info['priority_sla'])
        if d:
            break
    return dict(E=float(np.mean(E)), T=float(np.mean(T)),
                suc=float(np.mean(S)), sla=float(np.mean(SL)))


def build(sc, env):
    return {
        "LLM种子(HMA)": lambda: (HMAAgentRunner(env=env, mode="Distill"), None),
        "启发式种子(MPC)": lambda: (MPCBaseline(), env),
        "随机种子+Refiner": lambda: (RandomSeedRefiner(), env),
    }


def main():
    from scipy.stats import wilcoxon
    builders = {
        "LLM种子(HMA)": lambda env: (HMAAgentRunner(env=env, mode="Distill"), "runner"),
        "MPC(启发式)": lambda env: (MPCBaseline(), "pred"),
        "Random+Refiner": lambda env: (RandomSeedRefiner(), "pred"),
    }
    summary = {}
    for sc in SCENARIOS:
        print(f"\n===== 场景 {sc} =====")
        summary[sc] = {}
        per_method = {label: [] for label in builders}
        _rec = Recorder("e11_necessity")
        for sd in range(N_SEEDS):
            env = make_env(sd, sc)
            for label, mk in builders.items():
                obj, kind = mk(env)
                if kind == "runner":
                    r = run(None, env, runner=obj, scenario=sc)
                else:
                    r = run(obj, env, scenario=sc)
                per_method[label].append(r)
                _rec.add(method=label, seed=sd, episode=ep, metrics=r)
        for label, rows in per_method.items():
            mean = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
            std = {k: float(np.std([r[k] for r in rows])) for k in rows[0]}
            summary[sc][label] = {"mean": mean, "std": std}
            print(f"  {label:16s} E={mean['E']:.3f} T={mean['T']:.3f}±{std['T']:.3f} "
                  f"suc={mean['suc']:.1%} sla={mean['sla']:.1%}")
        # Wilcoxon: LLM vs MPC 配对 (per-seed)
        stats = {}
        aT = [r['T'] for r in per_method["LLM种子(HMA)"]]
        bT = [r['T'] for r in per_method["MPC(启发式)"]]
        aS = [r['suc'] for r in per_method["LLM种子(HMA)"]]
        bS = [r['suc'] for r in per_method["MPC(启发式)"]]
        def _w(x, y):
            if len(set(np.round(x,6))) < 2 or len(set(np.round(y,6))) < 2:
                return float('nan'), 1.0
            try:
                w, p = wilcoxon(x, y)
                return float(w), float(p)
            except Exception:
                return float('nan'), 1.0
        wt, pt = _w(aT, bT); ws, ps = _w(aS, bS)
        stats = {
            "llm_vs_mpc_latency_gain_pct": float(np.mean(aT) / np.mean(bT) - 1.0),
            "llm_vs_mpc_suc_gain_pp": float((np.mean(aS) - np.mean(bS)) * 100),
            "wilcoxon_latency_p": float(pt),
            "wilcoxon_success_p": float(ps),
        }
        summary[sc]["_stats"] = stats
        print(f"  >> LLM vs MPC: T增益={stats['llm_vs_mpc_latency_gain_pct']*100:+.1f}% "
              f"(p={stats['wilcoxon_latency_p']:.4f}), "
              f"suc增益={stats['llm_vs_mpc_suc_gain_pp']:+.1f}pp (p={stats['wilcoxon_success_p']:.4f})")

    os.makedirs(RES, exist_ok=True)
    json.dump(summary, open(f"{RES}/e1_llm_necessity_v2.json", "w"),
              indent=2, ensure_ascii=False)
    print(f"\n  保存 -> {RES}/e1_llm_necessity_v2.json")


if __name__ == "__main__":
    main()

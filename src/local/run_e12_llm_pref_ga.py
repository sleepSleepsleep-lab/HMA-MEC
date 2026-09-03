# -*- coding: utf-8 -*-
"""
================================================================
B12 (2026-08): CORE-LEO 路线复现 — LLM 偏好推断 + 遗传调度器
================================================================
对应 introduction 对 refCORELEO 的定位:"LLM 推偏好 + 传统求解器"。

基线: 每 episode 开始由 LLM 从全局状态推断能耗-时延偏好权重
      (w_e, w_t), 随后每步以该偏好为适应度目标用遗传算法 GA
      求解卸载方案 (GA 直接产决策, LLM 不参与逐方案推理)。

对比 (公平目标 J=0.5E+0.5T 评估):
  HMA-Distill : LLM 辩论语义压缩 + 验证器精化 (本文)
  GA-fixed    : 固定 ω=(0.5,0.5) 的离线遗传搜索
  LLMPref+GA  : B12 (LLM 推断偏好 + 遗传调度)  [CORE-LEO 路线]
  MPC         : 启发式种子 + 精化 (去 LLM 最小对照)

规模: 2 seeds × 6 ep × 150 步 (GA 每步 ~0.75s, 控制总时长)
依赖: 本地 vLLM (每 episode 1 次偏好推断)。
================================================================
"""
import os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)
sys.path.insert(0, os.path.join(SRC, "local"))

from config import NUM_USERS, NUM_EDGE_SERVERS, SEED, RESULTS_DIR, CHECKPOINT_DIR
from environment import MECEnvironment
from agent_runner import HMAAgentRunner
from local.baseline_mpc import MPCBaseline
from local.results_store import Recorder
from local.ga_baseline import GAOffloadBaseline as GABaseline
from local.experiment_common import compose_action

K, M = NUM_USERS, NUM_EDGE_SERVERS
POLICY_PATH = os.path.join(CHECKPOINT_DIR, "distilled_policy.pth")
N_SEEDS, N_EPISODES, N_STEPS = 2, 6, 150
OUT_JSON = os.path.join(RESULTS_DIR, "e12_llm_pref_ga.json")

SYSTEM_PREF = (
    "You are an MEC orchestrator. Given the current network state, "
    "infer two preference weights w_e (energy) and w_t (latency), "
    "w_e + w_t = 1, that best match the situation: heavier load / "
    "tight deadlines -> favor latency (higher w_t); energy-constrained "
    "or light load -> favor energy. Output ONLY JSON: "
    '{"w_e": 0.0-1.0, "w_t": 0.0-1.0}, summing to 1.')


def state_summary(env):
    load = np.asarray(env.server_load)
    fe = np.asarray(env.f_edge)
    ch = np.abs(np.asarray(env.channels)) ** 2
    taus = [t['tau'] for t in env.tasks]
    C = [t['C'] for t in env.tasks]
    return (f"servers={M} users={K} "
            f"load/edge_ratio={[round(float(load[m])/max(float(fe[m]),1e9),3) for m in range(M)]} "
            f"best_channel_mean={float(ch.max(1).mean()):.4f} "
            f"mean_C={float(np.mean(C)):.3e} mean_tau={float(np.mean(taus)):.3f}")


def infer_omega(llm, env, sd):
    try:
        r = llm.chat_json(SYSTEM_PREF, state_summary(env))
        we = float(r['w_e']); wt = float(r['w_t'])
        s = we + wt
        if s <= 0:
            return (0.5, 0.5)
        return (we / s, wt / s)
    except Exception:
        return (0.5, 0.5)


def run_ep(env, kind, obj):
    E, T, S, SL = [], [], [], []
    for _ in range(N_STEPS):
        st = env._get_state()
        if kind == "runner":
            out = obj.run_step(state=st, agents_reuse=True)
            a = compose_action(out['plan'], K, M)
        else:
            a = obj.predict(st, env)
        ns, _, d, info = env.step(a)
        E.append(info['energy']); T.append(info['latency'])
        S.append(info['success_rate']); SL.append(info['priority_sla'])
        if d:
            break
    return dict(E=float(np.mean(E)), T=float(np.mean(T)),
                suc=float(np.mean(S)), sla=float(np.mean(SL)))


def main():
    from llm_client import get_llm_client
    llm = get_llm_client()
    print("=" * 62)
    print("  B12 (CORE-LEO 路线): LLM 偏好推断 + 遗传调度 vs HMA")
    print("=" * 62)

    methods = {
        "HMA-Distill": "runner",
        "GA-fixed": "ga",
        "LLMPref+GA": "ga_pref",
        "MPC": "pred",
    }
    summary = {mname: [] for mname in methods}
    _rec = Recorder("e12")
    for sd in range(N_SEEDS):
        for ep in range(N_EPISODES):
            base_seed = SEED + sd + ep
            env_h = MECEnvironment(num_users=K, num_servers=M, seed=base_seed)
            env_h.reset()
            omega_pref = infer_omega(llm, env_h, sd)
            hma = HMAAgentRunner(env=env_h, mode="Distill",
                                 policy_path=POLICY_PATH, agents=None)
            for mname in ["HMA-Distill", "GA-fixed", "LLMPref+GA", "MPC"]:
                kind = methods[mname]
                if kind == "runner":
                    r = run_ep(env_h, "runner", hma)
                else:
                    e2 = MECEnvironment(num_users=K, num_servers=M, seed=base_seed)
                    e2.reset()
                    if kind == "ga":
                        obj = GABaseline(omega=(0.5, 0.5))
                    elif kind == "ga_pref":
                        obj = GABaseline(omega=omega_pref)
                    else:
                        obj = MPCBaseline()
                    r = run_ep(e2, "pred", obj)
                summary[mname].append(r)
                _rec.add(method=mname, seed=sd, episode=ep, metrics=r,
                         omega_pref=list(omega_pref))
            print(f"  seed{sd} ep{ep} ω_pref=({omega_pref[0]:.2f},{omega_pref[1]:.2f}) 完成")

    _rec.close()
    out = {}
    for mname, rows in summary.items():
        mean = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        std = {k: float(np.std([r[k] for r in rows])) for k in rows[0]}
        out[mname] = {"mean": mean, "std": std}
        print(f"  {mname:12s} E={mean['E']:.4f} T={mean['T']:.4f} "
              f"suc={mean['suc']:.1%} sla={mean['sla']:.1%}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    json.dump(out, open(OUT_JSON, "w"), indent=2, ensure_ascii=False)
    print(f"\n  保存 -> {OUT_JSON}")


if __name__ == "__main__":
    main()

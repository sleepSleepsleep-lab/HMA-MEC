# -*- coding: utf-8 -*-
"""
================================================================
E15 (2026-08): 多智能体辩论 vs 单 LLM 直接决策 —— 同一验证器精化器下
================================================================
回答审稿关键问题: "优势来自验证器精化与 LLM 语义先验，那么
多智能体五轮辩论（CW-Debate）相对单次 LLM 直接决策有何独立增益?"

三种种子生成器（spawner），统一经同一 PlanRefiner 精化后决策:
  A) Debate     : 完整 CW-Debate 五轮多智能体辩论 (65 次 LLM 调用/步)
  B) SingleLLM  : 单次 LLM 调用直接输出卸载方案 (1 次调用/步)
  C) Heuristic  : 固定 alpha=0.5 + 信道最优 (0 次调用/步)  [MPC]

场景: normal / multi_fault(双故障,t=6) / high_load / dishonest
指标: episode 时延/成功率/SLA (multi_fault 取扰动后), 每步 LLM 调用数。
依赖: 本地 vLLM (A,B 需调用; 每场景 ~32 辩论步 ≈ 35 分钟)。
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
from local.plan_refiner import PlanRefiner
from local.results_store import Recorder
from local.experiment_common import compose_action

K, M = NUM_USERS, NUM_EDGE_SERVERS
N_SEEDS = 10          # P1-2: 由 2 扩至 10 (与 E11 同口径)
N_STEPS = 120         # P1-2: 由 16 扩至 120 (与 E11 同口径)
PERTURB_STEP = 6
FAULT = [0, 1]
OUT_JSON = os.path.join(RESULTS_DIR, "e15_debate_vs_single.json")

SYSTEM_PLAN = (
    "You are an MEC task-offloading orchestrator. Given the current system "
    "state, decide for each user k the fraction alpha_k (0=all offload to "
    "edge, 1=all local) and the server m_k in 0..%d. "
    "Favor low latency for tight deadlines. " % (M - 1) +
    'Output ONLY JSON: {"alpha": [.. 8 floats ..], "server": [.. 8 ints ..]}')


def make_env(sd, scenario):
    env = MECEnvironment(num_users=K, num_servers=M, seed=SEED + sd)
    env.reset()
    if scenario == "high_load":
        for t in env.tasks:
            t['C'] *= 2.0
            t['D'] *= 2.0
    elif scenario == "dishonest":
        env.tasks[0]['priority'] = 3
        env.tasks[1]['C'] *= 3.0
    return env


def apply_mid(scenario, env, i):
    if scenario == "multi_fault" and i == PERTURB_STEP:
        for m in FAULT:
            env.f_edge[m] = 1.0


class Spawners:
    def __init__(self, env, llm):
        self.env = env
        self.llm = llm
        self.refiner = PlanRefiner(omega=(0.5, 0.5))
        self.full = HMAAgentRunner(env=env, mode="FullLLM", llm=llm,
                                   agents=None, verbose=False, use_refiner=False)
        self.calls = 0

    def seed_debate(self, st):
        out = self.full.run_step(state=st, agents_reuse=False)
        self.calls += 65                     # (K+M+2)*R_max 次 LLM 调用
        return np.asarray(out['plan']['alpha'], dtype=float), \
               np.asarray(out['plan']['server'], dtype=int)

    def seed_single(self, st):
        txt = self.env.state_to_text(state=st)
        r = self.llm.chat_json(SYSTEM_PLAN, txt)
        self.calls += 1
        a = np.asarray(r.get('alpha', []), dtype=float)
        s = np.asarray(r.get('server', []), dtype=int)
        if len(a) != K or len(s) != K:
            # 解析失败 -> 退回启发式
            a = np.full(K, 0.5); s = np.abs(self.env.channels).argmax(1)
        a = np.clip(a, 0.01, 1.0)
        s = np.clip(s, 0, M - 1).astype(int)
        return a, s

    def seed_heuristic(self, st):
        return np.full(K, 0.5), np.abs(self.env.channels).argmax(1).astype(int)


def run_ep(env, scenario, spawner, kind):
    E, T, S, SL = [], [], [], []
    for i in range(N_STEPS):
        apply_mid(scenario, env, i)
        st = env._get_state()
        if kind == 'Debate':
            a, s = spawner.seed_debate(st)
        elif kind == 'SingleLLM':
            a, s = spawner.seed_single(st)
        else:
            a, s = spawner.seed_heuristic(st)
        ar, sr = spawner.refiner.refine(env, a.copy(), np.asarray(s, int).copy())
        act = np.zeros(2 * K, np.float32)
        act[0::2] = np.clip(ar, 0.01, 1.0)
        act[1::2] = (np.clip(sr, 0, M - 1) + 0.5) / M
        ns, _, d, info = env.step(act)
        E.append(info['energy']); T.append(info['latency'])
        S.append(info['success_rate']); SL.append(info['priority_sla'])
        if d:
            break
    E, T, S, SL = map(np.array, (E, T, S, SL))
    sl = slice(PERTURB_STEP if scenario == "multi_fault" else 0, None)
    return dict(E=float(E[sl].mean()), T=float(T[sl].mean()),
                suc=float(S[sl].mean()), sla=float(SL[sl].mean()),
                calls=spawner.calls)


def main():
    import threading
    import concurrent.futures
    from llm_client import get_llm_client
    llm = get_llm_client()
    # 本地 vLLM 无速率限制: 关闭客户端节流以最大化 8 线程并行吞吐
    llm._respect_rate_limit = lambda: None
    print("=" * 64)
    print("  E15: 多智能体辩论 vs 单 LLM vs 启发式（同一验证器精化器）")
    print("  [P1-2 扩样] 种子级并行: 10 seeds x 120 步 x 4 场景")
    print("=" * 64)
    scenarios = ["normal", "multi_fault", "high_load", "dishonest"]
    kinds = ["Debate", "SingleLLM", "Heuristic"]
    out = {}
    _lock = threading.Lock()
    _rec = Recorder("e15")

    def run_one(sc, kind, sd):
        env = make_env(sd, sc)
        sp = Spawners(env, llm)
        r = run_ep(env, sc, sp, kind)
        r['calls'] = sp.calls
        with _lock:
            _rec.add(method=kind, seed=sd, episode=None,
                     metrics={k: r[k] for k in ('E', 'T', 'suc', 'sla')},
                     scenario=sc)
        return (sc, kind, sd, r)

    tasks = [(sc, kind, sd)
             for sc in scenarios for kind in kinds for sd in range(N_SEEDS)]
    # 快任务优先提交: Heuristic(0次LLM) > SingleLLM(1次/步) > Debate(65次/步),
    # 使监控数据尽早可用; 总耗时由 Debate 任务主导, 不受顺序影响
    tasks.sort(key=lambda t: 0 if t[1] == 'Heuristic'
               else (1 if t[1] == 'SingleLLM' else 2))
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
        futs = [ex.submit(run_one, sc, kind, sd) for sc, kind, sd in tasks]
        for f in concurrent.futures.as_completed(futs):
            sc, kind, sd, r = f.result()
            out.setdefault(sc, {}).setdefault(kind, []).append(r)
            print(f"  [{sc:10s}] {kind:10s} seed{sd:2d}: "
                  f"T={r['T']:.4f} suc={r['suc']:.1%} calls={r['calls']}",
                  flush=True)

    for sc in scenarios:
        print(f"\n  -- 场景 {sc} --")
        for kind in kinds:
            rows = out[sc].get(kind, [])
            if not rows:
                continue
            mean = {k: float(np.mean([r[k] for r in rows]))
                    for k in ('E', 'T', 'suc', 'sla')}
            mean['calls_per_step'] = float(np.mean([r['calls'] for r in rows])) / N_STEPS
            out[sc][kind] = mean
            print(f"     {kind:10s} E={mean['E']:.4f} T={mean['T']:.4f} "
                  f"suc={mean['suc']:.1%} sla={mean['sla']:.1%} "
                  f"calls/step={mean['calls_per_step']:.0f}")
    _rec.close()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    json.dump(out, open(OUT_JSON, "w"), indent=2, ensure_ascii=False)
    print(f"\n  保存 -> {OUT_JSON}")


if __name__ == "__main__":
    main()

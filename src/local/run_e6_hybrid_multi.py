# -*- coding: utf-8 -*-
"""
================================================================
E13b (2026-08): Hybrid 触发的正例 —— 多服务器并发故障
================================================================
单点故障下验证器精化（逐用户贪心）已充分兜底 (E13: Distill≈Hybrid-F)。
本实验制造"精化可能失效"的场景: 4 台服务器中 2 台 (server0,1) 在
t=100 同时宕机, 剩余负载向 server2,3 挤兑 -> 逐用户贪心精化易陷于
局部次优（反复尝试故障服务器或超载服务器）, 而 Hybrid 强制触发
FullLLM 从全局语义视角重新协商分配, 预期带来可测增益。

变体: A) Distill  B) Hybrid-Forced (显式异常检测触发)  C) MPC
指标: 扰动后 100 步 T/suc/sla; 记录触发次数。
依赖: 本地 vLLM (每 ep 触发 1 次 FullLLM ~60s)。
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
from local.plan_refiner import PlanRefiner
from local.experiment_common import compose_action

K, M = NUM_USERS, NUM_EDGE_SERVERS
POLICY_PATH = os.path.join(CHECKPOINT_DIR, "distilled_policy.pth")
PERTURB_STEP = 100
N_STEPS = 200
N_EPISODES = 3
FAULT_SERVERS = [0, 1]                  # 双服务器并发宕机
OUT_JSON = os.path.join(RESULTS_DIR, "e6_hybrid_multi.json")


class ForcedHybridMulti:
    def __init__(self, env, llm):
        self.refiner = PlanRefiner(omega=(0.5, 0.5))
        self.distill = HMAAgentRunner(env=env, mode="Distill",
                                      policy_path=POLICY_PATH, agents=None,
                                      use_refiner=False)
        self.full = HMAAgentRunner(env=env, mode="FullLLM", llm=llm,
                                   agents=None, verbose=False, use_refiner=False)
        self.triggered = False
        self.n_trigger = 0
        self.trigger_steps = []

    def _anomaly(self, env):
        low = np.where(np.asarray(env.f_edge) < 1e6)[0]
        return low.tolist()

    def act(self, st, env, step_idx):
        if not self.triggered:
            low = self._anomaly(env)
            if len(low) >= 2:                      # 并发多故障才触发
                t0 = time.time()
                out = self.full.run_step(state=st, agents_reuse=False)
                plan = out['plan']
                self.triggered = True
                self.n_trigger += 1
                self.trigger_steps.append(dict(step=step_idx,
                                               elapsed_s=round(time.time() - t0, 1),
                                               servers=low))
            else:
                od = self.distill.run_step(state=st, agents_reuse=True)
                plan = od['plan']
        else:
            od = self.distill.run_step(state=st, agents_reuse=True)
            plan = od['plan']
        a, s = self.refiner.refine(env, np.asarray(plan['alpha']).copy(),
                                   np.asarray(plan['server'], dtype=int).copy())
        act = np.zeros(2 * K, np.float32)
        act[0::2] = np.clip(a, 0.01, 1.0)
        act[1::2] = (np.clip(s, 0, M - 1) + 0.5) / M
        return act


def run_ep(sd, variant, llm):
    env = MECEnvironment(num_users=K, num_servers=M, seed=SEED + sd)
    env.reset()
    E, T, S, SL = [], [], [], []
    if variant == "Distill":
        runner = HMAAgentRunner(env=env, mode="Distill",
                                policy_path=POLICY_PATH, agents=None)
        actor = None
    elif variant == "Hybrid-Forced":
        actor = ForcedHybridMulti(env, llm)
        runner = None
    else:
        actor = MPCBaseline()
        runner = None
    for i in range(N_STEPS):
        if i == PERTURB_STEP:
            for m in FAULT_SERVERS:
                env.f_edge[m] = 1.0
        st = env._get_state()
        if runner is not None:
            out = runner.run_step(state=st, agents_reuse=True)
            a = compose_action(out['plan'], K, M)
        else:
            a = (actor.act(st, env, i) if hasattr(actor, 'act')
                 else actor.predict(st, env))
        ns, _, d, info = env.step(a)
        E.append(info['energy']); T.append(info['latency'])
        S.append(info['success_rate']); SL.append(info['priority_sla'])
        if d:
            break
    return dict(E=np.array(E), T=np.array(T), S=np.array(S), SL=np.array(SL),
                triggers=getattr(actor, 'trigger_steps', None))


def main():
    from llm_client import get_llm_client
    llm = get_llm_client()
    print("=" * 62)
    print(f"  E13b: 双服务器并发故障 (server{FAULT_SERVERS} @ t=100) — Hybrid 触发正例")
    print("=" * 62)
    out = {}
    _rec = Recorder("e6_hybrid_multi")
    for var in ["Distill", "Hybrid-Forced", "MPC"]:
        recs = []
        total_trig = 0
        for sd in range(N_EPISODES):
            r = run_ep(sd, var, llm)
            post = dict(E=float(r['E'][PERTURB_STEP:].mean()),
                        T=float(r['T'][PERTURB_STEP:].mean()),
                        suc=float(r['S'][PERTURB_STEP:].mean()),
                        sla=float(r['SL'][PERTURB_STEP:].mean()))
            trig = r['triggers'] or []
            total_trig += len(trig)
            for t in trig:
                print(f"    [seed{sd}] 触发@step{t['step']} 耗时{t['elapsed_s']:.0f}s 故障={t['servers']}")
            recs.append(dict(post=post))
            _rec.add(method=var, seed=sd, episode=None, metrics=post)
            print(f"    seed{sd} post: E={post['E']:.4f} T={post['T']:.3f} "
                  f"suc={post['suc']:.1%} sla={post['sla']:.1%}")
        agg = {k: float(np.mean([r['post'][k] for r in recs]))
               for k in ('E', 'T', 'suc', 'sla')}
        agg['n_triggers_total'] = total_trig
        out[var] = agg
        print(f"  => [{var}] post T={agg['T']:.3f} suc={agg['suc']:.1%} "
              f"sla={agg['sla']:.1%} 触发={total_trig}次")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    _rec.close()
    json.dump(out, open(OUT_JSON, "w"), indent=2, ensure_ascii=False)
    print(f"  保存 -> {OUT_JSON}")


if __name__ == "__main__":
    main()

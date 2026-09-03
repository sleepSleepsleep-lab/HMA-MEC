# -*- coding: utf-8 -*-
"""
================================================================
M1 补强 (2026-08): 显式异常检测 + Hybrid 强制触发 (server_fail)
================================================================
目的: 让 Hybrid 的"困难状态在线辩论兜底"真正发生并量化其价值。
此前 Hybrid 触发率 0.0% (c_min>tau_low 不触发)。本实验注入
服务器宕机 (f_edge[0]->0 @ t=100) 并对比三种变体扰动后性能:

  A. HMA-Distill  : 无兜底 (仅蒸馏先验+精化)           [references E6]
  B. Hybrid-F      强制触发 Hybrid: 异常检测(服务容量骤降)
                    → 在线 FullLLM 辩论重新协商 → 精化
  C. MPC           : 去 LLM 的启发式种子+精化对照

指标: 扰动后 (t>100) energy / latency / success / SLA;
      并记录 B 的触发次数与触发后单步时延 (证明 Hybrid 真实发生)。

依赖: 本地 vLLM。每 episode 扰动后触发 1 次 FullLLM (~60s)。
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
from local.results_store import Recorder

K, M = NUM_USERS, NUM_EDGE_SERVERS
POLICY_PATH = os.path.join(CHECKPOINT_DIR, "distilled_policy.pth")
PERTURB_STEP = 100
N_STEPS = 200
N_EPISODES = 3
OUT_JSON = os.path.join(RESULTS_DIR, "e6_hybrid_trigger.json")


class ForcedHybridAgent:
    """蒸馏 + 显式异常检测: 检测到服务器容量骤降 -> 强制在线 FullLLM 辩论.

    只在线路故障被检测到时触发一次, 随后各步沿用
    (重新协商后的初始方案 + 逐用户验证器精化).
    """
    def __init__(self, env, llm):
        self.refiner = PlanRefiner(omega=(0.5, 0.5))
        self.distill = HMAAgentRunner(env=env, mode="Distill",
                                      policy_path=POLICY_PATH, agents=None,
                                      use_refiner=False)
        self.full = HMAAgentRunner(env=env, mode="FullLLM", llm=llm,
                                   agents=None, verbose=False,
                                   use_refiner=False)
        self.triggered = False
        self.n_trigger = 0
        self.trigger_steps = []

    def _anomaly(self, env):
        # 服务器计算容量骤降 (宕机/断电) 检测
        low = np.where(np.asarray(env.f_edge) < 1e6)[0]
        return len(low) > 0, low

    def act(self, st, env, step_idx):
        if not self.triggered:
            fault, low = self._anomaly(env)
            if fault:
                t0 = time.time()
                out = self.full.run_step(state=st, agents_reuse=False)
                dt = time.time() - t0
                plan = out['plan']
                self.triggered = True
                self.n_trigger += 1
                self.trigger_steps.append(dict(step=step_idx,
                                               elapsed_s=round(dt, 1),
                                               servers=low.tolist()))
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


def run_episode(sd, variant, llm=None):
    env = MECEnvironment(num_users=K, num_servers=M, seed=SEED + sd)
    env.reset()
    energy, lat, suc, sla = [], [], [], []
    if variant == "Distill":
        runner = HMAAgentRunner(env=env, mode="Distill",
                                policy_path=POLICY_PATH, agents=None)
        actor = None
    elif variant == "Hybrid-Forced":
        actor = ForcedHybridAgent(env, llm)
        runner = None
    else:  # MPC
        actor = MPCBaseline()
        runner = None

    for i in range(N_STEPS):
        if i == PERTURB_STEP:
            env.f_edge[0] = 1.0                      # 服务器 0 宕机
        st = env._get_state()
        if runner is not None:
            out = runner.run_step(state=st, agents_reuse=True)
            a = compose_action(out['plan'], K, M)
        else:
            a = (actor.act(st, env, i) if hasattr(actor, 'act')
                 else actor.predict(st, env))
        ns, _, d, info = env.step(a)
        energy.append(info['energy']); lat.append(info['latency'])
        suc.append(info['success_rate']); sla.append(info['priority_sla'])
        if d:
            break
    return dict(energy=np.array(energy), latency=np.array(lat),
                success=np.array(suc), sla=np.array(sla),
                triggers=getattr(actor, 'trigger_steps', None))


def main():
    from llm_client import get_llm_client
    llm = get_llm_client()
    print("=" * 60)
    print("  M1 补强: 异常检测 + Hybrid 强制触发 (server_fail @ t=100)")
    print("=" * 60)
    variants = ["Distill", "Hybrid-Forced", "MPC"]
    out = {}
    _rec = Recorder("e6_hybrid_trigger")
    for var in variants:
        print(f"\n  -- 变体 {var} --")
        recs = []
        for sd in range(N_EPISODES):
            r = run_episode(sd, var, llm)
            pre = dict(E=float(r['energy'][:PERTURB_STEP].mean()),
                       T=float(r['latency'][:PERTURB_STEP].mean()),
                       suc=float(r['success'][:PERTURB_STEP].mean()),
                       sla=float(r['sla'][:PERTURB_STEP].mean()))
            post = dict(E=float(r['energy'][PERTURB_STEP:].mean()),
                        T=float(r['latency'][PERTURB_STEP:].mean()),
                        suc=float(r['success'][PERTURB_STEP:].mean()),
                        sla=float(r['sla'][PERTURB_STEP:].mean()))
            trig = None
            if r['triggers'] is not None:
                trig = r['triggers']
                for t in trig:
                    print(f"    [seed{sd}] 触发 @step{t['step']} 耗时{t['elapsed_s']:.0f}s 故障服务器={t['servers']}")
            recs.append(dict(pre=pre, post=post, triggers=trig))
            _rec.add(method=var, seed=sd, episode=None, metrics=post,
                     n_triggers=len(trig))
            print(f"    seed{sd}  post: E={post['E']:.4f} T={post['T']:.3f} "
                  f"suc={post['suc']:.1%} sla={post['sla']:.1%}")
        # 汇总 (3 ep 平均)
        agg = {}
        for key in ('E', 'T', 'suc', 'sla'):
            agg['post_' + key] = float(np.mean([r['post'][key] for r in recs]))
            agg['pre_' + key] = float(np.mean([r['pre'][key] for r in recs]))
        n_trig = sum(len(r['triggers'] or []) for r in recs)
        agg['n_triggers_total'] = n_trig
        out[var] = agg
        print(f"  => [{var}] post T={agg['post_T']:.3f} suc={agg['post_suc']:.1%} "
              f"sla={agg['post_sla']:.1%} 触发={n_trig}次")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    _rec.close()
    json.dump(out, open(OUT_JSON, "w"), indent=2, ensure_ascii=False)
    print(f"\n  保存 -> {OUT_JSON}")


if __name__ == "__main__":
    main()

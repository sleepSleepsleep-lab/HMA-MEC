# -*- coding: utf-8 -*-
"""
================================================================
E19 (2026-08): 真实业务 trace 泛化验证 —— Milano 基站流量
================================================================
在 E18 统计分布漂移的基础上, 用真实城市蜂窝流量记录 (Telecom Italia
Big Data Challenge, 米兰 100×100 网格, 2013-11-01 至 11-07, 每小时
Internet 流量, Barlacchi et al., Scientific Data 2015) 驱动任务负载:

  - 全局负载系数 λ(t) = I(t)/mean(I) (全网聚合的 10 分钟流量序列,
    以小时为一步, 7 天共 168 步), clip 到 [0.3, 3.0];
  - 任务数据量 D_k(t) = D_base_k × λ(t), 模拟真实城市级"高峰/低谷"
    负载周期 (峰均比约 7.3, 夜间低谷 / 白天高峰);
  - 每 episode 为完整 7 天 (168 步), 2 组独立种子。

方法: HMA-Distill (现有 K8 策略不重训) / MPC / GA / Greedy,
对照: 同结构无调制的均匀任务 (uniform), 报告成功率保持率。
================================================================
"""
import os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)
sys.path.insert(0, os.path.join(SRC, "local"))

from config import (NUM_USERS, NUM_EDGE_SERVERS, RESULTS_DIR, SEED,
                    CHECKPOINT_DIR, TASK_CYCLES_PER_BIT,
                    TASK_DEADLINE_MIN, TASK_DEADLINE_MAX, TASK_PRIORITY_PROB)
from environment import MECEnvironment
from agent_runner import HMAAgentRunner
from local.ga_baseline import GAOffloadBaseline as GABaseline
from local.baseline_mpc import MPCBaseline
from local.experiment_common import compose_action
from local.results_store import Recorder

K, M = NUM_USERS, NUM_EDGE_SERVERS
POLICY_PATH = os.path.join(CHECKPOINT_DIR, "distilled_policy.pth")
TRACE_PATH = os.path.join(RESULTS_DIR, "milano_data", "trace_internet.npy")
OUT_JSON = os.path.join(RESULTS_DIR, "e19_trace.json")

N_PHASES = 8             # 8 个均匀起始相位 (覆盖一周各时段) × 独立种子
LAM_MIN, LAM_MAX = 0.3, 3.0


def load_lam():
    """全局负载系数序列: 全网每小时流量归一化, clip 后重归一化.
    最终 mean=1 (与 uniform 平均负载一致), 仅保留真实时间波动模式."""
    t = np.load(TRACE_PATH)                 # (7*144, 10000)
    g = t.sum(axis=1)                       # 每 10 分钟全网流量 (1008,)
    # 每 6 槽聚合为小时 (每小时一个数据点, 其余槽为 0 的时间戳占位)
    hourly = g.reshape(-1, 6).sum(axis=1)
    lam = np.clip(hourly / max(hourly.mean(), 1e-9), LAM_MIN, LAM_MAX)
    lam = lam / max(lam.mean(), 1e-9)       # 重归一化: 平均负载与 uniform 对齐
    return lam.astype(np.float32)           # 168 步 (7 天)


LAM = load_lam()


class TraceEnv(MECEnvironment):
    """真实 trace 负载: 全局负载系数调制任务数据量."""

    def __init__(self, use_trace=True, seed=0, start_phase=0):
        self.use_trace = use_trace
        self._start_phase = start_phase % len(LAM)
        self._t = self._start_phase
        super().__init__(num_users=K, num_servers=M, seed=seed)

    def _modulate(self):
        # 用基准任务 × λ(t) 覆盖 (非累积), 每步反映真实负载周期
        if self.use_trace:
            lam = float(LAM[self._t % len(LAM)])
            for k in range(self.K):
                b = self._base_tasks[k]
                self.tasks[k]["D"] = b["D"] * lam
                self.tasks[k]["C"] = b["C"] * lam
        else:
            for k in range(self.K):
                b = self._base_tasks[k]
                self.tasks[k]["D"] = b["D"]
                self.tasks[k]["C"] = b["C"]

    def reset(self):
        self._t = self._start_phase
        st = super().reset()
        self._base_tasks = [dict(t) for t in self.tasks]
        self._modulate()
        return st

    def step(self, action, intrinsic_reward_fn=None):
        self._t += 1
        self._modulate()
        return super().step(action, intrinsic_reward_fn)


def run_env(env, kind, obj, n_steps):
    E, T, S, SL = [], [], [], []
    st = env.reset()
    for _ in range(n_steps):
        if kind == "runner":
            out = obj.run_step(state=env._get_state(), agents_reuse=True)
            a = compose_action(out['plan'], env.K, env.M)
        else:
            a = obj.predict(env._get_state(), env)
        ns, _, d, info = env.step(a)
        E.append(info['energy']); T.append(info['latency'])
        S.append(info['success_rate']); SL.append(info['priority_sla'])
        if d:
            break
    return dict(E=float(np.mean(E)), T=float(np.mean(T)),
                suc=float(np.mean(S)), sla=float(np.mean(SL)))


def make_method(env, name):
    if name == "HMA-Distill":
        return HMAAgentRunner(env=env, mode="Distill",
                              policy_path=POLICY_PATH, agents=None), "runner"
    elif name == "MPC":
        return MPCBaseline(), "pred"
    elif name == "GA":
        return GABaseline(), "pred"
    else:
        class _G:
            def predict(self, st, e):
                act = np.zeros(2 * e.K, np.float32)
                act[0::2] = 0.5
                act[1::2] = (np.abs(e.channels).argmax(1) + 0.5) / e.M
                return act
        return _G(), "pred"


def main():
    t0 = time.time()
    n_steps = len(LAM)
    print("=" * 62)
    print(f"  E19: Milano 真实 trace 泛化 (全局负载曲线 {n_steps} 步/7 天, "
          f"{N_PHASES} 起始相位 × 独立种子)")
    print(f"  λ: min={LAM.min():.2f} max={LAM.max():.2f} "
          f"mean={LAM.mean():.2f} std={LAM.std():.2f}")
    print("=" * 62)
    phases = [int(x) for x in np.linspace(0, n_steps, N_PHASES, endpoint=False)]
    _rec = Recorder("e19")
    out = {"n_steps": n_steps, "lam_min": float(LAM.min()),
           "lam_max": float(LAM.max()), "lam_std": float(LAM.std()),
           "phases": phases}
    for name in ["HMA-Distill", "MPC", "GA", "Greedy"]:
        recs_u, recs_t = [], []
        for ph in phases:
            sd = SEED + ph * 137
            for use_trace in [False, True]:
                env = TraceEnv(use_trace=use_trace, seed=sd,
                               start_phase=ph)
                obj, kind = make_method(env, name)
                r = run_env(env, kind, obj, n_steps)
                (recs_u if not use_trace else recs_t).append(r)
                _rec.add(method=name, seed=sd, episode=ph,
                         metrics=r, use_trace=use_trace)
        mu = {k: float(np.mean([r[k] for r in recs_u])) for k in recs_u[0]}
        mt = {k: float(np.mean([r[k] for r in recs_t])) for k in recs_t[0]}
        keep = mt['suc'] / max(mu['suc'], 1e-9)
        out[name] = {"uniform": mu, "trace": mt,
                     "suc_keep_rate": keep, "n": len(recs_t),
                     "per_run": {"uniform": recs_u, "trace": recs_t}}
        print(f"  {name:14s} uniform: E={mu['E']:.3f} T={mu['T']:.3f} "
              f"suc={mu['suc']:.1%} | trace: E={mt['E']:.3f} T={mt['T']:.3f} "
              f"suc={mt['suc']:.1%} | 保持率={keep:.1%}")
    _rec.close()
    json.dump(out, open(OUT_JSON, "w"), indent=2, ensure_ascii=False)
    print(f"  保存 -> {OUT_JSON} (总耗时 {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
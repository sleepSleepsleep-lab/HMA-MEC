# -*- coding: utf-8 -*-
"""
================================================================
E18 (2026-08): 分布外泛化验证 —— 不重训, 现有 K=8 蒸馏策略
================================================================
审稿风险: 全部结果在单一仿真设定 (均匀任务画像 + 瑞利信道) 下取得,
可能被质疑"参数过拟合"。本实验在【不重新训练】的前提下, 把现有
K=8 蒸馏策略 + 验证器精化闭环直接部署到 5 类分布外变体:

  G-base   基准场景 (对照组)
  G1a      任务画像对数正态 (长尾大任务)
  G1b      突发混合画像 (30% 用户任务 ×3, 模拟突发批次到达)
  G2a      莱斯信道 (K 因子 5 dB, LOS 主径)
  G2b      快衰落 (AR 相关系数 0.95 → 0.7)
  G3       M=3 服务器 (拓扑变化, 策略输出层失配→如实报告回退)

方法: HMA-Distill (策略不重训) / MPC / GA / Greedy。
指标: 能耗/时延/成功率/SLA + 相对 base 的成功率保持率。
问题: 验证器闭环的分布无关性 + 框架对设定漂移的鲁棒性。
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
                    TASK_DEADLINE_MIN, TASK_DEADLINE_MAX, TASK_PRIORITY_PROB,
                    CHANNEL_COEFF_SCALE)
from environment import MECEnvironment
from agent_runner import HMAAgentRunner
from local.ga_baseline import GAOffloadBaseline as GABaseline
from local.baseline_mpc import MPCBaseline
from local.experiment_common import compose_action
from local.results_store import Recorder

K, M = NUM_USERS, NUM_EDGE_SERVERS
POLICY_PATH = os.path.join(CHECKPOINT_DIR, "distilled_policy.pth")
OUT_JSON = os.path.join(RESULTS_DIR, "e18_generalization.json")
N_SEEDS, N_EPISODES, N_STEPS = 5, 6, 100
GA_SEEDS = 3          # GA 慢, 降采样 (n=3/变体)
KRICIAN_DB = 5.0       # 莱斯 K 因子


class GenEnv(MECEnvironment):
    """分布外变体环境: 覆盖 reset() 中的任务/信道生成."""

    def __init__(self, variant, num_users=K, num_servers=M, seed=0):
        self.variant = variant
        super().__init__(num_users=num_users, num_servers=num_servers,
                         seed=seed)

    def reset(self):
        super().reset()
        v = self.variant
        if v == "lognorm_task":
            self.tasks = []
            for k in range(self.K):
                D = float(np.clip(np.exp(self._rng.normal(np.log(2.2e6), 0.8)),
                                  1e5, 2e7))
                C = D * float(self._rng.uniform(*TASK_CYCLES_PER_BIT))
                tau = float(self._rng.uniform(TASK_DEADLINE_MIN,
                                              TASK_DEADLINE_MAX))
                p = int(self._rng.choice([1, 2, 3], p=TASK_PRIORITY_PROB))
                self.tasks.append({'D': D, 'C': C, 'tau': tau, 'priority': p})
        elif v == "burst_mix":
            # 30% 用户携带 3× 大任务 (突发批次混合画像)
            for t in self.tasks:
                if self._rng.random() < 0.3:
                    t['D'] *= 3.0
                    t['C'] *= 3.0
                    t['tau'] *= 1.5      # 大任务容忍时延略宽
        elif v == "rician":
            # LOS 主径: g = sqrt(K/(K+1))·u + sqrt(1/(K+1))·scattered
            kk = 10 ** (KRICIAN_DB / 10.0)
            los = np.sqrt(kk / (kk + 1.0)) * CHANNEL_COEFF_SCALE
            scattered = np.sqrt(1.0 / (kk + 1.0)) * (
                (self._rng.randn(self.K, self.M)
                 + 1j * self._rng.randn(self.K, self.M))
                * np.sqrt(0.5) * CHANNEL_COEFF_SCALE)
            phase = np.exp(1j * self._rng.uniform(0, 2 * np.pi,
                                                  (self.K, self.M)))
            self.channel_coeffs = los * phase + scattered
            self.channels = np.abs(self.channel_coeffs) ** 2
        elif v == "fast_fading":
            self.channel_corr = 0.7
        elif v == "pathloss":
            # 大尺度路径损耗变体 (P1-7): 自由空间 PL ∝ d^2
            # 用户-服务器距离差异化采样 (30-500 m), 以 100 m 参考距离归一化,
            # 使平均路径损耗与基准等距口径 (≈100 m 量级) 可比。
            dist = self._rng.uniform(30.0, 500.0, (self.K, self.M))
            self.pathloss_gain = (100.0 / dist) ** 2
            self.channels = np.abs(self.channel_coeffs) ** 2 * self.pathloss_gain
        return self._get_state()

    def step(self, action, intrinsic_reward_fn=None):
        """路径损耗为时不变乘性因子: 在信道 AR(1) 演进后重新施加."""
        out = super().step(action, intrinsic_reward_fn=intrinsic_reward_fn)
        if self.variant == "pathloss" and hasattr(self, "pathloss_gain"):
            self.channels = np.abs(self.channel_coeffs) ** 2 * self.pathloss_gain
        return out


def run_env(env, kind, obj):
    """在 env 上跑一个 episode, 返回指标均值."""
    E, T, S, SL = [], [], [], []
    st = env.reset()
    for _ in range(N_STEPS):
        if kind == "runner":
            out = obj.run_step(state=env._get_state(), agents_reuse=True)
            a = compose_action(out['plan'], env.K, env.M)
        else:
            a = obj.predict(env._get_state(), env)   # 直接返回动作向量
        ns, _, d, info = env.step(a)
        E.append(info['energy']); T.append(info['latency'])
        S.append(info['success_rate']); SL.append(info['priority_sla'])
        if d:
            break
    return dict(E=float(np.mean(E)), T=float(np.mean(T)),
                suc=float(np.mean(S)), sla=float(np.mean(SL)))


def main():
    t0 = time.time()
    print("=" * 62)
    print("  E18: 分布外泛化验证 (现有 K8 策略, 不重训)")
    print("=" * 62)
    variants = ["base", "lognorm_task", "burst_mix", "rician",
                "fast_fading", "pathloss", "M3"]
    out = {}
    _rec = Recorder("e18")
    for vi, variant in enumerate(variants):
        num_servers = 3 if variant == "M3" else M
        print(f"\n-- 变体: {variant} (M={num_servers}) --")
        rows = {"HMA-Distill": [], "MPC": [], "GA": [], "Greedy": []}
        for sd in range(N_SEEDS):
            for ep in range(N_EPISODES):
                ga_ok = (sd < GA_SEEDS and ep == 0)
                for name in rows:
                    if name == "GA" and not ga_ok:
                        continue
                    env = GenEnv(variant, num_users=K,
                                 num_servers=num_servers, seed=SEED + sd + ep)
                    if name == "HMA-Distill":
                        obj = HMAAgentRunner(env=env, mode="Distill",
                                             policy_path=POLICY_PATH,
                                             agents=None)
                        kind = "runner"
                    elif name == "MPC":
                        obj = MPCBaseline(); kind = "pred"
                    elif name == "GA":
                        obj = GABaseline(); kind = "pred"
                    else:
                        class _G:
                            def predict(self, st, e):
                                act = np.zeros(2 * e.K, np.float32)
                                act[0::2] = 0.5
                                act[1::2] = (np.abs(e.channels).argmax(1)
                                             + 0.5) / e.M
                                return act
                        obj = _G(); kind = "pred"
                    r = run_env(env, kind, obj)
                    rows[name].append(r)
                    _rec.add(method=name, seed=sd, episode=ep,
                             metrics=r, variant=variant)
        vout = {}
        for name, recs in rows.items():
            if not recs:
                continue
            mean = {k: float(np.mean([r[k] for r in recs])) for k in recs[0]}
            vout[name] = {"mean": mean, "n": len(recs),
                          "per_run": recs}
            print(f"    {name:14s} E={mean['E']:.3f} T={mean['T']:.3f} "
                  f"suc={mean['suc']:.1%} sla={mean['sla']:.1%} (n={len(recs)})")
        out[variant] = vout
    _rec.close()
    # 成功率保持率 (相对 base)
    print("\n" + "=" * 62)
    print("  成功率保持率 (suc_variant / suc_base)")
    print("=" * 62)
    base_suc = {m: out["base"][m]["mean"]["suc"] for m in out["base"]}
    for variant in variants[1:]:
        for m in out[variant]:
            r = out[variant][m]["mean"]["suc"] / max(base_suc.get(m, 1e-9), 1e-9)
            print(f"    {variant:14s} {m:14s} 保持率={r:.1%} "
                  f"(suc {base_suc.get(m, float('nan')):.1%} -> "
                  f"{out[variant][m]['mean']['suc']:.1%})")
    json.dump(out, open(OUT_JSON, "w"), indent=2, ensure_ascii=False)
    print(f"\n  保存 -> {OUT_JSON} (总耗时 {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
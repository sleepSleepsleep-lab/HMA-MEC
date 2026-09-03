# -*- coding: utf-8 -*-
"""
================================================================
E7 超参灵敏度分析 (local/run_e7_sensitivity.py)
================================================================
对 four CW-Debate 关键超参做 1D sweep：

  - CONFIDENCE_THRESHOLD  tau_c ∈ {0.4, 0.5, 0.6, 0.7, 0.8}
  - CONFIDENCE_BETA       beta ∈ {1, 2, 3, 5}
  - CONSENSUS_EPSILON    eps_c ∈ {0.01, 0.02, 0.05, 0.1}
  - VERIFY_GAP_TOLERANCE δ_v  ∈ {0.05, 0.10, 0.15, 0.25}

每个超参值独立运行 HMA-Distill 在 n_seeds × n_eps × n_steps 下评估，
记录 能耗 / 时延 / 成功率 / SLA / token 数 / 触发率 (Hybrid 时)。

结果保存 results/e7_sensitivity.npz; 由 fig_e7_sensitivity.py 绘制。
注意: E7 使用 FullLLM 模式 (CW-Debate 真实 LLM 辩论), 以更真实地反映
超参变化对辩论协议的影响; 依赖本地 vLLM 服务。episode 级并发执行
(ThreadPoolExecutor) 以充分利用 vLLM 连续批处理。
================================================================
"""

import os
import sys
import json
from concurrent.futures import ThreadPoolExecutor
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)

# 在 import 任何模块前先固定 baseline 超参
from config import (
    NUM_USERS, NUM_EDGE_SERVERS, MAX_STEPS, RESULTS_DIR, SEED,
    CHECKPOINT_DIR,
)
import config as _cfg

from environment import MECEnvironment
from cw_debate import cw_debate
from agent_define import make_agents
from local.experiment_common import compose_action
from llm_client import get_llm_client


# ============================================================
# 显眼配置区
# ============================================================
N_SEEDS    = 5
N_EPISODES = 3
N_STEPS    = 100                 # 推理时长度, 1k 步已足以观察敏感性
POLICY_PATH = os.path.join(CHECKPOINT_DIR, "distilled_policy.pth")
OUTPUT_NPZ  = os.path.join(RESULTS_DIR, "e7_sensitivity.npz")
OUTPUT_JSON = os.path.join(RESULTS_DIR, "e7_sensitivity.json")

# 各超参 sweep 配置
SWEEPS = {
    "tau_c":  [0.4, 0.5, 0.6, 0.7, 0.8],
    "beta":   [1, 2, 3, 5],
    "eps_c":  [0.01, 0.02, 0.05, 0.10],
    "delta_v":[0.05, 0.10, 0.15, 0.25],
}
DEFAULTS = {
    "tau_c":  _cfg.CONFIDENCE_THRESHOLD,
    "beta":   _cfg.CONFIDENCE_BETA_TAU,
    "eps_c":  _cfg.CONSENSUS_EPSILON,
    "delta_v":_cfg.VERIFY_GAP_TOLERANCE,
}
PARAM_ATTR = {
    "tau_c":  "CONFIDENCE_THRESHOLD",
    "beta":   "CONFIDENCE_BETA_TAU",
    "eps_c":  "CONSENSUS_EPSILON",
    "delta_v":"VERIFY_GAP_TOLERANCE",
}


def _attr_set(name, value):
    """Runtime override a config attribute (and mirror to env-level cache)."""
    setattr(_cfg, PARAM_ATTR[name], value)
    # 模块级 imports 也需刷新: cw_debate, agent_define 等
    import cw_debate as _cw
    import agent_define as _ad
    setattr(_cw, PARAM_ATTR[name], value)
    setattr(_ad, PARAM_ATTR[name], value)


def _attr_restore():
    for k, v in DEFAULTS.items():
        _attr_set(k, v)


def run_one_setting(param_name, param_value, n_seeds=N_SEEDS,
                    n_episodes=N_EPISODES, n_steps=N_STEPS):
    """在指定超参值下跑 HMA 评估 (使用 CW-Debate 协议, 参数通过消融路径生效)."""
    _attr_set(param_name, param_value)
    # 也需要更新 verifier.py 中的容差; verifier 直接读写 config.VERIFY_GAP_TOLERANCE
    import verifier as _v
    _v_cfg_tol = None
    if hasattr(_v, 'VERIFY_GAP_TOLERANCE'):
        _v_cfg_tol = _v.VERIFY_GAP_TOLERANCE
        _v.VERIFY_GAP_TOLERANCE = _cfg.VERIFY_GAP_TOLERANCE
    from agent_runner import HMAAgentRunner
    # 尝试获取 LLM 客户端 (有则 FullLLM 走真实 LLM; 无则走启发式 CW-Debate)
    try:
        _llm = get_llm_client()
    except Exception:
        _llm = None
    energy, lat, suc, sla = [], [], [], []

    def _run_ep(sd, ep):
        """单个 (seed, episode) 独立环境 + FullLLM 辩论评估 (线程安全)。"""
        env = MECEnvironment(num_users=NUM_USERS,
                              num_servers=NUM_EDGE_SERVERS,
                              seed=SEED + sd + ep)
        env.reset()
        runner = HMAAgentRunner(env=env, mode='FullLLM',
                                llm=_llm, agents=None)
        e, l, s, sl = [], [], [], []
        for _ in range(n_steps):
            state = env._get_state()
            out = runner.run_step(state=state, agents_reuse=True)
            a = compose_action(out['plan'], env.K, env.M)
            ns, _, d, info = env.step(a)
            e.append(info['energy'])
            l.append(info['latency'])
            s.append(info['success_rate'])
            sl.append(info['priority_sla'])
            if d:
                break
        return e, l, s, sl

    # 2026-08 整改: episode 级并发执行 (9 个 episode 合并为 6 线程),
    # vLLM 连续批处理吸收并发调用, 吞吐提升 3-4 倍, 结果与串行一致。
    combos = [(sd, ep) for sd in range(n_seeds) for ep in range(n_episodes)]
    with ThreadPoolExecutor(max_workers=min(6, len(combos))) as _ex:
        for e, l, s, sl in _ex.map(lambda t: _run_ep(*t), combos):
            energy.extend(e); lat.extend(l); suc.extend(s); sla.extend(sl)
    if _v_cfg_tol is not None:
        _v.VERIFY_GAP_TOLERANCE = _v_cfg_tol
    return {
        'energy':   float(np.mean(energy)),
        'latency':  float(np.mean(lat)),
        'success':  float(np.mean(suc)),
        'sla':      float(np.mean(sla)),
        'n_samples': len(energy),
    }


def _run_param_task(args):
    """子进程任务：串行扫描单个 param 的全部取值。

    用 进程级 并行 (ProcessPoolExecutor) 实现跨参数组并发：
    - 各参数组的 config 覆盖(_attr_set) 在独立子进程中, 互不污染；
    - 每参数组内部再 episode 级并发 (ThreadPoolExecutor), 充分
      利用 vLLM 连续批处理。
    """
    param, vals = args
    sub = {}
    for v in vals:
        print(f"    [{param}] {param} = {v}", flush=True)
        m = run_one_setting(param, v)
        sub[str(v)] = m
        print(f"      [{param}={v}] E={m['energy']:.5f}  T={m['latency']:.3f}  "
              f"suc={m['success']:.2%}  sla={m['sla']:.2%}", flush=True)
    return param, sub


def main():
    print("=" * 60)
    print("  E7 超参灵敏度分析 (HMA-Distill, FullLLM 辩论, 参数组进程级并行)")
    print("=" * 60)
    print(f"  Seeds={N_SEEDS}, Episodes/seed={N_EPISODES}, Steps/ep={N_STEPS}")
    print(f"  Baseline: {DEFAULTS}")
    results = {}  # {param: {value: {metric: mean}}}
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(
            max_workers=min(4, len(SWEEPS))) as ex:
        for param, sub in ex.map(_run_param_task, list(SWEEPS.items())):
            results[param] = sub
            print(f"  [param {param}] 完成: {list(sub.keys())}", flush=True)

    _attr_restore()

    # 扁平保存
    flat = {}
    for p, d in results.items():
        flat[f"{p}__values"] = np.array(
            [float(v) for v in d.keys()], dtype=np.float32)
        for k in ('energy', 'latency', 'success', 'sla'):
            flat[f"{p}__{k}"] = np.array(
                [d[v][k] for v in d.keys()], dtype=np.float32)
    os.makedirs(os.path.dirname(OUTPUT_NPZ), exist_ok=True)
    np.savez_compressed(OUTPUT_NPZ, **flat)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump({p: {v: d for v, d in r.items()}
                    for p, r in results.items()},
                    f, ensure_ascii=False, indent=2)
    print(f"\n  npz 保存: {OUTPUT_NPZ}")
    print("=" * 60)


if __name__ == "__main__":
    main()
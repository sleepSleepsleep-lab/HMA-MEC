# -*- coding: utf-8 -*-
"""
历史结果转档 (migrate_results.py)
=================================
把整改前各实验脚本产出的 npz/json 结果一次性转档为统一 records
(results/records/<exp>.jsonl), 使旧结果与 results_store 记录层同构, 供
load_records/summarize 统一读取 (绘图/统计/审计)。

支持的源格式:
  A) eX_result.npz: 键 "<method>__<metric>__vals|mean|std"
  B) json 含 per_run / per_episode / per_run.uniform|trace (E18/E19/E21)
  C) json 仅 mean/std (无 per-run) -> 只写 1 条聚合记录 (config.source=json)

用法:
    python local/migrate_results.py [--all] [--exp e6]
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)                       # .../src
ROOT = os.path.dirname(SRC)                       # 仓库根
sys.path.insert(0, HERE)
sys.path.insert(0, SRC)

from results_store import Recorder, DEFAULT_RECORDS_DIR, load_records
import numpy as np

RESULTS = os.path.join(ROOT, "results")

SPECIAL_EXPS = {"e4": "e4_efficiency", "e5": "e5_pareto", "e6": "e6_robust",
                "e7": "e7_sensitivity", "e8": "e8_distill_size"}

# (实验名, npz 路径, json 路径) 三元组; npz/json 可缺省
PLAN = [
    ("e1",  "e1_comparison.npz",          "e1_comparison.json"),
    ("e2",  "e2_scalability.npz",         None),
    ("e3",  "e3_ablation.npz",            "e3_ablation.json"),
    ("e3c", None,                         "e3_component_ablation.json"),
    ("e4",  "e4_efficiency.npz",          "e4_efficiency.json"),
    ("e5",  "e5_pareto.npz",              "e5_pareto.json"),
    ("e6",  "e6_robust.npz",              None),
    ("e7",  "e7_sensitivity.npz",         "e7_sensitivity.json"),
    ("e8",  "e8_distill_size.npz",        None),
    ("e18", None,                         "e18_generalization.json"),
    ("e19", None,                         "e19_trace.json"),
    ("e21", None,                         "e21_refined_baselines_full.json"),
]

JSON_METRIC_ALIAS = {"E": "energy", "T": "latency", "suc": "success_rate",
                     "sla": "priority_sla"}


def _record_one(rec, method, v, extra=None):
    """单方法层: per_run/per_episode 优先, 否则聚合单条。返回条数。"""
    n = 0
    per = v.get("per_run") or v.get("per_episode")
    kw = dict(extra or {})
    if isinstance(per, dict):          # E19: {uniform: [...], trace: [...]}
        for tag, runs in per.items():
            for i, r in enumerate(runs):
                rec.add(method=method, seed=i // max(len(runs), 1), episode=i,
                        metrics=_norm(r), tag=tag, **kw)
                n += 1
    elif isinstance(per, list) and per and isinstance(per[0], dict):
        for i, r in enumerate(per):
            rec.add(method=method, seed=i // max(len(per), 1), episode=i,
                    metrics=_norm(r), **kw)
            n += 1
    elif isinstance(v, dict) and "mean" in v and isinstance(v["mean"], dict):
        rec.add(method=method, seed=None, episode=None,
                metrics=_norm(v["mean"]), agg_only=True, **kw)
        n += 1
    return n


# 显式双层结构 (variant -> method -> 数据) 的实验
NESTED_EXPS = {"e18"}


def _migrate_json(exp, jpath):
    """B/C 型: per_run/per_episode 优先, 否则聚合单条。仅对显式标记的
    实验 (NESTED_EXPS, 如 E18: variant -> method) 做双层展开。"""
    d = json.load(open(os.path.join(RESULTS, jpath), encoding="utf-8"))
    rec = Recorder(exp, config={"source": jpath})
    n = 0
    for method, v in d.items():
        if not isinstance(v, dict):
            continue
        if exp in NESTED_EXPS:
            for m2, v2 in v.items():
                n += _record_one(rec, m2, v2, extra={"group": method})
        else:
            n += _record_one(rec, method, v)
    rec.close()
    return n


def _norm(m):
    """统一指标名: E/T/suc/sla -> energy/latency/success_rate/priority_sla。"""
    out = {}
    for k, val in m.items():
        kk = JSON_METRIC_ALIAS.get(k, k)
        if isinstance(val, (int, float)):
            out[kk] = float(val)
    return out


def _migrate_npz(exp, zpath):
    """A 型: __vals 数组逐 episode。"""
    d = np.load(os.path.join(RESULTS, zpath), allow_pickle=True)
    keys = [k for k in d.files if k.endswith("__vals")]
    # 键形: <prefix>__<method>__<metric>__vals (方法名在倒数第 3 段, 兼容
    # E1: method__metric / E2: K__method__metric 两种前缀长度)
    groups = {}
    for k in keys:
        parts = k.split("__")
        method, metric = parts[-3], parts[-2]
        prefix = "__".join(parts[:-3])
        groups.setdefault((prefix, method), {})[metric] = d[k]
    rec = Recorder(exp, config={"source": zpath})
    n = 0
    for (prefix, m), sub in sorted(groups.items()):
        n_ep = len(list(sub.values())[0])
        extra = {"group": prefix} if prefix else {}
        for ep in range(n_ep):
            metrics = {mt: float(v[ep]) for mt, v in sub.items()}
            rec.add(method=m, seed=None, episode=ep, metrics=metrics, **extra)
            n += 1
    rec.close()
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default=None, help="只转指定实验 (如 e6)")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    for exp, zpath, jpath in PLAN:
        if args.exp and exp != args.exp:
            continue
        cnt = 0
        _target = os.path.join(DEFAULT_RECORDS_DIR, exp + ".jsonl")
        if os.path.exists(_target):          # 重建语义, 避免重复追加
            os.remove(_target)
        if zpath and os.path.exists(os.path.join(RESULTS, zpath)):
            cnt += _migrate_npz(exp, zpath)
        if jpath and os.path.exists(os.path.join(RESULTS, jpath)):
            cnt += _migrate_json(exp, jpath)
        if exp in [e[0] for e in PLAN if e[0] in SPECIAL_EXPS] and not cnt:
            cnt += _migrate_special(exp, SPECIAL_EXPS[exp])
        if cnt:
            print(f"  e[{exp}] -> {cnt} 条记录")
        else:
            print(f"  e[{exp}] 无文件, 跳过")




# ---------------------------------------------------------------
# 专用转档: e4/e5/e6/e7/e8 (曲线/嵌套结构, 无 __vals)
# ---------------------------------------------------------------
SPECIAL_JSON = ["e4_efficiency", "e5_pareto", "e6_robust",
                "e7_sensitivity", "e8_distill_size"]


def _migrate_special(exp, stem):
    d = json.load(open(os.path.join(RESULTS, stem + ".json"),
                       encoding="utf-8"))
    rec = Recorder(exp, config={"source": stem + ".json"})
    n = 0
    if stem == "e4_efficiency":
        for mode, v in d.items():
            rec.add(method=mode, seed=None, episode=None, metrics=v.get("stats", {}))
            n += 1
    elif stem == "e5_pareto":
        for m, v in d.items():
            for i, w in enumerate(v.get("omegas", [])):
                rec.add(method=m, seed=None, episode=i, metrics={
                    "omega": float(w),
                    "energy": float(v["energy"][i]),
                    "latency": float(v["latency"][i]),
                    "conf_min": float(v["conf_min"][i]),
                    "fallback_rate": float(v["fallback_rate"][i])})
                n += 1
    elif stem == "e6_robust":
        for ptype, v in d.items():
            m = {k: float(np.mean(vv)) for k, vv in v.items() if vv}
            rec.add(method=ptype, seed=None, episode=None, metrics=m,
                    curves={k: list(vv) for k, vv in v.items()})
            n += 1
    elif stem == "e7_sensitivity":
        for param, vals in d.items():
            for val, m in vals.items():
                rec.add(method=param, seed=None, episode=float(val), metrics=m)
                n += 1
    elif stem == "e8_distill_size":
        for i, sz in enumerate(d["size"]):
            rec.add(method="D{}".format(sz), seed=None, episode=i, metrics={
                "energy": float(d["energy"][i]),
                "latency": float(d["latency"][i]),
                "success_rate": float(d["success"][i]),
                "priority_sla": float(d["sla"][i]),
                "train_dt_s": float(d["train_dt"][i])})
            n += 1
    rec.close()
    return n

if __name__ == "__main__":
    main()

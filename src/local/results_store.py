# -*- coding: utf-8 -*-
"""
统一结果记录层 (results_store.py)
=================================
所有实验脚本经 Recorder 在**运行中**把每个 episode/run 的指标增量写入
results/records/<experiment>.jsonl (一行一个 JSON 记录, 含实验名/方法/种子/
episode/配置/相对时间戳)。特性:

  - 追加式落盘 (每 episode 即 flush): 中途崩溃不丢已跑数据, 运行中即可读取绘图;
  - 统一 schema: {experiment, method, seed, episode, config, metrics, started..};
  - summarize(): 按方法聚合 mean/std/n + bootstrap 95% CI, 供论文/绘图直接使用;
  - migrate_npz_json(): 把历史 npz/json 结果一次性转档为 records, 兼容旧数据。

用法 (实验脚本内):
    from local.results_store import Recorder
    rec = Recorder("e18", config={"n_steps": 100})
    ...
    for ...:
        r = run_episode(...)
        rec.add(method="HMA-Distill", seed=sd, episode=ep, metrics=r)
    rec.close()

用法 (绘图/分析):
    from local.results_store import load_records, summarize
    recs = load_records("e18")
    tab  = summarize(recs, metric="suc", groupby="method")
"""
import json
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))            # .../src/local
SRC = os.path.dirname(HERE)                                   # .../src
ROOT = os.path.dirname(SRC)                                   # 仓库根
DEFAULT_RECORDS_DIR = os.path.join(ROOT, "results", "records")


class Recorder:
    """追加式 episode 记录器。close() 后不可再 add。"""

    def __init__(self, experiment, out_dir=None, config=None):
        self.experiment = experiment
        self.out_dir = out_dir or DEFAULT_RECORDS_DIR
        os.makedirs(self.out_dir, exist_ok=True)
        self.path = os.path.join(self.out_dir, experiment + ".jsonl")
        self.config = dict(config or {})
        self._start = time.time()
        self._count = 0
        self._fh = open(self.path, "a", encoding="utf-8")

    def add(self, method=None, seed=None, episode=None, metrics=None, **kw):
        """记录一个 episode/run 的指标。metrics 为 dict（标量或列表）。"""
        rec = {
            "experiment": self.experiment,
            "method": method,
            "seed": seed,
            "episode": episode,
            "t": round(time.time() - self._start, 3),
            "config": self.config,
            "metrics": dict(metrics) if metrics else {},
        }
        rec.update(kw)
        self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._fh.flush()
        self._count += 1

    @property
    def count(self):
        return self._count

    def close(self):
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def load_records(experiment_or_path, out_dir=None):
    """读取 records 文件, 返回 dict 列表 (保持写入顺序)。"""
    if os.path.exists(experiment_or_path):
        path = experiment_or_path
    else:
        path = os.path.join(out_dir or DEFAULT_RECORDS_DIR,
                            experiment_or_path + ".jsonl")
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # 容忍半个写入的尾巴
    return out


def _metric_val(rec, metric):
    m = rec.get("metrics", {})
    if metric in m:
        return m[metric]
    # 支持 "suc" -> "success_rate" 等别名
    alias = {"suc": "success_rate", "sla": "priority_sla",
             "T": "latency", "E": "energy"}
    if metric in alias and alias[metric] in m:
        return m[alias[metric]]
    return None


def summarize(records, metric="suc", groupby="method",
              n_boot=10000, seed_boot=0):
    """按 groupby 分组聚合: mean/std/n + bootstrap 95% CI (均值比亦支持)。

    返回 {group: {"n":.., "mean":.., "std":.., "ci95":(lo,hi), "vals":[...]}}
    """
    import numpy as np
    rng = np.random.default_rng(seed_boot)
    groups = {}
    for r in records:
        v = _metric_val(r, metric)
        if v is None:
            continue
        g = r.get(groupby, "?")
        groups.setdefault(g, []).append(float(v))
    out = {}
    for g, vals in groups.items():
        a = np.asarray(vals)
        mean, std = float(a.mean()), float(a.std())
        boots = [float(rng.choice(a, len(a)).mean()) for _ in range(n_boot)]
        out[g] = {"n": len(a), "mean": mean, "std": std,
                  "ci95": (float(np.percentile(boots, 2.5)),
                           float(np.percentile(boots, 97.5))),
                  "vals": vals}
    return out


def migrate_npz_json(npz_path, json_path, experiment, out_dir=None,
                     method_order=None, npz_metric_map=None):
    """把旧式结果 (eX_result.npz 的 __vals 数组 + json 摘要) 转档为 records。

    npz_metric_map: {"success_rate": "suc", ...} 顺带把指标名统一。
    返回写入条数。若 npz 含 per_run/per_episode (E18/E19/E21 新格式) 则优先用其。
    """
    import numpy as np
    rec = Recorder(experiment, out_dir=out_dir,
                   config={"source": os.path.basename(npz_path)})
    n = 0
    try:
        d = np.load(npz_path, allow_pickle=True)
        keys = [k for k in d.files]
        methods = method_order or sorted({k.split("__")[0] for k in keys})
        for m in methods:
            vals = None
            sub = {k: d[k] for k in keys if k.startswith(m + "__") and
                   k.endswith("__vals")}
            if sub:
                n_ep = len(list(sub.values())[0])
                for ep in range(n_ep):
                    metrics = {}
                    for k, v in sub.items():
                        metric = k.split("__")[1]
                        metrics[metric] = float(v[ep])
                    rec.add(method=m, seed=None, episode=ep, metrics=metrics)
                    n += 1
    finally:
        rec.close()
    return n

def tabulate(experiment, metric="suc", groupby="method", tag=None,
             out_dir=None):
    """绘图/论文用便捷聚合: records -> {group: (mean, std, ci95, n)}。

    tag: 只取该 tag 的记录 (如 e21 的 raw/refined, e19 的 uniform/trace)。
    返回按组排序的 dict[group] = {"mean","std","ci95","n","vals"}。
    """
    recs = load_records(experiment, out_dir=out_dir)
    if tag is not None:
        recs = [r for r in recs if r.get("tag") == tag]
    return summarize(recs, metric=metric, groupby=groupby)

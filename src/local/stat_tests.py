# -*- coding: utf-8 -*-
"""
================================================================
统计显著性检验 (local/stat_tests.py)
================================================================
Wilcoxon 符号秩检验：HMA (Distill/Hybrid) vs 各基线。
读取 e1_comparison.npz 的 per-episode 成对样本（同一批 episode 下
两种方法的指标即为配对样本），对四个评估指标分别输出 p 值。

用法:
    python local/stat_tests.py [npz_path]

依赖: scipy (pip install scipy)
注意: 显著性结论只有在 D3 重跑（50 ep × 5 seed）后才具备统计力；
      当前 6 样本数据仅用于验证管线。
================================================================
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)

import numpy as np
from scipy import stats

from config import RESULTS_DIR
from local.experiment_common import load_npz

METRICS = ('energy', 'latency', 'success_rate', 'priority_sla')


def _paired(name_a, name_b, res, metric):
    """提取两个方法的成对样本 (n,) 数组；样本数不同或缺失时返回 None。"""
    try:
        a = [r[metric] for r in res[name_a]['per_seed']]
        b = [r[metric] for r in res[name_b]['per_seed']]
    except (KeyError, TypeError):
        return None
    if len(a) != len(b) or len(a) == 0:
        return None
    return np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)


def run_tests(npz_path, methods=None):
    """对每个基线输出 4 指标 × p 值的表格。

    返回: {method_name: {metric: p_value or None}}
    """
    if not os.path.exists(npz_path):
        print(f"[stat_tests] 文件不存在: {npz_path}")
        return {}
    res = load_npz(npz_path)
    hma = None
    for cand in ('HMA-Distill', 'HMA-Hybrid'):
        if cand in res:
            hma = cand
            break
    if hma is None:
        print("[stat_tests] npz 中未找到 HMA-Distill/HMA-Hybrid")
        return {}

    targets = methods or [m for m in res if m != hma]
    print(f"[stat_tests] 参考方法: {hma}  (n = {len(res[hma]['per_seed'])} 样本)")
    header = "  基线            " + "".join(f"{m:>14s}" for m in METRICS)
    print(header)
    out = {}
    for name in sorted(targets):
        row = []
        for metric in METRICS:
            paired = _paired(hma, name, res, metric)
            p = None
            if paired is not None:
                try:
                    _, p = stats.wilcoxon(paired[0], paired[1])
                except ValueError:
                    p = None      # 全平局等无法检验的情形
            row.append(p)
            if p is not None:
                print(f"  {name:16s} {metric:>9s}: p={p:.4f}"
                      + (" *" if p < 0.05 else ""))
        out[name] = dict(zip(METRICS, row))
    return out


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        RESULTS_DIR, "e1_comparison.npz")
    run_tests(path)

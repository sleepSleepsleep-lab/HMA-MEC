# -*- coding: utf-8 -*-
"""E17-v2 结果分析: 汇总 + 配对 Wilcoxon（Debate vs DebateFB vs SingleLLM1）.
数据源: 优先 e17_feedback_v2_raw.json（新版脚本）; 否则解析运行日志逐状态行.
"""
import json, os, re, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULTS_DIR

PATH = os.path.join(RESULTS_DIR, "e17_feedback_v2.json")
RAW = os.path.join(RESULTS_DIR, "e17_feedback_v2_raw.json")
LOG = os.path.join(RESULTS_DIR, "logs", "e17_feedback_v2.log")


def load_raw():
    if os.path.exists(RAW):
        return json.load(open(RAW))
    # 从日志解析: "[state0] scale=0.30 | Debate J=11.49 suc=12% T=1.52 inf=1 | ..."
    raw, pat = [], re.compile(
        r"\[state\d+\] scale=([\d.]+)((?:\s*\|\s*\w+ J=[\d.]+ suc=\d+% T=[\d.]+ inf=\d+)+)")
    for line in open(LOG, encoding="utf-8", errors="ignore"):
        m = pat.search(line)
        if not m:
            continue
        ps = {"scale": float(m.group(1))}
        for seg in m.group(2).split("|"):
            seg = seg.strip()
            if not seg:
                continue
            gm = re.match(r"(\w+) J=([\d.]+) suc=(\d+)% T=([\d.]+) inf=(\d)", seg)
            if gm:
                g, J, suc, T = gm.group(1), float(gm.group(2)), \
                               float(gm.group(3)) / 100.0, float(gm.group(4))
                ps[g] = {"J": J, "suc": suc, "T": T}
                ps[g]["E"] = ps[g]["sla"] = float("nan")
        raw.append(ps)
    return raw


def main():
    if not os.path.exists(PATH):
        print(f"未找到 {PATH}")
        sys.exit(1)
    out = json.load(open(PATH))
    print("=" * 70)
    print("E17-v2 聚合（表用）")
    print("=" * 70)
    for g, agg in out.items():
        if "n_parsed" not in agg:
            print(f"  {g}: 全部失败")
            continue
        print(f"  {g:12s} n={agg['n_parsed']} 解析失败={agg.get('n_parse_fail', 0)} "
              f"VA硬罚拒绝={agg['infeasible_frac']:.1%} "
              f"J={agg['mean_J']:.3f} suc={agg['mean_suc']:.1%} "
              f"T={agg['mean_T']:.3f}s E={agg['mean_E']:.3f} "
              f"sla={agg['mean_sla']:.1%}")
    from scipy.stats import wilcoxon
    raw = load_raw()
    print("=" * 70)
    print(f"配对 Wilcoxon（按 scale 对齐, 共 {len(raw)} 状态）")
    print("=" * 70)

    def paired(ga, gb, k):
        a, b = [], []
        for ps in raw:
            if ps.get(ga) and ps.get(gb) and k in ps[ga] and k in ps[gb]:
                x, y = ps[ga][k], ps[gb][k]
                if not (np.isnan(x) or np.isnan(y)):
                    a.append(x)
                    b.append(y)
        return np.array(a), np.array(b)

    for ga, gb in [("Debate", "DebateFB"), ("Debate", "SingleLLM1")]:
        for k in ['J', 'T', 'suc']:
            a, b = paired(ga, gb, k)
            if len(a) < 5:
                print(f"  {ga}-vs-{gb} [{k}]: 配对样本不足 ({len(a)})")
                continue
            d = a - b
            if np.allclose(d, 0):
                print(f"  {ga}-vs-{gb} [{k}]: n={len(a)} 完全相同")
                continue
            st, p = wilcoxon(d, zero_method="wilcox")
            n_b = int(np.mean(b < a) * len(a) + 0.5)   # 后者更优(值更小)的状态数
            print(f"  {ga}-vs-{gb} [{k}]: n={len(a)} {ga}={a.mean():.4f} "
                  f"{gb}={b.mean():.4f} diff={d.mean():+.4f} "
                  f"{gb}更优={n_b}/{len(a)} p={p:.4f}")


if __name__ == "__main__":
    main()
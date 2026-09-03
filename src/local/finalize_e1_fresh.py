# -*- coding: utf-8 -*-
"""E1 重跑收尾 (finalize_e1_fresh.py): 等待全部方法完成 -> 合并 -> 统计检验 -> 汇总. """
import os, sys, time, json
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_REPO_ROOT)
_REPO_ROOT = os.path.dirname(_REPO_ROOT)

import numpy as np
from scipy.stats import wilcoxon

RESULTS = os.path.join(_REPO_ROOT, "results")
LOG_DIR = "/tmp"

REQUIRED = {
    'e1_fresh_basic.npz': ['/tmp/e1r_basic.log'],
    'e1_fresh_singlellm.json': ['/tmp/e1r_singlellm.log'],
    'e1_fresh_ledrl.json': ['/tmp/e1r_ledrl.log'],
}


def all_done():
    for f in REQUIRED:
        if not os.path.exists(os.path.join(RESULTS, f)):
            return False
    return True


def wait(timeout=6 * 3600, interval=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if all_done():
            print(f"[{time.strftime('%H:%M:%S')}] 全部方法完成", flush=True)
            return True
        # 打印进度
        status = []
        for f in REQUIRED:
            status.append(f"{f}: {'OK' if os.path.exists(os.path.join(RESULTS, f)) else '...'}")
        llm_requests = 0
        try:
            llm_requests = sum(1 for _ in open('/tmp/vllm_server.log'))
        except Exception:
            pass
        print(f"[{time.strftime('%H:%M:%S')}] " + " | ".join(status) +
              f" | LLM 请求≈{llm_requests}", flush=True)
        time.sleep(interval)
    return False


def merge():
    files = ['e1_fresh_basic.npz', 'e1_fresh_sac.npz', 'e1_fresh_ddpg.npz',
             'e1_fresh_dqn.npz', 'e1_fresh_maddpg.npz'] + \
            [f'e1_fresh_ga_s{i}.npz' for i in range(5)] + \
            ['e1_fresh_singlellm.npz', 'e1_fresh_ledrl.npz']
    out = {}
    for f in files:
        p = os.path.join(RESULTS, f)
        if not os.path.exists(p):
            print("缺失:", f); continue
        z = np.load(p, allow_pickle=False)
        for k in z.files:
            if k.endswith('__vals'):
                if k in out:
                    out[k] = np.concatenate([out[k], z[k]])
                else:
                    out[k] = z[k]
    methods = sorted({k.split('__')[0] for k in out})
    for m in methods:
        for met in ['energy', 'latency', 'success_rate', 'priority_sla']:
            v = out[f'{m}__{met}__vals']
            out[f'{m}__{met}__mean'] = np.array([v.mean()], dtype=np.float32)
            out[f'{m}__{met}__std'] = np.array([v.std()], dtype=np.float32)
    np.savez(os.path.join(RESULTS, 'e1_fresh.npz'), **out)
    print(f"\n===== E1 重跑汇总 (真独立 n=250) =====", flush=True)
    for m in methods:
        e = out[f'{m}__energy__mean'][0]; t = out[f'{m}__latency__mean'][0]
        s = out[f'{m}__success_rate__mean'][0]; sl = out[f'{m}__priority_sla__mean'][0]
        es = out[f'{m}__energy__std'][0]; ts = out[f'{m}__latency__std'][0]
        ss = out[f'{m}__success_rate__std'][0]; sls = out[f'{m}__priority_sla__std'][0]
        print(f"  {m:14s} E={e:.4f}±{es:.4f} T={t:.4f}±{ts:.4f} "
              f"suc={s*100:.2f}±{ss*100:.2f}% sla={sl*100:.2f}±{sls*100:.2f}%",
              flush=True)

    # 统计检验 (HMA-Distill vs 各基线, 配对 Wilcoxon, n=250 真独立)
    HMA = 'HMA-Distill'
    print("\n===== 配对 Wilcoxon (n=250 真独立) =====", flush=True)
    stats = {}
    for met, lab in [('latency', '时延'), ('success_rate', '成功率'),
                     ('priority_sla', 'SLA'), ('energy', '能耗')]:
        a = out[f'{HMA}__{met}__vals']
        for m in methods:
            if m == HMA:
                continue
            b = out[f'{m}__{met}__vals']
            n = min(len(a), len(b))
            stat, p = wilcoxon(a[:n], b[:n])
            diff = (a[:n].mean() - b[:n].mean())
            if met in ('success_rate', 'priority_sla'):
                dstr = f"{diff*100:+.1f}pp"
            else:
                dstr = f"{diff/abs(b[:n].mean())*100:+.1f}%"
            sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
            stats[(met, m)] = p
            print(f"  {lab} vs {m:12s} {dstr:>10s} p={p:.4f} {sig}", flush=True)

    # Holm 校正 (HMA vs 全部基线 × 4 指标)
    print("\n===== Holm 校正 (检验族 = HMA vs 12 基线 × 4 指标 = 48) =====", flush=True)
    family = [(met, m) for met in ['latency', 'success_rate', 'priority_sla', 'energy']
              for m in methods if m != HMA]
    ps = np.array([stats[k] for k in family])
    order = np.argsort(ps)
    m_cnt = len(family)
    holm = {}
    for rank, idx in enumerate(order):
        holm[family[idx]] = min(ps[idx] * (m_cnt - rank), 1.0)
    for met, lab in [('latency', '时延'), ('success_rate', '成功率'),
                     ('priority_sla', 'SLA'), ('energy', '能耗')]:
        for m in methods:
            if m == HMA:
                continue
            p = stats[(met, m)]; h = holm[(met, m)]
            if h < 0.05:
                print(f"  {lab} vs {m:12s} raw p={p:.4f} Holm p={h:.4f} 显著", flush=True)
    with open(os.path.join(RESULTS, 'e1_fresh_stats.json'), 'w') as f:
        json.dump({'raw': {f'{k[0]}|{k[1]}': v for k, v in stats.items()},
                   'holm': {f'{k[0]}|{k[1]}': v for k, v in holm.items()}},
                  f, indent=1)
    print("\n统计结果已保存 -> e1_fresh_stats.json", flush=True)


if __name__ == "__main__":
    ok = wait()
    if ok:
        merge()
    else:
        print("等待超时")

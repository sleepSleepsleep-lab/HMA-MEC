# -*- coding: utf-8 -*-
"""E1 重跑最终合并 (merge_e1_fresh_final.py): 15 方法 -> e1_fresh.npz/.json
键名映射: B7-LeDRL -> LeDRL, B8-SingleLLM -> SingleLLM (与论文基线编号 B10/B11 一致)
"""
import os, json
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_REPO_ROOT)
_REPO_ROOT = os.path.dirname(_REPO_ROOT)

import numpy as np

RESULTS = os.path.join(_REPO_ROOT, "results")
RENAME = {'B7-LeDRL': 'LeDRL', 'B8-SingleLLM': 'SingleLLM'}

FILES = ['e1_fresh_basic.npz', 'e1_fresh_sac.npz', 'e1_fresh_ddpg.npz',
         'e1_fresh_dqn.npz', 'e1_fresh_maddpg.npz', 'e1_fresh_singlellm.npz',
         'e1_fresh_ledrl_s0.npz', 'e1_fresh_ledrl_s1.npz',
         'e1_fresh_ledrl_s2.npz', 'e1_fresh_ledrl_s3.npz',
         'e1_fresh_ledrl_s4.npz'] + [f'e1_fresh_ga_s{i}.npz' for i in range(5)]


def main():
    out = {}
    for f in FILES:
        p = os.path.join(RESULTS, f)
        if not os.path.exists(p):
            print("缺失:", f)
            continue
        z = np.load(p, allow_pickle=False)
        for k in z.files:
            if not k.endswith('__vals'):
                continue
            parts = k.split('__')
            m = RENAME.get(parts[0], parts[0])
            nk = f"{m}__{parts[1]}__{parts[2]}"
            if nk in out:
                out[nk] = np.concatenate([out[nk], z[k]])
            else:
                out[nk] = z[k]
    methods = sorted({k.split('__')[0] for k in out})
    for m in methods:
        for met in ['energy', 'latency', 'success_rate', 'priority_sla']:
            v = out[f'{m}__{met}__vals']
            out[f'{m}__{met}__mean'] = np.array([v.mean()], dtype=np.float32)
            out[f'{m}__{met}__std'] = np.array([v.std()], dtype=np.float32)
    np.savez(os.path.join(RESULTS, 'e1_fresh.npz'), **out)
    print(f"已保存 e1_fresh.npz: {len(methods)} 个方法", flush=True)
    for m in methods:
        e = out[f'{m}__energy__mean'][0]; t = out[f'{m}__latency__mean'][0]
        s = out[f'{m}__success_rate__mean'][0]; sl = out[f'{m}__priority_sla__mean'][0]
        es = out[f'{m}__energy__std'][0]; ts = out[f'{m}__latency__std'][0]
        ss = out[f'{m}__success_rate__std'][0]; sls = out[f'{m}__priority_sla__std'][0]
        print(f"  {m:14s} E={e:.4f}±{es:.4f} T={t:.4f}±{ts:.4f} "
              f"suc={s*100:.2f}±{ss*100:.2f}% sla={sl*100:.2f}±{sls*100:.2f}%",
              flush=True)

    # 同时输出 LaTeX 表格行 (供表 1 更新)
    print("\n===== LaTeX 表格行 =====", flush=True)
    order = ['Greedy', 'AllLocal', 'AllEdge', 'Random', 'SAC', 'DDPG',
             'MADDPG', 'DQN', 'GA', 'MPC', 'LeDRL', 'SingleLLM',
             'HMA-Distill', 'HMA-Hybrid']
    for m in order:
        e = out[f'{m}__energy__mean'][0]; t = out[f'{m}__latency__mean'][0]
        s = out[f'{m}__success_rate__mean'][0]; sl = out[f'{m}__priority_sla__mean'][0]
        es = out[f'{m}__energy__std'][0]; ts = out[f'{m}__latency__std'][0]
        ss = out[f'{m}__success_rate__std'][0]; sls = out[f'{m}__priority_sla__std'][0]
        bold_t = r"\mathbf" if m.startswith('HMA') else ""
        print(f"{m} & ${e:.4f} \\pm {es:.4f}$ & ${bold_t}{{{t:.4f} \\pm {ts:.4f}}}$ "
              f"& ${bold_t}{{{s*100:.2f}\\% \\pm {ss*100:.2f}\\%}}$ "
              f"& ${sl*100:.2f}\\% \\pm {sls*100:.2f}\\%$ \\\\", flush=True)


if __name__ == "__main__":
    main()

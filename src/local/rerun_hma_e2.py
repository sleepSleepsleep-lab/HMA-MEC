# -*- coding: utf-8 -*-
"""重跑 E2 的 HMA-Distill/Hybrid (修复默认权重后), 更新 e2_scalability.npz."""
import os, sys, json
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_REPO_ROOT)
_REPO_ROOT = os.path.dirname(_REPO_ROOT)

HERE = _REPO_ROOT
sys.path.insert(0, os.path.join(HERE, "src"))
sys.path.insert(0, os.path.join(HERE, "src/local"))
import numpy as np
from config import RESULTS_DIR, SEED
from local.experiment_common import run_multi_episodes

OUTPUT_NPZ = os.path.join(RESULTS_DIR, "e2_scalability.npz")
K_LIST = [4, 8, 12, 16, 24, 32]
M_FIX = 4
N_SEEDS, N_EPISODES, N_STEPS = 2, 3, 100

raw = np.load(OUTPUT_NPZ, allow_pickle=False)
flat = dict(raw)

for K in K_LIST:
    print(f"  -- K = {K} --")
    specs = [{'name': 'HMA-Distill'}, {'name': 'HMA-Hybrid'}]
    results = run_multi_episodes(specs, K=K, M=M_FIX,
                                     record_experiment="e2",
                                 n_seeds=N_SEEDS, n_episodes=N_EPISODES,
                                 n_steps=N_STEPS, verbose=False)
    for m, d in results.items():
        for k in ('energy','latency','success_rate','priority_sla'):
            flat[f"K{K}__{m}__{k}__mean"] = np.array([d['mean'][k]], dtype=np.float32)
            flat[f"K{K}__{m}__{k}__vals"] = np.array([r[k] for r in d['per_seed']],
                                                      dtype=np.float32)
        dm = d['mean']
        print(f"     {m}: E={dm['energy']:.3f} T={dm['latency']:.3f} "
              f"suc={dm['success_rate']:.2%}")

np.savez_compressed(OUTPUT_NPZ, **flat)
print(f"  已保存 {len(flat)} keys -> {OUTPUT_NPZ}")

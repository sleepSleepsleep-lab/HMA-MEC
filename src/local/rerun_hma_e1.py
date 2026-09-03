# -*- coding: utf-8 -*-
"""重跑 E1 的 HMA-Distill/Hybrid (修复默认权重 + policy_path 后), 更新 e1_comparison."""
import os, sys, json
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_REPO_ROOT)
_REPO_ROOT = os.path.dirname(_REPO_ROOT)

HERE = _REPO_ROOT
sys.path.insert(0, os.path.join(HERE, "src"))
sys.path.insert(0, os.path.join(HERE, "src/local"))

from config import NUM_USERS, NUM_EDGE_SERVERS, RESULTS_DIR, SEED
from local.experiment_common import run_multi_episodes, save_npz, load_npz

OUTPUT_NPZ = os.path.join(RESULTS_DIR, "e1_comparison.npz")
OUTPUT_JSON = os.path.join(RESULTS_DIR, "e1_comparison.json")

print("=" * 60)
print("  重跑 E1 HMA-Distill/Hybrid (5 seed × 50 ep × 200 步, 修复后权重)")
print("=" * 60)
specs = [{'name': 'HMA-Distill'}, {'name': 'HMA-Hybrid'}]
out = run_multi_episodes(specs, K=NUM_USERS, M=NUM_EDGE_SERVERS,
                          record_experiment="e1",
                         n_seeds=5, n_episodes=50, n_steps=200, verbose=False)

old = load_npz(OUTPUT_NPZ)
for m in ('HMA-Distill', 'HMA-Hybrid'):
    if m in old:
        old[m] = out[m]
        print(f"  更新 {m}: E={out[m]['mean']['energy']:.4f} "
              f"T={out[m]['mean']['latency']:.3f} "
              f"suc={out[m]['mean']['success_rate']:.2%} "
              f"sla={out[m]['mean']['priority_sla']:.2%}")

save_npz(OUTPUT_NPZ, old)
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump({m: {'mean': d['mean'], 'std': d['std'],
                   'n_samples': len(d['per_seed'])}
               for m, d in old.items()}, f, ensure_ascii=False, indent=2)
print(f"  已保存 {len(old)} 方法 -> {OUTPUT_NPZ} / {OUTPUT_JSON}")

print("\n  更新后 HMA 对比:")
for m in ('Greedy','AllEdge','GA','HMA-Distill','HMA-Hybrid'):
    if m in old:
        d = old[m]['mean']
        print(f"  {m:12s} E={d['energy']:.4f} T={d['latency']:.3f} "
              f"suc={d['success_rate']:.2%} sla={d['priority_sla']:.2%}")

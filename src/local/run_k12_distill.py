# -*- coding: utf-8 -*-
"""
================================================================
方案A (2026-08): K=12 规模蒸馏策略训练与评估
================================================================
解决 E2 可扩展性缺陷: K>8 回退随机初始化 (输出层硬编码 K×M)。
用既有 debate_dataset 中 K=12-M=4 的 2039 条样本训练 K=12 策略网
(PolicyAgentNet 参数化 K/M/state_dim), 然后在 K=12 基准上评估
Distill+验证器精化 vs GA/MPC/Greedy, 检验"按规模蒸馏"是否恢复
成功率至与 K=8 相当水平 (旧版随机初始化 K=12 Distill 仅 67%)。

输出: results/e2_k12_distilled.json
================================================================
"""
import os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)
sys.path.insert(0, os.path.join(SRC, "local"))

from config import RESULTS_DIR, CHECKPOINT_DIR, SEED, POLICY_NET_BATCH
from distill_agent import DistillAgentTrainer, PolicyAgentRunner
from environment import MECEnvironment
from agent_runner import HMAAgentRunner
from local.ga_baseline import GAOffloadBaseline as GABaseline
from local.baseline_mpc import MPCBaseline
from local.plan_refiner import PlanRefiner
from local.experiment_common import compose_action

K12, M12 = 12, 4
DATASET = os.path.join(RESULTS_DIR, "debate_dataset.jsonl")
CKPT_K12 = os.path.join(CHECKPOINT_DIR, "distilled_policy_k12.pth")
OUT_JSON = os.path.join(RESULTS_DIR, "e2_k12_distilled.json")
N_SEEDS, N_EPISODES, N_STEPS = 3, 5, 100


def load_k12(path, K=K12, M=M12):
    sd_exp = K * 4 + M * 2
    states, alphas, servers, confs = [], [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            s = np.asarray(r["state"], dtype=np.float32)
            a = np.asarray(r["alpha"], dtype=np.float32)
            m = np.asarray(r["server"], dtype=int)
            c = r.get("confidence")
            if len(s) != sd_exp or len(a) != K or len(m) != K:
                continue
            if not (np.isfinite(s).all() and np.isfinite(a).all()):
                continue
            if c is not None and not np.isfinite(np.asarray(c)).all():
                continue
            states.append(s); alphas.append(a); servers.append(m)
            # 置信度监督: 保存完整 C-dim 向量 (trainer 内取 min, 与 K=8 一致)
            confs.append(np.asarray(c, dtype=np.float32) if c is not None
                         else np.full(K, 0.0, dtype=np.float32))
    return (np.stack(states), np.stack(alphas), np.stack(servers),
            np.stack(confs) if confs else np.zeros((0, K), dtype=np.float32))


def run(env, obj, kind):
    E, T, S, SL = [], [], [], []
    for _ in range(N_STEPS):
        st = env._get_state()
        if kind == "runner":
            out = obj.run_step(state=st, agents_reuse=True)
            a = compose_action(out['plan'], K12, M12)
        else:
            a = obj.predict(st, env)
        ns, _, d, info = env.step(a)
        E.append(info['energy']); T.append(info['latency'])
        S.append(info['success_rate']); SL.append(info['priority_sla'])
        if d:
            break
    return dict(E=float(np.mean(E)), T=float(np.mean(T)),
                suc=float(np.mean(S)), sla=float(np.mean(SL)))


def main():
    t0 = time.time()
    print("=" * 60)
    print(f"  方案A: K=12 蒸馏策略训练 + 评估")
    print("=" * 60)
    states, alphas, servers, confs = load_k12(DATASET)
    print(f"  K=12 样本: N={len(states)}, state_dim={states.shape[1]}")
    trainer = DistillAgentTrainer(K=K12, M=M12, state_dim=states.shape[1],
                                  epochs=100, batch=POLICY_NET_BATCH,
                                  save_path=CKPT_K12)
    model, history = trainer.train(states, alphas, servers, confidences=confs)
    np.save(os.path.splitext(CKPT_K12)[0] + "_history.npy", np.array(history))
    print(f"  训练完成 best_epoch={trainer.best_epoch} "
          f"(耗时 {time.time()-t0:.0f}s)  -> {CKPT_K12}")

    # ---- K=12 评估 ----
    methods = {}
    for sd in range(N_SEEDS):
        for ep in range(N_EPISODES):
            env = MECEnvironment(num_users=K12, num_servers=M12,
                                 seed=SEED + sd + ep)
            env.reset()
            for name, kind in [("HMA-Distill-K12", "runner"),
                               ("GA", "ga"), ("MPC", "pred"),
                               ("Greedy", "greedy")]:
                e2 = MECEnvironment(num_users=K12, num_servers=M12,
                                    seed=SEED + sd + ep)
                e2.reset()
                if kind == "runner":
                    obj = HMAAgentRunner(env=e2, mode="Distill",
                                         policy_path=CKPT_K12, agents=None)
                elif kind == "ga":
                    obj = GABaseline()
                elif kind == "pred":
                    obj = MPCBaseline()
                else:
                    obj = None  # Greedy: 用辅助
                if kind == "greedy":
                    class _G:
                        def predict(self, st, env):
                            act = np.zeros(2 * K12, np.float32)
                            act[0::2] = 0.5
                            act[1::2] = (np.abs(env.channels).argmax(1) + 0.5) / M12
                            return act
                    obj = _G()
                r = run(e2, obj, "runner" if kind == "runner" else "pred")
                methods.setdefault(name, []).append(r)
    out = {}
    for name, rows in methods.items():
        mean = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        std = {k: float(np.std([r[k] for r in rows])) for k in rows[0]}
        out[name] = {"mean": mean, "std": std}
        print(f"  {name:18s} E={mean['E']:.4f} T={mean['T']:.4f} "
              f"suc={mean['suc']:.1%} sla={mean['sla']:.1%}")
    json.dump(out, open(OUT_JSON, "w"), indent=2, ensure_ascii=False)
    print(f"\n  保存 -> {OUT_JSON}  (总耗时 {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
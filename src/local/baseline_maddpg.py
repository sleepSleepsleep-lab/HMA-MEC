# -*- coding: utf-8 -*-
"""
================================================================
MADDPG 基线 (local/baseline_maddpg.py)
================================================================
MARL 数值对照基线 (B13-MADDPG): 集中式训练-分布式执行 (CTDE)。

与引言第~\ref{sec:related} 节对 MARL 的评述对应:
  - 每个用户 k 一个确定性 actor (局部决策, 输入全局状态向量),
  - 一个集中式 critic (输入全局状态 + 全部 K 个用户的动作) 提供
    全局梯度信号,
  - 训练时执行 CTDE: 各 actor 依据集中 critic 的 Q 值更新,
    部署时仅用各 actor 前向 (无 critic)。

超参与训练预算与 SAC/DDPG 完全对齐 (config.py 同一组常量,
500 episode / 200 步 / buffer 1e5 / batch 128 / 高斯探索噪声 0.1),
保证对比公平。评估口径与 E1 一致: 每 seed 训练后跑
N_EPISODES × N_STEPS, 汇总 mean/std 至 results/e1_maddpg.json。

依赖: torch (CPU/GPU 均可)。5 seeds 并行训练约 30-60 分钟。
================================================================
"""
import os
import sys
import json
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)

from config import (NUM_USERS, NUM_EDGE_SERVERS, RESULTS_DIR, SEED,
                    LR_ACTOR, LR_CRITIC, GAMMA, TAU, BUFFER_CAPACITY,
                    BATCH_SIZE, HIDDEN_DIM, MAX_STEPS)
from environment import MECEnvironment

import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

K, M = NUM_USERS, NUM_EDGE_SERVERS
MADDPG_EPOCHS = 500        # 与 SAC/DDPG 同预算
N_SEEDS, N_EPISODES, N_STEPS = 5, 50, MAX_STEPS
WARMUP_STEPS = 1000
GRAD_CLIP = 1.0
NOISE_SCALE = 0.1
EARLY_STOP_PAT = 30
OUT_JSON = os.path.join(RESULTS_DIR, "e1_maddpg.json")
AD_K = 2                    # 每用户动作维数 (alpha, server 连续)


class ReplayBuffer:
    def __init__(self, capacity=BUFFER_CAPACITY):
        self.cap = capacity
        self.s, self.a, self.r, self.ns, self.d = [], [], [], [], []

    def push(self, s, a, r, ns, d):
        self.s.append(s); self.a.append(a); self.r.append(r)
        self.ns.append(ns); self.d.append(d)
        if len(self.s) > self.cap:
            self.s.pop(0); self.a.pop(0); self.r.pop(0)
            self.ns.pop(0); self.d.pop(0)

    def sample(self, batch):
        idx = np.random.choice(len(self.s), batch, replace=False)
        def pick(x):
            x = np.asarray(x, dtype=np.float32)
            return torch.FloatTensor(x[idx]).to(DEVICE)
        return (pick(self.s), pick(self.a), pick(self.r).unsqueeze(1),
                pick(self.ns), pick(self.d).unsqueeze(1))

    def __len__(self):
        return len(self.s)


class MADDPGActor(nn.Module):
    """用户 k 的确定性 actor: 全局状态 -> (alpha_k, server_k 连续)."""
    def __init__(self, sd, hd=HIDDEN_DIM):
        super().__init__()
        self.fc1 = nn.Linear(sd, hd)
        self.fc2 = nn.Linear(hd, hd)
        self.mu = nn.Linear(hd, AD_K)

    def forward(self, s):
        h = F.leaky_relu(self.fc1(s), 0.2)
        h = F.leaky_relu(self.fc2(h), 0.2)
        return torch.sigmoid(self.mu(h))


class MADDPGCritic(nn.Module):
    """集中式 critic: 全局状态 + 全部动作 -> Q."""
    def __init__(self, sd, ad_total, hd=HIDDEN_DIM):
        super().__init__()
        self.fc1 = nn.Linear(sd + ad_total, hd)
        self.fc2 = nn.Linear(hd, hd)
        self.fc3 = nn.Linear(hd, 1)

    def forward(self, s, a_all):
        h = F.leaky_relu(self.fc1(torch.cat([s, a_all], -1)), 0.2)
        h = F.leaky_relu(self.fc2(h), 0.2)
        return self.fc3(h)


class MADDPGAgent:
    """MADDPG: K 个独立 actor + 单个集中 critic (CTDE)."""

    def __init__(self, env, lr_a=LR_ACTOR, lr_c=LR_CRITIC,
                 gamma=GAMMA, tau=TAU):
        self.env = env
        self.K, self.sd = env.K, env.state_dim
        self.gamma, self.tau = gamma, tau
        self.actors = [MADDPGActor(self.sd).to(DEVICE) for _ in range(self.K)]
        self.actors_t = [MADDPGActor(self.sd).to(DEVICE)
                         for _ in range(self.K)]
        for a, at in zip(self.actors, self.actors_t):
            at.load_state_dict(a.state_dict())
        self.critic = MADDPGCritic(self.sd, self.K * AD_K).to(DEVICE)
        self.critic_t = MADDPGCritic(self.sd, self.K * AD_K).to(DEVICE)
        self.critic_t.load_state_dict(self.critic.state_dict())
        self.opt_a = [torch.optim.Adam(a.parameters(), lr=lr_a)
                      for a in self.actors]
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=lr_c)
        self.buf = ReplayBuffer()
        self.noise_scale = NOISE_SCALE
        self.name = "MADDPG"

    def _act_all(self, state, noise=False):
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            acts = [a(s) for a in self.actors]
            a_all = torch.cat(acts, dim=-1).squeeze(0).cpu().numpy()
        if noise:
            a_all = a_all + np.random.normal(0, self.noise_scale, a_all.shape)
        return np.clip(a_all, 0, 1).astype(np.float32)

    def select_action(self, state, evaluate=False):
        return self._act_all(state, noise=not evaluate)

    def predict(self, state, env):
        return self.select_action(state, evaluate=True)

    def update(self):
        if len(self.buf) < max(BATCH_SIZE, 1000):
            return
        s, a, r, ns, d = self.buf.sample(BATCH_SIZE)
        with torch.no_grad():
            na = torch.cat([at(ns) for at in self.actors_t], dim=-1)
            tgt = r + (1 - d) * self.gamma * self.critic_t(ns, na)
        q = self.critic(s, a)
        loss = F.mse_loss(q, tgt)
        self.opt_c.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), GRAD_CLIP)
        self.opt_c.step()
        # 各 actor 的确定性策略梯度 (DPG)
        for k, (actor, opt) in enumerate(zip(self.actors, self.opt_a)):
            a_new = torch.cat([actor(s) if j == k else
                               a[:, j * AD_K:(j + 1) * AD_K]
                               for j, actor in enumerate(self.actors)], dim=-1)
            a_loss = -self.critic(s, a_new).mean()
            opt.zero_grad(); a_loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), GRAD_CLIP)
            opt.step()
        # 软更新
        for actor, at in zip(self.actors, self.actors_t):
            for p, tp in zip(actor.parameters(), at.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
        for p, tp in zip(self.critic.parameters(), self.critic_t.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

    def train(self, env, episodes=MADDPG_EPOCHS,
              early_stop_pat=EARLY_STOP_PAT, verbose=True):
        history = []
        best = -np.inf; pat = 0
        for ep in range(episodes):
            s = env.reset(); total = 0.0
            for _ in range(MAX_STEPS):
                if len(self.buf) < WARMUP_STEPS:
                    a = np.random.uniform(0, 1, self.K * AD_K).astype(np.float32)
                else:
                    a = self.select_action(s, evaluate=False)
                ns, r, d, _ = env.step(a)
                self.buf.push(s, a, r, ns, d)
                if len(self.buf) >= BATCH_SIZE * 10:
                    self.update()
                s = ns; total += r
                if d:
                    break
            history.append(total)
            if total > best:
                best = total; pat = 0
            else:
                pat += 1
            if verbose and (ep + 1) % 10 == 0:
                print(f"  MADDPG ep {ep+1:3d}  ep_reward={total:.3f}  best={best:.3f}")
            if pat >= early_stop_pat:
                if verbose:
                    print(f"  MADDPG early stop at ep {ep+1}")
                break
        return history


def evaluate(method, env, n_steps=N_STEPS):
    """与 E1 相同的评估口径: 一个 episode 的均值指标."""
    s = env.reset()
    energy, lat, suc, sla = [], [], [], []
    for _ in range(n_steps):
        a = method.predict(s, env)
        ns, _, d, info = env.step(a)
        energy.append(info['energy']); lat.append(info['latency'])
        suc.append(info['success_rate']); sla.append(info['priority_sla'])
        s = ns
        if d:
            break
    return (float(np.mean(energy)), float(np.mean(lat)),
            float(np.mean(suc)), float(np.mean(sla)))


def run_seed(sd):
    """单个 seed: 训练 MADDPG 500ep 并在 N_EPISODES 个 episode 上评估."""
    env = MECEnvironment(num_users=K, num_servers=M, seed=SEED + sd)
    agent = MADDPGAgent(env)
    t0 = time.time()
    hist = agent.train(env, episodes=MADDPG_EPOCHS, verbose=False)
    dt = time.time() - t0
    E, T, S, SL = [], [], [], []
    for ep in range(N_EPISODES):
        e = MECEnvironment(num_users=K, num_servers=M, seed=SEED + sd + ep)
        r = evaluate(agent, e)
        E.append(r[0]); T.append(r[1]); S.append(r[2]); SL.append(r[3])
    print(f"  [seed {sd}] 训练 {len(hist)}ep ({dt:.0f}s)  "
          f"E={np.mean(E):.4f} T={np.mean(T):.3f} suc={np.mean(S):.2%}")
    return (float(np.mean(E)), float(np.mean(T)),
            float(np.mean(S)), float(np.mean(SL)))


def main():
    print("=" * 60)
    print(f"  MADDPG 基线 (B13): {N_SEEDS} seeds × {MADDPG_EPOCHS}ep 训练 "
          f"+ {N_EPISODES}ep 评估")
    print(f"  设备: {DEVICE}")
    print("=" * 60)
    t0 = time.time()
    ctx = __import__("multiprocessing").get_context("spawn")
    with ProcessPoolExecutor(max_workers=N_SEEDS, mp_context=ctx) as ex:
        per_seed = list(ex.map(run_seed, range(N_SEEDS)))
    E = np.array([r[0] for r in per_seed])
    T = np.array([r[1] for r in per_seed])
    S = np.array([r[2] for r in per_seed])
    SL = np.array([r[3] for r in per_seed])
    out = {
        "MADDPG": {
            "mean": {"energy": float(E.mean()), "latency": float(T.mean()),
                     "success_rate": float(S.mean()),
                     "priority_sla": float(SL.mean())},
            "std": {"energy": float(E.std()), "latency": float(T.std()),
                    "success_rate": float(S.std()),
                    "priority_sla": float(SL.std())},
            "n_samples": N_SEEDS * N_EPISODES,
            "per_seed": per_seed,
            "epochs": MADDPG_EPOCHS,
        }
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    json.dump(out, open(OUT_JSON, "w"), indent=2, ensure_ascii=False)
    print(f"  MADDPG: E={out['MADDPG']['mean']['energy']:.4f} "
          f"T={out['MADDPG']['mean']['latency']:.3f} "
          f"suc={out['MADDPG']['mean']['success_rate']:.2%} "
          f"sla={out['MADDPG']['mean']['priority_sla']:.2%}")
    print(f"  保存 -> {OUT_JSON} (总耗时 {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
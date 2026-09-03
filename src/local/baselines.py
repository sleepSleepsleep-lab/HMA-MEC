# -*- coding: utf-8 -*-
"""
================================================================
基线方法实现 (local/baselines.py)
================================================================
本文件实现论文中所有基线方法的本地 CPU 推理（对比 HMA-MEC 用）：

  1. Greedy           —— 贪心: 按信道最优服务器; alpha=0.5
  2. AllLocal         —— 全本地; alpha=0.99
  3. AllEdge          —— 全卸载至最优信道服务器; alpha=0.01
  4. Random           —— 均匀随机采样
  5. SAC              —— 单体软演员-评论家 (迭代训练)
  6. DDPG             —— 单体 DDPG (迭代训练)

所有方法统一接口:
  predict(state, env) -> action np.array(2K,)
  train(env, episodes) -> history (仅 SAC/DDPG)

随机种子、训练超参数均控制在文件顶部「显眼配置区」, 便于修改。
================================================================
"""

import os
import sys
import time
import random
import logging
from typing import Optional, List, Dict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)

from config import (
    NUM_USERS, NUM_EDGE_SERVERS, STATE_DIM, ACTION_DIM,
    LR_ACTOR, LR_CRITIC, GAMMA, TAU, BUFFER_CAPACITY, BATCH_SIZE,
    HIDDEN_DIM, MAX_EPISODES, MAX_STEPS, SEED,
    KAPPA_LOCAL, P_IDLE, BANDWIDTH, NOISE_POWER, TX_POWER_USER,
)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Normal
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    torch = None; nn = object; F = None; DEVICE = "cpu"

logger = logging.getLogger(__name__)

# ============================================================
# >>>>>>>>>>>>>>>>>>  显眼配置区  <<<<<<<<<<<<<<<<<<
# ============================================================
# 这里集中所有可能需要修改的参数,特别是训练时长。
# 实际实验请把 SAC_EPOCHS / DDPG_EPOCHS 调到 500+; 自检 50 即可。
# ============================================================
SAC_EPOCHS = 50               # SAC 训练 episode 数 (小规模自检用 50; 论文 500)
DDPG_EPOCHS = 50              # DDPG 训练 episode 数
WARMUP_STEPS = 1000           # 随机探索步数 (在 SAC/DDPG 训练前)
EARLY_STOP_PAT = 30           # 早停 patience
GRAD_CLIP = 1.0               # 梯度裁剪范数


# ============================================================
# 启发式基线
# ============================================================
class GreedyBaseline:
    """贪心: 每用户选信道增益最大的服务器, alpha=0.5."""
    def __init__(self, env=None):
        self.env = env
        self.name = "Greedy"

    def predict(self, state, env):
        K, M = env.K, env.M
        action = np.zeros(env.action_dim, dtype=np.float32)
        for k in range(K):
            best_m = int(np.argmax(env.channels[k]))
            action[2 * k]     = 0.5              # alpha (本地比例)
            action[2 * k + 1] = (best_m + 0.5) / M  # 归一化服务器选择
        return action


class AllLocalBaseline:
    """所有任务本地执行; alpha=0.99, 服务器任意."""
    def __init__(self, env=None):
        self.env = env; self.name = "AllLocal"

    def predict(self, state, env):
        K = env.K; M = env.M
        action = np.zeros(env.action_dim, dtype=np.float32)
        for k in range(K):
            action[2 * k]     = 0.99
            action[2 * k + 1] = 0.5 / M
        return action


class AllEdgeBaseline:
    """所有任务卸载至每用户最优信道服务器; alpha=0.01."""
    def __init__(self, env=None):
        self.env = env; self.name = "AllEdge"

    def predict(self, state, env):
        K, M = env.K, env.M
        action = np.zeros(env.action_dim, dtype=np.float32)
        for k in range(K):
            best_m = int(np.argmax(env.channels[k]))
            action[2 * k]     = 0.01
            action[2 * k + 1] = (best_m + 0.5) / M
        return action


class RandomBaseline:
    """均匀随机决策."""
    def __init__(self, env=None):
        self.env = env; self.name = "Random"

    def predict(self, state, env):
        return np.random.uniform(0, 1, env.action_dim).astype(np.float32)


# ============================================================
# 简化 SAC (单智能体)
# ============================================================
if torch is not None:

    class ReplayBuffer:
        def __init__(self, capacity=BUFFER_CAPACITY):
            from collections import deque
            self.buf = deque(maxlen=capacity)
        def push(self, *args):
            self.buf.append(args)
        def sample(self, n):
            batch = random.sample(self.buf, min(len(self.buf), n))
            s, a, r, ns, d = zip(*batch)
            return (torch.FloatTensor(np.array(s)).to(DEVICE),
                    torch.FloatTensor(np.array(a)).to(DEVICE),
                    torch.FloatTensor(np.array(r)).unsqueeze(1).to(DEVICE),
                    torch.FloatTensor(np.array(ns)).to(DEVICE),
                    torch.FloatTensor(np.array(d)).unsqueeze(1).to(DEVICE))
        def __len__(self): return len(self.buf)


    class SACActor(nn.Module):
        def __init__(self, sd, ad, hd=HIDDEN_DIM):
            super().__init__()
            self.fc1 = nn.Linear(sd, hd); self.ln1 = nn.LayerNorm(hd)
            self.fc2 = nn.Linear(hd, hd);  self.ln2 = nn.LayerNorm(hd)
            self.mu  = nn.Linear(hd, ad)
            self.log_std = nn.Linear(hd, ad)
            nn.init.constant_(self.log_std.bias, -0.5)
        def forward(self, s):
            h = F.leaky_relu(self.ln1(self.fc1(s)), 0.2)
            h = F.leaky_relu(self.ln2(self.fc2(h)), 0.2)
            mu = torch.sigmoid(self.mu(h))
            log_std = torch.clamp(self.log_std(h), -20, 2)
            return mu, log_std
        def sample(self, s):
            mu, log_std = self.forward(s)
            std = log_std.exp()
            dist = Normal(mu, std)
            z = dist.rsample()
            a = torch.sigmoid(z)
            log_p = dist.log_prob(z) - torch.log(a*(1-a) + 1e-6)
            return a, log_p.sum(-1, keepdim=True), mu


    class SACCritic(nn.Module):
        def __init__(self, sd, ad, hd=HIDDEN_DIM):
            super().__init__()
            self.fc1 = nn.Linear(sd + ad, hd)
            self.fc2 = nn.Linear(hd, hd)
            self.fc3 = nn.Linear(hd, 1)
        def forward(self, s, a):
            h = F.leaky_relu(self.fc1(torch.cat([s, a], -1)), 0.2)
            h = F.leaky_relu(self.fc2(h), 0.2)
            return self.fc3(h)


    class SACAgent:
        """简化 SAC, 训练 -> 推理."""
        def __init__(self, env,
                     lr_actor=LR_ACTOR, lr_critic=LR_CRITIC,
                     gamma=GAMMA, tau=TAU, alpha=0.2):
            self.env = env
            self.sd = env.state_dim; self.ad = env.action_dim
            self.gamma, self.tau, self.alpha = gamma, tau, alpha
            self.actor = SACActor(self.sd, self.ad).to(DEVICE)
            self.c1 = SACCritic(self.sd, self.ad).to(DEVICE)
            self.c2 = SACCritic(self.sd, self.ad).to(DEVICE)
            self.c1t = SACCritic(self.sd, self.ad).to(DEVICE)
            self.c2t = SACCritic(self.sd, self.ad).to(DEVICE)
            self.c1t.load_state_dict(self.c1.state_dict())
            self.c2t.load_state_dict(self.c2.state_dict())
            self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
            self.opt_c1 = torch.optim.Adam(self.c1.parameters(), lr=lr_critic)
            self.opt_c2 = torch.optim.Adam(self.c2.parameters(), lr=lr_critic)
            self.log_alpha = torch.tensor(np.log(alpha), requires_grad=True,
                                           device=DEVICE)
            self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=lr_critic)
            self.target_entropy = -float(self.ad)
            self.buf = ReplayBuffer()
            self.name = "SAC"

        def select_action(self, state, evaluate=False):
            with torch.no_grad():
                s = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                if evaluate:
                    mu, _ = self.actor(s); return mu.squeeze(0).cpu().numpy()
                a, _, _ = self.actor.sample(s); return a.squeeze(0).cpu().numpy()

        def predict(self, state, env):
            return self.select_action(state, evaluate=True)

        def update(self):
            if len(self.buf) < max(BATCH_SIZE, 1000): return
            s, a, r, ns, d = self.buf.sample(BATCH_SIZE)
            with torch.no_grad():
                na, nlog_p, _ = self.actor.sample(ns)
                tgt1 = self.c1t(ns, na); tgt2 = self.c2t(ns, na)
                tgt = torch.min(tgt1, tgt2) - self.log_alpha.exp() * nlog_p
                y = r + (1 - d) * self.gamma * tgt
            for c, opt, ct in [(self.c1, self.opt_c1, self.c1t),
                               (self.c2, self.opt_c2, self.c2t)]:
                q = c(s, a); loss = F.mse_loss(q, y)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(c.parameters(), GRAD_CLIP)
                opt.step()
            a_new, log_p, _ = self.actor.sample(s)
            q1 = self.c1(s, a_new); q2 = self.c2(s, a_new)
            a_loss = (self.log_alpha.exp() * log_p -
                      torch.min(q1, q2)).mean()
            self.opt_a.zero_grad(); a_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), GRAD_CLIP)
            self.opt_a.step()
            # 温度自适应
            alpha_loss = -(self.log_alpha * (log_p + self.target_entropy)
                            .detach()).mean()
            self.opt_alpha.zero_grad(); alpha_loss.backward(); self.opt_alpha.step()
            for c, ct in [(self.c1, self.c1t), (self.c2, self.c2t)]:
                for p, tp in zip(c.parameters(), ct.parameters()):
                    tp.data.copy_(self.tau * p.data + (1-self.tau)*tp.data)

        def train(self, env, episodes=SAC_EPOCHS,
                  early_stop_pat=EARLY_STOP_PAT, verbose=True):
            history = []
            best = -np.inf; pat = 0
            for ep in range(episodes):
                s = env.reset(); total = 0.0
                for _ in range(MAX_STEPS):
                    if len(self.buf) < WARMUP_STEPS:
                        a = np.random.uniform(0, 1, self.ad).astype(np.float32)
                    else:
                        a = self.select_action(s, evaluate=False)
                    ns, r, d, _ = env.step(a)
                    self.buf.push(s, a, r, ns, d)
                    if len(self.buf) >= BATCH_SIZE * 10:
                        self.update()
                    s = ns; total += r
                    if d: break
                history.append(total)
                if total > best: best = total; pat = 0
                else: pat += 1
                if verbose and (ep+1) % 10 == 0:
                    print(f"  SAC ep {ep+1:3d}  ep_reward={total:.3f}  best={best:.3f}")
                if pat >= early_stop_pat:
                    if verbose: print(f"  SAC early stop at ep {ep+1}")
                    break
            return history


    # ============================================================
    # 简化 DDPG
    # ============================================================
    class DDPGActor(nn.Module):
        def __init__(self, sd, ad, hd=HIDDEN_DIM):
            super().__init__()
            self.fc1 = nn.Linear(sd, hd); self.fc2 = nn.Linear(hd, hd)
            self.mu = nn.Linear(hd, ad)
        def forward(self, s):
            h = F.leaky_relu(self.fc1(s), 0.2)
            h = F.leaky_relu(self.fc2(h), 0.2)
            return torch.sigmoid(self.mu(h))


    class DDPGCritic(nn.Module):
        def __init__(self, sd, ad, hd=HIDDEN_DIM):
            super().__init__()
            self.fc1 = nn.Linear(sd + ad, hd)
            self.fc2 = nn.Linear(hd, hd)
            self.fc3 = nn.Linear(hd, 1)
        def forward(self, s, a):
            h = F.leaky_relu(self.fc1(torch.cat([s, a], -1)), 0.2)
            h = F.leaky_relu(self.fc2(h), 0.2)
            return self.fc3(h)


    class DDPGAgent:
        def __init__(self, env, lr_a=LR_ACTOR, lr_c=LR_CRITIC,
                     gamma=GAMMA, tau=TAU):
            self.env = env
            self.sd, self.ad = env.state_dim, env.action_dim
            self.gamma, self.tau = gamma, tau
            self.actor = DDPGActor(self.sd, self.ad).to(DEVICE)
            self.actor_t = DDPGActor(self.sd, self.ad).to(DEVICE)
            self.actor_t.load_state_dict(self.actor.state_dict())
            self.critic = DDPGCritic(self.sd, self.ad).to(DEVICE)
            self.critic_t = DDPGCritic(self.sd, self.ad).to(DEVICE)
            self.critic_t.load_state_dict(self.critic.state_dict())
            self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=lr_a)
            self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=lr_c)
            self.buf = ReplayBuffer()
            self.noise_scale = 0.1
            self.name = "DDPG"

        def select_action(self, state, evaluate=False):
            with torch.no_grad():
                s = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                a = self.actor(s).squeeze(0).cpu().numpy()
            if not evaluate:
                a = a + np.random.normal(0, self.noise_scale, a.shape)
            return np.clip(a, 0, 1).astype(np.float32)

        def predict(self, state, env):
            return self.select_action(state, evaluate=True)

        def update(self):
            if len(self.buf) < max(BATCH_SIZE, 1000): return
            s, a, r, ns, d = self.buf.sample(BATCH_SIZE)
            with torch.no_grad():
                na = self.actor_t(ns)
                tgt = r + (1 - d) * self.gamma * self.critic_t(ns, na)
            q = self.critic(s, a); loss = F.mse_loss(q, tgt)
            self.opt_c.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), GRAD_CLIP)
            self.opt_c.step()
            a_new = self.actor(s)
            a_loss = -self.critic(s, a_new).mean()
            self.opt_a.zero_grad(); a_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), GRAD_CLIP)
            self.opt_a.step()
            for c, ct in [(self.actor, self.actor_t),
                           (self.critic, self.critic_t)]:
                for p, tp in zip(c.parameters(), ct.parameters()):
                    tp.data.copy_(self.tau * p.data + (1-self.tau)*tp.data)

        def train(self, env, episodes=DDPG_EPOCHS,
                  early_stop_pat=EARLY_STOP_PAT, verbose=True):
            history = []
            best = -np.inf; pat = 0
            for ep in range(episodes):
                s = env.reset(); total = 0.0
                for _ in range(MAX_STEPS):
                    if len(self.buf) < WARMUP_STEPS:
                        a = np.random.uniform(0, 1, self.ad).astype(np.float32)
                    else:
                        a = self.select_action(s, evaluate=False)
                    ns, r, d, _ = env.step(a)
                    self.buf.push(s, a, r, ns, d)
                    if len(self.buf) >= BATCH_SIZE * 10:
                        self.update()
                    s = ns; total += r
                    if d: break
                history.append(total)
                if total > best: best = total; pat = 0
                else: pat += 1
                if verbose and (ep+1) % 10 == 0:
                    print(f"  DDPG ep {ep+1:3d}  ep_reward={total:.3f}  best={best:.3f}")
                if pat >= early_stop_pat:
                    if verbose: print(f"  DDPG early stop at ep {ep+1}")
                    break
            return history


# ============================================================
# 统一 eval 接口: 给定方法, 返回评估指标
# ============================================================
def evaluate(method, env, n_steps=MAX_STEPS) -> Dict:
    """统一评估接口: 跑一个 episode, 返回能耗/时延/成功率/SLA 均值."""
    s = env.reset()
    energy, lat, suc, sla = [], [], [], []
    for _ in range(n_steps):
        a = method.predict(s, env)
        ns, _, d, info = env.step(a)
        sla.append(info.get('priority_sla', 0.0))
        energy.append(info['energy']); lat.append(info['latency'])
        suc.append(info['success_rate'])
        s = ns
        if d: break
    return {
        'energy': float(np.mean(energy)),
        'latency': float(np.mean(lat)),
        'success_rate': float(np.mean(suc)),
        'priority_sla': float(np.mean(sla)),
    }


def get_all_baselines(env, trainable=True):
    """返回所有 baseline 的实例列表."""
    methods = [
        GreedyBaseline(env),
        AllLocalBaseline(env),
        AllEdgeBaseline(env),
        RandomBaseline(env),
    ]
    if trainable and torch is not None:
        methods.append(SACAgent(env))
        methods.append(DDPGAgent(env))
    return methods


# ============================================================
# 自检
# ============================================================
if __name__ == "__main__":
    from environment import MECEnvironment
    print("=" * 60)
    print("  baselines 自检 (CPU, 5 epoch)")
    print("=" * 60)
    env = MECEnvironment(num_users=8, num_servers=4, seed=42)
    env.reset()

    # 启发式直接评估
    for m in [GreedyBaseline(env), AllLocalBaseline(env),
              AllEdgeBaseline(env), RandomBaseline(env)]:
        r = evaluate(m, env, n_steps=50)
        print(f"  {m.name:>10s}  E={r['energy']:.5f}  "
              f"T={r['latency']:.3f}  suc={r['success_rate']:.2%}  "
              f"sla≈{r['priority_sla']:.2%}")

    if torch is not None:
        sac = SACAgent(env)
        sac.train(env, episodes=5, verbose=False)
        r = evaluate(sac, env, n_steps=50)
        print(f"  {'SAC':>10s}  E={r['energy']:.5f}  "
              f"T={r['latency']:.3f}  suc={r['success_rate']:.2%}  "
              f"sla≈{r['priority_sla']:.2%}")
        ddpg = DDPGAgent(env)
        ddpg.train(env, episodes=5, verbose=False)
        r = evaluate(ddpg, env, n_steps=50)
        print(f"  {'DDPG':>10s}  E={r['energy']:.5f}  "
              f"T={r['latency']:.3f}  suc={r['success_rate']:.2%}  "
              f"sla≈{r['priority_sla']:.2%}")
    print("=" * 60)
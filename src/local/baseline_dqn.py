# -*- coding: utf-8 -*-
"""
================================================================
DQN 基线 (local/baseline_dqn.py)
================================================================
2026-08 新增: 深强化学习 (离散) 基线, 与 SAC/DDPG (连续) 互为补充。
对齐问题形式化: 奖励 = -(λe·E + λt·T)/scale, 与评估目标一致 (M2 整改)。

设计 (CTDE 简化):
  - 每用户独立决策动作 (alpha 离散 bin × server), K 个用户共享 Q 网络,
    参数输入含 (任务特征 + 全部服务器特征);
  - 联合动作施加到环境, 每步共享全局奖励 (E+T 加权) 归入各用户经验;
  - 标准 DQN: epsilon-greedy / replay buffer / target network / TD(0)。
训练规模与 SAC/DDPG 一致 (500 ep, E1 论文规模)。
================================================================
"""

import numpy as np

from config import (HIDDEN_DIM, LR_ACTOR, GAMMA, BUFFER_CAPACITY)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except Exception:                       # pragma: no cover
    torch = None; nn = object; F = None; optim = None; DEVICE = "cpu"

ALPHA_BINS = (0.05, 0.2, 0.4, 0.6, 0.8, 0.95)
N_ACTIONS = len(ALPHA_BINS) * 4          # alpha bin × server(4)

DQN_BUFFER = 20000
DQN_BATCH = 64
TARGET_UPDATE = 100
EPS_START, EPS_END, EPS_DECAY = 0.9, 0.05, 0.995


def _user_state(env, k):
    """每用户决策子状态: 任务特征 + 各服务器 (算力, 负载, 信道)。"""
    t = env.tasks[k]
    feats = [t['D'], t['C'], t['tau'], t['priority']]
    for m in range(env.M):
        feats += [np.log10(max(env.f_edge[m], 1e-9)),
                  float(env.server_load[m]),
                  float(env.channels[k, m])]
    return np.asarray(feats, dtype=np.float32)


def _decode(idx):
    a_i, s_i = divmod(int(idx), 4)
    return ALPHA_BINS[a_i], s_i


if torch is not None:
    class QuantNet(nn.Module):
        def __init__(self, sd, na, hd=HIDDEN_DIM):
            super().__init__()
            self.fc1 = nn.Linear(sd, hd); self.ln1 = nn.LayerNorm(hd)
            self.fc2 = nn.Linear(hd, hd); self.ln2 = nn.LayerNorm(hd)
            self.out = nn.Linear(hd, na)
        def forward(self, x):
            h = F.relu(self.ln1(self.fc1(x)))
            h = F.relu(self.ln2(self.fc2(h)))
            return self.out(h)


class DQNAgent:
    def __init__(self, env=None):
        self.name = "DQN"
        self.env = env
        if env is None:
            self.sd = 16; self.M = 4; self.Q = None; return
        self.sd = len(_user_state(env, 0))
        self.M = env.M
        self.Q = QuantNet(self.sd, N_ACTIONS).to(DEVICE)
        self.Qt = QuantNet(self.sd, N_ACTIONS).to(DEVICE)
        self.Qt.load_state_dict(self.Q.state_dict())
        self.opt = optim.Adam(self.Q.parameters(), lr=LR_ACTOR)
        self.buf = []
        self.buf_cap = DQN_BUFFER
        self.gamma = GAMMA
        self.steps = 0

    def _q(self, env, k):
        s = torch.FloatTensor(_user_state(env, k)).unsqueeze(0).to(DEVICE)
        return self.Q(s).squeeze(0)

    def _sample_q(self, env, k, eps):
        if np.random.rand() < eps:
            return int(np.random.randint(N_ACTIONS))
        return int(self._q(env, k).detach().cpu().argmax().item())

    def predict(self, state, env, eps=0.0):
        K = env.K; M = env.M
        action = np.zeros(2 * K, dtype=np.float32)
        with torch.no_grad():
            for k in range(K):
                idx = self._sample_q(env, k, eps)
                a_i, s_i = _decode(idx)
                action[2 * k] = a_i
                action[2 * k + 1] = (s_i + 0.5) / M
        return action

    def _train_step(self, batch):
        s = torch.FloatTensor(np.array([b[0] for b in batch])).to(DEVICE)
        a = torch.LongTensor([b[1] for b in batch]).unsqueeze(1).to(DEVICE)
        r = torch.FloatTensor([b[2] for b in batch]).unsqueeze(1).to(DEVICE)
        ns = torch.FloatTensor(np.array([b[3] for b in batch])).to(DEVICE)
        d = torch.FloatTensor([b[4] for b in batch]).unsqueeze(1).to(DEVICE)
        q = self.Q(s).gather(1, a)
        with torch.no_grad():
            tgt = r + self.gamma * (1 - d) * self.Qt(ns).max(1, keepdim=True)[0]
        loss = F.mse_loss(q, tgt)
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        return float(loss.item())

    def train(self, env, episodes=500, verbose=True):
        K, M = env.K, env.M
        eps = EPS_START
        for ep in range(episodes):
            ns = env.reset()
            done = False
            total_r = 0.0
            while not done:
                # 每用户独立经验: 决策联合动作
                s_states = [_user_state(env, k) for k in range(K)]
                a_idxs = [self._sample_q(env, k, eps) for k in range(K)]
                action = np.zeros(2 * K, dtype=np.float32)
                for k in range(K):
                    a_i, s_i = _decode(a_idxs[k])
                    action[2 * k] = a_i
                    action[2 * k + 1] = (s_i + 0.5) / M
                ns, _, done, info = env.step(action)
                # 奖励对齐评估目标 (无 ad hoc bonus): r = -(E + T)
                r = -(info['energy'] + info['latency'])
                total_r += r
                ns_states = [_user_state(env, k) for k in range(K)]
                for k in range(K):
                    self.buf.append((s_states[k], a_idxs[k], r, ns_states[k], float(done)))
                if len(self.buf) > self.buf_cap:
                    self.buf = self.buf[-self.buf_cap:]
                # 每步训练
                if len(self.buf) >= DQN_BATCH:
                    idx = np.random.choice(len(self.buf), DQN_BATCH, replace=False)
                    self._train_step([self.buf[i] for i in idx])
                self.steps += 1
                if self.steps % TARGET_UPDATE == 0:
                    self.Qt.load_state_dict(self.Q.state_dict())
            eps = max(EPS_END, eps * EPS_DECAY)
            if verbose and (ep + 1) % 50 == 0:
                print(f"    [DQN] ep {ep+1}/{episodes} R={total_r:.2f} eps={eps:.3f}")
        return self

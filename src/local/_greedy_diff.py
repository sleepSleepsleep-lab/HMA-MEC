# -*- coding: utf-8 -*-
"""临时差分验证脚本:对比 src 与 src_backup 的 Greedy 数值是否一致(C1/C2 向后兼容)。
用法: python _greedy_diff.py <src_dir> <seeds...>
"""
import sys
import numpy as np

SRC = sys.argv[1]
sys.path.insert(0, SRC)

from environment import MECEnvironment
from config import NUM_USERS, NUM_EDGE_SERVERS


def greedy_eval(seed):
    env = MECEnvironment(num_users=NUM_USERS, num_servers=NUM_EDGE_SERVERS,
                         seed=seed)
    env.reset()
    e = t = suc = sla = 0.0
    for _ in range(20):
        action = np.zeros(env.action_dim, dtype=np.float32)
        for k in range(env.K):
            action[k * 2] = 0.5
            action[k * 2 + 1] = (float(np.argmax(env.channels[k])) + 0.5) / env.M
        _, _, _, info = env.step(action)
        e += info['energy']
        t += info['latency']
        suc += info['success_rate']
        sla += info['priority_sla']
    return round(e / 20, 6), round(t / 20, 6), round(suc / 20, 6), round(sla / 20, 6)


if __name__ == "__main__":
    seeds = [int(x) for x in sys.argv[2:]] or [0, 1]
    for seed in seeds:
        print(seed, greedy_eval(seed))

# -*- coding: utf-8 -*-
"""
================================================================
MEC / VEC 仿真环境 (environment.py)
================================================================
本文件实现端-边-云三层计算卸载仿真环境，主要新增内容 (2026.08)：
  1. 有限码长(FBL)通信模型：替代 Shannon 无限码长假设
  2. 云端卸载通道：边缘过载时任务自动溢出到云
  3. VEC 移动场景环境（vec_environment.py 中 MECEnvironment 子类）
  4. simulate(plan) 反事实仿真接口：供 Verifier Agent 评估候选方案
  5. state_to_text() 自然语言转换接口

调用模式：
  - step(action):  真实推进系统状态
  - simulate(plan): 在当前状态副本上预演，不改变真实状态
================================================================
"""

import copy
import numpy as np
try:
    from scipy.stats import norm as _scipy_norm
except ImportError:
    _scipy_norm = None
from config import (
    NUM_USERS, NUM_EDGE_SERVERS, F_LOCAL, F_EDGE, F_CLOUD,
    BANDWIDTH, NOISE_POWER, TX_POWER_USER, KAPPA_LOCAL, P_IDLE,
    CHANNEL_CORR_COEF, CHANNEL_COEFF_SCALE,
    TASK_DATA_MIN, TASK_DATA_MAX, TASK_CYCLES_PER_BIT,
    TASK_DEADLINE_MIN, TASK_DEADLINE_MAX, TASK_PRIORITY_PROB,
    MAX_STEPS, ACCOUNT_EDGE_ENERGY, KAPPA_EDGE,
    FBL_ENABLED, FBL_BLOCKLENGTH, FBL_MAX_ERROR_PROB,
    ENABLE_CLOUD_OFFLOAD, CLOUD_LATENCY_BASE, CLOUD_TRANSMISSION_FACTOR,
)


def compute_fbl_rate(h, B, P_tx, N0, n_block, epsilon_max):
    """有限码长 (FBL) 可达速率 / Shannon 无限码长速率。

    参考：Y. Polyanskiy et al., "Channel coding rate in the finite
    blocklength regime," IEEE TIT, 2010.

    FBL 可达速率:
        R = B*log2(1+SNR) - B*sqrt(V/n)*Q^{-1}(epsilon)
    其中 V = 1 - 1/(1+SNR)^2 为信道散度, n 为码长, epsilon 为误码率。

    参数:
        h:          信道功率增益
        B:          带宽 (Hz)
        P_tx:       发射功率 (W)
        N0:         噪声功率谱密度 (W/Hz)  [注: 此处实为总噪声功率]
        n_block:    码元数
        epsilon_max: 最大可容忍误码率
    返回:
        rate_shannon:  Shannon 理论速率 (bps)
        rate_fbl:      FBL 可达速率 (bps, >=0)
        outage_flag:   bool, 表示在当前衰落和误码率要求下是否不可达
    """
    snr = P_tx * h / max(N0, 1e-12)
    C_per_hz = np.log2(1.0 + snr)
    rate_shannon = B * C_per_hz

    if not FBL_ENABLED or _scipy_norm is None:
        return rate_shannon, rate_shannon, False

    V = 1.0 - 1.0 / (1.0 + snr) ** 2
    V = max(V, 1e-8)
    Q_inv = -_scipy_norm.ppf(float(epsilon_max))
    penalty = B * np.sqrt(V / max(n_block, 1)) * Q_inv
    rate_fbl = max(rate_shannon - penalty, 1.0)
    # 标记衰落太深导致 FBL 速率为零（有效丢包）
    outage = rate_fbl <= 1.0 and snr < 0.1
    return rate_shannon, rate_fbl, outage


class MECEnvironment:
    """多用户多服务器 MEC 仿真环境。

    与 Gymnasium 接口风格一致：reset() / step(action) / _get_state()。
    state_dim = 4K + 2M，action_dim = 2K。
    """

    def __init__(self, num_users=NUM_USERS,
                 num_servers=NUM_EDGE_SERVERS, seed=None):
        self.K = num_users
        self.M = num_servers
        self.state_dim = self.K * 4 + self.M * 2
        self.action_dim = self.K * 2

        self._rng = np.random.RandomState(seed) if seed is not None else np.random

        if self.K <= len(F_LOCAL):
            self.f_local = np.array(F_LOCAL[:self.K], dtype=np.float64)
        else:
            self.f_local = self._rng.uniform(0.8e9, 1.8e9, self.K)
        if self.M <= len(F_EDGE):
            self.f_edge = np.array(F_EDGE[:self.M], dtype=np.float64)
        else:
            self.f_edge = self._rng.uniform(7e9, 15e9, self.M)
        self.f_cloud = F_CLOUD

        self.reset()

    # ==================== 状态管理 ====================

    def reset(self):
        """重置环境到新 episode 的初始状态。

        返回：初始状态向量 np.ndarray(shape=(state_dim,), dtype=float32)
        """
        self.step_count = 0
        # 任务画像：每个用户随机生成 (数据量, 计算量, 时延约束, 优先级)
        self.tasks = []
        for k in range(self.K):
            D = float(self._rng.uniform(TASK_DATA_MIN, TASK_DATA_MAX))
            C = D * float(self._rng.uniform(*TASK_CYCLES_PER_BIT))
            tau = float(self._rng.uniform(TASK_DEADLINE_MIN, TASK_DEADLINE_MAX))
            p = int(self._rng.choice([1, 2, 3], p=TASK_PRIORITY_PROB))
            self.tasks.append({'D': D, 'C': C, 'tau': tau, 'priority': p})

        # 信道增益矩阵 (K, M)，复高斯 AR(1) 模型
        # 复信道系数 g ~ CN(0, 1)，功率增益 |g|^2
        self.channel_coeffs = ((self._rng.randn(self.K, self.M)
                                + 1j * self._rng.randn(self.K, self.M))
                               * np.sqrt(0.5) * CHANNEL_COEFF_SCALE)
        self.channels = np.abs(self.channel_coeffs) ** 2
        # 服务器已加载负载（CPU 周期数）
        self.server_load = np.zeros(self.M)
        return self._get_state()

    def _get_state(self):
        """把内部状态编码为状态向量。

        编码规则：前 4K 维为每用户的 (D/D_max, C/(D_max·cycle_max),
        tau/tau_max, p/3)；后 2M 维为每服务器的 (L/F, min_h/h_scale)。
        """
        state = []
        for k in range(self.K):
            t = self.tasks[k]
            state.extend([
                t['D'] / TASK_DATA_MAX,
                t['C'] / (TASK_DATA_MAX * TASK_CYCLES_PER_BIT[1]),
                t['tau'] / TASK_DEADLINE_MAX,
                t['priority'] / 3.0,
            ])
        for m in range(self.M):
            state.extend([
                self.server_load[m] / self.f_edge[m],
                float(np.min(self.channels[:, m])) / 1e-5,
            ])
        return np.array(state, dtype=np.float32)

    # ==================== 在线推进 ====================

    def step(self, action, intrinsic_reward_fn=None):
        """按给定的动作推进一个时间步，并更新系统状态。

        参数：
            action:             1D 数组，长度为 action_dim = 2K，
                                每用户：(alpha_k, server_select_k)
            intrinsic_reward_fn: 可选的内在奖励回调（旧项目沿用，
                                Agent 版本下也可由 Verifier 提供）

        返回：(next_state, reward, done, info)
        """
        pre_state = self._get_state()
        act = np.asarray(action)
        n_cols = 3 if hasattr(self, 'action_dim') and self.action_dim % self.K == 0 else 2
        if act.ndim == 1:
            if act.shape[0] == self.K * 3:
                n_cols = 3
            elif act.shape[0] == self.K * 2:
                n_cols = 2
            act = act.reshape(self.K, n_cols)
        else:
            n_cols = act.shape[1]
            act = act.reshape(self.K, n_cols)
        total_energy, total_latency, success_count = 0.0, 0.0, 0
        priority_bonus = 0.0
        es_load = np.zeros(self.M)
        es_task_count = np.zeros(self.M)
        user_energy = np.zeros(self.K)
        edge_server_energy = np.zeros(self.M)

        sla_count = 0       # 高优先级 (p=3) 任务按时完成的个数
        sla_total = 0        # 高优先级 (p=3) 任务总数

        # ---- 排队模型：先按优先级排序，再在服务器上 FIFO 执行 ----
        # 2026.07 修改：引入基于优先级的 FIFO 排队模型替代简单的处理器共享。
        #
        # 原理：在多用户 MEC 场景中，任务以不同优先级卸载至同一服务器时，
        # 高优先级任务应获得优先处理权。排队时延导致卸载部分的总完成时间
        # 不再仅由传输时延 + 纯计算时延决定，还与同服务器上的竞争任务相关。
        # 这使得"卸载到低负载服务器"与"卸载到近距离服务器"之间产生真正
        # 的决策矛盾——前者计算快但传输远（传输能耗升），后者传输近但排队
        # 长（时延升），从而在能耗-时延平面上制造 Pareto 前沿。
        #
        # 具体实现：
        #   1. 每服务器维护一个按 (优先级降序, 用户索引升序) 排序的队列
        #   2. 每个用户的执行时延 = 前面所有用户的任务执行时间之和
        #   3. 总时延 = max(本地执行时延, 传输时延 + 排队时延 + 自身计算时延)

        # 先构建 (user, server, priority, alpha) 列表以便排序
        user_task_list = []
        for k in range(self.K):
            t = self.tasks[k]
            alpha = float(np.clip(act[k, 0], 0.01, 1.0))
            cloud_k = bool(np.round(act[k, 2])) if n_cols >= 3 and ENABLE_CLOUD_OFFLOAD else False
            server_idx = int(np.clip(np.floor(act[k, 1] * self.M), 0, self.M - 1))
            user_task_list.append((k, server_idx, t['priority'], alpha, t, cloud_k))

        # 按服务器分组（云端用户不参与边缘排队），组内按优先级降序、用户索引升序
        server_queues = {m: [] for m in range(self.M)}
        for k, server_idx, prio, alpha, t, cloud_k in user_task_list:
            if not cloud_k:
                server_queues[server_idx].append((k, prio, alpha, t))
        for m in range(self.M):
            # 优先级降序 (数值大优先)，同优先级按用户索引升序 (先到先得)
            server_queues[m].sort(key=lambda x: (-x[1], x[0]))

        # 计算每服务器的累积排队时延
        server_acc_delay = np.zeros(self.M)
        cloud_offload_info = []  # 记录路由到云端的用户信息
        fbl_outage_count = 0     # FBL 传输中断计数
        for k in range(self.K):
            t = self.tasks[k]
            cloud_k = bool(np.round(act[k, 2])) if n_cols >= 3 and ENABLE_CLOUD_OFFLOAD else False
            server_idx = int(np.clip(np.floor(act[k, 1] * self.M), 0, self.M - 1))
            alpha = float(np.clip(act[k, 0], 0.01, 1.0))
            local_cycles = alpha * t['C']
            local_time = local_cycles / self.f_local[k]
            local_energy = KAPPA_LOCAL * (self.f_local[k] ** 2) * local_cycles

            off_cycles = (1 - alpha) * t['C']
            off_data = (1 - alpha) * t['D']
            h = self.channels[k, server_idx]

            # FBL 通信模型：用有限码长速率替代 Shannon 理想速率
            _, rate, fbl_outage = compute_fbl_rate(
                h, BANDWIDTH, TX_POWER_USER, NOISE_POWER,
                FBL_BLOCKLENGTH, FBL_MAX_ERROR_PROB)
            if fbl_outage:
                fbl_outage_count += 1

            tx_time = off_data / (rate + 1e-9)
            tx_energy = TX_POWER_USER * tx_time

            # 云端卸载分支 (2026.08 新增)
            if ENABLE_CLOUD_OFFLOAD and cloud_k:
                # 云端：传输到 RSU 后经核心网上传云端
                cloud_tx_energy = CLOUD_TRANSMISSION_FACTOR * tx_energy
                cloud_time = tx_time + CLOUD_LATENCY_BASE + off_cycles / self.f_cloud
                cloud_edge_energy = 0.0
                edge_energy_k = 0.0
                offload_time = cloud_time
                if ACCOUNT_EDGE_ENERGY:
                    edge_energy_k = KAPPA_EDGE * (self.f_cloud ** 2) * off_cycles * 0.01
                total_time = max(local_time, offload_time)
                idle_time = max(0.0, offload_time - local_time)
                user_k_energy = local_energy + cloud_tx_energy + edge_energy_k + P_IDLE * idle_time
                cloud_offload_info.append(k)
                es_load[server_idx] += 0  # 云端处理不占用边缘算力
            else:
                # 边缘卸载分支 (原逻辑 + FBL 速率)
                queue_delay = 0.0
                for other_k, other_prio, other_alpha, other_t in server_queues.get(server_idx, []):
                    if other_k == k:
                        break
                    other_off = (1 - other_alpha) * other_t['C']
                    queue_delay += other_off / self.f_edge[server_idx]
                self_comp_time = off_cycles / (self.f_edge[server_idx] + 1e-9)
                offload_time = tx_time + queue_delay + self_comp_time

                if ACCOUNT_EDGE_ENERGY:
                    edge_energy_k = KAPPA_EDGE * (self.f_edge[server_idx] ** 2) * off_cycles
                else:
                    edge_energy_k = 0.0
                total_time = max(local_time, offload_time)
                idle_time = max(0.0, offload_time - local_time)
                user_k_energy = local_energy + tx_energy + edge_energy_k + P_IDLE * idle_time
                edge_server_energy[server_idx] += edge_energy_k
            total_energy += user_k_energy
            total_latency += total_time
            user_energy[k] = user_k_energy
            if not cloud_k:
                es_load[server_idx] += off_cycles
                es_task_count[server_idx] += 1

            ok = total_time <= t['tau']
            if ok:
                success_count += 1
                priority_bonus += t['priority'] * 2.0
            # SLA 统计：仅对高优先级任务 (p=3)
            if t['priority'] == 3:
                sla_total += 1
                if ok:
                    sla_count += 1

        self.server_load = es_load
        # 信道一阶自回归更新（复高斯 AR(1)，保证功率平稳）
        self.channel_coeffs = (CHANNEL_CORR_COEF * self.channel_coeffs
            + np.sqrt(1.0 - CHANNEL_CORR_COEF ** 2)
            * ((self._rng.randn(self.K, self.M)
                + 1j * self._rng.randn(self.K, self.M))
               * np.sqrt(0.5) * CHANNEL_COEFF_SCALE))
        self.channels = np.abs(self.channel_coeffs) ** 2

        info = {
            'energy': total_energy / 1e3,
            'latency': total_latency / self.K,
            'success_rate': success_count / self.K,
            'priority_sla': sla_count / max(sla_total, 1),
            'user_energy': user_energy / 1e3,
            'edge_server_energy': edge_server_energy / 1e3,
            'fbl_outage_rate': fbl_outage_count / self.K,
            'cloud_offload_count': len(cloud_offload_info),
        }
        r_env = (-total_energy / 1e3 - total_latency / self.K + priority_bonus) / 10.0

        r_llm = 0.0
        if intrinsic_reward_fn is not None:
            r_llm = float(intrinsic_reward_fn(pre_state, self, info))
            info['r_env'] = r_env
            info['r_llm'] = r_llm
            reward = r_env + r_llm
        else:
            reward = r_env

        self.step_count += 1
        done = self.step_count >= MAX_STEPS
        return self._get_state(), reward, done, info

    def sample_action(self):
        """均匀随机采样一个动作，便于 baseline 对照。"""
        return np.random.uniform(0, 1, self.action_dim).astype(np.float32)

    # ==================== 反事实仿真（A4 Verifier 用） ====================

    def simulate(self, plan):
        """反事实仿真：在当前状态的副本上预演给定方案，
        但不修改环境的真实状态。

        参数：
            plan: dict 形式：{'alpha': np.ndarray(K,), 'server': np.ndarray(K,)}
                  alpha[k] 为用户 k 的本地比例，server[k] 为所选服务器索引

        返回：dict {
            'energy': ..., 'latency': ..., 'success_rate': ...,
            'priority_sla': ..., 'per_user_time': np.ndarray(K,),
            'per_user_ok': np.ndarray(K,)
        }
        """
        # 用 copy 模拟，避免污染真实状态
        sim_channels = self.channels.copy()
        sim_tasks = copy.deepcopy(self.tasks)
        sim_f_local = self.f_local.copy()
        sim_f_edge = self.f_edge.copy()

        alpha = np.clip(np.asarray(plan['alpha']), 0.01, 1.0)
        server = np.clip(np.asarray(plan['server']).astype(int), 0, self.M - 1)
        cloud = np.zeros(self.K, dtype=bool)
        if 'cloud' in plan:
            cloud = np.asarray(plan['cloud']).astype(bool)

        total_energy, total_latency, success_count = 0.0, 0.0, 0
        per_user_time = np.zeros(self.K)
        per_user_ok = np.zeros(self.K, dtype=bool)
        es_task_count = np.zeros(self.M)
        fbl_outage_count = 0
        cloud_count = 0
        sla_count, sla_total = 0, 0

        # ---- 云端用户先处理（不参与边缘排队）----
        for k in range(self.K):
            if not cloud[k]:
                continue
            t = sim_tasks[k]
            a_k = alpha[k]
            local_cycles = a_k * t['C']
            local_time = local_cycles / sim_f_local[k]
            local_energy = KAPPA_LOCAL * (sim_f_local[k] ** 2) * local_cycles
            off_cycles = (1 - a_k) * t['C']
            off_data = (1 - a_k) * t['D']
            s_k = int(server[k])
            h = sim_channels[k, s_k]
            _, rate, fbl_outage = compute_fbl_rate(
                h, BANDWIDTH, TX_POWER_USER, NOISE_POWER,
                FBL_BLOCKLENGTH, FBL_MAX_ERROR_PROB)
            if fbl_outage:
                fbl_outage_count += 1
            tx_time = off_data / (rate + 1e-9)
            tx_energy = TX_POWER_USER * tx_time
            cloud_tx_energy = CLOUD_TRANSMISSION_FACTOR * tx_energy
            cloud_time_c = tx_time + CLOUD_LATENCY_BASE + off_cycles / self.f_cloud
            ec_k = KAPPA_EDGE * (self.f_cloud ** 2) * off_cycles * 0.01 if ACCOUNT_EDGE_ENERGY else 0.0
            offload_time = cloud_time_c
            user_k_energy = local_energy + cloud_tx_energy + ec_k + P_IDLE * max(0.0, offload_time - local_time)
            total_time = max(local_time, offload_time)
            total_energy += user_k_energy
            total_latency += total_time
            per_user_time[k] = total_time
            ok = total_time <= t['tau']
            if ok:
                success_count += 1
                per_user_ok[k] = True
            if t['priority'] >= 3:
                sla_total += 1
                if ok:
                    sla_count += 1
            cloud_count += 1

        # ---- 排队模型（仅边缘用户）----
        sim_server_queues = {m: [] for m in range(self.M)}
        for k in range(self.K):
            if cloud[k]:
                continue
            a_k = alpha[k]
            s_k = int(server[k])
            p_k = sim_tasks[k]['priority']
            sim_server_queues[s_k].append((k, p_k, a_k))
        for m in range(self.M):
            sim_server_queues[m].sort(key=lambda x: (-x[1], x[0]))
        sim_queue_delays = {m: 0.0 for m in range(self.M)}

        for m in range(self.M):
            for k, p_k, a_k in sim_server_queues[m]:
                t = sim_tasks[k]

                local_cycles = a_k * t['C']
                local_time = local_cycles / sim_f_local[k]
                local_energy = KAPPA_LOCAL * (sim_f_local[k] ** 2) * local_cycles

                off_cycles = (1 - a_k) * t['C']
                off_data = (1 - a_k) * t['D']
                h = sim_channels[k, m]

                # FBL 通信模型
                _, rate, fbl_outage = compute_fbl_rate(
                    h, BANDWIDTH, TX_POWER_USER, NOISE_POWER,
                    FBL_BLOCKLENGTH, FBL_MAX_ERROR_PROB)
                if fbl_outage:
                    fbl_outage_count += 1
                tx_time = off_data / (rate + 1e-9)
                tx_energy = TX_POWER_USER * tx_time

                queue_delay = sim_queue_delays[m]
                self_comp_time = off_cycles / (sim_f_edge[m] + 1e-9)
                sim_queue_delays[m] += self_comp_time
                offload_time = tx_time + queue_delay + self_comp_time
                edge_energy_k = (KAPPA_EDGE * (sim_f_edge[m] ** 2) * off_cycles
                                 if ACCOUNT_EDGE_ENERGY else 0.0)
                user_k_energy = (local_energy + tx_energy + edge_energy_k
                                 + P_IDLE * max(0.0, offload_time - local_time))

                total_time = max(local_time, offload_time)
                total_energy += user_k_energy
                total_latency += total_time
                per_user_time[k] = total_time
                ok = total_time <= t['tau']
                if ok:
                    success_count += 1
                    per_user_ok[k] = True
                if t['priority'] >= 3:
                    sla_total += 1
                    if ok:
                        sla_count += 1

        return {
            'energy': total_energy / 1e3,
            'latency': total_latency / self.K,
            'success_rate': success_count / self.K,
            'priority_sla': (sla_count / sla_total) if sla_total > 0 else 1.0,
            'per_user_time': per_user_time,
            'per_user_ok': per_user_ok,
            'fbl_outage_rate': fbl_outage_count / self.K if self.K > 0 else 0.0,
            'cloud_offload_count': cloud_count,
        }

    # ==================== 状态自然语言化（供 LLM Agent 阅读） ====================

    def state_to_text(self, state=None):
        """把当前状态转换为中文自然语言描述，供 LLM Agent 阅读理解。

        参数：state：可选。若为 None，则使用当前内部状态。
        返回：str
        """
        if state is None:
            state = self._get_state()
        lines = []
        lines.append(f"当前 MEC 系统状态（用户数 {self.K}，边缘服务器数 {self.M}）：")
        for k in range(self.K):
            base = k * 4
            lines.append(
                f"  用户 {k}: 数据量={state[base]:.2f}, 计算量={state[base+1]:.2f}, "
                f"时延约束={state[base+2]:.2f}, 优先级={int(state[base+3]*3)}"
            )
        base_u = self.K * 4
        for m in range(self.M):
            lines.append(
                f"  服务器 {m}: 负载率={state[base_u + m*2]:.2f}, "
                f"最差信道={state[base_u + m*2 + 1]:.2f}"
            )
        return "\n".join(lines)


# ============================================================
# 派生类：用于消融实验，移除信道状态
# ============================================================
class MECEnvironmentNoChannel(MECEnvironment):
    """去除信道状态的变体，用于消融研究。"""
    def _get_state(self):
        state = super()._get_state()
        base = self.K * 4
        for m in range(self.M):
            state[base + m * 2 + 1] = 0.0
        return state


# ============================================================
# Gym 风格包装类（兼容旧 baseline 调用习惯）
# ============================================================
class MECEnvWrapper:
    def __init__(self, num_users=NUM_USERS, num_servers=NUM_EDGE_SERVERS):
        self.env = MECEnvironment(num_users=num_users,
                                  num_servers=num_servers)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def simulate(self, plan):
        return self.env.simulate(plan)

    def state_to_text(self):
        return self.env.state_to_text()

    @property
    def state_dim(self):
        return self.env.state_dim

    @property
    def action_dim(self):
        return self.env.action_dim
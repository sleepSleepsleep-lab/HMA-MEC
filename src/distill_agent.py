# -*- coding: utf-8 -*-
"""
================================================================
A3: 蒸馏式零开销推理 (distill_agent.py)
================================================================
本模块实现 HMA-MEC 中蒸馏推理层 (Distill-Agent) 的策略网络与训练流程。
包含两个核心组件:

  1. PolicyAgentNet     —— 蒸馏策略网络
       输入:  原始状态 s_t  ((4K+2M) 维)
       输出:  分布参数 (alpha_mean, alpha_logstd, server_logits)
       解码:  每用户 alpha_k 为 Beta 分布样本 (软剪裁到 [0.01, 1.0])
                server_k 为 softmax 上的范畴分布
       同时输出对自身置信度的预测 c_min ∈ [0, 1] (用于 Hybrid 模式判别困难状态)

  2. DistillAgentTrainer —— 离线蒸馏训练器
       输入:  离线生成的 (s_t, a_t^star) 数据集 D_debate (jsonl 格式)
       输出:  训练好的 PolicyAgentNet 权重 (results/checkpoints/distilled_policy.pth)
       支持:  断点续训 (读取已训练 epoch)

  3. PolicyAgentRunner   —— 在线推理调用接口
       实现 agent_runner.py 中 Distill / Hybrid 两种模式所需的 ``单次前向'' 逻辑

注意: 本模块本身可在 CPU 上小规模训练; 蒸馏数据生成 (调 LLM 进行
CW-Debate) 与大规模训练 (batch ≥ 256, EPOCH ≥ 100) 应在 GPU 服务器上执行。
================================================================
"""

import os
import sys
import json
import time
import logging
import numpy as np

from typing import Optional, Tuple, List, Dict

from config import (
    NUM_USERS, NUM_EDGE_SERVERS, STATE_DIM,
    POLICY_NET_HIDDEN, POLICY_NET_LR, POLICY_NET_EPOCHS, POLICY_NET_BATCH,
    DISTILL_DATASET_SIZE,
    HYBRID_CONFIDENCE_LOW, CONFIDENCE_THRESHOLD,
    RESULTS_DIR, CHECKPOINT_DIR, SEED,
)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Beta, Categorical
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    torch = None
    nn = None
    F = None
    DEVICE = "cpu"


logger = logging.getLogger(__name__)


# ============================================================
# 蒸馏策略网络 PolicyAgentNet
# ============================================================
class PolicyAgentNet(nn.Module if torch else object):
    """蒸馏策略网络。

    架构：
        state_dim -> 256 -> 256 -> 五路输出头
        Head A: alpha_mean(k)  ∈ (0,1)        (K 头)
        Head B: alpha_logstd(k) clamped       (K 头)
        Head C: server_logits(k, m)           (K × M)
        Head D: cloud_logits(k, 2)            (K × 2, 新增云端选择)
        Head E: confidence_min  ∈ [0, 1]      (单值, Hybrid 模式用)

    损失:  Laplace NLL on alpha (回归 to a^star)
           + CrossEntropy on server  (分类 to m^star)
           + CrossEntropy on cloud   (分类 to cloud^star, 新增)
           + MSE on confidence estimate vs 真实置信度
    """

    def __init__(self, K: int = NUM_USERS, M: int = NUM_EDGE_SERVERS,
                 state_dim: Optional[int] = None,
                 hidden: int = POLICY_NET_HIDDEN):
        super().__init__()
        # 从 K, M 自动推导 state_dim, 避免硬编码 40 导致 K!=8 时崩溃
        if state_dim is None:
            state_dim = K * 4 + M * 2
        self.K, self.M, self.state_dim = K, M, state_dim
        # 共享主干
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.LayerNorm(hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden),
            nn.LeakyReLU(0.2),
        )
        # 头 A/B: 每用户 alpha 的 Beta 分布参数 (mean, log_std)
        self.alpha_mean   = nn.Linear(hidden, K)
        self.alpha_logstd = nn.Linear(hidden, K)
        # 头 C: 每用户服务器选择的 logits (K*M)
        self.server_logits = nn.Linear(hidden, K * M)
        # 头 D: 每用户云端选择 logits (K*2), 0=边缘 1=云端
        self.cloud_logits = nn.Linear(hidden, K * 2)
        # 头 E: 自评最小置信度 (1)
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.LeakyReLU(0.2),
            nn.Linear(hidden // 2, 1), nn.Sigmoid(),
        )
        self._initialize()

    def _initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        nn.init.constant_(self.alpha_logstd.bias, -1.0)  # 初始方差较小

    def forward(self, state):
        """前向，返回各分布参数。

        参数:  state: (B, state_dim) 张量
        返回:
            alpha_mean:   (B, K) ∈ (0,1)
            alpha_std:    (B, K) ∈ (0, +∞)
            server_logits: (B, K, M)
            cloud_logits:  (B, K, 2)
            conf_min:     (B,) ∈ (0,1)
        """
        h = self.shared(state)
        alpha_mean = torch.sigmoid(self.alpha_mean(h))
        log_std = torch.clamp(self.alpha_logstd(h), -5.0, 2.0)
        alpha_std = log_std.exp()
        server_logits = self.server_logits(h).view(-1, self.K, self.M)
        cloud_logits = self.cloud_logits(h).view(-1, self.K, 2)
        conf_min = self.confidence_head(h).squeeze(-1)
        return alpha_mean, alpha_std, server_logits, cloud_logits, conf_min

    def sample_action(self, state,
                      deterministic: bool = False
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """从策略网络采样动作。

        参数:
            state:          np.ndarray (state_dim,)
            deterministic:  True 时取 alpha 的众数与 argmax，用于评估
        返回:
            alpha:          (K,) float32 ∈ [0.01, 1.0]
            server:         (K,) int ∈ [0, M-1]
            cloud:          (K,) bool
            conf_min:       float, 对当前状态的最小置信度估计
        """
        s_t = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            a_mean, a_std, s_logits, c_logits, conf = self.forward(s_t)
        if deterministic:
            alpha = a_mean.squeeze(0)
            conf_min = float(conf.squeeze().item())
        else:
            eps = 1e-4
            m = torch.clamp(a_mean.squeeze(0), eps, 1 - eps)
            kappa = 1.0 / (a_std.squeeze(0) ** 2 + eps) - 1
            kappa = torch.clamp(kappa, 1.0, 100.0)
            a_param = m * kappa
            b_param = (1 - m) * kappa
            dist = Beta(a_param, b_param)
            alpha = dist.sample()
            conf_min = float(conf.squeeze().item())
        server = s_logits.squeeze(0).argmax(dim=-1).cpu().numpy().astype(int)
        cloud = c_logits.squeeze(0).argmax(dim=-1).cpu().numpy().astype(bool)
        alpha = alpha.cpu().numpy().astype(np.float32)
        alpha = np.clip(alpha, 0.01, 1.0)
        return alpha, server, cloud, conf_min


# ============================================================
# 蒸馏数据集 (jsonl 格式) 工具
# ============================================================
def save_debate_record(log_path: str, state: np.ndarray,
                       alpha: np.ndarray, server: np.ndarray,
                       cloud: Optional[np.ndarray] = None,
                       confidence: Optional[np.ndarray] = None,
                       fingerprint: Optional[str] = None):
    """追加一条蒸馏样本 (jsonl 格式),便于断点续存。"""
    rec = {
        'state':      np.asarray(state, dtype=np.float32).tolist(),
        'alpha':      np.asarray(alpha, dtype=np.float32).tolist(),
        'server':     np.asarray(server, dtype=int).tolist(),
        'cloud':      (np.asarray(cloud, dtype=int).tolist()
                       if cloud is not None else None),
        'confidence': (np.asarray(confidence, dtype=np.float32).tolist()
                       if confidence is not None else None),
        'fingerprint': fingerprint or '',
    }
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(rec, ensure_ascii=False) + '\n')


def load_debate_dataset(log_path: str,
                        max_samples: Optional[int] = None,
                        target_K: Optional[int] = None,
                        target_M: Optional[int] = None
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                   Optional[np.ndarray], Optional[np.ndarray]]:
    """读取蒸馏数据集, 支持按 (K, M) 配置筛选.

    参数:
        log_path:     jsonl 文件路径
        max_samples:  最大样本数 (None 表示全部)
        target_K:     目标用户数, 仅保留该配置的样本
        target_M:     目标服务器数, 与 target_K 配合使用
    返回: (states, alphas, servers, clouds, confidences)
        states shape (N, state_dim)  -- float32
        alphas shape (N, K)           -- float32
        servers shape (N, K)          -- int
        clouds shape (N, K) or None   -- int
        confidences shape (N, K) or None
    """
    if not os.path.exists(log_path):
        return None, None, None, None, None
    states, alphas, servers, clouds_list, confs = [], [], [], [], []
    n_skipped = 0

    # 如果指定了 target_K 和 fingerprint 筛选, 优先使用正则
    _target_fp = None
    if target_K is not None and target_M is not None:
        import re
        _target_fp = re.compile(rf'K{target_K}-M{target_M}')

    target_state_dim = None
    target_K_data = None
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            # fingerprint 筛选
            fp = r.get('fingerprint', '')
            if _target_fp is not None and not _target_fp.search(fp):
                n_skipped += 1
                continue

            s = np.asarray(r['state'], dtype=np.float32)
            a = np.asarray(r['alpha'], dtype=np.float32)
            sv = np.asarray(r['server'], dtype=int)
            cl = r.get('cloud')
            if cl is not None:
                cl = np.asarray(cl, dtype=int)
            cf = r.get('confidence')
            if cf is not None:
                cf = np.asarray(cf, dtype=np.float32)
            # 形状一致性
            if target_state_dim is None:
                target_state_dim = s.shape[0]
                target_K_data = a.shape[0]
            if s.shape[0] != target_state_dim or a.shape[0] != target_K_data or sv.shape[0] != target_K_data:
                n_skipped += 1
                continue
            if cf is not None and cf.shape[0] != target_K_data:
                n_skipped += 1
                continue
            states.append(s); alphas.append(a); servers.append(sv)
            if cl is not None:
                clouds_list.append(cl)
            if cf is not None:
                confs.append(cf)
    if not states:
        return None, None, None, None, None
    if max_samples is not None and len(states) > max_samples:
        # 均匀采样
        idx = np.linspace(0, len(states) - 1, max_samples).astype(int)
    else:
        idx = slice(None)
    confidences = np.stack(confs, axis=0) if confs else None
    clouds_out = np.stack(clouds_list, axis=0) if clouds_list else None
    return (np.stack(states, axis=0)[idx],
            np.stack(alphas, axis=0)[idx],
            np.stack(servers, axis=0)[idx],
            (clouds_out[idx] if clouds_out is not None else None),
            (confidences[idx] if confidences is not None else None))


def count_debate_records(log_path: str) -> int:
    """统计已完成的蒸馏样本数 (断点续存用)."""
    if not os.path.exists(log_path):
        return 0
    count = 0
    with open(log_path, 'r', encoding='utf-8') as f:
        for _ in f:
            if _.strip():
                count += 1
    return count


# ============================================================
# 蒸馏训练器
# ============================================================
class DistillAgentTrainer:
    """离线蒸馏训练器。

    训练目标:
        L = L_alpha + λ_s · L_server + λ_c · L_conf
    其中:
        L_alpha   : 对 (alpha_k^star) 的 Beta 分布 NLL  (回归)
        L_server  : 对 (m_k^star) 的 CrossEntropy       (分类)
        L_conf    : min_c 估计值与辩论轮最小置信度的 MSE
    """

    def __init__(self, K=NUM_USERS, M=NUM_EDGE_SERVERS,
                 state_dim=None,
                 lr: float = POLICY_NET_LR,
                 epochs: int = POLICY_NET_EPOCHS,
                 batch: int = POLICY_NET_BATCH,
                 lambda_server: float = 1.0,
                 lambda_conf: float = 0.1,
                 save_path: str = os.path.join(CHECKPOINT_DIR,
                                               "distilled_policy.pth")):
        self.K, self.M = K, M
        self.state_dim = state_dim
        self.lr = lr
        self.epochs = epochs
        self.batch = batch
        self.lambda_server = lambda_server
        self.lambda_conf = lambda_conf
        self.save_path = save_path
        self.model = PolicyAgentNet(K=K, M=M, state_dim=state_dim).to(DEVICE)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.history: List[List[float]] = []
        self.best_epoch: int = -1
        self.best_val: float = float('inf')

    def _loss(self, batch):
        s, a_star, m_star, c_star, conf_gt = (
            batch[0], batch[1], batch[2],
            batch[3] if len(batch) > 3 else None,
            batch[4] if len(batch) > 4 else None,
        )
        s = s.to(DEVICE)
        a_star = a_star.to(DEVICE)
        m_star = m_star.to(DEVICE).long().view(-1, self.K)
        a_mean, a_std, s_logits, c_logits, conf_pred = self.model.forward(s)
        # --- alpha NLL (近似为 Laplace) ---
        eps = 1e-4
        a_mean_c = torch.clamp(a_mean, eps, 1 - eps)
        log_a_std = torch.log(a_std + eps)
        L_alpha = (log_a_std + torch.abs(a_star - a_mean_c) / (a_std + eps)).mean()
        # --- server CE ---
        s_logits_v = s_logits.view(-1, self.M).float()
        m_target = m_star.view(-1)
        L_server = F.cross_entropy(s_logits_v, m_target)
        # --- cloud CE (新增) ---
        if c_star is not None and c_star.numel() > 0:
            c_star = c_star.to(DEVICE).long().view(-1)
            L_cloud = F.cross_entropy(c_logits.view(-1, 2), c_star)
        else:
            L_cloud = torch.zeros((), device=DEVICE)
        # --- conf MSE ---
        if conf_gt is not None and conf_gt.numel() > 0:
            conf_gt = conf_gt.to(DEVICE).float().view(-1, self.K)
            conf_min_gt = conf_gt.min(dim=-1, keepdim=True)[0].view(-1)
            L_conf = F.mse_loss(conf_pred, conf_min_gt)
        else:
            L_conf = torch.zeros((), device=DEVICE)
        loss = L_alpha + self.lambda_server * L_server + 0.5 * L_cloud + self.lambda_conf * L_conf
        return loss, (float(L_alpha.detach()), float(L_server.detach()),
                       float(L_cloud.detach()), float(L_conf.detach()))

    def train(self, states, alphas, servers,
              clouds: Optional[np.ndarray] = None,
              confidences: Optional[np.ndarray] = None,
              val_ratio: float = 0.1):
        """从头或断点续训。

        参数:
            states:      (N, state_dim)
            alphas:      (N, K)
            servers:     (N, K)
            clouds:      (N, K) or None, 云端卸载标志
            confidences: (N, K) or None
            val_ratio:   验证集比例
        返回:
            self.model, history
        """
        # 尝试续训
        if os.path.exists(self.save_path):
            try:
                self.model.load_state_dict(torch.load(self.save_path,
                                                      map_location=DEVICE))
                logger.info(f"[Distill] 续训: 已加载 {self.save_path}")
                print(f"[Distill] 续训: 已加载 {self.save_path}")
            except Exception as e:
                logger.warning(f"[Distill] 续训加载失败: {e}")

        # 划分 train/val
        N = len(states)
        idx = np.random.permutation(N)
        n_val = max(1, int(N * val_ratio))
        val_idx, train_idx = idx[:n_val], idx[n_val:]
        tr_s = torch.from_numpy(states[train_idx]).float()
        tr_a = torch.from_numpy(alphas[train_idx]).float()
        tr_m = torch.from_numpy(servers[train_idx]).long()
        va_s = torch.from_numpy(states[val_idx]).float()
        va_a = torch.from_numpy(alphas[val_idx]).float()
        va_m = torch.from_numpy(servers[val_idx]).long()
        if clouds is not None and clouds.shape[0] == N:
            tr_cl = torch.from_numpy(clouds[train_idx]).long()
            va_cl = torch.from_numpy(clouds[val_idx]).long()
        else:
            tr_cl, va_cl = None, None
        if confidences is not None and confidences.shape[0] == N:
            tr_c = torch.from_numpy(confidences[train_idx]).float()
            va_c = torch.from_numpy(confidences[val_idx]).float()
        else:
            tr_c, va_c = None, None

        n_tr = len(train_idx)
        print(f"[Distill] Train={n_tr}, Val={n_val}, Epochs={self.epochs}, Batch={self.batch}")
        patience, pat_wait = 10, 0
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

        for epoch in range(self.epochs):
            self.model.train()
            perm = torch.randperm(n_tr)
            total = [0.0, 0.0, 0.0, 0.0]
            for i in range(0, n_tr, self.batch):
                bi = perm[i:i + self.batch]
                # 数据增强: 状态向量附加高斯噪声（提升蒸馏鲁棒性, 弥补 5000 样本量）
                noise = torch.randn_like(tr_s[bi]) * 0.015
                s_noisy = (tr_s[bi] + noise).clamp(0.0, 1.0)
                batch = (s_noisy, tr_a[bi], tr_m[bi],
                         tr_cl[bi] if tr_cl is not None else None,
                         tr_c[bi] if tr_c is not None else None)
                loss, parts = self._loss(batch)
                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optim.step()
                for j, p in enumerate(parts):
                    total[j] += p * len(bi) / n_tr
            # 验证
            self.model.eval()
            with torch.no_grad():
                loss_val, _ = self._loss((va_s, va_a, va_m, va_cl, va_c))
            self.history.append([epoch + 1, total[0], total[1],
                                 total[2], total[3], float(loss_val)])
            if loss_val < self.best_val:
                self.best_val = float(loss_val)
                self.best_epoch = epoch + 1
                torch.save(self.model.state_dict(), self.save_path)
                pat_wait = 0
                marker = " *"
            else:
                pat_wait += 1
                marker = ""
            if (epoch + 1) % 10 == 0 or epoch < 5:
                print(f"  [Epoch {epoch+1:3d}] "
                      f"Lα={total[0]:.4f} Ls={total[1]:.4f} "
                      f"Lc={total[2]:.4f} Lval={float(loss_val):.4f}{marker}")
            if pat_wait >= patience:
                print(f"  [Early Stop] 第{epoch+1}轮, best epoch={self.best_epoch}")
                break

        # 重新加载最佳模型
        if os.path.exists(self.save_path):
            self.model.load_state_dict(torch.load(self.save_path, map_location=DEVICE))
        return self.model, self.history


# ============================================================
# 在线推理运行器 (Distill / Hybrid 模式共享)
# ============================================================
class PolicyAgentRunner:
    """对 PolicyAgentNet 的简洁在线推理包装。

    三种模式对应:
        Distill :  仅前向,无 LLM 调用
        Hybrid  :  前向后,若 conf_min < tau_low,降级到 CW-Debate
        FullLLM :  不使用本网,直接调用 CW-Debate

    加载 checkpoint 时自动适应 (K, M) 差异：若形状不匹配则给出提示并随机初始化。
    """

    def __init__(self, model_path: Optional[str] = None,
                 K: int = NUM_USERS, M: int = NUM_EDGE_SERVERS):
        self.K, self.M = K, M
        self.model = PolicyAgentNet(K=K, M=M).to(DEVICE)
        if model_path and os.path.exists(model_path):
            try:
                ckpt = torch.load(model_path, map_location=DEVICE)
                # 检查键形状是否完全匹配
                model_keys = self.model.state_dict()
                mismatch = False
                for k in ckpt:
                    if k in model_keys and ckpt[k].shape != model_keys[k].shape:
                        mismatch = True
                        break
                if mismatch:
                    # 形状不匹配（不同 K/M），提示后使用随机初始化的模型
                    self.model = PolicyAgentNet(K=K, M=M).to(DEVICE)
                    print(f"[PolicyAgentRunner] K={K},M={M} 与权重训练规模不同, 使用随机初始化")
                else:
                    self.model.load_state_dict(ckpt)
                    self.model.eval()
                    print(f"[PolicyAgentRunner] 已加载 {model_path}")
            except Exception as e:
                print(f"[PolicyAgentRunner] 权重加载异常 ({e}), 使用随机初始化")
        elif model_path is None:
            print("[PolicyAgentRunner] 未指定权重路径, 使用随机初始化模型")
        else:
            print(f"[PolicyAgentRunner] 未见 {model_path}, 使用随机初始化模型")

    def infer(self, state: np.ndarray,
              deterministic: bool = True) -> Dict:
        """单次前向 -> 输出 plan 与 conf_min (Distill 模式专用)."""
        alpha, server, cloud, conf_min = self.model.sample_action(
            state, deterministic=deterministic)
        plan = {'alpha': alpha, 'server': server, 'cloud': cloud}
        return {
            'plan':       plan,
            'conf_min':   conf_min,
            'mode':       'Distill',
        }


# ============================================================
# 自检 (CPU 可运行: 用合成数据训练 5 epoch)
# ============================================================
if __name__ == "__main__":
    if torch is None:
        print("torch 未安装, 跳过自检"); sys.exit()
    print("=" * 60)
    print("  PolicyAgentNet 蒸馏自检 (CPU, 合成数据 1000 样本)")
    print("=" * 60)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    N = 1000
    K, M = NUM_USERS, NUM_EDGE_SERVERS
    states = np.random.uniform(0, 1, (N, STATE_DIM)).astype(np.float32)
    # 模拟 a^star: alpha ~ Beta(2,2); server 偏向负载低的服务器
    alphas = np.random.beta(2, 2, (N, K)).astype(np.float32)
    servers = np.random.randint(0, M, (N, K))
    confs = np.random.uniform(0.2, 1.0, (N, K)).astype(np.float32)

    trainer = DistillAgentTrainer(K=K, M=M, epochs=5, batch=64,
                                  save_path=os.path.join(CHECKPOINT_DIR,
                                                       "smoke_policy.pth"))
    trainer.train(states, alphas, servers, confidences=confs)
    # 验证 PolicyAgentRunner
    runner = PolicyAgentRunner(model_path=trainer.save_path, K=K, M=M)
    s_test = np.random.uniform(0, 1, STATE_DIM).astype(np.float32)
    out = runner.infer(s_test, deterministic=False)
    print(f"  infer 测试: alpha={out['plan']['alpha']}, "
          f"server={out['plan']['server']}, conf_min={out['conf_min']:.3f}")
    print("=" * 60)
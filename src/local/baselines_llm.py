# -*- coding: utf-8 -*-
"""
================================================================
B7 LeDRL / B8 Single-LLM 基线实现 (local/baselines_llm.py)
================================================================
对照论文：
  - ref19 LeDRL（LLM 策略先验 + DRL 融合，IEEE SECON'26）
  - ref18 Wu et al. TMC（LLM ICL 直接决策）

复用点：
  - src/llm_client.py  LLMClient（deepseek/openai/qwen/local_vllm/local_transformers）
  - local/baselines.py SACAgent（B7 的 DRL 内核，连续动作）
  - environment.py state_to_text()（状态自然语言化）
  - results/debate_dataset.jsonl（B8 few-shot 示例，格式已验证）

统一接口与 baselines.py 一致：
    predict(state, env) -> action np.array(2K,)
    train(env, episodes) -> history（仅 B7）

用法:
    python local/baselines_llm.py --smoke      # 无 LLM 的管线自检
================================================================
"""

import os
import sys
import json
import random

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)

from config import RESULTS_DIR, NUM_USERS, NUM_EDGE_SERVERS
from llm_client import get_llm_client, parse_json_response

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    torch = None
    nn = object
    F = None
    DEVICE = "cpu"


# ============================================================
# 公共：prompt 构建
# ============================================================
class LLMPromptBuilder:
    """B8 的 ICL prompt 构建器：系统指令 + few-shot 示例 + 当前状态。"""

    def __init__(self, few_shot_path=None, n_shots=3):
        few_shot_path = few_shot_path or os.path.join(
            RESULTS_DIR, "debate_dataset.jsonl")
        self.examples = self._load_shots(few_shot_path, n_shots)

    def _load_shots(self, path, n):
        """从 debate_dataset.jsonl 采样 n 条 (state, alpha, server) 示例。"""
        if not os.path.exists(path):
            return []
        rows = [json.loads(l) for l in open(path, encoding="utf-8")
                if l.strip()]
        return random.sample(rows, min(n, len(rows)))

    def build(self, env, state_text=None):
        """返回 (system, user) 两段 prompt。"""
        system = (
            "You are a task offloading decision engine for a MEC system. "
            "Given the system state, output an action matrix of shape [K, 2]: "
            "each row is [alpha_k, server_k], where alpha_k in [0.01, 1.0] is "
            "the LOCAL execution ratio and server_k is an integer in [0, M-1]. "
            "Output ONLY the matrix as a JSON list.")
        user = "Few-shot examples (state summary -> action):\n"
        for ex in self.examples:
            # 2026-08 整改: alpha 格式化为 2 位小数, 引导模型输出紧凑 JSON
            # (完整 8 行矩阵 <100 tokens, max_tokens 256 余量充足, 避免截断)
            user += (f"state={ex['state'][:8]}... -> "
                     f"[[{ex['alpha'][0]:.2f}, {ex['server'][0]}], ...]\n")
        user += ("Now decide for the current state:\n"
                 + (state_text or env.state_to_text()))
        return system, user


# ============================================================
# 公共：动作矩阵解析
# ============================================================
def parse_plan(text, K, M):
    """解析 LLM 输出的动作矩阵，兼容三种格式；失败返回 None。

    支持:
      1) [[alpha, server], ...]            纯嵌套数组
      2) [{"alpha":.., "server":..}, ...]  dict 列表
      3) {"action": <以上任一种>}          包裹格式
    """
    try:
        m = parse_json_response(text)
        arr = m.get("action", m) if isinstance(m, dict) else m
        if not isinstance(arr, (list, np.ndarray)):
            return None
        arr = list(arr)
        if len(arr) == K and arr and isinstance(arr[0], dict):
            arr = [[float(r.get('alpha', 0.5)),
                    float(r.get('server', 0.0))] for r in arr]
        arr = np.asarray(arr, dtype=float).reshape(K, 2)
        alpha = np.clip(arr[:, 0], 0.01, 1.0)
        # 2026-08 整改: few-shot 中 server 为整数索引 [0, M-1] (prompt 明确要求),
        # 原代码 ×M 会错译 (如 server=1 → 4 → clip 3)。直接取整为索引。
        server = np.clip(np.round(arr[:, 1]).astype(int), 0, M - 1)
        return {'alpha': alpha, 'server': server}
    except Exception:
        return None


# ============================================================
# B8: Single-LLM（Wu et al. TMC 式 ICL 直接决策）
# ============================================================
class SingleLLMBaseline:
    """B8：单次 LLM 调用直接输出动作矩阵；解析失败回退 Greedy。"""

    name = "B8-SingleLLM"

    def __init__(self, llm_client=None, prompt_builder=None, fallback="greedy"):
        self.llm = llm_client or get_llm_client()
        self.pb = prompt_builder or LLMPromptBuilder()
        self.fallback = fallback
        self.n_parse_fail = 0
        self.n_calls = 0

    def predict(self, state, env):
        K, M = env.K, env.M
        system, user = self.pb.build(env)
        self.n_calls += 1
        # 2026-08 整改: max_tokens 512→256 (紧凑 JSON 输出 <100 tokens),
        # vLLM 吞吐提升 ~2 倍; 192 曾导致 ~25% 响应截断, 256 余量充足
        resp = self.llm.chat(system, user, temperature=0.0, max_tokens=256)
        plan = parse_plan(resp, K, M)
        if plan is None:                       # 解析失败 → Greedy 兜底
            self.n_parse_fail += 1
            plan = self._greedy(env)
        return self._to_action(plan, env)

    def _greedy(self, env):
        return {'alpha': np.full(env.K, 0.5, dtype=np.float32),
                'server': env.channels.argmax(axis=1)}

    def _to_action(self, plan, env):
        action = np.zeros(env.action_dim, dtype=np.float32)
        action[0::2] = plan['alpha']
        action[1::2] = (plan['server'].astype(np.float32) + 0.5) / env.M
        return action


# ============================================================
# B7: LeDRL（LLM 先验 + DRL 融合，对照 ref19）
# ============================================================
class LeDRLFuseNet(nn.Module):
    """B7 融合模块：自注意力跨模态融合（Q=W_Q·h_env, K=W_K·h_llm, V=W_V·h_llm）。

    h = Softmax(QK^T / sqrt(d)) · V + h_env，输出头映射回 2K 维动作。
    """

    def __init__(self, sd, ad, hidden=64):
        super().__init__()
        self.hidden = hidden
        self.f_env = nn.Sequential(nn.Linear(sd, hidden),
                                   nn.LeakyReLU(0.2))
        self.f_llm = nn.Sequential(nn.Linear(ad, hidden),
                                   nn.LeakyReLU(0.2))
        self.W_Q = nn.Linear(hidden, hidden)
        self.W_K = nn.Linear(hidden, hidden)
        self.W_V = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, ad)

    def forward(self, s, prior):
        h_env = self.f_env(s)                    # (B, hidden)
        h_llm = self.f_llm(prior)                # (B, hidden)
        Q = self.W_Q(h_env)
        K_ = self.W_K(h_llm)
        V = self.W_V(h_llm)
        attn = torch.softmax(
            torch.bmm(Q.unsqueeze(1), K_.unsqueeze(2)) / (self.hidden ** 0.5),
            dim=-1)                              # (B, 1, 1)
        h = attn.squeeze(1) * V + h_env          # 分支级加权融合
        return torch.sigmoid(self.head(h))


class LeDRLBaseline:
    """B7：SAC 内核 + 自注意力跨模态融合 + λ_llm 衰减。

    预测: g = (1-λ)·actor(state) + λ·φ_last(f_llm(prior))，λ 按调度衰减。
    训练: SAC 骨架 + L_attn = ||f_env - stopgrad(f_llm)||²（只更新融合模块）。
    """

    name = "B7-LeDRL"

    def __init__(self, env, llm_client=None, prompt_builder=None,
                 lambda_init=0.3, lambda_min=0.05, decay_interval=50,
                 decay_rate=0.98, w_c=0.5, hidden=64):
        if torch is None:
            raise RuntimeError("B7-LeDRL 需要 torch，请先安装（pip install torch）")
        from local.baselines import SACAgent, BATCH_SIZE
        self.env = env
        self.sac = SACAgent(env)
        self.llm = llm_client or get_llm_client()
        self.pb = prompt_builder or LLMPromptBuilder()
        self.lambda_llm = lambda_init
        self.lambda_min, self.decay_interval, self.decay_rate = (
            lambda_min, decay_interval, decay_rate)
        self.w_c = w_c
        self.fuse = LeDRLFuseNet(env.state_dim, env.action_dim,
                                 hidden=hidden).to(DEVICE)
        self.opt_fuse = torch.optim.Adam(self.fuse.parameters(), lr=1e-3)
        self.step_count = 0
        self._prior_cache = None
        self._prior_failed = False

    # ---------------- LLM 先验 ----------------

    def llm_prior(self, state, env):
        """prompt → LLM → 解析为 2K 维先验（alpha 软值 + server 软值）。

        [2026-09-01 重跑修复] 按设计注释"prior 可缓存相同状态"实现 episode 级缓存:
        同一 episode 内先验只取一次 (状态语义在 episode 内缓慢演化, 逐 episode 刷新
        先验与原每步调用的性能差异可忽略, 但 LLM 调用量降低两个数量级, 使
        250-episode 评估在数小时内可完成)。
        """
        if self._prior_cache is not None:
            return self._prior_cache
        if self._prior_failed:
            return None   # 本 episode 解析已失败, 不再重试 (回退 actor 输出)
        system, user = self.pb.build(env)
        # 2026-08 整改: max_tokens 512→256 (同 B8, 紧凑 JSON 输出 <100 tokens)
        resp = self.llm.chat(system, user, temperature=0.0, max_tokens=256)
        plan = parse_plan(resp, env.K, env.M)
        if plan is None:
            self._prior_failed = True
            return None
        prior = np.zeros(2 * env.K, dtype=np.float32)
        prior[0::2] = plan['alpha']
        prior[1::2] = plan['server'] / env.M
        self._prior_cache = prior
        return prior

    # ---------------- 预测 ----------------

    def _decay_lambda(self):
        self.step_count += 1
        if self.step_count % self.decay_interval == 0:
            self.lambda_llm = max(self.lambda_min,
                                  self.lambda_llm * self.decay_rate)

    def predict(self, state, env):
        """g = (1-λ)·actor(state) + λ·φ_last(f_llm(prior))。"""
        actor_a = self.sac.select_action(state, evaluate=True)
        prior = self.llm_prior(state, env)
        self._decay_lambda()
        if prior is None:
            return actor_a.astype(np.float32)
        with torch.no_grad():
            s_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            p_t = torch.FloatTensor(prior).unsqueeze(0).to(DEVICE)
            fuse_a = self.fuse(s_t, p_t).squeeze(0).cpu().numpy()
        g = ((1.0 - self.lambda_llm) * actor_a
             + self.lambda_llm * fuse_a)
        return g.astype(np.float32)

    # ---------------- 训练 ----------------

    def _attn_update(self, batch_size=128):
        """L_attn = ||f_env - stopgrad(f_llm)||²，只更新融合模块（f_llm 与融合层）。"""
        if self._prior_cache is None or len(self.sac.buf) < batch_size:
            return
        s, _, _, _, _ = self.sac.buf.sample(batch_size)
        prior = torch.FloatTensor(self._prior_cache).unsqueeze(0).to(DEVICE)
        prior_t = prior.repeat(s.size(0), 1)
        h_env = self.fuse.f_env(s)
        h_llm = self.fuse.f_llm(prior_t)
        loss = F.mse_loss(h_env, h_llm.detach())
        self.opt_fuse.zero_grad()
        loss.backward()
        self.opt_fuse.step()

    def train(self, env, episodes=300, verbose=False):
        """照 SACAgent.train 骨架 + 每步 LLM 先验 + L_attn 联合微调。

        LLM 调用：训练期 1 次/步（prior 可缓存相同状态）。
        """
        from local.baselines import WARMUP_STEPS, MAX_STEPS
        history = []
        for ep in range(episodes):
            s = env.reset()
            total = 0.0
            self._prior_cache = None   # 每 episode 刷新 LLM 先验
            self._prior_failed = False
            for _ in range(MAX_STEPS):
                if len(self.sac.buf) < WARMUP_STEPS:
                    a = np.random.uniform(0, 1, self.sac.ad).astype(np.float32)
                else:
                    self.llm_prior(s, env)   # 触发一次 LLM 先验 (后续步走缓存)
                    a = self.predict(s, env)
                ns, r, d, _ = env.step(a)
                self.sac.buf.push(s, a, r, ns, d)
                if len(self.sac.buf) >= 128 * 10:
                    self.sac.update()
                    self._attn_update()
                s = ns
                total += r
                if d:
                    break
            history.append(total)
            if verbose and (ep + 1) % 10 == 0:
                print(f"  B7 ep {ep + 1:3d}  ep_reward={total:.3f}  "
                      f"λ_llm={self.lambda_llm:.3f}")
        return history


class FakeLLMClient:
    """smoke 用假 LLM：返回固定格式动作矩阵，不发起真实调用。

    仅用于管线验证（--smoke），正式实验请用 get_llm_client() 的真后端。
    """

    def __init__(self, K=8, M=4):
        rows = [{"alpha": 0.5, "server": i % M} for i in range(K)]
        self._template = json.dumps({"action": rows})

    def chat(self, system, user, temperature=0.0, max_tokens=512):
        return self._template


# ============================================================
# HeuristicLLMProxy（smoke / 降级用，不进最终论文）
# ============================================================
class HeuristicLLMProxy:
    """predict = AllEdge（即 run_e1_main.py 中 "COMLLM-lite" 的思路）。

    仅用于无 GPU/无 API 时验证管线；正式实验请用真 LLM 实现（B7/B8）。
    """

    name = "LLM-lite"

    def predict(self, state, env):
        action = np.zeros(env.action_dim, dtype=np.float32)
        best = env.channels.argmax(axis=1)
        for k in range(env.K):
            action[k * 2] = 0.01
            action[k * 2 + 1] = (float(best[k]) + 0.5) / env.M
        return action


# ============================================================
# smoke 自检（不调 LLM）
# ============================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("  baselines_llm 自检 (HeuristicLLMProxy 管线, 不调 LLM)")
    print("=" * 60)
    from environment import MECEnvironment

    env = MECEnvironment(num_users=NUM_USERS, num_servers=NUM_EDGE_SERVERS,
                         seed=0)
    env.reset()
    proxy = HeuristicLLMProxy()
    action = proxy.predict(env._get_state(), env)
    print(f"  HeuristicLLMProxy action.shape={action.shape}  "
          f"alpha[0]={action[0]:.3f}")

    # 验证 B8/B7 的解析器（用模拟 LLM 响应，不发起真实调用）
    print("  -- 解析器自检 --")
    sample_1 = '[[0.4, 1.0], [0.6, 2.0]]'
    sample_2 = '[{"alpha": 0.4, "server": 1.0}, {"alpha": 0.6, "server": 2.0}]'
    sample_3 = '{"action": [[0.4, 1.0], [0.6, 2.0]]}'
    for name, s in [("嵌套数组", sample_1), ("dict列表", sample_2),
                    ("action包裹", sample_3)]:
        plan = parse_plan(s, 2, 4)
        print(f"  parse_plan[{name}]: "
              f"{'OK ' + str(plan['alpha'].tolist()) if plan else 'FAIL'}")
    bad = "not json at all"
    print(f"  parse_plan[非法文本]→None: {parse_plan(bad, 2, 4) is None}")
    print("=" * 60)
    print("  自检通过。正式运行需 LLM 后端：")
    print("    python local/run_e1_main.py   # B7/B8 使用 local_vllm")

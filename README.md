> **迁移说明**：本仓库合并历史含旧骨架提交（README 一度为三层建模旧版描述）；发布主仓库为干净单提交的 https://github.com/sleepSleepsleep-lab/hma-mec-2026 ，论文引用请使用新地址。

# HMA-MEC：面向移动边缘计算任务卸载的混合多智能体大模型框架

> **Hybrid Multi-Agent framework with LLM for Mobile Edge Computing task offloading**

HMA-MEC 将**多智能体系统（MAS）**与**大语言模型（LLM）**结合，用于解决端-边-云三层移动边缘计算（MEC）网络中的**任务卸载决策**问题。框架在论文提出的五轮 **CW-Debate** 置信度加权辩论协议之上，进一步通过**离线蒸馏**获得零 LLM 开销的策略网络，并配合**反事实验证**智能体抑制奖励作弊（reward hacking），形成「提出 — 批判 — 验证 — 蒸馏」的完整闭环。


- **代码语言**：Python
- **运行环境**：CPU（local/visual 部分）＋ GPU / vLLM（gpu 部分）

---

## 1. 项目亮点

| 编号 | 创新点 | 实现模块 |
| --- | --- | --- |
| A1 | **Agent 定义模型（S-F Model）**：以「场景集 S、功能集 F、边界集 B」三元组形式化定义各类智能体的身份与权限边界 | `agent_define.py` |
| A2 | **CW-Debate 置信度加权多轮辩论协议**：Propose → Critique → Arbitrate → Verify → Consensus 五轮交互，含置信度门控、批判子图修剪与 ToM 预测 | `cw_debate.py`、`orchestrator.py` |
| A3 | **蒸馏式推理智能体（Distill-Agent）**：将在线辩论产生的 `(s, a*, c*)` 数据离线蒸馏为策略网络 `PolicyAgentNet`，推理零 API 开销 | `distill_agent.py` |
| A4 | **验证智能体（Verifier-Agent）**：反事实仿真 + 拒绝采样，拦截物理不可行或奖励作弊方案 | `verifier.py` |
| A5 | **三模式在线推理**：`FullLLM`（在线辩论）/ `Distill`（纯蒸馏）/ `Hybrid`（困难状态自动降级触发辩论） | `agent_runner.py` |

配套通信/系统模型：

- **有限码长（FBL）通信模型**：替代 Shannon 无限码长假设，显式建模信道散度与误码率（`environment.py`）
- **端-边-云三层架构**：边缘过载任务自动溢出至云端，云端卸载通道独立建模（`environment.py`）
- **优先级 FIFO 排队模型**：多用户竞争同一服务器时按优先级调度，制造真实的能耗-时延 Pareto 张力
- **VEC 车联网场景**：支持无人驾驶出租车移动场景（位置/速度感知、RSU 覆盖、路径损耗）

---

## 2. 系统架构

```
                     ┌─────────────────────────────────────────┐
  Distill 层          │  PolicyAgentNet:  s_t → (α, m, cloud)   │  ← 零 LLM 调用
   (在线推理)          │  困难状态 conf_min < τ_low → 降级在线辩论 │
                     └───────────────────▲─────────────────────┘
                                         │ 硬状态回退
                     ┌───────────────────┴─────────────────────┐
  多智能体辩论层       │   R0 状态广播 ─ R1 提议 ─ R2 交叉批判 ─   │
   (CW-Debate)       │   R3 OA 仲裁 + ToM ─ R4 VA 反事实验证 ─  │
                     │   R5 共识终止                              │
                     │   UA×K / EA×M / OA×1 / VA×1 / CA×1      │
                     └───────────────────▲─────────────────────┘
                                         │ a_t = (α_k, m_k, cloud_k)
                     ┌───────────────────┴─────────────────────┐
  MEC 环境层          │   任务卸载仿真 / FBL 信道 / 优先级排队 /    │
                     │   云端溢出 / simulate() 反事实仿真接口     │
                     └─────────────────────────────────────────┘
```

四类核心智能体：

| 智能体 | 数量 | 职责 |
| --- | --- | --- |
| **UA** 用户智能体 | K | 任务画像、生成候选卸载方案（α_k, m_k, 置信度）、交叉批判 |
| **EA** 边缘智能体 | M | 容量申报、负载接纳校验、短时负载预测 |
| **OA** 编排智能体 | 1 | 偏好推断 ω_t、冲突仲裁、Theory-of-Mind 置信度预测、共识终止 |
| **VA** 验证智能体 | 1 | 反事实仿真、拒绝采样、偏差校验（防 reward hacking） |
| **CA** 云端智能体 | 1 | 接收边缘溢出任务、云端成本申报 |

CW-Debate 五轮协议每步决策流程：

```
Round 0  状态广播：OA 将状态向量 + 自然语言描述分发给所有 Agent
Round 1  局部提议：每 UA 提交 (alpha, server, confidence, reason)，每 EA 提交容量申报
Round 2  交叉批判：同服务器冲突提示（置信度门控 + 负载子图修剪）+ EA 接纳校验
Round 3  OA 仲裁：基于偏好 + 批判修正方案，并用 ToM 预测每用户置信度
Round 4  反事实验证：VA 在当前状态副本上仿真，与 OA 自评估对比，偏差过大则拒绝
Round 5  共识终止：置信度变化 < ε 或达到最大轮次则收敛，否则回退重仲裁
```

---

## 3. 目录结构

```
.
├── src/                            # 全部源代码
│   ├── config.py                   # 全局配置（系统规模/LLM 后端/超参数）
│   ├── environment.py              # MEC / VEC 仿真环境（FBL 信道、云端、反事实仿真）
│   ├── llm_client.py               # LLM 客户端抽象（deepseek/openai/qwen/vLLM/transformers）
│   ├── agent_define.py             # A1：Agent 定义模型（S-F 三元组）
│   ├── cw_debate.py                # A2：CW-Debate 五轮辩论协议
│   ├── orchestrator.py             # OA 编排智能体（偏好推断/仲裁/ToM/共识）
│   ├── verifier.py                 # A4：VA 验证智能体（反事实 + 拒绝采样）
│   ├── distill_agent.py            # A3：蒸馏策略网络 PolicyAgentNet 与训练器
│   ├── agent_runner.py             # 三模式在线推理运行器（FullLLM/Distill/Hybrid）
│   ├── train_k8_policy.py          # 用 K=8 样本快速训练蒸馏策略
│   ├── requirements.txt            # Python 依赖
│   │
│   ├── gpu/                        # 需在 GPU / vLLM 服务器运行的代码
│   │   ├── gen_distill_dataset.py  #   离线蒸馏数据生成（并行、断点续存）
│   │   └── train_distill_policy.py #   蒸馏策略网络训练（自动续训）
│   │
│   ├── local/                      # 本地 CPU 可运行的实验代码
│   │   ├── baselines.py            #   Greedy/AllLocal/AllEdge/Random/SAC/DDPG
│   │   ├── experiment_common.py    #   实验公共框架（评估/多种子/npz 存取）
│   │   ├── run_all_experiments.sh  #   一键跑全部实验
│   │   └── run_e1_*.py ... run_e9_*.py   # 实验 E1-E9
│   │
│   └── visual/                     # 论文绘图脚本（输出 1200 DPI PDF/TIFF）
│       ├── fig_arch.py             #   图1：三层架构与 CW-Debate 协议
│       ├── fig_distill_net.py      #   图2：蒸馏策略网络结构
│       ├── fig_e1_*.py ... fig_e9_*.py  #   各实验对应图表
│       └── fig_train_history.py    #   蒸馏训练损失曲线
└── results/                        # 运行时自动生成（实验数据/日志/checkpoints）
```

## 4. 快速开始

### 4.1 环境依赖

```bash
pip install -r src/requirements.txt
```

- `torch>=2.0`：蒸馏训练必需（本地 CPU 安装 CPU 版本即可；GPU 服务器装 CUDA 版本）
- `openai>=1.0`：DeepSeek / OpenAI / vLLM 共用此包
- 仅使用 `local_transformers` 后端时需额外安装 `transformers`；仅使用 `local_vllm` 时需安装 `vllm`

### 4.2 最小验证（无需 LLM、无需 GPU）

各核心模块均内置 `__main__` 自检，可直接运行：

```bash
# 环境仿真
python src/environment.py

# Agent 定义模型自检
python src/agent_define.py

# CW-Debate 协议自检（Distill 启发式模式，不调用 LLM）
python src/cw_debate.py

# 蒸馏策略网络 CPU 小规模训练自检
python src/distill_agent.py

# 基线方法自检
python src/local/baselines.py
```

### 4.3 完整流程（离线蒸馏 → 在线推理）

```bash
# 1. （GPU + LLM）离线生成辩论数据 D_debate，写入 results/debate_dataset.jsonl
python src/gpu/gen_distill_dataset.py --smoke        # 先试跑 3 条验证流水线
python src/gpu/gen_distill_dataset.py                # 正式生成（--workers 32 --target 50000）

# 2. （GPU）训练蒸馏策略网络，保存 results/checkpoints/distilled_policy.pth
python src/gpu/train_distill_policy.py --epochs 100 --batch 128

# 3. （CPU）在线推理：三模式任选
python src/agent_runner.py        # Distill 模式自检
```

### 4.4 切换 LLM 后端

在 `src/config.py` 中修改 `LLM_BACKEND` 即可，业务代码无需改动：

```python
LLM_BACKEND = "local_vllm"        # 可选：deepseek / openai / qwen / local_vllm / local_transformers
LLM_API_KEY = ""                  # 商用 API 时填写
LLM_MODEL   = "Qwen3.5-9B"        # 本地模型即 HuggingFace 仓库名
```

---

## 5. 实验（E1-E9）

实验脚本位于 `src/local/`，运行后结果以 `.npz` / `.json` 保存到 `results/`，再由 `src/visual/` 的绘图脚本生成论文图。

| 实验 | 脚本 | 内容 |
| --- | --- | --- |
| E1 主对比 | `run_e1_main.py` | 基准 8×4 场景下对比 Greedy/AllLocal/AllEdge/Random/SAC/DDPG/HMA-Distill/HMA-Hybrid |
| E2 可扩展性 | `run_e2_scalability.py` | 固定 M=4，K∈{4,8,12,16,24,32} 的性能与 token/时延增长 |
| E3 消融 | `run_e3_ablation.py` | 12 组消融变体（关闭置信度门控/ToM/验证/边界/优先级/偏好/各轮次等） |
| E4 推理效率 | `run_e4_efficiency.py` | Distill/Hybrid/FullLLM 的 token 数、单步时延与困难状态触发率 |
| E5 Pareto 前沿 | `run_e5_pareto.py` | 注入不同 OA 偏好 ω_e 扫描真实能耗-时延 Pareto 曲线 |
| E6 鲁棒性 | `run_e6_robust.py` | 信道突变/服务器宕机/不诚实 UA 三类扰动下验证 VA 与 ToM 能力 |
| E7 灵敏度 | `run_e7_sensitivity.py` | 对 τ_c、β、ε_c、δ_v 四个关键超参做 1D sweep |
| E8 蒸馏数据量 | `run_e8_distill_size.py` | 数据集规模 {1k,2k,5k,10k,20k} 对策略性能的影响 |
| E9 多 LLM 对比 | `run_e9_multi_llm.py` | 4 个 LLM 后端（Qwen/Llama/Mistral）分阶段生成→训练→评估 |

一键运行全部实验：

```bash
bash src/local/run_all_experiments.sh
```

> 注：`run_all_experiments.sh` 会先探测 `localhost:8000` 的 vLLM 服务；部分实验（如 E5/E9）需要 LLM 服务在线，否则相关方法会退化为启发式。

---

## 6. 关键配置（`src/config.py`）

```python
# 系统规模
NUM_USERS = 8; NUM_EDGE_SERVERS = 4; NUM_CLOUD_SERVERS = 1
SCENE_TYPE = "MEC"            # "MEC" 固定用户 / "VEC" 车联网移动场景

# LLM 后端
LLM_BACKEND = "local_vllm"    # deepseek | openai | qwen | local_vllm | local_transformers
LLM_TEMPERATURE = 0.0         # 确定性输出，辩论一致性更好

# FBL 通信
FBL_ENABLED = True            # True=有限码长模型；False=Shannon 无限码长
FBL_BLOCKLENGTH = 168         # 每帧码元数 n_k（5G NR mini-slot）

# 云端卸载
ENABLE_CLOUD_OFFLOAD = True   # 边缘过载自动溢出至云端

# 辩论协议
CONFIDENCE_THRESHOLD = 0.6    # 置信度门控阈值
DEBATE_MAX_ROUNDS = 5         # 最大辩论轮次
CONSENSUS_EPSILON = 0.05      # 共识终止阈值

# 蒸馏
DISTILL_DATASET_SIZE = 5000   # 离线蒸馏数据集规模
POLICY_NET_HIDDEN = 256       # 策略网络隐藏层维度
```

---

## 7. 复现说明

- 所有模块设置全局随机种子 `SEED = 42`（`config.py`），保证实验可复现。
- 蒸馏数据生成支持**断点续存**：已写入 `debate_dataset.jsonl` 的样本自动跳过，中断后重跑即可。
- 论文用 SAC/DDPG 建议训练 500+ episode；代码默认 30-50 便于 CPU 自检（见 `local/baselines.py` 与各实验「显眼配置区」）。

---

## 8. 引用

如本项目对你的研究有帮助，欢迎引用：

```bibtex
@misc{hme-mec,
  author = {},
  title  = {HMA-MEC: A Hybrid Multi-Agent LLM Framework for Mobile Edge Computing Task Offloading},
  year   = {2026},
  howpublished = {\url{https://github.com/sleepSleepsleep-lab/HMA-MEC}}
}
```

---

## 9. 开源协议

本项目采用 [MIT License](LICENSE)，可自由使用、修改与分发，请保留版权声明。

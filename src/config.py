# -*- coding: utf-8 -*-
"""
================================================================
全局配置文件 (config.py)
================================================================
本文件集中管理 MEC 系统参数、Agent 框架超参数、LLM 客户端配置、
蒸馏与训练超参数等。所有可能需要用户根据自身硬件 / API 配额 /
实验设计进行调整的参数，均放置在本文件的「显眼配置区」中，
并以中文注释解释每个参数的含义与典型取值范围。

阅读建议：先看「显眼配置区」，再按需查看其他细分配置。
================================================================
"""

import os
import random
import numpy as np

# ============================================================
# 全局随机种子：保证实验可复现
# ============================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

try:
    import torch
    torch.manual_seed(SEED)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    torch = None
    DEVICE = "cpu"


# ============================================================
# >>>>>>>>>>>>>>>>>>  显眼配置区  <<<<<<<<<<<<<<<<<<
# ============================================================
# 以下参数最常被用户修改：系统规模、LLM 方案、API 密钥、路径。
# 修改这些参数即可适配绝大多数实验需求，无需改动其他配置块。
# ============================================================

# ---- 系统规模 ----
NUM_USERS = 8                  # 用户设备数量 K，典型范围 4~32
NUM_EDGE_SERVERS = 4           # 边缘服务器数量 M，典型范围 2~8

# ---- LLM 客户端方案 ----
# 可选值："deepseek" | "openai" | "qwen" | "local_vllm" | "local_transformers"
# "deepseek"         : 调用 DeepSeek 商用 API（旧项目沿用）
# "openai"           : 调用 OpenAI 官方 API
# "qwen"             : 调用阿里云百炼平台通义千问 API (OpenAI 兼容格式)
# "local_vllm"       : 本地 vLLM 部署的开源模型（需 GPU），E9 多模型对比使用
# "local_transformers": 直接 transformers 推理（开发调试用，速度较慢）
LLM_BACKEND = "local_vllm"

# API 密钥与端点（仅在使用商用 API 时生效，本地部署可留空）
LLM_API_KEY = ""
LLM_API_BASE = ""

# LLM 模型名（不同后端对应不同模型取值）
# vLLM:  模型名即 HuggingFace 仓库名，如 "Qwen/Qwen2.5-7B"
# DeepSeek:  "deepseek-v4-flash" / "deepseek-v4-pro"
# Qwen:      "qwen3.7-plus" / "qwen3.7-max"
LLM_MODEL = "Qwen/Qwen2.5-7B"
# 2026.07 修改：从 Qwen3.6-27B 降级为 Qwen2.5-7B。
# 理由：优先使用小尺寸模型验证框架有效性。
# 实验表明 HMA-MEC 在不同 LLM 规模下性能差异小于 5%，
# 框架优势源于多 Agent 辩论协议与蒸馏压缩范式，而非大模型参数规模。
# 7B 级模型可在消费级 GPU（RTX 3090/4090）上运行，降低实验门槛。

# ----- DeepSeek 专属配置 -----
LLM_THINKING_ENABLED = False

# ----- 阿里云百炼 (Qwen) 专属配置 -----
QWEN_API_KEY = ""
QWEN_API_BASE = ""

# ----- 本地 vLLM / transformers 配置 -----
# 主模型本地权重路径与 vLLM 服务端口
# 2026-08 整改: 原 "/models/Qwen3.6-27B" 为旧机器路径, 已迁移至本服务器
# (本地模型权重目录, 模型经 ModelScope 下载, _VLLMClient 用其 basename 作模型名)
LLM_LOCAL_MODEL_PATH = os.environ.get("HMA_LLM_MODEL_PATH", "models/Qwen2.5-7B")
LLM_LOCAL_PORT = 8000

# E9 多模型对比的端口定义
# 每个模型独立启动一个 vLLM 服务，各占一个端口。
# 2026.07 修改：主力模型从 27B 降级为 7B，新增 3B 级极轻量模型用于树莓派部署对比。
LLM_PORT_QWEN25_7B = 8000    # 主力：7B 级
LLM_PORT_LLAMA_8B  = 8001    # 对比：英文 8B
LLM_PORT_MISTRAL_7B= 8002    # 对比：英文 7B
LLM_PORT_QWEN25_3B = 8003    # 对比：3B 级极轻量（树莓派部署测试）

# E9 模型注册表：name → (port, is_primary)
# 注意：port 8004 及以上为预留端口，兼容旧版 Qwen3.6-27B 的可选扩展实验
LLM_MODEL_REGISTRY = {
    "Qwen2.5-7B":    {"port": LLM_PORT_QWEN25_7B, "primary": True},
    "Llama-3.1-8B":  {"port": LLM_PORT_LLAMA_8B,  "primary": False},
    "Mistral-7B":    {"port": LLM_PORT_MISTRAL_7B, "primary": False},
    "Qwen2.5-3B":    {"port": LLM_PORT_QWEN25_3B, "primary": False},
}

# 调用配置
LLM_TEMPERATURE = 0.0        # 0 表示近乎确定性输出，辩论一致性更好
LLM_MAX_TOKENS = 512         # 单次响应最大 token 数
LLM_REQUEST_DELAY = 0.3      # 两次 API 调用之间的最小间隔（秒），速率限制器会覆盖此值
LLM_RATE_LIMIT_RPM = 2500    # 全局速率限制：每分钟最多调用次数（DeepSeek API 配额）

# ---- 工作目录 ----
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "checkpoints")


# ============================================================
# 通信模型参数
# ============================================================
BANDWIDTH = 10e6        # 每个子信道带宽 10 MHz
NOISE_POWER = 1e-9      # 加性高斯白噪声功率 -60 dBm
TX_POWER_USER = 0.2     # 终端上行发射功率 200 mW
TX_POWER_ES = 1.0       # 边缘服务器下行功率 1 W（本文忽略下行）

# 信道时间相关性系数（一阶自回归）
CHANNEL_CORR_COEF = 0.95
# 复高斯信道系数幅度标尺（使 |g|^2 的期望值在约 1e-6 量级）
CHANNEL_COEFF_SCALE = 1e-3


# ============================================================
# 计算模型参数
# ============================================================
# 终端本地 CPU 频率（CPU 周期/秒），与旧项目保持一致
F_LOCAL = [1.0e9, 1.5e9, 0.8e9, 1.2e9, 1.0e9, 1.3e9, 0.9e9, 1.1e9]
# 边缘服务器 CPU 频率
F_EDGE = [10e9, 8e9, 12e9, 9e9]
# 云端 CPU 频率（预留，本文不使用云端）
F_CLOUD = 100e9

# 任务参数
TASK_DATA_MIN = 0.5e6     # 任务最小数据量 0.5 MB
TASK_DATA_MAX = 5e6       # 任务最大数据量 5 MB
TASK_CYCLES_PER_BIT = [500, 1500]   # 每比特所需 CPU 周期数范围
TASK_DEADLINE_MIN = 0.5    # 任务最小容忍时延 0.5 s
TASK_DEADLINE_MAX = 3.0    # 任务最大容忍时延 3.0 s
# 任务优先级概率分布：低=0.5, 中=0.3, 高=0.2
TASK_PRIORITY_PROB = [0.5, 0.3, 0.2]

# 能耗系数
KAPPA_LOCAL = 1e-27   # CMOS 有效开关电容
P_IDLE = 0.05         # 终端空闲功率（W）

# 边缘服务器有效能耗开关与系数（用于计入服务器侧 CPU 能量，避免
# "AllEdge" 在未计量服务器能耗时被无差别偏向）。
# 当 ACCOUNT_EDGE_ENERGY = True 时，系统总能耗把
#   E_edge_server = KAPPA_EDGE * F_m^2 * off_cycles_m
# 计入用户口径以反映边缘算力消耗的代价。
# 关闭时与原项目建模一致。
#
# 重要设计决策（2026.07 修改）：
#   原始版本 KAPPA_EDGE = 1e-30（为 KAPPA_LOCAL 的 1/1000），
#   导致服务器侧能耗可忽略，AllEdge 策略在能耗指标上呈现"近似免费"的假象。
#   这掩盖了本地执行与边缘卸载之间真正的能耗-时延 trade-off：
#   当服务器能耗被低估时，优化目标本质上等效于仅优化时延，
#   实验中 E5 Pareto 前沿无法展开正是在于缺少这一矛盾张力。
#
#   将 KAPPA_EDGE 提升至与 KAPPA_LOCAL 同量级（1e-27），
#   使"卸载至服务器"产生与"本地执行"量级相当的能耗代价，
#   从而在目标函数中恢复能耗-时延的竞争关系。
ACCOUNT_EDGE_ENERGY = True
KAPPA_EDGE = 1e-27    # 边缘服务器 CMOS 有效能量系数
# 取值与终端 KAPPA_LOCAL = 1e-27 同量级，使边缘服务器能耗
# 不再是可忽略项，恢复本地执行与边缘卸载之间的真实 trade-off。


# ============================================================
# 环境与训练时长参数
# ============================================================
MAX_EPISODES = 1000
MAX_STEPS = 200       # 每个 episode 的时间步数

# ---- 基线训练奖励对齐开关（M2 修复，2026-08 整改）----
# True:  保留 priority_bonus（旧行为，SAC/DDPG 训练含优先级奖励）
# False: 移除 bonus，训练奖励 = -(E + T)/10，与评估指标对齐（推荐，E1 重跑用）
REWARD_INCLUDE_PRIORITY_BONUS = False

# ---- （可选，Q2）链路误帧率支持（默认全 0 = 现有行为不变）----
LINK_ERROR_RATE = 0.0            # 全局默认误帧率，E6 link_fail 场景单独设置

# ---- （可选，Q5）GA 基线超参数 ----
GA_POP_SIZE = 30                 # 种群大小
GA_GENERATIONS = 50              # 迭代代数
GA_MUTATION_RATE = 0.1           # 变异概率
GA_ELITE = 2                     # 精英保留数
GA_SEED_MIX = 0.3                # 初始种群中启发式种子（Greedy/AllEdge）比例

# ---- （可选，Q6）边缘中继开关（06 文档方案 B，默认关）----
ENABLE_RELAY = False
RELAY_LINK_RATE = 5e6            # 边缘间链路速率 bps
RELAY_PROP_DELAY = 0.0001        # 边缘间传播时延 s（100μs 量级）

# 状态与动作维度（由系统规模推导）
STATE_DIM = NUM_USERS * 4 + NUM_EDGE_SERVERS * 2
ACTION_DIM = NUM_USERS * 2  # 每用户两维：(卸载比例, 服务器选择)


# ============================================================
# A1: Agent 定义模型超参数
# ============================================================
# Agent 类型数量：K 个 UA + M 个 EA + 1 个 OA + 1 个 VA
AGENT_TYPES = ["UA", "EA", "OA", "VA"]

# 置信度门控阈值（A2）：低于该值的 UA 进入批判轮
CONFIDENCE_THRESHOLD = 0.6

# ---- 置信度灵敏度系数（2026.07 扩展为三维） ----
# 原始版本仅使用时延因子：c_k = sigmoid(CONFIDENCE_BETA · Δ_t)。
# 修改后将 MEC 物理量显式嵌入置信度建模，使置信度反映信道质量
# 与服务器负载率：
#   c_k = sigmoid(β1·Δ_t + β2·SINR_k + β3·(1-ρ_{m_k}))
# 其中 Δ_t 为时延余量，SINR 为信道质量，ρ 为服务器负载率。
# 该修改使低置信度不再仅由"时延超标"触发，还覆盖"信道较差"
# 和"服务器过载"两种 MEC 特有的不可行场景。
CONFIDENCE_BETA_TAU = 2.0    # 时延余量权重 β1（原 CONFIDENCE_BETA）
CONFIDENCE_BETA_SINR = 1.0   # 信道质量权重 β2
CONFIDENCE_BETA_LOAD = 1.5   # 服务器充裕度权重 β3
# 共识终止阈值：置信度变化小于该值则终止辩论
CONSENSUS_EPSILON = 0.05
# 最大辩论轮次
DEBATE_MAX_ROUNDS = 5
# 反事实验证允许的能耗/时延与 OA 预测差异上限（A4）
VERIFY_GAP_TOLERANCE = 0.15
# 反事实回退最大次数
VERIFY_MAX_FALLBACK = 2


# ============================================================
# A3: 蒸馏式推理 Agent 超参数
# ============================================================
# 离线蒸馏数据集大小
DISTILL_DATASET_SIZE = 5000
# 蒸馏策略网络结构
POLICY_NET_HIDDEN = 256
POLICY_NET_LR = 1e-3
POLICY_NET_EPOCHS = 100
POLICY_NET_BATCH = 128
POLICY_NOISE_STD = 0.015    # 蒸馏训练时状态向量高斯噪声标准差（数据增强）
# 困难状态触发在线辩论的置信度下限（混合模式）
HYBRID_CONFIDENCE_LOW = 0.3


# ============================================================
# 可视化配置
# ============================================================
# 图像 DPI（IEEE 期刊要求线图 ≥ 600，本方案统一 1000 以适配常见审稿规范）
TIFF_DPI = 1000
PDF_DPI = 1000
# 颜色调色板（与旧项目视觉一致性）
COLORS = {
    'green':  '#71C58B',
    'red':    '#EF8979',
    'teal':   '#8DCCC2',
    'gold':   '#EAC49A',
    'purple': '#8A6C87',
    'blue':   '#7BA7D9',
    'gray':   '#9AA0A6',
}
# 中文字体（matplotlib 显示中文用）
PLOT_FONT_FAMILY = "SimHei"
PLOT_FONT_SIZE = 12


# ============================================================
# 旧 SAC baseline 超参数（保留作对比）
# ============================================================
LR_ACTOR = 1e-4
LR_CRITIC = 3e-4
GAMMA = 0.99
TAU = 0.005
BUFFER_CAPACITY = 100000
BATCH_SIZE = 128
HIDDEN_DIM = 256


# ============================================================
# 工具函数：保证输出目录存在
# ============================================================
def ensure_dir(path):
    """若目录不存在则递归创建。"""
    os.makedirs(path, exist_ok=True)


ensure_dir(RESULTS_DIR)
ensure_dir(CHECKPOINT_DIR)
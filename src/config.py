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
NUM_CLOUD_SERVERS = 1          # 云端服务器数量

# ---- 场景模式选择 ----
# "MEC": 原固定用户场景； "VEC": 车联网移动场景（无人驾驶出租车）
SCENE_TYPE = "MEC"

# ---- LLM 客户端方案 ----
# 可选值："deepseek" | "openai" | "qwen" | "local_vllm" | "local_transformers"
# "deepseek"         : 调用 DeepSeek 商用 API（旧项目沿用）
# "openai"           : 调用 OpenAI 官方 API
# "qwen"             : 调用阿里云百炼平台通义千问 API (OpenAI 兼容格式)
# "local_vllm"       : 本地 vLLM 部署的开源模型（需 GPU），用于离线蒸馏数据生成
# "local_transformers": 直接 transformers 推理（无需 vLLM，直接加载模型推理）
LLM_BACKEND = "local_vllm"

# API 密钥与端点（仅在使用商用 API 时生效，本地部署可留空）
LLM_API_KEY = ""
LLM_API_BASE = ""

# LLM 模型名（不同后端对应不同模型取值）
# vLLM:  模型名即 HuggingFace 仓库名，如 "Qwen/Qwen2.5-7B"
# DeepSeek:  "deepseek-v4-flash" / "deepseek-v4-pro"
# Qwen:      "qwen3.7-plus" / "qwen3.7-max"
LLM_MODEL = "Qwen3.5-9B"
# 2026.08 修改：切换为通义千问 Qwen3.5-9B 本地推理。
# 模型路径：/root/autodl-tmp/model_origin/qwen3.5
# vLLM 服务通过 --served-model-name Qwen3.5-9B 注册。
# 该模型在 MEC 任务卸载的 JSON 格式输出与推理方面
# 与 DeepSeek-v4-flash 质量接近，适合离线蒸馏数据生成。

# ----- DeepSeek 专属配置 -----
LLM_THINKING_ENABLED = False

# ----- 阿里云百炼 (Qwen) 专属配置 -----
QWEN_API_KEY = ""
QWEN_API_BASE = ""

# ----- 本地 vLLM / transformers 配置 -----
# 主模型（Qwen3.5-9B）的本地权重路径与 vLLM 服务端口
LLM_LOCAL_MODEL_PATH = "/root/autodl-tmp/model_origin/qwen3.5"
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

# ---- 有限码长 (FBL) 通信模型参数 ----
FBL_ENABLED = True                # True=FBL模型; False=Shannon无限码长
FBL_BLOCKLENGTH = 168             # 每帧码元数 n_k (5G NR mini-slot)
FBL_MAX_ERROR_PROB = 0.01         # 每用户最大解码错误概率 ε_max

# ---- VEC 移动场景参数（SCENE_TYPE="VEC" 时生效）----
ROAD_LENGTH = 500.0               # 单向道路总长 (m)
RSU_COVERAGE_RADIUS = 100.0       # 每 RSU 覆盖半径 (m)
VEHICLE_SPEED_MIN = 30.0          # 最低车速 (km/h)
VEHICLE_SPEED_MAX = 60.0          # 最高车速 (km/h)
PATH_LOSS_ALPHA = 3.0             # 路径损耗指数（城市宏蜂窝）
PATH_LOSS_REF = 1e-3              # 参考距离路径损耗
CARRIER_FREQ = 5.9e9              # 载波频率 5.9 GHz (DSRC/C-V2X)
CONNECTION_TIMEOUT_PENALTY = 5.0  # 连接超时惩罚系数

# ---- 云端卸载层参数 ----
CLOUD_LATENCY_BASE = 0.05         # 核心网传播+排队时延 (50 ms)
CLOUD_TRANSMISSION_FACTOR = 2.0   # 云端上行能耗倍率（相对于 RSU 上行）
ENABLE_CLOUD_OFFLOAD = True       # 是否启用云端卸载通道


# ============================================================
# 计算模型参数
# ============================================================
# 终端本地 CPU 频率（CPU 周期/秒），与旧项目保持一致
F_LOCAL = [1.0e9, 1.5e9, 0.8e9, 1.2e9, 1.0e9, 1.3e9, 0.9e9, 1.1e9]
# 边缘服务器 CPU 频率
F_EDGE = [10e9, 8e9, 12e9, 9e9]
# 云端 CPU 频率（端-边-云三层架构：算力无限但时延较高）
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

# 状态与动作维度（由系统规模推导）
# 2026.08: 状态空间可含 6K+2M (VEC场景: 每用户额外位置+速度)
#          动作空间可含 3K (每用户额外 cloud 选择标志)
# MEC 场景: state_dim=4K+2M, action_dim=2K+(cloud_idx*1K)
# VEC 场景: state_dim=6K+2M, action_dim=3K
STATE_DIM = NUM_USERS * 4 + NUM_EDGE_SERVERS * 2
ACTION_DIM = NUM_USERS * 2  # (卸载比例, 服务器选择)
_ACTION_DIM_VEC = NUM_USERS * 3  # (卸载比例, 服务器选择, cloud标志)


# ============================================================
# A1: Agent 定义模型超参数
# ============================================================
# Agent 类型数量：K 个 UA + M 个 EA + 1 个 OA + 1 个 VA
AGENT_TYPES = ["UA", "EA", "OA", "VA", "CA"]

# 置信度门控阈值（A2）：低于该值的 UA 进入批判轮
CONFIDENCE_THRESHOLD = 0.6

# ---- 置信度灵敏度系数（2026.08 扩展为四维） ----
# v1 (2026.06): c_k = sigmoid(β·Δ_t)                 ——仅时延
# v2 (2026.07): c_k = sigmoid(β1·Δ_t+β2·SINR+β3·(1-ρ)) ——三维MEC物理量
# v3 (2026.08): c_k = sigmoid(β1·Δ_t+β2·SINR+β3·(1-ρ)+β4·(1-ε_k))
#                ——四维: 增加FBL解码可靠性分量, 覆盖有限码长传输误差
CONFIDENCE_BETA_TAU = 2.0    # 时延余量权重 β1
CONFIDENCE_BETA_SINR = 1.0   # 信道质量权重 β2
CONFIDENCE_BETA_LOAD = 1.5   # 服务器充裕度权重 β3
CONFIDENCE_BETA_FBL = 0.8    # FBL 传输可靠性权重 β4（新增）
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
HIDDEN_DIM = 256


# ============================================================
# 工具函数：保证输出目录存在
# ============================================================
def ensure_dir(path):
    """若目录不存在则递归创建。"""
    os.makedirs(path, exist_ok=True)


ensure_dir(RESULTS_DIR)
ensure_dir(CHECKPOINT_DIR)
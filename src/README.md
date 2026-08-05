# src 目录说明

本目录按硬件环境分三部分组织代码：

| 目录 | 运行环境 | 用途 |
| --- | --- | --- |
| `local/` | 本地 CPU | baseline 算法、实验脚本（E1-E9）、公共实验框架 |
| `gpu/` | NVIDIA GPU + vLLM 服务器 | LLM 离线蒸馏数据生成、蒸馏策略网络训练 |
| `visual/` | 本地 CPU | 论文绘图脚本，输出 1200 DPI PDF/TIFF |

公共模块（`src/` 根目录，被三部分共享）：

| 文件 | 说明 |
| --- | --- |
| `config.py` | 全局配置（系统规模 / LLM 后端 / 超参数） |
| `environment.py` | MEC / VEC 仿真环境（FBL 信道、云端卸载、反事实仿真接口） |
| `llm_client.py` | LLM 客户端抽象层（deepseek / openai / qwen / local_vllm / local_transformers） |
| `agent_define.py` | A1：Agent 定义模型（场景-功能-边界三元组） |
| `cw_debate.py` | A2：CW-Debate 置信度加权多轮辩论协议 |
| `orchestrator.py` | OA 编排智能体实现（偏好推断 / 仲裁 / ToM / 共识） |
| `verifier.py` | A4：验证智能体（反事实仿真 + 拒绝采样） |
| `distill_agent.py` | A3：蒸馏策略网络 PolicyAgentNet 与训练器 |
| `agent_runner.py` | 三模式在线推理运行器（FullLLM / Distill / Hybrid） |
| `train_k8_policy.py` | 用 K=8 样本快速训练蒸馏策略 |

详细使用方法见项目根目录 [README.md](../README.md)。

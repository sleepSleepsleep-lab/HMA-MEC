# -*- coding: utf-8 -*-
"""
================================================================
src 目录结构说明 (README.md 的代码版)
================================================================
本目录分三部分组织代码，便于在不同硬件环境下分发执行：

  local/    —— 本地 CPU 可运行的代码：
              baseline 算法、可视化、小规模自检、推理调用脚本

  gpu/      —— 需上传到带 NVIDIA GPU 的高性能机器运行的代码：
              LLM 离线蒸馏数据生成、蒸馏策略网络训练、
              大规模 LLM 在线辩论实验

  visual/   —— 本地 CPU 运行的可视化脚本：
              每个图一个 .py 文件，同时输出 1200 DPI TIFF 与 1200 DPI PDF

公共模块（位于 src 根目录，被三部分共享）：
  config.py            全局配置
  environment.py       MEC 仿真环境
  llm_client.py        LLM 客户端抽象层
  agent_define.py      A1: Agent 定义模型
  verifier.py          A4: 验证智能体（P2 完善）
  cw_debate.py         A2: 多轮辩论协议（P2 实现）
  orchestrator.py      OA 实现（P2 实现）
  distill_agent.py     A3: 蒸馏策略网络（P3 实现）
  agent_runner.py      在线推理运行器（P3 实现）
================================================================
"""
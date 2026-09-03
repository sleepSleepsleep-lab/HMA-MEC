# HMA-MEC：面向移动边缘计算任务卸载的层级式多智能体大模型协同推理框架

> 论文配套实验代码与结果数据（companion repository）。
>
> HMA-MEC is a hierarchical multi-agent large-language-model (LLM)
> collaborative decision-making framework for joint task offloading and
> resource allocation in multi-user multi-server Mobile Edge Computing (MEC).
> This repository contains the simulator, the four agents (UA/EA/OA/VA),
> the CW-Debate protocol, the counterfactual-simulation verifier, the
> distillation pipeline, and all result files behind experiments E1–E21
> of the paper.

## 框架与代码对应（论文 §2–§3）

| 论文组件 | 说明 | 主要代码 |
|---|---|---|
| A1 场景–功能–边界 Agent 定义 | 四类 Agent（UA/EA/OA/VA） | `src/agent_define.py`, `src/agent_runner.py` |
| A2 CW-Debate 置信度加权多轮辩论协议 | 五轮交互、置信度门控、共识终止 | `src/cw_debate.py`, `src/orchestrator.py` |
| A3 Distill-Agent 离线蒸馏推理 | 辩论 → 单次前向策略网络（Distill/Hybrid/FullLLM 三模式） | `src/distill_agent.py`, `src/gpu/`（数据集生成与训练） |
| A4 反事实仿真验证器（VA） | 环境副本反事实评估 + 绝对底线/相对偏差判据 + 回退控制 | `src/verifier.py`, `src/local/plan_refiner.py` |
| MEC 物理环境 | 优先级 FIFO 排队、ARQ、能耗核算 | `src/environment.py`, `src/config.py` |

## 目录结构

```
src/                核心代码（环境、Agent、协议、验证器、蒸馏、LLM 客户端）
  local/            E1–E21 实验脚本（run_eX_*.py）、基线实现、结果取数入口 results_store.py
  visual/           论文图表生成脚本
  gpu/              蒸馏教师数据采集与策略网络训练脚本
results/            实验结果
  *.json            各实验汇总指标（mean/std/ci95/n）
  *.npz             时序原始数据
  records/*.jsonl   逐 episode 记录（所有汇总数字的单一事实源）
  checkpoints/*.pth 蒸馏策略权重（含 E9 三后端、K12 重蒸馏等）
```

## 环境

- 实验完成于 Ubuntu 22.04 + NVIDIA RTX 4090 24GB（论文口径）；
  纯 Distill 模式与绘图脚本在 Windows/Linux CPU 环境亦可运行。
- Python 3.12；`pip install -r requirements.txt`
- 复现 FullLLM 辩论/E9 需本地 vLLM 服务：
  `python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-7B --port 8000`
  （模型本地权重目录可用环境变量 `HMA_LLM_MODEL_PATH` 指定）

## 快速开始

```bash
pip install -r requirements.txt
# 量级自检：蒸馏前向 / 验证器闭环 / GA 单步计时（无需 GPU 与 LLM）
python src/local/_verify_timing_local.py
# 取数入口：从 records 复算任意实验的 mean/std/ci95
python -c "import sys; sys.path.insert(0,'src/local'); import results_store; print([m for m in dir(results_store) if not m.startswith('_')])"
# 各实验（需 vLLM 后端的除外）
python src/local/run_e1_main.py        # E1 主对比
python src/local/run_e3_ablation.py    # E3 组件消融
python src/local/run_e6_robust.py      # E6 五类扰动鲁棒性
```

## 数据说明

- 统计口径：评估种子为 `S+s*100+e`（5 种子 × 50 episode，250 个互不碰撞环境，真独立样本）；
  检验为配对 Wilcoxon / Mann-Whitney U（双侧），多重比较族 Holm 校正。
- **米兰（Telecom Italia）原始数据集因第三方许可未随仓库分发**（约 1.5 GB）。
  E19 复现请按 [Barlacchi et al., Sci. Data 2015](https://doi.org/10.1038/sdata.2015.55)
  自行下载，预处理流程见 `src/local/preprocess_milano.py`（10 分钟粒度全网流量 → 小时聚合
  → clip[0.3, 3] → 均值归一化）。
- 蒸馏教师数据集（`debate_dataset*.jsonl`，约 30 MB，含 LLM 对话轨迹）未随仓库分发，
  可用 `src/gpu/gen_distill_dataset.py` 在 vLLM 后端上重新生成。

## Citation

论文投稿中，BibTeX 将在录用后更新至此。

## License

MIT

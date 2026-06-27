# GRPO

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.7](https://img.shields.io/badge/PyTorch-2.7-red.svg)](https://pytorch.org/)
[![CUDA 12.6](https://img.shields.io/badge/CUDA-12.6-green.svg)](https://developer.nvidia.com/cuda-downloads)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> 基于 **veRL + vLLM V1** 在 τ-bench airline（多轮、多工具对话智能体）上训练 **GRPO（Group Relative Policy Optimization）** 的工程实现。
> 采用 **SFT 冷启动 → GRPO 强化** 的两阶段流程，reward 为 τ-bench 任务级二元 outcome（成功 1 / 失败 0）。

---

## 流程

1. **SFT 数据采集**：用 72B 教师在 τ-bench airline 上 rollout，收集成功轨迹（含污染检测与多轮工具调用掩码）。
2. **SFT 训练**：对 7B 策略做 LoRA 微调，作为 GRPO 的冷启动起点。
3. **构建 GRPO 数据**：把采集的轨迹整理成 veRL 所需的 parquet，并生成 τ-bench 工具 schema。
4. **GRPO 训练**：以二元 outcome reward 做 group-relative 策略优化。
5. **评测**：独立 pass@k 评测器，对检查点做逐步评测。

---

## 项目结构

```raw
📦 grpo/
├── ⚙️ configs/
│   ├── train/
│   │   ├── grpo/vanilla_grpo.yaml     # GRPO 训练配置
│   │   ├── mock/mock_grpo.yaml        # 冒烟测试配置
│   │   └── sft/                       # SFT 采集 + LoRA 训练配置
│   ├── eval/                          # 评测配置
│   ├── interaction_config/            # τ-bench interaction 接入
│   ├── tool_config/                   # τ-bench 工具 schema
│   └── baseline_airline.yaml
├── 💻 src/
│   ├── envs/                          # τ-bench interaction / tools / context
│   ├── evaluation/pass_k_eval.py      # 独立 pass@k 评测器
│   ├── models/vllm_policy.py          # vLLM 策略包装
│   ├── training/sft_dataset.py        # SFT 数据集
│   └── utils/
├── 📜 scripts/
│   ├── train/{grpo,sft,mock}/         # 训练启动脚本
│   ├── eval/                          # 评测脚本
│   └── vllm_server/                   # vLLM 服务脚本
├── 📚 docs/                           # 工程设计 / 优化记录
└── 🧪 experiments/                    # 检查点、评测输出
```

仓库根目录还包含 vendored 的 `verl/`（训练框架）与 `tau-bench/`（评测基准），按本地 editable 安装使用。

---

## 快速开始

### 1. 环境搭建

```bash
bash setup.sh                  # conda env + PyTorch 2.7 + CUDA 12.6 + 依赖
conda activate agentrl
cd grpo
# veRL 与 τ-bench 按本地 editable 安装（见 requirements.txt）
```

### 2. SFT 冷启动

```bash
# 采集教师轨迹
python scripts/train/sft/collect_sft_data.py --config-name=sft_collect_airline
# LoRA 训练
python scripts/train/sft/sft_train.py --config-name=sft_airline_lora
# （可选）合并 LoRA 权重
python scripts/train/sft/merge_lora.py
```

### 3. 构建 GRPO 数据

```bash
python scripts/train/grpo/gen_tool_config.py     # 生成 τ-bench 工具 schema
python scripts/train/grpo/build_grpo_parquet.py  # 构建 GRPO 训练 parquet
```

### 4. GRPO 训练

```bash
cd scripts/train/grpo
bash run_vanilla.sh
# 冒烟测试：
# bash scripts/train/mock/run_vanilla_grpo_mock.sh
```

### 5. 评测

```bash
cd scripts/eval
bash eval_vanilla.sh
```

> **硬件**：2×A800（80GB）。GPU 0 运行 7B 策略 vLLM；GPU 1 运行 72B-AWQ 用户模拟器。
> **离线模式**：所有训练 / 评测脚本注入 `HF_HUB_OFFLINE=1` 与 `TRANSFORMERS_OFFLINE=1`，适用于隔离环境。

---

## 技术栈

- **训练框架**: [veRL](https://github.com/volcengine/verl) 0.6.1（FSDP + vLLM V1）
- **策略模型**: Qwen2.5-7B-Instruct
- **用户模拟器**: Qwen2.5-72B-Instruct-AWQ
- **评测基准**: [τ-bench](https://github.com/sierra-research/tau-bench) airline
- **推理引擎**: vLLM V1（Hermes tool-call parsing）
- **注意力**: FlashAttention-2

---

## 致谢

- [veRL](https://github.com/volcengine/verl) —— 开源 RL 训练框架
- [τ-bench](https://github.com/sierra-research/tau-bench) —— 评测基准
- [Qwen](https://github.com/QwenLM/Qwen) —— 基座模型

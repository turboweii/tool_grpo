# GRPO

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.7](https://img.shields.io/badge/PyTorch-2.7-red.svg)](https://pytorch.org/)
[![CUDA 12.6](https://img.shields.io/badge/CUDA-12.6-green.svg)](https://developer.nvidia.com/cuda-downloads)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> An engineering implementation of **GRPO (Group Relative Policy Optimization)** on τ-bench airline (multi-turn, multi-tool conversational agents), built on **veRL + vLLM V1**.
> Two-stage pipeline: **SFT warm-start → GRPO RL**. Reward is the τ-bench task-level binary outcome (1 = success, 0 = failure).

---

## Pipeline

1. **SFT data collection**: roll out a 72B teacher on τ-bench airline and collect successful trajectories (with contamination detection and multi-turn tool-call masking).
2. **SFT training**: LoRA fine-tune the 7B policy as the GRPO warm start.
3. **Build GRPO data**: pack the collected trajectories into veRL parquet and generate the τ-bench tool schema.
4. **GRPO training**: group-relative policy optimization with binary outcome reward.
5. **Evaluation**: an independent pass@k evaluator that scores checkpoints step by step.

---

## Project structure

```raw
📦 grpo/
├── ⚙️ configs/
│   ├── train/
│   │   ├── grpo/vanilla_grpo.yaml     # GRPO training config
│   │   ├── mock/mock_grpo.yaml        # smoke-test config
│   │   └── sft/                       # SFT collect + LoRA training configs
│   ├── eval/                          # evaluation configs
│   ├── interaction_config/            # τ-bench interaction adapter
│   ├── tool_config/                   # τ-bench tool schema
│   └── baseline_airline.yaml
├── 💻 src/
│   ├── envs/                          # τ-bench interaction / tools / context
│   ├── evaluation/pass_k_eval.py      # standalone pass@k evaluator
│   ├── models/vllm_policy.py          # vLLM policy wrapper
│   ├── training/sft_dataset.py        # SFT dataset
│   └── utils/
├── 📜 scripts/
│   ├── train/{grpo,sft,mock}/         # training launch scripts
│   ├── eval/                          # evaluation scripts
│   └── vllm_server/                   # vLLM server scripts
├── 📚 docs/                           # design / optimization notes
└── 🧪 experiments/                    # checkpoints, eval outputs
```

The repo root also vendors `verl/` (training framework) and `tau-bench/` (benchmark), installed editable locally.

---

## Quick start

### 1. Environment

```bash
bash setup.sh                  # conda env + PyTorch 2.7 + CUDA 12.6 + deps
conda activate agentrl
cd grpo
# Install veRL and τ-bench editable locally (see requirements.txt)
```

### 2. SFT warm start

```bash
# Collect teacher trajectories
python scripts/train/sft/collect_sft_data.py --config-name=sft_collect_airline
# LoRA training
python scripts/train/sft/sft_train.py --config-name=sft_airline_lora
# (optional) merge LoRA weights
python scripts/train/sft/merge_lora.py
```

### 3. Build GRPO data

```bash
python scripts/train/grpo/gen_tool_config.py     # generate τ-bench tool schema
python scripts/train/grpo/build_grpo_parquet.py  # build GRPO training parquet
```

### 4. GRPO training

```bash
cd scripts/train/grpo
bash run_vanilla.sh
# Smoke test:
# bash scripts/train/mock/run_vanilla_grpo_mock.sh
```

### 5. Evaluation

```bash
cd scripts/eval
bash eval_vanilla.sh
```

> **Hardware**: 2×A800 (80GB). GPU 0 runs the 7B policy vLLM; GPU 1 runs the 72B-AWQ user simulator.
> **Offline mode**: all training / eval scripts inject `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` for isolated environments.

---

## Tech stack

- **Training framework**: [veRL](https://github.com/volcengine/verl) 0.6.1 (FSDP + vLLM V1)
- **Policy model**: Qwen2.5-7B-Instruct
- **User simulator**: Qwen2.5-72B-Instruct-AWQ
- **Benchmark**: [τ-bench](https://github.com/sierra-research/tau-bench) airline
- **Inference engine**: vLLM V1 (Hermes tool-call parsing)
- **Attention**: FlashAttention-2

---

## Acknowledgements

- [veRL](https://github.com/volcengine/verl) — open-source RL training framework
- [τ-bench](https://github.com/sierra-research/tau-bench) — benchmark
- [Qwen](https://github.com/QwenLM/Qwen) — base models

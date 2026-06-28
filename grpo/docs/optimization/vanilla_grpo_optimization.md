# Vanilla GRPO 工程优化

> 在 veRL + vLLM V1 + τ-bench airline 上做多轮工具调用 GRPO 训练。
>
> **核心**：TP=2 双 GPU Policy + `bypass_mode` + `use_fused_kernels` + 显存参数调优，把峰值控制在 ~73GB，稳定跑完 500 步。

---

## 1. 硬件拓扑

| GPU | 用途 | 关键配置 |
|-----|------|---------|
| GPU 0 + GPU 1 | Policy（FSDP actor/ref + vLLM RolloutEngine） | `TP=2`, `CUDA_VISIBLE_DEVICES=0,1` |
| GPU 2 | 72B-AWQ User Simulator（OpenAI API, port 8001） | `max_num_seqs=8` |

> Policy 的 TP=2 跨 GPU0+1；user simulator 作为独立 vLLM 服务跑在另一张卡上，互不争抢。SFT 采集阶段是另一套拓扑（policy/user-sim 各一张、无 TP），见 [collect_sft_optimization.md](collect_sft_optimization.md)。

---

## 2. 核心工程优化

### 2.1 TP=2 双 GPU Policy
单卡 80GB 在长序列多轮场景下无法同时容纳 vLLM 池子 + FSDP all_gather + logits 峰值。`tensor_model_parallel_size=2` 把 Policy 分布到 GPU0+GPU1。

### 2.2 bypass_mode + calculate_log_probs
FSDP `compute_log_prob` 会创建 full logits `[micro_batch, seq_len, vocab]`，bf16 下 13.9GB，叠加 softmax 同尺寸张量峰值 ~27.8GB，单卡爆。处理方式：
1. vLLM 在 rollout 生成时计算 `log_probs`（采样本来就需要，零额外成本）：`rollout.calculate_log_probs: true`。
2. `algorithm.rollout_correction.bypass_mode: true` 直接用 rollout_log_probs 当 `old_log_probs`，跳过 FSDP actor 的 `compute_log_prob`。

actor `compute_log_prob` 的 ~20–40GB 峰值降为 0。GRPO 同一步内 rollout→update，`π_rollout` 与 `π_old` 差异极小，效果与标准 GRPO 基本一致。

### 2.3 use_fused_kernels
bypass 只跳过 actor 的 old_log_prob，**ref log_prob 和 actor update 仍需 forward**。`use_fused_kernels=true`（model + actor + ref 三处）调用 flash-attn 的 `cross_entropy_loss`，在 CUDA kernel 内直接算 log_prob/entropy，不创建完整 logits 张量，把 ref/actor update 的 logits 峰值从 ~27.8GB 压到 ~2GB。

> Qwen2.5 兼容：veRL 通过 `monkey_patch.py` 把 `forward_with_torch_backend` patch 到 `Qwen2ForCausalLM`。

### 2.4 训练步数
`train.parquet` 仅 40 行（seen tasks），`train_batch_size=4`，一个 epoch = 10 步。靠 `total_epochs=50` 达到 500 步（40×50/4）。GRPO 里 steps 和 epoch 先到阈值就停。

### 2.5 显存参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `gpu_memory_utilization` | 0.50 | vLLM 池子 40GB，PyTorch 可用 40GB |
| `max_num_seqs` | 12 | TP=2 下每卡分摊，KV cache 约 5–8GB |
| `ppo_micro_batch_size_per_gpu` | 2 | actor update 平衡速度与峰值 |
| `ref.log_prob_micro_batch_size_per_gpu` | 4 | fused kernel 下安全 |

---

## 3. Rollout Prompt

GRPO rollout 的 prompt 与 SFT 采集一致（见 [collect_sft_optimization.md §3](collect_sft_optimization.md)）：system = airline wiki + 日期锚定，tools = 14 个 airline 工具，user = τ-bench reset 的首条消息。由 `build_grpo_parquet.py` 把 `SYSTEM_PROMPT = WIKI + date` 写进 parquet 的 prompt 列。

---

## 4. 可选扩展模块（env 开关，默认关）

vanilla GRPO 之外，训练循环支持两个 opt-in 模块，互不依赖、可任意组合：

| 模块 | 开关 | 作用 |
|------|------|------|
| **B-NDSR** | `B_NDSR_ENABLED=true` | 预算化非退化采样 + 全失败 task 的 suffix replay。group 不再固定 n=8，改为 sequential（4→+2+2→8）+ 全 0 task 从 checkpoint 分叉续采 |
| **LLM-Judge** | `LLM_JUDGE_ENABLED=true` | 用 72B 给失败轨迹的关键 step 打分，把 binary reward 密化（outcome 锚定：成功=1.0 不 judge，失败=α·过程分）。反 hacking 监控 `llm_judge/outcome_rate` vs `mean_failure_process` |

两者都接在 ray_tracer 的 reward 计算前后，关闭时是 no-op、与 vanilla 行为完全一致。细节见 [b_ndsr.py](../../src/training/b_ndsr.py) / [llm_judge.py](../../src/training/llm_judge.py)。

---

## 5. veRL 适配

vendored 的 `verl/` 包含以下针对本项目的适配：

| 文件 | 内容 |
|------|------|
| `trainer/ppo/ray_trainer.py` | bypass fallback（batch 有 rollout_log_probs 时跳过 compute_log_prob）+ B-NDSR/judge 钩子 |
| `trainer/ppo/rollout_corr_helper.py` | `open_dict(policy_loss_config)` 绕过 OmegaConf struct 限制 |
| `workers/fsdp_workers.py` | `compute_ref_log_prob` 返回 `meta_info={"temperature":...}`（修复 LoRA 路径 temperature 丢失）|
| `experimental/agent_loop/tool_parser.py` | `_repair_json()` 修复截断 JSON |
| `experimental/agent_loop/tool_agent_loop.py` | τ-bench interaction 接入 + B-NDSR trace 追踪 |
| `utils/tracking.py` | SwanLab resume（`SWANLAB_RUN_ID` / `SWANLAB_RESUME`）|

---

## 6. 稳定配置

完整配置见 [`configs/train/grpo/vanilla_grpo.yaml`](../../configs/train/grpo/vanilla_grpo.yaml)。关键项：

```yaml
actor_rollout_ref:
  model: { enable_gradient_checkpointing: true, lora_rank: 16, lora_alpha: 32, use_fused_kernels: true }
  actor:
    fsdp_config: { optimizer_offload: true, param_offload: true, model_dtype: bf16 }
    optim: { lr: 5.0e-6 }
    use_kl_loss: true, kl_loss_coef: 0.01, kl_loss_type: low_var_kl
    ppo_mini_batch_size: 4, ppo_micro_batch_size_per_gpu: 2, use_fused_kernels: true
  rollout:
    mode: async, default_agent_loop: tool_agent
    multi_turn: { enable: true, format: hermes, max_user_turns: 15, max_assistant_turns: 15 }
    n: 8, temperature: 0.7, top_p: 0.9
    tensor_model_parallel_size: 2, calculate_log_probs: true
    gpu_memory_utilization: 0.50, max_num_seqs: 12, max_model_len: 24576
  ref: { fsdp_config: {optimizer_offload, param_offload, bf16}, use_fused_kernels: true, log_prob_micro_batch_size_per_gpu: 4 }

algorithm:
  adv_estimator: grpo
  rollout_correction: { bypass_mode: true }

trainer:
  total_epochs: 50, total_training_steps: 500, save_freq: 50
  project_name: grpo, experiment_name: vanilla_grpo_500step
  n_gpus_per_node: 2
```

---

## 7. 性能基准

### 7.1 显存（稳定步）
| 指标 | 数值 | 余量 |
|------|------|------|
| Peak Memory Allocated | ~73.2 GB | ~6.8 GB |
| Peak Memory Reserved | ~73.4 GB | ~6.6 GB |

峰值构成：vLLM 池子 40GB + FSDP all_gather 14GB + 激活值（seq≈20000）~12GB + fused kernel 临时 ~2GB + 碎片 ~5GB。

### 7.2 单步耗时
| 阶段 | 耗时 | 占比 |
|------|------|------|
| Rollout (gen) | ~720s | ~86% |
| Ref log_prob | ~27s | 3% |
| Actor update | ~93s | 11% |
| **总计** | ~840s | — |

Rollout 是最大瓶颈，受 72B user-sim 延迟和多轮对话（平均 27 turns）主导。500 步约 ~117 小时。

---

## 8. 快速启动

```bash
# 1. 72B User Simulator（GPU 2）
bash scripts/vllm_server/72b.sh

# 2. 训练（GPU 0+1）
export CUDA_VISIBLE_DEVICES=0,1
export VLLM_USE_V1=1
export HF_HUB_OFFLINE=1
bash scripts/train/grpo/run_vanilla.sh

# 可选扩展：
# B_NDSR_ENABLED=true        bash scripts/train/grpo/run_vanilla.sh
# LLM_JUDGE_ENABLED=true     bash scripts/train/grpo/run_vanilla.sh
```

---

## 9. Checklist

- `CUDA_VISIBLE_DEVICES=0,1`，`tensor_model_parallel_size: 2`
- `calculate_log_probs: true`，`bypass_mode: true`
- `use_fused_kernels: true`（model + actor + ref）
- `total_epochs: 50`，`gpu_memory_utilization: 0.50`，`max_num_seqs: 12`
- `log_prob_micro_batch_size_per_gpu: 4`（ref）/ `2`（actor）
- 72B User Simulator 正常（`curl http://localhost:8001/v1/models`）

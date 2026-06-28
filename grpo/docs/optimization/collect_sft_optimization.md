# SFT 轨迹采集：工程设计与关键决策

> 使用 **Qwen2.5-72B-Instruct-AWQ** 在 τ-bench airline 领域采集工具调用轨迹，作为 Qwen2.5-7B-Instruct LoRA 检查点（GRPO 的 warm-start）的监督微调数据。
>
> **约束**：单机 2×A800-80GB，既要避免 72B 量化模型在长上下文下的 CUDA 稳定性问题，又要保证轨迹质量与吞吐。

---

## 核心决策速览

| 决策项 | 取值 | 理由 |
|--------|------|------|
| 上下文长度上限 | **16K** | Native AWQ 在 >16K 时存在非法内存访问；`awq_marlin` 在 16K 以内稳定 |
| 量化格式 | **`awq_marlin`** | Native AWQ kernel 在长上下文下有非法内存访问问题 |
| Best-of-N 采样 | **16** | 补偿因截断过滤丢弃的长轨迹，保证每个 task 有足够成功样本 |
| 采样温度 | **分层 16 档**：`0.0/0.5/0.8/1.0` 各 4 条 | 兼顾稳定性（greedy）与多样性（高温探索）|
| 主动截断阈值 | **35,000 字符**（≈11–12K tokens） | 为生成预留余量；触发的轨迹标记为污染并丢弃 |
| 每步生成长度 | **512 tokens** | 覆盖常规 tool call，同时为输入留出 15.6K 空间 |
| 双卡部署 | Policy @ GPU0（4 seqs）+ UserSim @ GPU1（8 seqs） | 不对称上限匹配各自负载特征，消除单卡 seq-slot 争抢 |
| 训练/保留集划分 | **40 seen / 10 unseen** | 最大化训练数据量，同时保留泛化评估 holdout |

---

## 1. 背景

- **环境**：[τ-bench](https://github.com/sierra-research/tau-bench) airline（50 个任务、有状态工具、基于 LLM 的用户模拟器）。
- **模型**：Qwen2.5-72B-Instruct-AWQ（权重约 40 GB），同时充当 **Agent 策略（teacher）** 和 **τ-bench 用户模拟器**。
- **部署**：两张 A800 各跑一个 vLLM 实例，分别承担 policy 与 user simulator，避免单卡争抢。
- **目标**：采集成功轨迹，SFT 出 7B 的 GRPO 冷启动检查点。

---

## 2. 硬件与部署

### 2.1 环境
- **GPU**：2× NVIDIA A800-SXM4-80GB
- **Conda 环境**：`agentrl`（Python 3.10）
- **模型路径**：`../models/Qwen2.5-72B-Instruct-AWQ`

### 2.2 双卡分离部署

| 角色 | GPU | 端口 | `max_num_seqs` | 典型上下文 | 16K 下每序列 KV Cache |
|------|-----|------|----------------|-----------|----------------------|
| **Policy** | 0 | 8000 | 4 | 8K–14K | ~5 GB |
| **User Sim** | 1 | 8001 | 8 | 2K–4K | ~1.25 GB |

Policy 请求携带完整工具调用历史（8K–14K），4 条并发序列已占用约 20 GB KV Cache；user-sim 请求是短回复，8 条序列仍有余量。分离部署消除单卡 seq-slot 争抢，稳定吞吐。

### 2.3 启动命令

```bash
# Policy（GPU 0）— 上下文长，KV Cache 压力大
CUDA_DEVICES=0 PORT=8000 bash scripts/vllm_server/72b.sh

# User Sim（GPU 1）— 上下文短，KV Cache 压力小
CUDA_DEVICES=1 PORT=8001 MAX_NUM_SEQS=8 bash scripts/vllm_server/72b.sh
```

---

## 3. Prompt 设计

每条 trajectory 的初始消息由 `tau_bench_wrapper.run_single_task` 构造：

```
system: <airline 政策 wiki（env.wiki，~1.5K tokens）>
        + "# Date Context\n今天是 2024-05-15；无年份的日期按 2024 处理"
tools:  14 个 airline 工具（env.tools_info，OpenAI function 格式）
user:   τ-bench reset 返回的用户第一句话
```

**airline wiki** 是 agent 的政策知识库（book/modify/cancel/refund 规则、payment 限制、baggage 配额、basic-economy 不可改签等）。τ-bench 的工具 API **不会替 agent 拦截违规操作**，所以 wiki 必须进 prompt，agent 才能遵守政策——否则会做出 env 不报错但评测判失败的动作。

**日期锚定**：Qwen2.5 默认把无年份日期补成 2023，而 τ-bench airline 数据基于 2024；显式锚定 2024-05-15，避免 `search_direct_flight` 因年份错位返回空。

---

## 4. 量化与上下文稳定性

使用 `awq_marlin` 量化 + `max_model_len=16384` + `enable_chunked_prefill=False`。Native AWQ kernel 在 16K 以上存在非法内存访问，`awq_marlin` 的内存访问更安全；关闭 chunked prefill 降低 prefill 阶段的不确定性。

### 16K 下的显存占用

```raw
A800 80GB × 0.90 util = 72GB 可用
├── AWQ Marlin 权重:          ~40 GB
├── CUDA context/workspace:    ~3 GB
├── Prefix caching 索引:       ~1 GB
└── KV Cache block pool:       ~22 GB
    └── 16K × 80层 × 8 KV heads × 128 dim × 2 bytes ≈ 5 GB/seq
        4 seqs (policy)   = 20 GB
        8 seqs (user sim) = 峰值 40GB，平均远低于（user sim 上下文通常 2–4K）

nvidia-smi 实际观察: ~68–70 GB
```

---

## 5. 上下文截断策略

**截断是质量过滤器**：一旦某条 trajectory 触发截断阈值，整条标记为**污染**并丢弃，即使最终回合本可成功。

**三层防护：**

1. **采集时主动截断**：阈值 35,000 字符（≈11–12K tokens）。触发后保留 `system` + 前 2 条 + 最近交互，逐步丢弃旧轮次；`VLLMPolicy.was_truncated` 永久置位，`TrajectoryResult.was_contaminated_from_turn` 非 None 的轨迹进入 `*_contaminated.jsonl`，永不进 `train.jsonl`。
2. **数量兜底**：`best_of_n=16` + 分层温度（greedy→1.0 多档），短轨迹有充足采样机会补成功样本。
3. **训练对齐**：SFT `max_length=16384`，rollout 同样 16K 约束，避免 train/serve 分布偏移。

| 阈值 | 取值 | 用途 |
|------|------|------|
| vLLM `max_model_len` | 16K | 硬约束，保证 CUDA 稳定 |
| 主动截断触发线 | 35,000 chars（≈11–12K tokens） | 留 4–5K 给生成 + 安全余量 |
| 进 `train.jsonl` 的最大长度 | ~12–13K tokens | `raw_messages` tokenize 后的上限 |

---

## 6. 并发：`num_workers` 与 `max_num_seqs`

```raw
num_workers  = 同时跑的 task 数（ThreadPoolExecutor 并发度）
max_num_seqs = vLLM 单卡同时处理的 request 上限

约束: max_num_seqs(policy)   ≥ num_workers
      max_num_seqs(user_sim) ≥ num_workers（通常有余量）
```

一个 worker 的交互链：`policy req → wait → user_sim req → wait → ...`，任意时刻只发 1 个 request，但 4 个 workers 的 peak 并发 = 4。

```yaml
num_workers: 4
# policy 端:   max_num_seqs=4（匹配）
# user_sim 端: max_num_seqs=8（有余量）
```

---

## 7. 数据质量

### 7.1 过滤规则
1. **Context 超限**：`[CONTEXT_LENGTH_EXCEEDED]` → 立即 abort trajectory。
2. **截断污染**：`was_contaminated_from_turn != None` → 整条作废，不进 train。
3. **API 错误**：`[POLICY_API_ERROR]` → 返回 fake assistant message，允许 trajectory 继续（收集诊断信息）。
4. **Success 筛选**：只有 `success=True` 且未被污染的轨迹进 `train.jsonl`。

### 7.2 输出文件
```raw
experiments/sft_collect_airline/
├── task_0000.jsonl                 # 干净 success trajectory
├── task_0000_contaminated.jsonl    # 截断污染轨迹（诊断用，不进 train）
├── task_0000.meta.json             # 该 task 统计（success/污染数）
├── train.jsonl                     # 合并后的 SFT 训练数据（seen task）
├── holdout_train.jsonl             # unseen task 数据（备用）
├── split.json                      # seen/unseen task 切分（40/10）
├── summary.json                    # 全局统计
└── collect_config.yaml             # 采集参数快照
```

### 7.3 Checkpoint Resume
- 按 task 粒度：`task_XXXX.meta.json` 存在则跳过。
- 重跑前删除空文件：`rm task_*.jsonl task_*.meta.json task_*_contaminated.jsonl`。

---

## 8. 采集参数（`configs/train/sft/sft_collect_airline.yaml`）

```yaml
env:
  name: airline
  user_model: "Qwen/Qwen2.5-72B-Instruct-AWQ"
  user_base_url: "http://localhost:8001/v1"
  task_split: test

policy:
  model_name: "Qwen/Qwen2.5-72B-Instruct-AWQ"
  base_url: "http://localhost:8000/v1"
  top_p: 0.9
  max_tokens: 512

collect:
  best_of_n: 16
  temperatures: [0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.8, 0.8, 0.8, 0.8, 1.0, 1.0, 1.0, 1.0]
  max_turns: 30
  num_workers: 4
  holdout_size: 10
```

---

## 9. 操作 SOP

### 9.1 启动 vLLM（双卡）
```bash
cd grpo
CUDA_DEVICES=0 PORT=8000 bash scripts/vllm_server/72b.sh           # policy
CUDA_DEVICES=1 PORT=8001 MAX_NUM_SEQS=8 bash scripts/vllm_server/72b.sh  # user sim
```

### 9.2 Tiny 验证
```bash
python scripts/train/sft/collect_sft_data.py \
  --config configs/train/sft/sft_collect_airline.yaml --tiny
```
验证：vLLM 16K 稳定（无 CUDA crash）、`task_0000.meta.json` 的 `num_contaminated=0`、`train.jsonl` 只含干净数据。

### 9.3 全量采集
```bash
python scripts/train/sft/collect_sft_data.py \
  --config configs/train/sft/sft_collect_airline.yaml
```

### 9.4 采后检查
1. **长度分布**：`train.jsonl` 中 trajectory token 数应在 8K–13K，无 14K+ 长尾。
2. **Task 覆盖率**：0 条成功的 task 是 GRPO 的诊断素材。
3. **抽样人工检查**：随机抽 3–5 条，看截断边界附近行为是否异常。

---

## 10. 采集结果

采集指标记录在 `experiments/sft_collect_airline/summary.json`（成功 task 数、成功轨迹数、污染轨迹数、耗时等），按每次实际采集更新。重点观察：
- **成功率**：成功 task / 50；带 wiki 后政策密集型 task（certificate 支付、basic-economy 改签、取消规则）成功率应明显改善。
- **污染率**：触截断的轨迹占比，预期极低（<1%）。
- **耗时**：50 task × best-of-16 = 800 rollout，单次约 7 小时量级。

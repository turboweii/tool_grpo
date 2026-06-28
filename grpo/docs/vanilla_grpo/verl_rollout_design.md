# veRL Multi-Turn Rollout Pipeline 设计

> τ-bench airline 的多轮工具调用 rollout 如何接入 veRL 的 async agent loop，并在 GRPO 训练里驱动采样与 reward。
>
> 相关代码：[tau_bench_context.py](../../src/envs/tau_bench_context.py)、[tau_bench_interaction.py](../../src/envs/tau_bench_interaction.py)、[tau_bench_tools.py](../../src/envs/tau_bench_tools.py)。

---

## 1. 目标与约束

- **任务**：τ-bench airline（50 个有状态任务、14 个工具、LLM 用户模拟器），多轮对话 + 工具调用。
- **训练**：vanilla GRPO，reward = τ-bench 任务级二元 outcome（成功 1 / 失败 0），不引入 shaping。
- **模型**：policy = Qwen2.5-7B-Instruct（SFT LoRA merged），user simulator = Qwen2.5-72B-Instruct-AWQ。
- **框架**：veRL（复用官方 `ToolAgentLoop`，不自写 agent loop）+ vLLM V1。

τ-bench 的 env 是**有状态**的：`env.reset(task_index)` 返回用户首条消息，`env.step(Action)` 处理 tool 调用或 `RESPOND_ACTION`（驱动 user sim）；env 状态（订单 DB、对话历史）绑定在实例内，跨 task 不能复用。核心工程问题：**在 veRL 的并发 async rollout 下，让每条 trajectory 的 Tool 和 Interaction 共享同一个 env 实例，且不同 trajectory 互不串扰。**

---

## 2. τ-bench → veRL 映射：contextvar 隔离

veRL 提供三个扩展点：

| 抽象 | 职责 |
|---|---|
| `ToolAgentLoop`（官方） | 一条 trajectory 的完整状态机循环（pending → generating → processing_tools → interacting）|
| `BaseTool` | 单个工具的 schema + 执行 |
| `BaseInteraction` | 非工具交互（用户回复）|

映射方式：**Interaction 持有 env 实例，Tool 通过 `contextvars.ContextVar` 读当前 trajectory 的 env。**

为什么 contextvar 成立：
1. veRL 为**每条 trajectory 单独 `asyncio.create_task`**（[agent_loop.py](verl/verl/experimental/agent_loop/agent_loop.py)），`create_task` 时 context 被 **fork**，每个 task 有自己的 contextvar 副本。
2. 一条 trajectory 的整个生命周期（start_interaction → tool.execute → generate_response）都在**同一个 task** 内顺序执行，`start_interaction` 里 `set` 的 env 对后续 `tool.execute` 一直可见。
3. 同一 task 的 group_size 条 rollout 各自一个 task → env **天然隔离**，无需显式锁或 registry。

**唯一的脆性点**：contextvar 不跨线程。`tool.execute` / `env.step` 必须留在 asyncio task 里，不能挪进 `run_in_executor`（线程池不继承 contextvar）。veRL 的 `run_in_executor` 只用于 tokenizer/processor 这类不碰 env 的 CPU 操作，所以当前是安全的。

---

## 3. 组件

### 3.1 context 注册表 — [tau_bench_context.py](../../src/envs/tau_bench_context.py)
两个模块级 `ContextVar`：
- `CURRENT_TAU_ENV` —— 当前 trajectory 的 τ-bench env 实例。
- `CURRENT_TAU_STATE` —— 当前 trajectory 的累计状态（`task_id` / `total_reward` / `num_tool_calls` / `num_user_turns` / `done` / `contaminated`）。

`make_initial_state(task_id)` 给出 state 初值。Interaction 在 `start_interaction` 里 `set`，Tool 在 `execute` 里 `get`。

### 3.2 Interaction — [tau_bench_interaction.py](../../src/envs/tau_bench_interaction.py)
`TauBenchInteraction(BaseInteraction)`，持有 env，驱动 user simulator：

| 回调 | 做什么 |
|---|---|
| `start_interaction(instance_id, task_id)` | `get_env()` + `env.reset(task_index)`，`set` contextvar，存进 `_instance_dict`。支持 `b_ndsr_replay_actions`：先把 checkpoint 前缀动作重放进 fresh env（B-NDSR suffix replay 用）。|
| `generate_response(instance_id, messages)` | assistant 输出无 tool_calls 时触发：取最新 assistant content → **污染检测**（含 forbidden template token 则 reward=0 终止）→ `env.step(RESPOND_ACTION)` 驱动 user sim → 累计 reward/turn；`done` 或超 `max_turns` 时返回最终 binary reward。|
| `calculate_score(instance_id)` | 返回 binary outcome（`total_reward >= 1.0 → 1.0 else 0.0`），供 GRPO advantage 使用。|
| `finalize_interaction(instance_id)` | 清 `_instance_dict`（contextvar 随 asyncio task 结束自动释放）。|

**污染检测**：assistant 输出含 `</tool_response>` / `<tool_response>`（长上下文 format drift 的信号）时，锁定 reward=0 并 terminate，避免带毒轨迹进训练。

### 3.3 Tools — [tau_bench_tools.py](../../src/envs/tau_bench_tools.py)
14 个 airline tool 各自一个静态子类（`TauBench_<name>_Tool`，继承 `TauBenchToolBase`），pickle-safe（Ray/FSDP 跨进程安全）。`execute` 从 contextvar 读 env → `env.step(Action(self.name, parameters))` → 把 observation 回传给模型，累计 `num_tool_calls`；step reward 恒为 0（reward 只走 Interaction 的 outcome）。env/state 缺失时 **fail-loud**（抛 RuntimeError，宁可整 batch 崩也不让带毒轨迹进 GRPO）。

`verify_tool_classes_match_env()` 启动时校验静态类名与 `env.tools_info` 完全对齐，不一致立刻报错。

### 3.4 ToolAgentLoop（veRL 官方）
驱动状态机；额外追踪 B-NDSR 所需的 flags（json_error / invalid_tool / read_count / write_executed / repeated_loop 等）和 checkpoints（write 之前、read 之后），挂到 `output.extra_fields['b_ndsr_trace']`。

---

## 4. Prompt

初始消息（与 SFT 采集一致，见 [collect_sft_optimization.md §3](../optimization/collect_sft_optimization.md)）：

```
system: <airline 政策 wiki（env.wiki，~1.5K tokens）>
        + "# Date Context\n今天是 2024-05-15；无年份日期按 2024 处理"
tools:  14 个 airline 工具（OpenAI function schema）
user:   τ-bench reset 返回的用户第一句话
```

由 [build_grpo_parquet.py](../../scripts/train/grpo/build_grpo_parquet.py) 把 `SYSTEM_PROMPT = WIKI + date` 写进 parquet 的 `prompt` 列。wiki 是 agent 的政策知识库——τ-bench 的工具 API 不替 agent 拦截违规操作（改 basic-economy、无保险取消等），所以 wiki 必须进 prompt，agent 才能遵守政策。

---

## 5. Reward 与可选扩展

**基础**：`calculate_score` 返回 binary outcome，veRL 据此算 GRPO group-relative advantage。reward 严格等于 τ-bench outcome，无 shaping。

**两个 opt-in 扩展**（env 开关，默认关，互不依赖）：

| 模块 | 开关 | 在 rollout 里的角色 |
|---|---|---|
| **B-NDSR** | `B_NDSR_ENABLED` | 改变采样：group 不固定 n=8，改 sequential（4→+2+2→8）；全失败 task 从 best-failed-prefix 的 checkpoint（`before_first_write`）分叉 suffix 续采。`b_ndsr_replay_actions` 由 Interaction 重放进 fresh env。|
| **LLM-Judge** | `LLM_JUDGE_ENABLED` | 在 B-NDSR 分组之后、advantage 之前，用 72B 给失败轨迹的关键 step 打分，把 binary reward 密化（outcome 锚定：成功=1.0 不 judge，失败=α·过程分）。监控 `llm_judge/outcome_rate` vs `mean_failure_process` 防 hacking。|

两者关闭时是 no-op，行为与 vanilla GRPO 完全一致。详见 [b_ndsr.py](../../src/training/b_ndsr.py) / [llm_judge.py](../../src/training/llm_judge.py)。

---

## 6. 数据格式（parquet）

每行一个 task，`rollout.n` 由 veRL 运行时展开成 group。schema：

| 列 | 内容 |
|---|---|
| `prompt` | `[{"role":"system","content": SYSTEM_PROMPT}]`（wiki + date）|
| `extra_info` | `{index, task_id, split:"seen"|"unseen", interaction_kwargs:{name, task_id}}` |
| `data_source` / `ability` | `"tau_bench_airline"` |

`extra_info.task_id` 决定 rollout 跑哪个 τ-bench 任务；`split` 用于 seen/unseen 分组评测。

---

## 7. 配置

- **工具 schema**：[configs/tool_config/tau_bench_airline_tools.yaml](../../configs/tool_config/tau_bench_airline_tools.yaml) —— 由 [gen_tool_config.py](../../scripts/train/grpo/gen_tool_config.py) 从 `env.tools_info` 动态生成，14 个 `class_name` 指向静态 tool 子类。
- **Interaction**：[configs/interaction_config/tau_bench_airline.yaml](../../configs/interaction_config/tau_bench_airline.yaml) —— `TauBenchInteraction` + env_name/user_model/user_base_url/task_split/max_turns。
- **训练**：[configs/train/grpo/vanilla_grpo.yaml](../../configs/train/grpo/vanilla_grpo.yaml)（正式）/ [mock_grpo.yaml](../../configs/train/mock/mock_grpo.yaml)（5-step 冒烟）。关键项：`rollout.mode=async`、`default_agent_loop=tool_agent`、`multi_turn.format=hermes`、`adv_estimator=grpo`、`rollout_correction.bypass_mode=true`。工程优化（TP=2 / fused kernels / 显存参数）见 [vanilla_grpo_optimization.md](../optimization/vanilla_grpo_optimization.md)。

---

## 8. 已知约束与注意事项

- **vLLM 长上下文稳定性**：72B-AWQ 用 `awq_marlin` + `max_model_len≤16K`；native AWQ 在 >16K 有非法内存访问。
- **contextvar 不跨线程**：别把 `tool.execute` / `env.step` 挪进 `run_in_executor`（veRL 当前只在 `run_in_executor` 里做 tokenizer/processor，不碰 env，安全）。
- **rollout 是吞吐瓶颈**：72B user-sim 每轮被调用，多轮对话（平均 ~27 turn）主导单步耗时；B-NDSR oversample / judge 会进一步加大 user-sim 负载（judge 已做并发 + 缓存 + near-miss-only 优化）。
- **可复现**：相同 seed + checkpoint 下 rollout 一致；污染轨迹 reward=0 并 terminate。

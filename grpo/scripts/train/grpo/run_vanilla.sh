#!/bin/bash
set -e

source /opt/conda/etc/profile.d/conda.sh
conda activate agentrl

# 注意: PyTorch 是 2.7.0+cu126(自带 CUDA 12.6 runtime),但系统 CUDA 工具链是 12.4
# CUDA_HOME 必须指向实际存在的系统目录,供 triton/deepspeed 找 ptxas
export CUDA_HOME=/usr/local/cuda-12.4
export TRITON_PTXAS_PATH=/usr/local/cuda-12.4/bin/ptxas
export CUDA_VISIBLE_DEVICES=0,1
export DS_SKIP_TRITON=1
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export VLLM_USE_V1=1
export HF_HUB_OFFLINE=1
export OPENAI_API_KEY=dummy
export LITELLM_LOCAL_MODEL_COST_MAP="True"
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export VLLM_LOGGING_LEVEL=ERROR
export SWANLAB_RUN_ID="6ewzzw71dojehfhws9a85"
export SWANLAB_RESUME="true"

# expandable_segments disabled: incompatible with vLLM memory pool
# export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

export B_NDSR_ENABLED="${B_NDSR_ENABLED:-false}"
export B_NDSR_ROOT_TEMPERATURES="${B_NDSR_ROOT_TEMPERATURES:-0.45,0.65,0.80,0.95,0.55,1.00,0.70,1.05}"
export B_NDSR_ROOT_TOP_PS="${B_NDSR_ROOT_TOP_PS:-0.85,0.90,0.92,0.95,0.88,0.96,0.90,0.97}"
export JASS_ENABLED="${JASS_ENABLED:-false}"
export JASS_JUDGE_MODEL="${JASS_JUDGE_MODEL:-Qwen/Qwen2.5-72B-Instruct-AWQ}"
export JASS_JUDGE_BASE_URL="${JASS_JUDGE_BASE_URL:-http://localhost:8001/v1}"
B_NDSR_EXTRA_ARGS=()
case "${B_NDSR_ENABLED,,}" in
    1|true|yes|y|on)
        B_NDSR_EXTRA_ARGS+=(actor_rollout_ref.rollout.n=1)
        B_NDSR_EXTRA_ARGS+=(actor_rollout_ref.actor.ppo_mini_batch_size=2)
        B_NDSR_EXTRA_ARGS+=(actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1)
        B_NDSR_EXTRA_ARGS+=(actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1)
        B_NDSR_EXTRA_ARGS+=(trainer.experiment_name=vanilla_grpo_b_ndsr)
        ;;
esac

cd /workspace/grpo
mkdir -p experiments/vanilla

# nohup python -m verl.trainer.main_ppo \
#     --config-path=$(pwd)/configs \
#     --config-name=vanilla_grpo \
#     > experiments/vanilla/training.log 2>&1 &
# echo "Training PID: $!"

python -m verl.trainer.main_ppo \
    --config-path=$(pwd)/configs \
    --config-name=vanilla_grpo \
    "${B_NDSR_EXTRA_ARGS[@]}"

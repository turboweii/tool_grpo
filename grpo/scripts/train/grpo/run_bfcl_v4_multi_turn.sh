#!/bin/bash
set -e

source /opt/conda/etc/profile.d/conda.sh
conda activate agentrl

export CUDA_HOME=/usr/local/cuda-12.4
export TRITON_PTXAS_PATH=/usr/local/cuda-12.4/bin/ptxas
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export DS_SKIP_TRITON=1
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export VLLM_USE_V1=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
export LITELLM_LOCAL_MODEL_COST_MAP="True"
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export VLLM_LOGGING_LEVEL=ERROR

export B_NDSR_ENABLED="${B_NDSR_ENABLED:-false}"
export B_NDSR_ROOT_TEMPERATURES="${B_NDSR_ROOT_TEMPERATURES:-0.45,0.65,0.80,0.95,0.55,1.00,0.70,1.05}"
export B_NDSR_ROOT_TOP_PS="${B_NDSR_ROOT_TOP_PS:-0.85,0.90,0.92,0.95,0.88,0.96,0.90,0.97}"
# BFCL functions are task-specific and not airline read/write names. For the
# first BFCL B-NDSR version, any successful function call can be an after-read
# replay checkpoint; before-write remains disabled unless explicitly set.
export B_NDSR_READ_TOOLS="${B_NDSR_READ_TOOLS:-*}"
export B_NDSR_WRITE_TOOLS="${B_NDSR_WRITE_TOOLS:-}"

export JASS_ENABLED="${JASS_ENABLED:-false}"
export JASS_JUDGE_MODEL="${JASS_JUDGE_MODEL:-Qwen/Qwen2.5-72B-Instruct-AWQ}"
export JASS_JUDGE_BASE_URL="${JASS_JUDGE_BASE_URL:-http://localhost:8001/v1}"
export LLM_JUDGE_ENABLED="${LLM_JUDGE_ENABLED:-false}"
export LLM_JUDGE_MODEL="${LLM_JUDGE_MODEL:-Qwen/Qwen2.5-72B-Instruct-AWQ}"
export LLM_JUDGE_BASE_URL="${LLM_JUDGE_BASE_URL:-http://localhost:8001/v1}"
export LLM_JUDGE_ALPHA="${LLM_JUDGE_ALPHA:-0.2}"
export LLM_JUDGE_N_SAMPLES="${LLM_JUDGE_N_SAMPLES:-3}"
export LLM_JUDGE_DOMAIN="${LLM_JUDGE_DOMAIN:-bfcl}"

B_NDSR_EXTRA_ARGS=()
case "${B_NDSR_ENABLED,,}" in
    1|true|yes|y|on)
        B_NDSR_EXTRA_ARGS+=(actor_rollout_ref.rollout.n=1)
        B_NDSR_EXTRA_ARGS+=(actor_rollout_ref.actor.ppo_mini_batch_size=2)
        B_NDSR_EXTRA_ARGS+=(actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1)
        B_NDSR_EXTRA_ARGS+=(actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1)
        B_NDSR_EXTRA_ARGS+=(trainer.experiment_name=bfcl_v4_multi_turn_b_ndsr)
        ;;
esac

cd /workspace/grpo
mkdir -p experiments/bfcl_v4_multi_turn

python -m verl.trainer.main_ppo \
    --config-path=$(pwd)/configs/train/grpo \
    --config-name=bfcl_v4_multi_turn_grpo \
    "${B_NDSR_EXTRA_ARGS[@]}"

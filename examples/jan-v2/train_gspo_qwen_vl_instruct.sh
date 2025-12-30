#!/usr/bin/env bash
set -e # Exit immediately if a command exits with a non-zero status.
set -u # Treat unset variables as an error.
set -o pipefail # Return the exit status of the last command in the pipe that failed.

# ========================================================================================
# R-HORIZON: Distributed Reinforcement Learning Training Script
#
# This script launches a distributed training job using Ray and the GRPO algorithm.
# It is designed to be run in a multi-GPU, multi-node environment.
#
# Usage:
#   - For single-node training: bash scripts/train_rl.sh
#   - For multi-node training: Ensure the Ray cluster is initialized, then run this
#     script on each node.
#   - Customize parameters by passing them as arguments, e.g.,
#     bash scripts/train_rl.sh --model_path "/path/to/your/model" --output_dir "/path/to/save"
# ========================================================================================

# ---
# ⚙️ Default Configuration
# ---
# These can be overridden by command-line arguments.
MODEL_PATH="/mnt/nas/alex/models/Qwen/Qwen3-VL-30B-A3B-Instruct" # Path to your base model
# MODEL_PATH="/mnt/nas/alex/models/Qwen/Qwen3-30B-A3B-Instruct-2507" # Path to your base model
PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/vllm_multiturn/config"


TRAIN_DATA_DIR="/mnt/nas/bachvd/Code-Agent/verl/data/janv2_searchqa/train.parquet"
EVAL_DATA_DIR="/mnt/nas/data/r-horizon-training-data"

TOOL_CONFIG="$CONFIG_PATH/tool_config/search_tool_config.yaml"
OUTPUT_DIR="./checkpoints/qwen3-vl-30b-a3b-instruct-tool-env-gspo-run-2" # Directory to save checkpoints and logs

WORLD_SIZE=1  # Number of nodes
GPUS_PER_NODE=8 # Number of GPUs per node
MASTER_PORT=29500

# ---
# ↗️ Command-line Argument Parsing
# ---
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model_path) MODEL_PATH="$2"; shift ;;
        --train_data_dir) TRAIN_DATA_DIR="$2"; shift ;;
        --eval_data_dir) EVAL_DATA_DIR="$2"; shift ;;
        --output_dir) OUTPUT_DIR="$2"; shift ;;
        --world_size) WORLD_SIZE="$2"; shift ;;
        --gpus_per_node) GPUS_PER_NODE="$2"; shift ;;
        --master_port) MASTER_PORT="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# ---
# 🛠️ Environment Setup
# ---
# Set PyTorch and NCCL environment variables for performance and debugging.
export WANDB_API_KEY=
export TORCH_CPP_LOG_LEVEL="INFO"
export TORCH_DISTRIBUTED_DEBUG="INFO"
export NCCL_DEBUG="WARN"
export NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:512'
export TOKENIZERS_PARALLELISM="false"
export OMP_NUM_THREADS=4
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NVLS_ENABLE=0
# export VLLM_ATTENTION_BACKEND="XFORMERS"
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

export NCCL_P2P_DISABLE=0
# export HF_HUB_ENABLE_HF_TRANSFER=1
export NCCL_SHM_DISABLE=0
export NCCL_DEBUG=INFO
export NCCL_IGNORE_DISABLED_P2P=1
export VLLM_USE_V1=1
# export ROLLOUT_QUANTIZATION=fp8
export VLLM_USE_DEEP_GEMM=0
export RAY_memory_usage_threshold=0.99
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=36000
export SGLANG_ENABLE_JIT_DEEPGEMM=0

# ---
# 🚀 Training Hyperparameters
# ---
# Rollout and PPO settings
ROLLOUT_BATCH_SIZE=64
PPO_MINI_BATCH=64
micro_batch_size_per_gpu=32
micro_batch_logprob_size_per_gpu=8
MAX_PROMPT_LENGTH=2048
RESPONSE_LENGTH=8192 # Renamed from RES_LENGTH for clarity
GROUP_SIZE=5
N_VAL_SAMPLES=8
TRAIN_TEMPERATURE=0.7
WANDB_MODE='online'

clip_ratio_low=3e-4
clip_ratio_high=4e-4

critic_lr=1e-6
gae_gamma=1.0
gae_lam=0.95
critic_warmup=0

enable_overlong_buffer=True
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=1.0

# Tensor/Sequence Parallelism (for very large models)
TP=1 # Tensor Parallelism
SP=4 # Sequence Parallelism
MAX_TOKEN_LEN=$(((64000 + MAX_PROMPT_LENGTH + 100) / SP))

# ---
# 📊 Dataset Configuration
# ---
# Assumes data files are in the specified directories.
# Modify the file names if your dataset structure is different.

train_files="[\"/mnt/nas/bachvd/Code-Agent/verl/data/janv2_searchr1/train.parquet\"]"
test_files="[\"$EVAL_DATA_DIR/aime24.parquet\",\"$EVAL_DATA_DIR/aime25.parquet\"]"

# ---
#  wandb Configuration (optional)
# ---
# Set to "online" to enable Weights & Biases logging.
# Ensure you have run `wandb login` first.
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${OUTPUT_DIR}/wandb"
export MAX_JOBS=4
# ---

# Create directories
mkdir -p "$OUTPUT_DIR"
STATS_DIR="${OUTPUT_DIR}/stats"
mkdir -p "$STATS_DIR"

# Project and Experiment Naming
PROJECT_NAME="Qwen3-VL-30B-A3B-Instruct-tool-gspo"
EXP_NAME="gspo-vision-mix-text-$(basename ${MODEL_PATH})-$(date +%Y%m%d-%H%M%S)"

echo "🚀 Submitting Ray job..."
echo "  - Model: ${MODEL_PATH}"
echo "  - Output Dir: ${OUTPUT_DIR}"
echo "  - Train Data: ${train_files}"
echo "  - Eval Data: ${test_files}"

# Submit the training job to the Ray cluster.
# The entry point is assumed to be `verl.trainer.main_ppo`.
# The configuration is passed using Hydra-style overrides.
# "/mnt/nas/bachvd/Code-Agent/verl/data/r-horizon-train/train.parquet" 
# "/mnt/nas/bachvd/Code-Agent/verl/data/r-horizon-vision-mixed-multi-label/train-k2.parquet"
PYTHONUNBUFFERED=1 python -m verl.trainer.main_ppo \
    reward_model.reward_manager=dapo \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward_model.reward_kwargs.max_resp_len=64048 \
    algorithm.adv_estimator=grpo \
    +algorithm.use_reward_clip=True \
    algorithm.use_kl_in_reward=False \
    algorithm.rollout_correction.rollout_is=token \
    algorithm.rollout_correction.rollout_is_threshold=2.0 \
    algorithm.rollout_correction.rollout_is_batch_normalize=False \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.gamma=$gae_gamma \
    algorithm.lam=$gae_lam \
    actor_rollout_ref.actor.policy_loss.loss_mode=gspo \
    critic.optim.lr=$critic_lr \
    critic.model.path=$MODEL_PATH \
    critic.model.use_remove_padding=True \
    critic.ppo_max_token_len_per_gpu=$MAX_TOKEN_LEN \
    critic.ulysses_sequence_parallel_size=$SP \
    data.train_files=$train_files \
    data.val_files=$test_files \
    data.train_batch_size=$ROLLOUT_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$RESPONSE_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    data.filter_overlong_prompts_workers=128 \
    actor_rollout_ref.nccl_timeout=36000 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_shm=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.actor.clip_ratio_low=$clip_ratio_low \
    actor_rollout_ref.actor.clip_ratio_high=$clip_ratio_high \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.clip_grad=0.1 \
    actor_rollout_ref.actor.freeze_vision_tower=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$MAX_TOKEN_LEN \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$SP \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$micro_batch_size_per_gpu \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.actor.grad_clip=0.1 \
    actor_rollout_ref.actor.loss_agg_mode='token-mean' \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TP \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
    actor_rollout_ref.rollout.temperature=$TRAIN_TEMPERATURE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.top_p=0.8 \
    +actor_rollout_ref.rollout.repetition_penalty=1.0 \
    +actor_rollout_ref.rollout.presence_penalty=0.0 \
    actor_rollout_ref.rollout.top_k=20 \
    actor_rollout_ref.rollout.max_num_seqs=512 \
    actor_rollout_ref.rollout.max_model_len=66048 \
    actor_rollout_ref.rollout.n=$GROUP_SIZE \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$micro_batch_logprob_size_per_gpu \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    trainer.use_legacy_worker_impl=auto \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXP_NAME \
    trainer.test_freq=10000 \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=$GPUS_PER_NODE \
    trainer.nnodes=$WORLD_SIZE \
    trainer.save_freq=5 \
    trainer.logger=['console','wandb'] \
    trainer.default_local_dir=$OUTPUT_DIR \
    trainer.total_epochs=10 \
    trainer.critic_warmup=$critic_warmup \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG" \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=50 \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=20 \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=32768 
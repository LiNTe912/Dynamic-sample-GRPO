#!/bin/bash
#export VLLM_ATTENTION_BACKEND=XFORMERS
unset VLLM_ATTENTION_BACKEND
export VLLM_USE_V1=1

# ------------------------------------------------------------

DATE=$(date +%m%d)
TIME_TAG=$(date +%H%M%S)

TASK="DS-GRPO"
BACKBONE="Qwen3-4B"
ADVANTAGE="grpo"

MAX_PROMPT_LENGTH=512
MAX_RESPONSE_LENGTH=2048

EPISODE=80
DATA_TRAIN_BATCH_SIZE=1
N_SAMPLES_PER_PROMPT=8
MINI_BATCH_SIZE=1
MICRO_BATCH_SIZE=2

DATA_LOCAL_DIR="data/nq_open"
BACKBONE_PATH="/dev_data/wy/zwt/qwen3-4b"
REWARD_FUNCTION_PATH="/dev_data/wy/zwt/verl/verl/utils/reward_score/ds.py"

MODEL="${TASK}-${BACKBONE}"
DATASET="nq-open"
EXPERIMENT="DS-GRPO"

WANDB_PROJECT="DS-verl"
LOG_NAME="${DATE}-${EXPERIMENT}-${MODEL}-${ADVANTAGE}"
OUTPUT_DIR="checkpoints/${MODEL}_${DATASET}"
COUNT_FILE="count/${MODEL}_${DATASET}_count.json"

# ------------------------------------------------------------
python -m verl.trainer.main_ppo \
  reward_model.reward_manager=batch \
  +reward_model.reward_kwargs.count_file=$COUNT_FILE \
  custom_reward_function.path=$REWARD_FUNCTION_PATH \
  data.train_files=$DATA_LOCAL_DIR/train.parquet \
  data.val_files=$DATA_LOCAL_DIR/test.parquet \
  data.max_prompt_length=$MAX_PROMPT_LENGTH \
  data.max_response_length=$MAX_RESPONSE_LENGTH \
  data.train_batch_size=$DATA_TRAIN_BATCH_SIZE \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  actor_rollout_ref.model.path=$BACKBONE_PATH \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
  actor_rollout_ref.model.lora_rank=64 \
  actor_rollout_ref.model.lora_alpha=32 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.optim.lr=5e-7 \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.03 \
  actor_rollout_ref.actor.optim.warmup_style='cosine' \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.temperature=0.6 \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.free_cache_engine=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.n=$N_SAMPLES_PER_PROMPT \
  actor_rollout_ref.rollout.max_model_len=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
  actor_rollout_ref.rollout.max_num_batched_tokens=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
  critic.optim.lr=9e-6 \
  critic.model.use_remove_padding=True \
  critic.model.path=$BACKBONE_PATH \
  critic.model.enable_gradient_checkpointing=True \
  critic.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
  critic.model.fsdp_config.param_offload=False \
  critic.model.fsdp_config.optimizer_offload=False \
  algorithm.kl_ctrl.kl_coef=0.00 \
  algorithm.adv_estimator=$ADVANTAGE \
  trainer.logger=['console'] \
  trainer.val_before_train=False \
  trainer.project_name=$WANDB_PROJECT \
  trainer.experiment_name=$LOG_NAME \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.test_freq=1000 \
  trainer.save_freq=500 \
  trainer.resume_mode=auto \
  trainer.default_local_dir=$OUTPUT_DIR \
  trainer.total_epochs=3

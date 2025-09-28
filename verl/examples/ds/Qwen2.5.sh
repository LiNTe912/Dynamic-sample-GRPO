#!/bin/bash
#export VLLM_ATTENTION_BACKEND=XFORMERS
unset VLLM_ATTENTION_BACKEND
export VLLM_USE_V1=1

# ------------------------------------------------------------

DATE=$(date +%m%d)
TIME_TAG=$(date +%H%M%S)

TASK="DS-GRPO"
BACKBONE="qwen2.5-14b"
ADVANTAGE="grpo"

MAX_PROMPT_LENGTH=512
MAX_RESPONSE_LENGTH=1024

EPISODE=80
DATA_TRAIN_BATCH_SIZE=32
N_SAMPLES_PER_PROMPT=8
MINI_BATCH_SIZE=16
MICRO_BATCH_SIZE=8

DATA_LOCAL_DIR="data/webquestions"
BACKBONE_PATH="/data2/wentao/qwen2.5-14b". # path to your model

MODEL="${TASK}-${BACKBONE}"
DATASET="webquestions"
EXPERIMENT="DS-GRPO"

WANDB_PROJECT="DS-verl"
LOG_NAME="${DATE}-${EXPERIMENT}-${MODEL}-${ADVANTAGE}"
OUTPUT_DIR="/data2/zwt/verl/checkpoints/${MODEL}_${DATASET}" # path to save the output
COUNT_FILE="count/${MODEL}_${DATASET}_count.json"

# ------------------------------------------------------------
python -m verl.trainer.main_ppo \
  reward_model.reward_manager=batch \
  +reward_model.reward_kwargs.count_file=$COUNT_FILE \
  +reward_model.reward_kwargs.batch_size=$DATA_TRAIN_BATCH_SIZE \
  +reward_model.reward_kwargs.n_samples=$N_SAMPLES_PER_PROMPT \
  custom_reward_function.path=verl/utils/reward_score/ds.py \
  data.train_files=$DATA_LOCAL_DIR/train_first10k.parquet \
  data.val_files=$DATA_LOCAL_DIR/train_first10k.parquet \
  data.max_prompt_length=$MAX_PROMPT_LENGTH \
  data.max_response_length=$MAX_RESPONSE_LENGTH \
  data.train_batch_size=$DATA_TRAIN_BATCH_SIZE \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  actor_rollout_ref.model.path=$BACKBONE_PATH \
  actor_rollout_ref.model.use_shm=True  \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
  actor_rollout_ref.model.lora_rank=32 \
  actor_rollout_ref.model.lora_alpha=64 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.optim.lr=2e-5 \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.03 \
  actor_rollout_ref.actor.optim.warmup_style='cosine' \
  actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.temperature=0.8 \
  actor_rollout_ref.rollout.top_p=0.95 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.dtype=bfloat16 \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
  actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.rollout.n=$N_SAMPLES_PER_PROMPT \
  actor_rollout_ref.rollout.max_model_len=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
  actor_rollout_ref.rollout.max_num_batched_tokens=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
  critic.optim.lr=9e-6 \
  critic.model.use_remove_padding=True \
  critic.model.path=$BACKBONE_PATH \
  critic.model.enable_gradient_checkpointing=True \
  critic.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
  critic.model.fsdp_config.param_offload=True \
  critic.model.fsdp_config.optimizer_offload=True \
  algorithm.kl_ctrl.kl_coef=0.00 \
  algorithm.adv_estimator=$ADVANTAGE \
  algorithm.norm_adv_by_std_in_grpo=False \
  trainer.logger=['console'] \
  trainer.val_before_train=False \
  trainer.project_name=$WANDB_PROJECT \
  trainer.experiment_name=$LOG_NAME \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1 \
  trainer.test_freq=-1\
  trainer.save_freq=2000 \
  trainer.resume_mode=auto \
  trainer.default_local_dir=$OUTPUT_DIR \
  trainer.total_epochs=1
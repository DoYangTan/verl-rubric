#!/bin/bash

# load .env
[ -f ".env" ] && set -a && . .env && set +a

EXP_NAME="ourmethod1_1"
PROJECT_NAME="rubrichub_v1_Medical"
ROLLOUT_N=8
USE_DYNAMIC_BSZ=True
MAX_TOKENS=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))

#############################
set -x

VAL_ONLY=${VAL_ONLY:-False}

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo_v1_1 \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.filter_overlong_prompts=True \
    reward_model.use_reward_loop=True \
    reward_model.reward_manager=dapo \
    custom_reward_function.path=verl/utils/reward_score/rubric_reward/ourmethod.py \
    custom_reward_function.name=compute_score \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.use_dynamic_bsz=${USE_DYNAMIC_BSZ} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.max_model_len=${MAX_TOKENS} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION} \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.n_gpus_per_node=${NUM_GPUS} \
    trainer.nnodes=1 \
    trainer.resume_mode=disable \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.val_only=${VAL_ONLY} \
    trainer.total_epochs=15 $@

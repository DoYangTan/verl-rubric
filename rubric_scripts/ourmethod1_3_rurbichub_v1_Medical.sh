#!/bin/bash

# Auto-select GPUs with low memory usage unless explicitly set by the user.
if [ -z "${CUDA_VISIBLE_DEVICES}" ]; then
    mem_threshold_mb=10000
    poll_interval_seconds=${GPU_POLL_INTERVAL_SECONDS:-15}
    while true; do
        free_gpus=$(nvidia-smi --query-gpu=index,memory.used \
            --format=csv,noheader,nounits | awk -F', *' -v thr="${mem_threshold_mb}" \
            '$2 < thr {print $1}' | paste -sd, -)
        if [ -n "${free_gpus}" ]; then
            export CUDA_VISIBLE_DEVICES="${free_gpus}"
            break
        fi
        echo "[INFO] No GPUs with memory used <${mem_threshold_mb}MB. Retrying in ${poll_interval_seconds}s..." >&2
        sleep "${poll_interval_seconds}"
    done
fi
NUM_GPUs=$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F, '{print NF}')

# Load local .env so Ray workers inherit grader/attribution settings.
if [ -f ".env" ]; then
    set -a
    . .env
    set +a
fi
export WANDB_API_KEY='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'
export WANDB_HTTP_TIMEOUT=60
export DEBUG_REWARD_FLOW=1
export DEBUG_REWARD_FLOW_LIMIT=200
export ATTRIBUTION_ENABLE=1

# proxy
export http_proxy="http://127.0.0.1:10086"
export https_proxy="http://127.0.0.1:10086"
export all_proxy="socks5://127.0.0.1:10086"
export no_proxy="localhost,127.0.0.1,0.0.0.0"

if [ -z "${RAY_TMPDIR}" ]; then
    mkdir -p /workspace/verl/verl-rubric-dev/tmp_ray
    export RAY_TMPDIR="/workspace/verl/verl-rubric-dev/tmp_ray"
fi
if [ -z "${TMPDIR}" ]; then
    mkdir -p /workspace/verl/verl-rubric-dev/tmp_ray
    export TMPDIR="/workspace/verl/verl-rubric-dev/tmp_ray"
fi

EXP_NAME="rubrichub_v1_Medical_ourmethod"
PROJECT_NAME="rubrichub_v1_Medical"
MODEL_PATH="model_weight/Qwen/Qwen2.5-7B-Instruct"
RUN_TAG=${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}

ttl_decay_factor=${GRPO_TTL_DECAY_FACTOR:-0.95}

max_prompt_length=4096
max_response_length=4096
use_dynamic_bsz=True
max_tokens=$((max_prompt_length + max_response_length))
enable_overlong_buffer=True
overlong_buffer_len=$((1024 * 2))
overlong_buffer_penalty_factor=1
actor_ppo_max_token_len=$((max_tokens * 1))
infer_ppo_max_token_len=$((max_tokens * 1))
max_num_batched_tokens=$((max_tokens * 1))
clip_ratio_low=0.2
clip_ratio_high=0.28 # clip high
train_batch_size=64
ppo_mini_batch_size=32

#############################
set -x

VAL_ONLY=${VAL_ONLY:-False}

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo_v1_3 \
    algorithm.grpo_v1_3_decay_factor=${ttl_decay_factor} \
    data.train_files=data/RubricHub_v1/RuRL/RubricHub_v1/RuRL/rurbichub_v1_Medical.parquet \
    data.val_files=data/health_bench/healthbench_val.parquet \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    reward_model.use_reward_loop=True \
    reward_model.num_workers=1 \
    reward_model.reward_manager=dapo \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_buffer_penalty_factor} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=True \
    +reward_model.reward_kwargs.max_resp_len=${max_response_length} \
    custom_reward_function.path=verl/utils/reward_score/rubric_reward/ourmethod.py \
    custom_reward_function.name=compute_score \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.agent.default_agent_loop=single_turn_agent \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.max_num_batched_tokens=${max_num_batched_tokens} \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.8 \
    actor_rollout_ref.rollout.val_kwargs.top_k=20 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.rollout_data_dir="log/rollout_log/${EXP_NAME}/${RUN_TAG}" \
    trainer.validation_data_dir="log/validation_log/${EXP_NAME}/${RUN_TAG}" \
    trainer.n_gpus_per_node=${NUM_GPUs} \
    trainer.nnodes=1 \
    trainer.resume_mode=disable \
    +ray_kwargs.ray_init.runtime_env.env_vars.DEBUG_REWARD_FLOW="\"${DEBUG_REWARD_FLOW}\"" \
    +ray_kwargs.ray_init.runtime_env.env_vars.DEBUG_REWARD_FLOW_LIMIT="\"${DEBUG_REWARD_FLOW_LIMIT}\"" \
    +ray_kwargs.ray_init.runtime_env.env_vars.ATTRIBUTION_ENABLE="\"${ATTRIBUTION_ENABLE}\"" \
    +ray_kwargs.ray_init.runtime_env.env_vars.VLLM_BASE_URL="\"${VLLM_BASE_URL}\"" \
    +ray_kwargs.ray_init.runtime_env.env_vars.VLLM_MODEL="\"${VLLM_MODEL}\"" \
    +ray_kwargs.ray_init.runtime_env.env_vars.VLLM_MAX_TOKENS="\"${VLLM_MAX_TOKENS}\"" \
    +ray_kwargs.ray_init.runtime_env.env_vars.VLLM_TEMPERATURE="\"${VLLM_TEMPERATURE}\"" \
    +ray_kwargs.ray_init.runtime_env.env_vars.ATTRIBUTION_VLLM_BASE_URL="\"${ATTRIBUTION_VLLM_BASE_URL}\"" \
    +ray_kwargs.ray_init.runtime_env.env_vars.ATTRIBUTION_VLLM_MODEL="\"${ATTRIBUTION_VLLM_MODEL}\"" \
    +ray_kwargs.ray_init.runtime_env.env_vars.ATTRIBUTION_VLLM_MAX_TOKENS="\"${ATTRIBUTION_VLLM_MAX_TOKENS}\"" \
    +ray_kwargs.ray_init.runtime_env.env_vars.ATTRIBUTION_VLLM_TEMPERATURE="\"${ATTRIBUTION_VLLM_TEMPERATURE}\"" \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.val_only=${VAL_ONLY} \
    trainer.total_epochs=15 $@

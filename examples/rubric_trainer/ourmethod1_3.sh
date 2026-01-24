#!/bin/bash

MAX_GPU_MEM_GB=30
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        gpu_list=$(nvidia-smi --query-gpu=index,memory.total --format=csv,noheader,nounits 2>/dev/null | \
            awk -F',' -v max_mem=$((MAX_GPU_MEM_GB * 1024)) '{gsub(/ /, "", $2); if ($2 + 0 < max_mem) print $1}' | \
            paste -sd, -)
        if [[ "${gpu_list}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
            export CUDA_VISIBLE_DEVICES="${gpu_list}"
        fi
    fi
fi

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    NUM_GPUs=$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}')
else
    NUM_GPUs=1
fi

echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>} NUM_GPUs=${NUM_GPUs}"

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

EXP_NAME="rubrichub_v1_Medical_ourmethod"
PROJECT_NAME="rubrichub_v1_Medical"
MODEL_PATH="model_weight/Qwen/Qwen2.5-7B-Instruct"
RUN_TAG=${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}

max_prompt_length=4096
max_response_length=4096
use_dynamic_bsz=True
max_tokens=$((max_prompt_length + max_response_length))
actor_ppo_max_token_len=$((max_tokens * 1))
infer_ppo_max_token_len=$((max_tokens * 1))
max_num_batched_tokens=$((max_tokens * 1))
clip_ratio_low=0.2
clip_ratio_high=0.28 # clip high
train_batch_size=8
ppo_mini_batch_size=8
debug_steps=${DEBUG_STEPS:-5}
train_max_samples=$((train_batch_size * debug_steps))
total_epochs=${TOTAL_EPOCHS:-1}
enable_overlong_buffer=True
overlong_buffer_len=$((1024 * 2))
overlong_penalty_factor=1
ttl_decay_factor=${GRPO_TTL_DECAY_FACTOR:-0.95}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.2}

RUN_LOG_DIR="log/run_logs/${EXP_NAME}"
mkdir -p "${RUN_LOG_DIR}"
RUN_LOG_FILE="${RUN_LOG_DIR}/train_${RUN_TAG}.log"
GPU_MONITOR_INTERVAL_SEC=${GPU_MONITOR_INTERVAL_SEC:-30}
GPU_MONITOR_LOG="${RUN_LOG_DIR}/gpu_${RUN_TAG}.log"
RETRY_ON_RESOURCE_ERROR=${RETRY_ON_RESOURCE_ERROR:-1}
RETRY_WAIT_SEC=${RETRY_WAIT_SEC:-60}
MAX_RETRIES=${MAX_RETRIES:-0}

start_gpu_monitor() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "[monitor] nvidia-smi not found; skipping GPU monitor."
        return
    fi
    (
        while true; do
            nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total \
                --format=csv,noheader,nounits
            sleep "${GPU_MONITOR_INTERVAL_SEC}"
        done
    ) >> "${GPU_MONITOR_LOG}" 2>/dev/null &
    GPU_MONITOR_PID=$!
}

stop_gpu_monitor() {
    if [[ -n "${GPU_MONITOR_PID:-}" ]]; then
        kill "${GPU_MONITOR_PID}" 2>/dev/null || true
    fi
}

trap stop_gpu_monitor EXIT

#############################
set -x

start_gpu_monitor

run_training() {
    python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo_v1_3 \
        algorithm.grpo_v1_3_decay_factor=${ttl_decay_factor} \
        data.train_files=data/RubricHub_v1/RuRL/RubricHub_v1/RuRL/rurbichub_v1_Medical.parquet \
        data.val_files=data/health_bench/healthbench_val.parquet \
        data.train_batch_size=${train_batch_size} \
        data.max_prompt_length=${max_prompt_length} \
        data.max_response_length=${max_response_length} \
        data.filter_overlong_prompts=True \
        data.train_max_samples=${train_max_samples} \
        data.truncation='error' \
        data.return_raw_chat=True \
        reward_model.use_reward_loop=True \
        reward_model.num_workers=1 \
        reward_model.reward_manager=dapo \
        +reward_model.reward_kwargs.max_resp_len=${max_response_length} \
        +reward_model.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
        +reward_model.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
        +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
        +reward_model.reward_kwargs.overlong_buffer_cfg.log=True \
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
        actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util} \
        actor_rollout_ref.rollout.n=8 \
        actor_rollout_ref.rollout.max_num_batched_tokens=${max_num_batched_tokens} \
        actor_rollout_ref.rollout.temperature=1.0 \
        actor_rollout_ref.rollout.top_p=1.0 \
        actor_rollout_ref.rollout.top_k=-1 \
        actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
        actor_rollout_ref.rollout.val_kwargs.top_p=0.8 \
        actor_rollout_ref.rollout.val_kwargs.top_k=20 \
        actor_rollout_ref.rollout.val_kwargs.n=1 \
        actor_rollout_ref.rollout.val_kwargs.do_sample=True \
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
        trainer.rollout_data_dir="log/rollout_log/${EXP_NAME}" \
        trainer.validation_data_dir="log/validation_log/${EXP_NAME}" \
        trainer.n_gpus_per_node=${NUM_GPUs} \
        trainer.nnodes=1 \
        trainer.resume_mode=disable \
        trainer.total_training_steps=${debug_steps} \
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
        trainer.save_freq=5 \
        trainer.test_freq=1 \
        trainer.total_epochs=${total_epochs} $@ 2>&1 | tee -a "${RUN_LOG_FILE}"
    return ${PIPESTATUS[0]}
}

is_resource_error() {
    rg -n -i "(out of memory|cuda|nccl|cublas|cudnn|cuda calloc|no cuda runtime)" "${RUN_LOG_FILE}" >/dev/null 2>&1
}

attempt=0
while true; do
    run_training "$@"
    status=$?
    if [[ ${status} -eq 0 ]]; then
        exit 0
    fi
    if [[ "${RETRY_ON_RESOURCE_ERROR}" = "1" ]] && is_resource_error; then
        attempt=$((attempt + 1))
        if [[ "${MAX_RETRIES}" -ne 0 && "${attempt}" -ge "${MAX_RETRIES}" ]]; then
            exit ${status}
        fi
        echo "[retry] resource error detected, waiting ${RETRY_WAIT_SEC}s before retry..." | tee -a "${RUN_LOG_FILE}"
        sleep "${RETRY_WAIT_SEC}"
        continue
    fi
    exit ${status}
done

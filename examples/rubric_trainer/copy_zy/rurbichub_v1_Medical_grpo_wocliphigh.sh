pip install --no-deps -e /opt/tiger/verl
pip install tensorboard
pip install -U flashinfer-python
pip install math-verify -U

export CUDA_VISIBLE_DEVICES="0,1,2,3"

NUM_GPUs=4
EXP_NAME="rubrichub_v1_Medical_Qwen2.5-7B-Instruct_GRPO"
PROJECT_NAME="rubrichub_v1_Medical"
MODEL_PATH="/mnt/hdfs/__MERLIN_USER_DIR__/models/Qwen2.5-7B-Instruct"

export TENSORBOARD_DIR="hdfs://harunawl/home/byte_data_seed_wl/user/zhouyang.1107/trial_verl/tensorboard_log/${PROJECT_NAME}/${EXP_NAME}"

max_prompt_length=4096
max_response_length=8192
use_dynamic_bsz=True
max_tokens=$((max_prompt_length + max_response_length))
enable_overlong_buffer=True
overlong_buffer_len=$((1024*4))
overlong_buffer_penalty_factor=0.5
actor_ppo_max_token_len=$((max_tokens * 1))
infer_ppo_max_token_len=$((max_tokens * 1))
max_num_batched_tokens=$((max_tokens * 1))
clip_ratio_low=0.2
clip_ratio_high=0.2
train_batch_size=128
ppo_mini_batch_size=128


#############################
set -x

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=hdfs://harunawl/home/byte_data_seed_wl/user/zhouyang.1107/data/rubric/RubricHub/rurbichub_v1_Medical.parquet \
    data.val_files=hdfs://harunawl/home/byte_data_seed_wl/user/zhouyang.1107/data/rubric/health_bench/healthbench_val.parquet \
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
    custom_reward_function.path=verl/utils/reward_score/rubric_reward/rurbichub_v1_Medical.py \
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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.prompt_length=${max_prompt_length} \
    actor_rollout_ref.rollout.response_length=${max_response_length} \
    actor_rollout_ref.rollout.max_model_len=${max_tokens} \
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
    trainer.logger=['console','tensorboard'] \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.default_local_dir="/mnt/hdfs/__MERLIN_USER_DIR__/trial_verl/checkpoints/${PROJECT_NAME}/${EXP_NAME}" \
    trainer.rollout_data_dir="/mnt/hdfs/__MERLIN_USER_DIR__/trial_verl/log/rollout_log/${PROJECT_NAME}/${EXP_NAME}" \
    trainer.validation_data_dir="/mnt/hdfs/__MERLIN_USER_DIR__/trial_verl/log/validation_log/${PROJECT_NAME}/${EXP_NAME}" \
    trainer.n_gpus_per_node=${NUM_GPUs} \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=15 $@

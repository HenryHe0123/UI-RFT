#!/bin/bash

# Usage: nohup bash train.sh > train.log 2>&1 &
# Merge model: python scripts/model_merger.py --local_dir .../actor/

source /inspire/hdd/global_user/liupengfei-24025/yhhe/miniconda3/bin/activate verl

verl_path=/inspire/hdd/global_user/liupengfei-24025/yhhe/code/V-RFT/verl
model_path=/inspire/hdd/global_user/liupengfei-24025/yhhe/model/Qwen2.5-VL-3B-Instruct
project_name=ui128
experiment_name=qwen2_5_vl_3b_grpo
save_path=$verl_path/checkpoints/$project_name/$experiment_name

set -x
export VLLM_ATTENTION_BACKEND=XFORMERS
cd $verl_path

PYTHONUNBUFFERED=1 WANDB_MODE=offline python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$verl_path/data/ui100/train.parquet \
    data.val_files=$verl_path/data/ui100/test.parquet \
    data.train_batch_size=32 \
    data.max_prompt_length=8192 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.image_key=images \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.max_num_batched_tokens=10240 \
    actor_rollout_ref.rollout.max_model_len=10240 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    +algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.default_local_dir=$save_path \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=8 \
    trainer.test_freq=8 \
    trainer.total_epochs=24 $@

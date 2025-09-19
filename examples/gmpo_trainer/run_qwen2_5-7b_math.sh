set -x

unset RAY_ADDRESS

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 动态计算可用 GPU 数
if [ -z "${CUDA_VISIBLE_DEVICES}" ]; then
N_GPUS=$(command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --list-gpus | wc -l || echo 1)
else
IFS=',' read -ra DEV_ARR <<< "$CUDA_VISIBLE_DEVICES"
N_GPUS=${#DEV_ARR[@]}
fi

gsm8k_train_path=$HOME/data/gsm8k/train.parquet
gsm8k_test_path=$HOME/data/gsm8k/test.parquet

train_files="['$gsm8k_train_path']"
test_files="['$gsm8k_test_path']"

use_kl_loss=False
loss_mode=geo_mean
clip_ratio=0.4
save_contents="['model', 'optimizer', 'extra']"

export WANDB_MODE=offline
save_contents="['hf_model']"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=32 \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=/root/et/checkpoints/Qwen2.5-VL-3B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.policy_loss.loss_mode=${loss_mode} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.checkpoint.save_contents=${save_contents} \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_gmpo_example_gsm8k_math' \
    trainer.experiment_name='qwen2_5_vl_3b_function_rm' \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=100 $@

LOG_DIR=${LOG_DIR:-"logs/verl"}
mkdir -p "$LOG_DIR"
TS=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/run_qwen2_5_vl-7b_${TS}.log"
# 将脚本所有输出同时写入终端与文件
exec > >(tee -a "$LOG_FILE") 2>&1
echo "日志输出到: $LOG_FILE"

set -x
ENGINE=${1:-vllm}

mkdir -p /home/torch_ipc/ray
export RAY_NODE_IP_ADDRESS=$(hostname -I | awk '{print $1}')
export RAY_TMPDIR=/home/torch_ipc/ray
export RAY_ADDRESS=auto
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

ray stop -f
ray start --head \
  --node-ip-address="$RAY_NODE_IP_ADDRESS" \
  --dashboard-host=0.0.0.0 \
  --dashboard-port=8265 \
  --temp-dir="$RAY_TMPDIR" \
  --plasma-directory="$RAY_TMPDIR" \
  --object-store-memory=$((8*1024*1024*1024))
  
# ray stop -f
# unset RAY_ADDRESS

export WANDB_MODE=offline
# export WANDB_BASE_URL="http://127.0.0.1:8080"
export WANDB_API_KEY="local-5d171bb31e4084b5258517989579e10ad7648d98"

# 本地可视化与持久化（TensorBoard 与 MLflow）
experiment_name="qwen2_5_vl_7b_function_rm_iceberg_no_kl_new_bs512"
export TENSORBOARD_DIR=${TENSORBOARD_DIR:-/data_ali/shunian/verl/tensorboard_log/verl_grpo_example_iceberg/${experiment_name}}
mkdir -p "$TENSORBOARD_DIR"

export LOCAL_METRICS_DIR=${LOCAL_METRICS_DIR:-/data_ali/shunian/verl/logs/local_metrics}
mkdir -p "$LOCAL_METRICS_DIR"

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 动态计算可用 GPU 数
if [ -z "${CUDA_VISIBLE_DEVICES}" ]; then
N_GPUS=$(command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --list-gpus | wc -l || echo 1)
else
IFS=',' read -ra DEV_ARR <<< "$CUDA_VISIBLE_DEVICES"
N_GPUS=${#DEV_ARR[@]}
fi

# geo3k_train_path=$HOME/data/geo3k/train.parquet
# geo3k_test_path=$HOME/data/geo3k/test.parquet
# train_files="['$geo3k_train_path']"
# test_files="['$geo3k_test_path']"

iceberg_train_path=/data_ali/shunian/data/rl_data/iceberg/train.parquet
iceberg_test_path=/data_ali/shunian/data/rl_data/iceberg/test.parquet
train_files="['$iceberg_train_path']"
test_files="['$iceberg_test_path']"

# iceberg_train_path=/data/rl_data/iceberg/train_simple.parquet
# iceberg_test_path=/data/rl_data/iceberg/test_simple.parquet
# train_files="['$iceberg_train_path']"
# test_files="['$iceberg_test_path']"

# python3 -m verl.trainer.main_ppo \
#     algorithm.adv_estimator=grpo \
#     data.train_files="$train_files" \
#     data.val_files="$test_files" \
#     data.train_batch_size=512 \
#     data.max_prompt_length=4096 \
#     data.max_response_length=2048 \
#     data.filter_overlong_prompts=True \
#     data.filter_overlong_prompts_workers=128 \
#     data.dataloader_num_workers=64 \
#     data.truncation='error' \
#     data.image_key=images \
#     actor_rollout_ref.model.path=/data_ali/shunian/models/Qwen2.5-VL-7B-Instruct \
#     actor_rollout_ref.actor.optim.lr=1e-6 \
#     actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
#     actor_rollout_ref.actor.optim.weight_decay=0.1 \
#     actor_rollout_ref.actor.checkpoint.save_contents=['model','hf_model','optimizer','extra'] \
#     actor_rollout_ref.model.use_remove_padding=True \
#     actor_rollout_ref.actor.ppo_mini_batch_size=256 \
#     actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
#     actor_rollout_ref.actor.entropy_coeff=0 \
#     actor_rollout_ref.model.enable_gradient_checkpointing=False \
#     actor_rollout_ref.actor.fsdp_config.param_offload=False \
#     actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
#     actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
#     actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
#     actor_rollout_ref.rollout.name=$ENGINE \
#     +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=False \
#     +actor_rollout_ref.rollout.engine_kwargs.vllm.max_num_batched_tokens=8192 \
#     +actor_rollout_ref.rollout.engine_kwargs.vllm.max_num_seqs=128 \
#     +actor_rollout_ref.rollout.engine_kwargs.vllm.max_model_len=12288 \
#     actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
#     actor_rollout_ref.rollout.disable_log_stats=False \
#     actor_rollout_ref.rollout.enable_chunked_prefill=True \
#     actor_rollout_ref.rollout.enforce_eager=False \
#     actor_rollout_ref.rollout.free_cache_engine=True \
#     actor_rollout_ref.rollout.n=16 \
#     actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
#     actor_rollout_ref.ref.fsdp_config.param_offload=False \
#     algorithm.use_kl_in_reward=False \
#     trainer.critic_warmup=0 \
#     trainer.logger='["console","tensorboard"]' \
#     trainer.project_name='verl_grpo_example_iceberg' \
#     trainer.experiment_name=${experiment_name} \
#     trainer.n_gpus_per_node=$N_GPUS \
#     trainer.nnodes=1 \
#     trainer.save_freq=100 \
#     trainer.test_freq=50 \
#     trainer.log_val_generations=10 \
#     trainer.total_epochs=100 $@
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=512 \
    data.max_prompt_length=4096 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=128 \
    data.dataloader_num_workers=64 \
    data.truncation='error' \
    data.image_key=images \
    actor_rollout_ref.model.path=/data_ali/shunian/models/Qwen2.5-VL-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.checkpoint.save_contents=['model','hf_model','optimizer','extra'] \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=False \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name='verl_grpo_example_iceberg' \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=2 \
    trainer.save_freq=100 \
    trainer.test_freq=50 \
    trainer.log_val_generations=10 \
    trainer.total_epochs=100 $@
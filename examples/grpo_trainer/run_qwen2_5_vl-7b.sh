#!/usr/bin/env bash

# ------------------------- 参数解析与运行标识 -------------------------
ENGINE="vllm"
DATASET=""
MODEL_ARG=""
SAVE_DIR=""
LR=""
KL_LOSS=0
# rollout.n 默认值
ROLLOUT_N=4
ROLLOUT_N_SET=0
# 长度奖励开关（默认关闭）以及可选的奖励最大长度（若未提供，则引用模型最大输出长度）
LENGTH_REWARD=0 # 0=关闭，1=开启 2=half length reward
LENGTH_REWARD_MAX_LEN=""
BATCH_SIZE=128
MAX_RESPONSE_LENGTH=1024
STRONG_TAG=0
REPETITION_PENALTY=1.0
FORMAT_ANSWER_PRODUCT=0

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --engine)
      ENGINE="$2"; shift 2 ;;
    --dataset)
      DATASET="$2"; shift 2 ;;
    --model)
      MODEL_ARG="$2"; shift 2 ;;
    --save_dir)
      SAVE_DIR="$2"; shift 2 ;;
    --length_reward)
      LENGTH_REWARD="$2"; shift 2 ;;
    --lr)
      LR="$2"; shift 2 ;;
    --kl_loss)
      KL_LOSS=1; shift ;;
    --rollout_n)
      ROLLOUT_N="$2"; ROLLOUT_N_SET=1; shift 2 ;;
    --batch_size)
      BATCH_SIZE="$2"; shift 2 ;;
    --max_response_length)
      MAX_RESPONSE_LENGTH="$2"; shift 2 ;;
    --reward_max_length)
      LENGTH_REWARD_MAX_LEN="$2"; shift 2 ;;
    --strong_tag)
      STRONG_TAG=1; shift ;;
    --repetition_penalty)
      REPETITION_PENALTY="$2"; shift 2 ;;
    --format_answer_product)
      FORMAT_ANSWER_PRODUCT=1; shift ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do
        EXTRA_ARGS+=("$1")
        shift
      done
      break ;;
    *)
      # 其余参数暂存，稍后仅保留 Hydra 覆盖（key=value 或以 + 开头）再传给 Python
      EXTRA_ARGS+=("$1"); shift ;;
  esac
done

# 仅保留 Hydra 覆盖样式的参数（避免把脚本自定义旗标传给 Python）
HYDRA_OVERRIDES=()
for arg in "${EXTRA_ARGS[@]}"; do
  if [[ "$arg" == *=* || "$arg" == +* ]]; then
    HYDRA_OVERRIDES+=("$arg")
  fi
done

# 默认保存根目录
SAVE_DIR=${SAVE_DIR:-/data_ali/shunian/verl}

# 模型路径（默认保持原脚本的 7B Instruct）
MODEL_PATH=${MODEL_ARG:-/data_ali/shunian/models/Qwen2.5-VL-7B-Instruct}

# 生成用于区分 run 的标识（包含 dataset 与模型名）
dataset_tag=${DATASET:-iceberg}
model_tag=$(basename "${MODEL_PATH}")
TS=$(date +"%Y%m%d_%H%M%S")

# 日志目录与文件名（包含关键信息）
LOG_DIR=${LOG_DIR:-"${SAVE_DIR}/logs/verl"}

TRACE_DIR=${TRACE_DIR:-/data_ali/shunian/verl/traces}
TRACE_DB_PATH=${TRACE_DB_PATH:-${TRACE_DIR}/mlflow_traces.db}
TRACE_ARTIFACT_DIR=${TRACE_ARTIFACT_DIR:-${TRACE_DIR}/artifacts}
mkdir -p "$TRACE_DIR" "$TRACE_ARTIFACT_DIR"
if [[ ! -f "$TRACE_DB_PATH" ]]; then
  TRACE_DB_PATH="$TRACE_DB_PATH" python3 - <<'PYTHON'
import os
import sqlite3

db_path = os.environ["TRACE_DB_PATH"]
os.makedirs(os.path.dirname(db_path), exist_ok=True)
sqlite3.connect(db_path).close()
PYTHON
fi
export MLFLOW_TRACKING_URI="sqlite:///${TRACE_DB_PATH}"
export MLFLOW_ARTIFACT_URI=${MLFLOW_ARTIFACT_URI:-"file://${TRACE_ARTIFACT_DIR}"}

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
  --object-store-memory=$((64*1024*1024*1024))
  
export WANDB_MODE=offline
# export WANDB_BASE_URL="http://127.0.0.1:8080"
export WANDB_API_KEY="local-5d171bb31e4084b5258517989579e10ad7648d98"
export SWANLAB_API_KEY="kY60KoAC6L73EYtx8liD7"

# 本地可视化与持久化（TensorBoard 与 MLflow）
project_name="verl_grpo_example_iceberg"
# 让实验名可体现数据集、模型及是否开启长度奖励
lr_tag=""
if [[ -n "${LENGTH_REWARD}" ]]; then
  if [[ "${LENGTH_REWARD}" == "1" ]]; then
    lr_tag="_lenrw"
  elif [[ "${LENGTH_REWARD}" == "2" ]]; then
    lr_tag="_half_lenrw"
  fi
fi
# KL 损失标签（若启用，则加入实验名）
kl_tag=""
if [[ "${KL_LOSS}" == "1" ]]; then
  kl_tag="_kl"
fi
# 学习率标签（若传入 --lr，则加入实验名。将小数点替换为 p，减号替换为 n）
lr_rate_tag=""
if [[ -n "${LR}" ]]; then
  lr_sanitized="${LR//./p}"
  lr_sanitized="${lr_sanitized//-/n}"
  lr_rate_tag="_lr${lr_sanitized}"
fi
rollout_n_tag=""
if [[ "${ROLLOUT_N_SET}" == "1" ]]; then
  rollout_n_tag="_rn${ROLLOUT_N}"
fi

batch_size_tag=""
if [[ -n "${BATCH_SIZE}" ]]; then
  batch_size_tag="_bs${BATCH_SIZE}"
fi

reward_max_length_tag=""
if [[ -n "${LENGTH_REWARD_MAX_LEN}" && "${LENGTH_REWARD_MAX_LEN}" != "${MAX_RESPONSE_LENGTH}" ]]; then
  reward_max_length_tag="_rmax${LENGTH_REWARD_MAX_LEN}"
fi

format_answer_product_tag=""
if [[ "${FORMAT_ANSWER_PRODUCT}" == "1" ]]; then
  format_answer_product_tag="_fmtprod"
fi

experiment_name="qwen2_5_vl_7b_${dataset_tag}_${model_tag}${lr_tag}${kl_tag}${lr_rate_tag}${rollout_n_tag}${batch_size_tag}${reward_max_length_tag}${format_answer_product_tag}_${MAX_RESPONSE_LENGTH}_${REPETITION_PENALTY}"

mkdir -p "$LOG_DIR/run_qwen2_5_vl-7b_${dataset_tag}_${model_tag}_${ROLLOUT_N}_${BATCH_SIZE}${reward_max_length_tag}${format_answer_product_tag}_${MAX_RESPONSE_LENGTH}_${REPETITION_PENALTY}"
LOG_FILE="$LOG_DIR/run_qwen2_5_vl-7b_${dataset_tag}_${model_tag}_${ROLLOUT_N}_${BATCH_SIZE}${reward_max_length_tag}${format_answer_product_tag}_${MAX_RESPONSE_LENGTH}_${REPETITION_PENALTY}/${TS}.log"

# 将脚本所有输出同时写入终端与文件
exec > >(tee -a "$LOG_FILE") 2>&1
echo "日志输出到: $LOG_FILE"

set -x
echo "[ARGS] ENGINE=$ENGINE DATASET=$DATASET MODEL_PATH=$MODEL_PATH SAVE_DIR=$SAVE_DIR LR=${LR:-default} KL_LOSS=$KL_LOSS ROLLOUT_N=$ROLLOUT_N MAX_RESPONSE_LENGTH=$MAX_RESPONSE_LENGTH REWARD_MAX_LENGTH=${LENGTH_REWARD_MAX_LEN:-$MAX_RESPONSE_LENGTH} FORMAT_ANSWER_PRODUCT=$FORMAT_ANSWER_PRODUCT"

export TENSORBOARD_DIR=${TENSORBOARD_DIR:-${SAVE_DIR}/tensorboard_log/${project_name}/${experiment_name}}
mkdir -p "$TENSORBOARD_DIR"

export LOCAL_METRICS_DIR=${LOCAL_METRICS_DIR:-/data_ali/shunian/verl/logs/local_metrics}
mkdir -p "$LOCAL_METRICS_DIR"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

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

# 数据集切换：--dataset simple 切到简易集，否则默认完整 iceberg
if [[ "${DATASET}" == "simple" ]]; then
  iceberg_train_path=/data_ali/shunian/data/rl_data/iceberg/train_simple.parquet
  iceberg_test_path=/data_ali/shunian/data/rl_data/iceberg/test_simple.parquet
elif [[ "${DATASET}" == "thinking" ]]; then
  iceberg_train_path=/data_ali/shunian/data/rl_data/iceberg/train_thinking.parquet
  iceberg_test_path=/data_ali/shunian/data/rl_data/iceberg/test_thinking.parquet
elif [[ "${DATASET}" == "normal" ]]; then
  iceberg_train_path=/data_ali/shunian/data/rl_data/iceberg/train_normal.parquet
  iceberg_test_path=/data_ali/shunian/data/rl_data/iceberg/test_normal.parquet
elif [[ "${DATASET}" == "extra_simple" ]]; then
  iceberg_train_path=/data_ali/shunian/data/rl_data/iceberg/train_extra_simple.parquet
  iceberg_test_path=/data_ali/shunian/data/rl_data/iceberg/test_extra_simple.parquet
elif [[ "${DATASET}" == "selected_simple" ]]; then
  iceberg_train_path=/data_ali/shunian/data/rl_data/iceberg/train_selected_simple.parquet
  iceberg_test_path=/data_ali/shunian/data/rl_data/iceberg/test_simple.parquet
elif [[ "${DATASET}" == "strong_tag" ]]; then
  iceberg_train_path=/data_ali/shunian/data/rl_data/iceberg/train_strong_tag.parquet
  iceberg_test_path=/data_ali/shunian/data/rl_data/iceberg/test_strong_tag.parquet
else
  iceberg_train_path=/data_ali/shunian/data/rl_data/iceberg/train.parquet
  iceberg_test_path=/data_ali/shunian/data/rl_data/iceberg/test.parquet
fi
train_files="['$iceberg_train_path']"
test_files="['$iceberg_test_path']"

# iceberg_train_path=/data/rl_data/iceberg/train_simple.parquet
# iceberg_test_path=/data/rl_data/iceberg/test_simple.parquet
# train_files="['$iceberg_train_path']"
# test_files="['$iceberg_test_path']"

# 根据开关构造长度奖励参数（默认关闭）。
LENGTH_REWARD_ARGS=()
model_max_response_length=${MAX_RESPONSE_LENGTH}
reward_max_length=${LENGTH_REWARD_MAX_LEN:-$model_max_response_length}

FORMAT_ANSWER_PRODUCT_ARGS=()
if [[ "${FORMAT_ANSWER_PRODUCT}" == "1" ]]; then
  FORMAT_ANSWER_PRODUCT_ARGS+=("+reward_model.reward_kwargs.format_answer_product=true")
fi

if [[ "$LENGTH_REWARD" == "1" ]]; then
  LENGTH_REWARD_ARGS+=("+reward_model.reward_kwargs.enable_length_reward=true")
  if [[ -n "${reward_max_length}" ]]; then
    LENGTH_REWARD_ARGS+=("+reward_model.reward_kwargs.reward_max_length=${reward_max_length}")
  fi
  LENGTH_REWARD_ARGS+=("+reward_model.reward_kwargs.length_reward_max_len=${reward_max_length}")
elif [[ "$LENGTH_REWARD" == "2" ]]; then
  # 半阈值模式：传入 2 以触发 iceberg.compute_score 的 half-threshold 分支
  LENGTH_REWARD_ARGS+=("+reward_model.reward_kwargs.enable_length_reward=2")
  if [[ -n "${reward_max_length}" ]]; then
    LENGTH_REWARD_ARGS+=("+reward_model.reward_kwargs.reward_max_length=${reward_max_length}")
  fi
  LENGTH_REWARD_ARGS+=("+reward_model.reward_kwargs.length_reward_max_len=${reward_max_length}")
fi

# 根据开关构造 strong_tag（默认关闭）
STRONG_TAG_ARGS=()
if [[ "$STRONG_TAG" == "1" ]]; then
  STRONG_TAG_ARGS+=("+reward_model.reward_kwargs.strong_tag=true")
fi

# 根据开关构造 KL 损失参数（默认关闭）。
KL_LOSS_ARGS=()
if [[ "$KL_LOSS" == "1" ]]; then
  KL_LOSS_ARGS+=("actor_rollout_ref.actor.use_kl_loss=True")
  KL_LOSS_ARGS+=("actor_rollout_ref.actor.kl_loss_coef=0.001")
  KL_LOSS_ARGS+=("actor_rollout_ref.actor.kl_loss_type=low_var_kl")
fi

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=${BATCH_SIZE} \
    data.max_prompt_length=4096 \
    data.max_response_length=${model_max_response_length} \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=128 \
    data.dataloader_num_workers=64 \
    data.truncation='error' \
    data.image_key=images \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=${LR:-1e-6} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.checkpoint.save_contents=['model','hf_model','optimizer','extra'] \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=False \
    +actor_rollout_ref.rollout.repetition_penalty=$REPETITION_PENALTY \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.rollout.max_num_seqs=2 \
    actor_rollout_ref.rollout.max_model_len=16384 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    +trainer.max_ckpt_to_keep=5 \
    trainer.logger='["console","tensorboard","swanlab"]' \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.default_local_dir=${SAVE_DIR}/checkpoints/${project_name}/${experiment_name} \
    +trainer.tensorboard_dir=${TENSORBOARD_DIR} \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=100 \
    trainer.log_val_generations=100 \
    trainer.total_epochs=20 \
    ${KL_LOSS_ARGS[@]} ${LENGTH_REWARD_ARGS[@]} ${STRONG_TAG_ARGS[@]} ${FORMAT_ANSWER_PRODUCT_ARGS[@]} ${HYDRA_OVERRIDES[@]}
    
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
#     actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
#     actor_rollout_ref.rollout.enable_chunked_prefill=True \
#     actor_rollout_ref.rollout.enforce_eager=True \
#     actor_rollout_ref.rollout.free_cache_engine=False \
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
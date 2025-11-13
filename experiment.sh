#!/usr/bin/env bash

# 实验启动脚本：封装 run_qwen2_5_vl-7b.sh，并预留评测占位
set -euo pipefail


# bash experiment.sh --dataset simple --model /data_ali/shunian/models/Qwen2.5-VL-7B-Instruct


# ------------------------- 参数解析 -------------------------
ENGINE="vllm"
DATASET=""
MODEL_PATH=""
SAVE_DIR=""
LENGTH_REWARD=0 # 0=关闭，1=开启 2=half length reward
LR=""
KL_LOSS=0
ROLLOUT_N=4
ROLLOUT_N_SET=0
BATCH_SIZE=128
MAX_RESPONSE_LENGTH=1024
REWARD_MAX_LENGTH=""
FORMAT_ANSWER_PRODUCT=0
STRONG_TAG=0
REPETITION_PENALTY=1.0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --engine)
      ENGINE="$2"; shift 2 ;;
    --dataset)
      DATASET="$2"; shift 2 ;;
    --model)
      MODEL_PATH="$2"; shift 2 ;;
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
      REWARD_MAX_LENGTH="$2"; shift 2 ;;
    --strong_tag)
      STRONG_TAG=1; shift ;;
    --repetition_penalty)
      REPETITION_PENALTY="$2"; shift 2 ;;
    --format_answer_product)
      FORMAT_ANSWER_PRODUCT=1; shift ;;
    --)
      shift; break ;;
    *)
      # 其余参数（Hydra 覆盖等）原样向后传递
      break ;;
  esac
done

# 与 run 脚本保持一致的默认值
SAVE_DIR=${SAVE_DIR:-/data_ali/shunian/verl}
MODEL_PATH=${MODEL_PATH:-/data_ali/shunian/models/Qwen2.5-VL-7B-Instruct}

# 构造与 run 脚本一致的 experiment_name，便于后续评测直接对齐
dataset_tag=${DATASET:-iceberg}
model_tag=$(basename "${MODEL_PATH}")
project_name="verl_grpo_example_iceberg"
lr_tag=""
if [[ "${LENGTH_REWARD}" == "1" ]]; then
  lr_tag="_lenrw"
fi
if [[ "${LENGTH_REWARD}" == "2" ]]; then
  lr_tag="_half_lenrw"
fi
# KL 损失标签（仅在传入 --kl_loss 时拼接）
kl_tag=""
if [[ "${KL_LOSS}" == "1" ]]; then
  kl_tag="_kl"
fi
# 学习率标签（仅在传入 --lr 时拼接）
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
reward_max_length_tag=""
if [[ -n "${REWARD_MAX_LENGTH}" && "${REWARD_MAX_LENGTH}" != "${MAX_RESPONSE_LENGTH}" ]]; then
  reward_max_length_tag="_rmax${REWARD_MAX_LENGTH}"
fi
format_answer_product_tag=""
if [[ "${FORMAT_ANSWER_PRODUCT}" == "1" ]]; then
  format_answer_product_tag="_fmtprod"
fi
experiment_name="qwen2_5_vl_7b_${dataset_tag}_${model_tag}${lr_tag}${kl_tag}${lr_rate_tag}${rollout_n_tag}${reward_max_length_tag}${format_answer_product_tag}"
checkpoint_dir="${SAVE_DIR}/checkpoints/${project_name}/${experiment_name}"

SCRIPT_DIR="examples/grpo_trainer"

echo "[INFO] 将启动训练：engine=${ENGINE} dataset=${DATASET:-iceberg} model=${MODEL_PATH} save_dir=${SAVE_DIR} rollout_n=${ROLLOUT_N} max_response_length=${MAX_RESPONSE_LENGTH} reward_max_length=${REWARD_MAX_LENGTH:-${MAX_RESPONSE_LENGTH}} format_answer_product=${FORMAT_ANSWER_PRODUCT}"

# 组合额外运行开关
EXTRA_FLAGS=()
if [[ -n "${LENGTH_REWARD}" ]]; then
  EXTRA_FLAGS+=("--length_reward" "${LENGTH_REWARD}")
fi
if [[ -n "${LR}" ]]; then
  EXTRA_FLAGS+=("--lr" "${LR}")
fi
if [[ "${KL_LOSS}" == "1" ]]; then
  EXTRA_FLAGS+=("--kl_loss")
fi
if [[ "${ROLLOUT_N_SET}" == "1" ]]; then
  EXTRA_FLAGS+=("--rollout_n" "${ROLLOUT_N}")
fi
if [[ -n "${BATCH_SIZE}" ]]; then
  EXTRA_FLAGS+=("--batch_size" "${BATCH_SIZE}")
fi
if [[ -n "${MAX_RESPONSE_LENGTH}" ]]; then
  EXTRA_FLAGS+=("--max_response_length" "${MAX_RESPONSE_LENGTH}")
fi
if [[ -n "${REWARD_MAX_LENGTH}" ]]; then
  EXTRA_FLAGS+=("--reward_max_length" "${REWARD_MAX_LENGTH}")
fi
if [[ "${FORMAT_ANSWER_PRODUCT}" == "1" ]]; then
  EXTRA_FLAGS+=("--format_answer_product")
fi
if [[ "${STRONG_TAG}" == "1" ]]; then
  EXTRA_FLAGS+=("--strong_tag")
fi
if [[ -n "${REPETITION_PENALTY}" ]]; then
  EXTRA_FLAGS+=("--repetition_penalty" "${REPETITION_PENALTY}")
fi

bash "${SCRIPT_DIR}/run_qwen2_5_vl-7b.sh" \
  --engine "${ENGINE}" \
  --dataset "${DATASET}" \
  --model "${MODEL_PATH}" \
  --save_dir "${SAVE_DIR}" \
  "${EXTRA_FLAGS[@]}" \
  -- "$@"

# ------------------------- 训练后自动测试 -------------------------
echo "[INFO] 训练结束，开始自动测试新模型"

# 依据数据集选择默认的 parquet（可被外部覆盖）
case "${dataset_tag}" in
  simple)
    TEST_PARQUET="/data_ali/shunian/data/rl_data/iceberg/test_simple.parquet" ;;
  extra_simple)
    TEST_PARQUET="/data_ali/shunian/data/rl_data/iceberg/test_extra_simple.parquet" ;;
  thinking)
    TEST_PARQUET="/data_ali/shunian/data/rl_data/iceberg/test_thinking.parquet" ;;
  selected_simple)
    TEST_PARQUET="/data_ali/shunian/data/rl_data/iceberg/test_simple.parquet" ;;
  normal)
    TEST_PARQUET="/data_ali/shunian/data/rl_data/iceberg/test_normal.parquet" ;;
  *)
    TEST_PARQUET="/data_ali/shunian/data/rl_data/iceberg/test.parquet" ;;

esac

# 若训练脚本产出的目录结构遵循 examples 逻辑，则最新 global_step_XXX/actor/huggingface 存在
# 这里将 ckpt 根目录传入 test.sh，由其自动解析并定位到最终 HF 模型目录
bash /data_ali/shunian/verl/test.sh \
  --ckpt_dir "${checkpoint_dir}" \
  --dataset "${dataset_tag}" \
  --parquet "${TEST_PARQUET}" \
  --out_dir "/data_ali/shunian/verl/outputs/inference_run" \
  --dtype "bfloat16" \
  --num_samples "100" \
  --max_new_tokens "${MAX_RESPONSE_LENGTH}" \
  --repetition_penalty "${REPETITION_PENALTY}" \
  --force "1"

# 可在此处追加自动收集指标、产出报告的逻辑
# EVAL_HOOK_START
# EVAL_HOOK_END

exit 0



#!/usr/bin/env bash
set -euo pipefail

# 用法:
#   bash scripts/merge_fsdp_actor.sh [GLOBAL_STEP_DIR] [TARGET_DIR] [--parquet PATH] [--dtype auto|bfloat16|float16|float32] [--num_samples N] [--max_new_tokens N]
# 例子(仅合并):
  # bash scripts/merge_fsdp_actor.sh \
  #   /root/et/verl/checkpoints/verl_grpo_example_iceberg/qwen2_5_vl_3b_function_rm_iceberg/global_step_1500 \
  #   /root/et/verl/checkpoints/verl_grpo_example_iceberg/qwen2_5_vl_3b_function_rm_iceberg/global_step_1500_merged_hf
# 例子(合并后顺带推理测试):
#   bash scripts/merge_fsdp_actor.sh \
#     /root/et/verl/checkpoints/.../global_step_700 \
#     /root/et/verl/checkpoints/.../global_step_700_merged_hf \
#     --parquet /path/to/data.parquet --dtype bfloat16 --num_samples 2 --max_new_tokens 64

DEFAULT_GLOBAL_STEP_DIR="/root/et/verl/checkpoints/verl_grpo_example_iceberg/qwen2_5_vl_3b_function_rm_iceberg/global_step_700"

# 解析参数: 支持位置参数(前两个) + 可选长参数
PARQUET="/root/data/iceberg/test.parquet"
DTYPE="auto"
NUM_SAMPLES="2"
MAX_NEW_TOKENS="1024"

POS_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --parquet)
      PARQUET="$2"; shift 2 ;;
    --dtype)
      DTYPE="$2"; shift 2 ;;
    --num_samples)
      NUM_SAMPLES="$2"; shift 2 ;;
    --max_new_tokens)
      MAX_NEW_TOKENS="$2"; shift 2 ;;
    -h|--help)
      echo "用法: bash scripts/merge_fsdp_actor.sh [GLOBAL_STEP_DIR] [TARGET_DIR] [--parquet PATH] [--dtype auto|bfloat16|float16|float32] [--num_samples N] [--max_new_tokens N]"
      exit 0 ;;
    *)
      POS_ARGS+=("$1"); shift ;;
  esac
done

GLOBAL_STEP_DIR="${POS_ARGS[0]:-${DEFAULT_GLOBAL_STEP_DIR}}"
GLOBAL_STEP_DIR="${GLOBAL_STEP_DIR%/}"
ACTOR_DIR="${GLOBAL_STEP_DIR}/actor"

if [[ ! -d "${ACTOR_DIR}" ]]; then
  echo "[错误] 未找到 actor 目录: ${ACTOR_DIR}" >&2
  exit 1
fi

if [[ ! -f "${ACTOR_DIR}/fsdp_config.json" ]]; then
  echo "[错误] 未找到 fsdp_config.json: ${ACTOR_DIR}/fsdp_config.json" >&2
  exit 1
fi

# 默认输出目录为 <GLOBAL_STEP_DIR>_merged_hf
TARGET_DIR_DEFAULT="${GLOBAL_STEP_DIR}_merged_hf"
TARGET_DIR="${POS_ARGS[1]:-${TARGET_DIR_DEFAULT}}"

mkdir -p "${TARGET_DIR}"

echo "[信息] 合并 FSDP 检查点 -> HuggingFace 格式"
echo "[信息] 输入(Actor): ${ACTOR_DIR}"
echo "[信息] 输出目录:    ${TARGET_DIR}"

python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir "${ACTOR_DIR}" \
  --target_dir "${TARGET_DIR}"

echo "[完成] 模型已合并到: ${TARGET_DIR}"

# 若提供 parquet，则进行推理测试
if [[ -n "${PARQUET}" ]]; then
  if [[ ! -e "${PARQUET}" ]]; then
    echo "[警告] 指定的 parquet 路径不存在: ${PARQUET} (跳过推理测试)" >&2
    exit 0
  fi
  echo "[信息] 开始基于 parquet 抽样推理测试"
  echo "[信息] 模型:   ${TARGET_DIR}"
  echo "[信息] parquet: ${PARQUET}"
  python /root/et/verl/scripts/test_hf_inference_from_parquet.py \
    --model_dir "${TARGET_DIR}" \
    --parquet "${PARQUET}" \
    --dtype "${DTYPE}" \
    --num_samples "${NUM_SAMPLES}" \
    --max_new_tokens "${MAX_NEW_TOKENS}"
fi



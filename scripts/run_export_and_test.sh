#!/bin/bash
# set -euo pipefail

# 默认路径（可通过环境变量覆盖）
ACTOR_DIR=/root/et/verl/checkpoints/verl_grpo_example_iceberg/qwen2_5_vl_3b_function_rm_iceberg/global_step_500/actor
MODEL_DIR="${ACTOR_DIR}/huggingface"
PARQUET=/root/data/iceberg/test.parquet
DTYPE=bfloat16
NUM_SAMPLES=2
MAX_NEW_TOKENS=1024
BACKEND=${BACKEND:-gloo}
SINGLE_PROCESS=1
MODEL_DEVICE=cuda
GPU_ID=0

EXPORT_SCRIPT=/root/et/verl/scripts/export_fsdp_actor_to_hf.py
TEST_SCRIPT=/root/et/verl/scripts/test_hf_inference_from_parquet.py

if [[ ! -d "$ACTOR_DIR" ]]; then
  echo "[ERROR] ACTOR_DIR 不存在: $ACTOR_DIR" >&2
  exit 1
fi

if [[ ! -f "$PARQUET" ]]; then
  echo "[ERROR] PARQUET 不存在: $PARQUET" >&2
  exit 1
fi

if [[ ! -f "$EXPORT_SCRIPT" ]]; then
  echo "[ERROR] 找不到导出脚本: $EXPORT_SCRIPT" >&2
  exit 1
fi

if [[ ! -f "$TEST_SCRIPT" ]]; then
  echo "[ERROR] 找不到测试脚本: $TEST_SCRIPT" >&2
  exit 1
fi

# 推断需要的进程数（优先从 fsdp_config.json 读取 world_size）
if [[ -z "${NPROC_PER_NODE:-}" ]]; then
  if [[ -f "${ACTOR_DIR}/fsdp_config.json" ]]; then
    NPROC_PER_NODE=$(python - "$ACTOR_DIR" <<'PY'
import json, os, sys
actor_dir = sys.argv[1]
path = os.path.join(actor_dir, 'fsdp_config.json')
with open(path, 'r', encoding='utf-8') as f:
    print(json.load(f).get('world_size', 8))
PY
)
  else
    NPROC_PER_NODE=8
  fi
fi

# 选择 torchrun 或回退到 python -m torch.distributed.run
if command -v torchrun >/dev/null 2>&1; then
  TORCHRUN=torchrun
else
  TORCHRUN="python -m torch.distributed.run"
fi

echo "[INFO] 使用进程数: ${NPROC_PER_NODE}"
if [[ "$SINGLE_PROCESS" == "1" ]]; then
  echo "[INFO] 导出 FSDP -> HF: 单进程模式 (dtype=$DTYPE)"
else
  echo "[INFO] 导出 FSDP -> HF: $ACTOR_DIR -> $MODEL_DIR (dtype=$DTYPE, backend=$BACKEND)"
fi


if [[ "$SINGLE_PROCESS" == "1" ]]; then
  if [[ "$MODEL_DEVICE" == "cuda" ]]; then
    CUDA_VISIBLE_DEVICES="$GPU_ID" \
    python "$EXPORT_SCRIPT" \
      --actor_dir "$ACTOR_DIR" \
      --dtype "$DTYPE" \
      --out_dir "$MODEL_DIR" \
      --single_process \
      --model_device "$MODEL_DEVICE"
  else
    CUDA_VISIBLE_DEVICES= \
    python "$EXPORT_SCRIPT" \
      --actor_dir "$ACTOR_DIR" \
      --dtype "$DTYPE" \
      --out_dir "$MODEL_DIR" \
      --single_process \
      --model_device "$MODEL_DEVICE"
  fi
elif [[ "$BACKEND" == "gloo" ]]; then
  CUDA_VISIBLE_DEVICES= \
  ${TORCHRUN} --nproc_per_node="${NPROC_PER_NODE}" "$EXPORT_SCRIPT" \
    --actor_dir "$ACTOR_DIR" \
    --dtype "$DTYPE" \
    --out_dir "$MODEL_DIR" \
    --backend "$BACKEND"
else
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  ${TORCHRUN} --nproc_per_node="${NPROC_PER_NODE}" "$EXPORT_SCRIPT" \
    --actor_dir "$ACTOR_DIR" \
    --dtype "$DTYPE" \
    --out_dir "$MODEL_DIR" \
    --backend "$BACKEND"
fi

echo "[INFO] 开始基于 parquet 的推理测试: model_dir=$MODEL_DIR, parquet=$PARQUET"
python "$TEST_SCRIPT" \
  --model_dir "$MODEL_DIR" \
  --parquet "$PARQUET" \
  --dtype "$DTYPE" \
  --num_samples "$NUM_SAMPLES" \
  --max_new_tokens "$MAX_NEW_TOKENS"

echo "[DONE] 导出与测试完成"



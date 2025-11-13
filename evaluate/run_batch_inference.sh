#!/usr/bin/env bash
set -euo pipefail

# 批量运行 inference_qwen_merged.py，按 evaluate/checkpoints.md 的分组依次推理
# 使用方法：
#   bash evaluate/run_batch_inference.sh [可选：透传给 inference_qwen_merged.py 的参数]
# 例如：
#   bash evaluate/run_batch_inference.sh --flash-attn2

CHECKPOINTS_MD=${CHECKPOINTS_MD:-"evaluate/checkpoints.md"}
INFER_SCRIPT=${INFER_SCRIPT:-"evaluate/inference_qwen_merged.py"}
PYTHON_BIN=${PYTHON_BIN:-python}
# 添加参数
USE_VLLM="1"
VLLM_DTYPE="bfloat16"
VLLM_TP=1
VLLM_GPU_MEM=0.8
FLASH_ATTN2="1"
BATCH_SIZE=64
export CUDA_VISIBLE_DEVICES="0,1,2,3"

# 与 inference 脚本中默认保存路径保持一致；如需自定义可通过环境变量覆盖
EVAL_RESULTS_DIR=${EVAL_RESULTS_DIR:-"/data_ali/shunian/verl/evaluate/results"}
LOG_ROOT=${LOG_ROOT:-"evaluate/logs/batch"}
INDEX_CSV=${INDEX_CSV:-"evaluate/results/index.csv"}

mkdir -p "${LOG_ROOT}" || true
mkdir -p "$(dirname "${INDEX_CSV}")" || true

if [[ ! -f "${CHECKPOINTS_MD}" ]]; then
  echo "找不到 checkpoints 列表: ${CHECKPOINTS_MD}" >&2
  exit 1
fi

if [[ ! -f "${INFER_SCRIPT}" ]]; then
  echo "找不到推理脚本: ${INFER_SCRIPT}" >&2
  exit 1
fi

sanitize() {
  # 将路径/名称转换为安全的文件名
  echo "$1" | sed 's/[\\/:*?"<>| ]/_/g'
}

# 基于 CUDA_VISIBLE_DEVICES 与 VLLM_TP 生成 GPU 分组（每组大小 = VLLM_TP）
# 例如：CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 且 VLLM_TP=4 ->
# GPU_GROUPS=("0,1,2,3" "4,5,6,7")
declare -a GPU_GROUPS
build_gpu_groups() {
  local csv_devices=${CUDA_VISIBLE_DEVICES:-}
  local tp_size=${VLLM_TP:-1}
  GPU_GROUPS=()
  if [[ -z "$csv_devices" ]]; then
    return
  fi
  IFS=',' read -ra all_gpus <<< "$csv_devices"
  local total=${#all_gpus[@]}
  if [[ -z "$tp_size" || "$tp_size" -lt 1 ]]; then
    tp_size=1
  fi
  local i=0
  while (( i + tp_size - 1 < total )); do
    local grp="${all_gpus[i]}"
    local j=1
    while (( j < tp_size )); do
      grp+=",${all_gpus[i+j]}"
      ((j++))
    done
    GPU_GROUPS+=("$grp")
    ((i+=tp_size))
  done
  # 若无法整除且没有形成任何完整分组，则退化为单组使用全部可见卡
  if (( ${#GPU_GROUPS[@]} == 0 )); then
    GPU_GROUPS+=("$csv_devices")
  fi
}

ARGS="--vllm-dtype ${VLLM_DTYPE} --vllm-tp ${VLLM_TP} --vllm-gpu-mem ${VLLM_GPU_MEM} --batch-size ${BATCH_SIZE}"

if [[ "${USE_VLLM}" == "1" ]]; then
  ARGS+=" --use-vllm"
fi
if [[ "${FLASH_ATTN2}" == "1" ]]; then
  ARGS+=" --flash-attn2"
fi

# 生成 组名|checkpoint 路径 的记录流
mapfile -t RECORDS < <(awk '
  BEGIN { section = "未分组" }
  /^##[[:space:]]+/ {
    sub(/\r$/, "", $0);                   # 去除标题行可能的 CR
    section = substr($0, index($0,$2));
    next
  }
  /^[[:space:]]*$/ { next }
  # 以 / 开头的行认为是 checkpoint 路径
  # 注意：去除行首/行尾空白与可能存在的 CR 字符（Windows CRLF）
  /^[[:space:]]*\// {
    gsub(/^[[:space:]]+|[[:space:]]+$/, "", $0);  # 修正原先使用 \s 的错误写法
    sub(/\r$/, "", $0);                           # 显式去除行末 CR
    print section "|" $0
  }
' "${CHECKPOINTS_MD}")

# 按 checkpoint 路径去重，保持首次出现的顺序
declare -A SEEN_CKPT
declare -a DEDUP_RECORDS
for rec in "${RECORDS[@]}"; do
  CKPT=${rec#*|}
  # 忽略空行
  if [[ -z "${CKPT}" ]]; then
    continue
  fi
  if [[ -z "${SEEN_CKPT[$CKPT]:-}" ]]; then
    SEEN_CKPT[$CKPT]=1
    DEDUP_RECORDS+=("$rec")
  else
    echo "跳过重复的 checkpoint: ${CKPT}"
  fi
done

# 写入索引头（如不存在）
if [[ ! -f "${INDEX_CSV}" ]]; then
  echo "timestamp,group,checkpoint,output_json,log_file" > "${INDEX_CSV}"
fi

echo "即将处理 ${#DEDUP_RECORDS[@]} 条去重后的模型路径..."

# 生成 GPU 分组，并确定并发槽位数
build_gpu_groups
concurrency=${#GPU_GROUPS[@]}
if (( concurrency < 1 )); then
  concurrency=1
fi
echo "使用并发槽位数: ${concurrency} (GPU 分组: ${GPU_GROUPS[*]})"

# 以步长 = 并发槽位数 处理记录
for ((i=0; i<${#DEDUP_RECORDS[@]}; i+=concurrency)); do
  declare -a pids=()
  declare -a metas=()

  # 启动一批并发任务（每个槽位对应一个 GPU 组）
  for ((slot=0; slot<concurrency; slot++)); do
    idx=$((i+slot))
    [[ $idx -ge ${#DEDUP_RECORDS[@]} ]] && break

    rec="${DEDUP_RECORDS[$idx]}"
    GROUP=${rec%%|*}
    CKPT=${rec#*|}
    [[ -z "${CKPT}" ]] && continue

    group_slug=$(sanitize "${GROUP}")
    ckpt_slug=$(sanitize "${CKPT}")

    log_dir="${LOG_ROOT}/${group_slug}"
    mkdir -p "${log_dir}"

    ts=$(date +%Y%m%d_%H%M%S)
    log_file="${log_dir}/${ckpt_slug}_${ts}.log"

    # 若对应结果文件已经存在（任意时间戳），则跳过并写入索引
    existing_result=$(ls -1t "${EVAL_RESULTS_DIR}"/evaluation_results_merged_${ckpt_slug}_*.json 2>/dev/null | head -n1 || true)
    if [[ -n "${existing_result}" ]]; then
      csv_ts=$(date -Iseconds)
      echo "${csv_ts},${GROUP},${CKPT},${existing_result},(skipped)" >> "${INDEX_CSV}"
      echo "发现已有结果，跳过: ${CKPT} -> ${existing_result}"
      continue
    fi

    gpu_set="${GPU_GROUPS[$slot]}"

    echo "[${ts}] 并发槽${slot} GPU(${gpu_set}) 运行分组: ${GROUP} | 模型: ${CKPT}"

    set +e
    CUDA_VISIBLE_DEVICES="${gpu_set}" \
    "${PYTHON_BIN}" -u "${INFER_SCRIPT}" -c "${CKPT}" ${ARGS} \
      --cuda-visible-devices "${gpu_set}" \
      2>&1 | tee "${log_file}" &
    pids[$slot]=$!
    metas[$slot]="${GROUP}|${CKPT}|${log_file}"
    set -e
  done

  # 回收本批并发任务并写入索引
  for ((slot=0; slot<concurrency; slot++)); do
    [[ -z "${pids[$slot]:-}" ]] && continue
    set +e
    wait "${pids[$slot]}"
    ret=$?
    set -e

    IFS='|' read -r GROUP CKPT log_file <<< "${metas[$slot]}"

    # 尝试从日志中提取输出 JSON
    output_json=""
    if grep -q "结果已保存到:" "${log_file}"; then
      output_json=$(grep -oE '结果已保存到: .*' "${log_file}" | tail -n1 | sed 's/结果已保存到: //')
    fi
    # 兜底：从结果目录取最新文件
    if [[ -z "${output_json}" ]]; then
      if ls -1t "${EVAL_RESULTS_DIR}"/evaluation_results_merged_*.json >/dev/null 2>&1; then
        output_json=$(ls -1t "${EVAL_RESULTS_DIR}"/evaluation_results_merged_*.json | head -n1)
      fi
    fi

    csv_ts=$(date -Iseconds)
    echo "${csv_ts},${GROUP},${CKPT},${output_json},${log_file}" >> "${INDEX_CSV}"

    if [[ ${ret} -ne 0 ]]; then
      echo "警告：并发槽${slot} 推理返回非零代码(${ret})，已记录到索引。" >&2
    fi
  done
done

echo "全部推理完成。索引: ${INDEX_CSV}"


python evaluate/visualize_group_results.py \
  --index evaluate/results/index.csv \
  --out evaluate/summary
#!/usr/bin/env bash
set -euo pipefail

# 单模型测试脚本：在训练完成后对最终(最新)的 checkpoint 执行推理
# 参考 scripts/inference.sh 的推理流程与命名规则，精简为单模型用法
#
# 用法示例：
#   bash /data_ali/shunian/verl/test.sh \
#     --ckpt_dir "/data_ali/shunian/verl/checkpoints/verl_grpo_example_iceberg/simple_Qwen2.5-VL-7B-Instruct" \
#     --dataset simple \
#     --out_dir "/data_ali/shunian/verl/outputs/inference_run" \
#     --dtype bfloat16 --num_samples 100 --max_new_tokens 1024

# ------------------------- 参数解析 -------------------------
CKPT_DIR=""
DATASET=""
PARQUET=""
OUT_DIR="/data_ali/shunian/verl/outputs/inference_run"
DTYPE="bfloat16"
NUM_SAMPLES="100"
MAX_NEW_TOKENS="1024"
REPETITION_PENALTY="1.2"
FORCE="1"       # 1=强制重做

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ckpt_dir)
      CKPT_DIR="$2"; shift 2 ;;
    --dataset)
      DATASET="$2"; shift 2 ;;
    --parquet)
      PARQUET="$2"; shift 2 ;;
    --out_dir)
      OUT_DIR="$2"; shift 2 ;;
    --dtype)
      DTYPE="$2"; shift 2 ;;
    --num_samples)
      NUM_SAMPLES="$2"; shift 2 ;;
    --max_new_tokens)
      MAX_NEW_TOKENS="$2"; shift 2 ;;
    --repetition_penalty)
      REPETITION_PENALTY="$2"; shift 2 ;;
    --force)
      FORCE="$2"; shift 2 ;;
    --)
      shift; break ;;
    *)
      # 其余参数忽略
      shift ;;
  esac
done

if [[ -z "${CKPT_DIR}" ]]; then
  echo "[错误] 必须提供 --ckpt_dir" >&2
  exit 1
fi

# ------------------------- 工具函数 -------------------------
abs_path() {
  python3 - "$1" <<'PY'
import os,sys
p=os.path.abspath(sys.argv[1])
print(p)
PY
}

detect_num_gpus() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --list-gpus 2>/dev/null | wc -l | awk '{print $1}'
  else
    echo 0
  fi
}

get_free_port() {
  python3 - <<'PY'
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(('127.0.0.1', 0))
print(s.getsockname()[1])
s.close()
PY
}

dataset_tag_from_path() {
  local f="$1"
  local base
  base="$(basename "${f}")"
  echo "${base%.*}"
}

extract_step_token() {
  local p="$1"
  local IFS='/'
  read -ra parts <<< "$p"
  local seg
  for seg in "${parts[@]}"; do
    if [[ "$seg" =~ ^global_step_[0-9]+(_merged_hf)?$ ]]; then
      echo "${seg%%_merged_hf}"
      return 0
    fi
  done
  local parent
  parent="$(basename "$(dirname "$p")")"
  local grand
  grand="$(basename "$(dirname "$(dirname "$p")")")"
  if [[ "$parent" == "huggingface" && "$grand" =~ ^global_step_[0-9]+$ ]]; then
    echo "$grand"
    return 0
  fi
  echo ""
}

extract_model_name() {
  local p="$1"
  local IFS='/'
  read -ra parts <<< "$p"
  local i
  for ((i=0; i<${#parts[@]}; i++)); do
    if [[ "${parts[$i]}" =~ ^global_step_[0-9]+(_merged_hf)?$ ]]; then
      if (( i-1 >= 0 )); then
        echo "${parts[$i-1]}"
        return 0
      fi
    fi
  done
  local base
  base="$(basename "${p}")"
  if [[ "${base}" == "huggingface" || "${base}" == "actor" ]]; then
    local parent
    parent="$(basename "$(dirname "${p}")")"
    if [[ -n "${parent}" && "${parent}" != "." && "${parent}" != "actor" && ! "${parent}" =~ ^global_step_[0-9]+$ ]]; then
      echo "${parent}"
      return 0
    fi
    local grand
    grand="$(basename "$(dirname "$(dirname "${p}")")")"
    if [[ -n "${grand}" && "${grand}" != "." && "${grand}" != "actor" && ! "${grand}" =~ ^global_step_[0-9]+$ ]]; then
      echo "${grand}"
      return 0
    fi
  fi
  echo "$(basename "${p}")"
}

sanitize_name() {
  local p="$1"
  local candidate
  candidate="$(extract_model_name "${p}")" || true
  if [[ -n "${candidate}" ]]; then
    local base_name="${candidate}"
    local step
    step="$(extract_step_token "${p}")"
    if [[ -n "${step}" && "${base_name}" != *"${step}"* ]]; then
      echo "${base_name}_${step}"
      return 0
    fi
    echo "${base_name}"
    return 0
  fi
  local base
  base="$(basename "${p}")"
  local step
  step="$(extract_step_token "${p}")"
  if [[ -n "${step}" && "${base}" != *"${step}"* ]]; then
    echo "${base}_${step}"
    return 0
  fi
  echo "${base}"
}

# ------------------------- 解析/定位模型目录 -------------------------
resolve_model_dir() {
  local cdir="$1"
  local abs="$(abs_path "${cdir}")"

  # 直接是 HF 目录
  if [[ -f "${abs}/config.json" ]]; then
    echo "${abs}"
    return 0
  fi

  # global_step/huggingface 或 global_step/actor/huggingface
  if [[ -d "${abs}/huggingface" && -f "${abs}/huggingface/config.json" ]]; then
    echo "${abs}/huggingface"
    return 0
  fi
  if [[ -d "${abs}/actor/huggingface" && -f "${abs}/actor/huggingface/config.json" ]]; then
    echo "${abs}/actor/huggingface"
    return 0
  fi

  # 在子目录中查找最新 global_step
  local latest_step=""
  local latest_num=-1
  local d
  shopt -s nullglob
  for d in "${abs}"/global_step_*; do
    if [[ -d "$d" ]]; then
      local name
      name="$(basename "$d")"
      if [[ "$name" =~ ^global_step_([0-9]+) ]]; then
        local num
        num="${BASH_REMATCH[1]}"
        if (( num > latest_num )); then
          latest_num=$num
          latest_step="$d"
        fi
      fi
    fi
  done
  shopt -u nullglob

  if [[ -n "${latest_step}" ]]; then
    if [[ -d "${latest_step}/actor/huggingface" && -f "${latest_step}/actor/huggingface/config.json" ]]; then
      echo "${latest_step}/actor/huggingface"
      return 0
    fi
    if [[ -d "${latest_step}/huggingface" && -f "${latest_step}/huggingface/config.json" ]]; then
      echo "${latest_step}/huggingface"
      return 0
    fi
  fi

  echo ""  # 未找到
  return 1
}

# ------------------------- 数据集到 parquet 映射 -------------------------
parquet_from_dataset() {
  local ds="$1"
  case "$ds" in
    simple)
      echo "/data_ali/shunian/data/rl_data/iceberg/test_simple.parquet" ;;
    extra_simple)
      echo "/data_ali/shunian/data/rl_data/iceberg/test_extra_simple.parquet" ;;
    thinking)
      echo "/data_ali/shunian/data/rl_data/iceberg/test_thinking.parquet" ;;
    iceberg|default|*)
      echo "/data_ali/shunian/data/rl_data/iceberg/test.parquet" ;;
    selected_simple)
      echo "/data_ali/shunian/data/rl_data/iceberg/test_simple.parquet" ;;
    strong_tag)
      echo "/data_ali/shunian/data/rl_data/iceberg/test_strong_tag.parquet" ;;
  esac
}

# ------------------------- 主流程 -------------------------
CKPT_DIR_ABS="$(abs_path "${CKPT_DIR}")"
MODEL_DIR="$(resolve_model_dir "${CKPT_DIR_ABS}" || true)"
if [[ -z "${MODEL_DIR}" ]]; then
  echo "[错误] 未能在 ${CKPT_DIR_ABS} 解析到 HF 模型目录(huggingface/config.json)。" >&2
  exit 1
fi

if [[ -z "${PARQUET}" ]]; then
  PARQUET="$(parquet_from_dataset "${DATASET:-default}")"
fi
if [[ ! -f "${PARQUET}" ]]; then
  echo "[错误] 指定的 parquet 不存在: ${PARQUET}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"
RESULTS_DIR="${OUT_DIR}/results"
LOGS_DIR="${OUT_DIR}/logs"
mkdir -p "${RESULTS_DIR}" "${LOGS_DIR}"

NAME="$(sanitize_name "${MODEL_DIR}")"
DTAG="$(dataset_tag_from_path "${PARQUET}")"
NAME_DS="${NAME}__${DTAG}"
RESULT_LOG="${LOGS_DIR}/${NAME_DS}.log"
RESULT_JSON="${RESULTS_DIR}/${NAME_DS}.jsonl"

echo "[信息] 测试模型: ${MODEL_DIR}"
echo "[信息] 数据集: ${PARQUET}"
echo "[信息] 输出: ${RESULT_JSON} (日志: ${RESULT_LOG})"

if [[ "${FORCE}" != "1" && -s "${RESULT_LOG}" ]]; then
  echo "[信息] 已存在推理结果日志，跳过。"
  exit 0
fi

# 并发设置：优先从 CUDA_VISIBLE_DEVICES 获取数量
NUM_GPUS_ENV="${CUDA_VISIBLE_DEVICES:-}"
if [[ -n "${NUM_GPUS_ENV}" ]]; then
  NP=$(echo "${NUM_GPUS_ENV}" | awk -F',' '{print NF}')
else
  NP=$(detect_num_gpus)
fi
if [[ -z "${NP}" || "${NP}" -lt 1 ]]; then NP=1; fi
MPORT=$(get_free_port)

status=0
set +e
if [[ "${NP}" -gt 1 ]]; then
  echo "[信息] 使用多卡并发: nproc_per_node=${NP}, master_port=${MPORT}" > "${RESULT_LOG}" 2>&1
  TORCH_COMMAND=(torchrun --standalone --nproc_per_node "${NP}" --master_port "${MPORT}" /data_ali/shunian/verl/scripts/test_hf_inference_from_parquet.py)
  "${TORCH_COMMAND[@]}" \
    --model_dir "${MODEL_DIR}" \
    --parquet "${PARQUET}" \
    --dtype "${DTYPE}" \
    --num_samples "${NUM_SAMPLES}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --repetition_penalty "${REPETITION_PENALTY}" \
    --trust_remote_code \
    --resume \
    --json_out "${RESULT_JSON}" \
    >> "${RESULT_LOG}" 2>&1
  status=$?
  # 合并分片
  if [[ ${status} -eq 0 ]]; then
    parts_merged=0
    echo "[信息] 合并 rank 分片 JSON..." >> "${RESULT_LOG}" 2>&1
    tmp_out="${RESULT_JSON}.tmp"
    rm -f "${tmp_out}"
    if [[ -f "${RESULT_JSON}" ]]; then
      cat "${RESULT_JSON}" >> "${tmp_out}"
    fi
    for ((r=0; r<NP; r++)); do
      part=$(printf "%s.part%02dof%02d" "${RESULT_JSON}" "${r}" "${NP}")
      if [[ -f "${part}" ]]; then
        cat "${part}" >> "${tmp_out}"
        parts_merged=$((parts_merged+1))
      fi
    done
    if [[ -f "${tmp_out}" ]]; then
      mv "${tmp_out}" "${RESULT_JSON}"
      echo "[信息] 合并完成，共合并 ${parts_merged}/${NP} 个分片 -> ${RESULT_JSON}" >> "${RESULT_LOG}" 2>&1
      for ((r=0; r<NP; r++)); do
        part=$(printf "%s.part%02dof%02d" "${RESULT_JSON}" "${r}" "${NP}")
        [[ -f "${part}" ]] && rm -f "${part}"
      done
    else
      echo "[警告] 未发现分片文件，跳过合并。" >> "${RESULT_LOG}" 2>&1
    fi
  fi
else
  python3 /data_ali/shunian/verl/scripts/test_hf_inference_from_parquet.py \
    --model_dir "${MODEL_DIR}" \
    --parquet "${PARQUET}" \
    --dtype "${DTYPE}" \
    --num_samples "${NUM_SAMPLES}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --repetition_penalty "${REPETITION_PENALTY}" \
    --trust_remote_code \
    --resume \
    --json_out "${RESULT_JSON}" \
    > "${RESULT_LOG}" 2>&1
  status=$?
fi
set -e

if [[ $status -ne 0 ]]; then
  echo "[错误] 推理失败，请查看日志: ${RESULT_LOG}" >&2
  exit $status
fi

echo "[完成] 推理结束，结果位于: ${RESULT_JSON}"



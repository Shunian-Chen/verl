#!/usr/bin/env bash
set -euo pipefail

# 用法:
#   直接在本脚本顶部“配置区”填写参数后执行（尤其是 MODEL_PATHS 列表）：
#   bash scripts/inference.sh
#
# 说明:
# - 通过“配置区”的 MODEL_PATHS 指定要处理的一组 checkpoint 目录
# - 每个路径可以是以下三种之一, 本脚本会自动识别:
#   a) HuggingFace 模型目录(包含 config.json)
#   b) FSDP actor 目录(包含 fsdp_config.json)
#   c) global_step 目录(包含 actor 子目录, 其下含 fsdp_config.json)
# - 若是 b/c, 将自动调用 `python -m verl.model_merger merge --backend fsdp` 合并为 HF
# - 推理调用 `scripts/test_hf_inference_from_parquet.py`，输出保存到 OUT_DIR/logs/<name>.log
# - 合并后的 HF 模型保存到 OUT_DIR/models/<name>_merged_hf

# ===================== 配置区 =====================
OUT_DIR="/root/et/verl/outputs/inference_run"
PARQUET="/root/data/iceberg/test.parquet"
DTYPE="bfloat16"           # auto|bfloat16|float16|float32
NUM_SAMPLES="100"
MAX_NEW_TOKENS="512"
FORCE="1"                  # 1=强制重做(合并/推理)

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 多卡并发配置：auto 自动探测；或设置为显式数字（如 4）
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')
# 合并 rank 分片 JSON（torchrun 执行后会生成 .partXXofYY 文件）
MERGE_PARTS="1"
REMOVE_PARTS_AFTER_MERGE="1"

# 模型路径列表：直接编辑此数组（可混合 HF、actor、global_step 目录）
# 示例：
# MODEL_PATHS=(
#   "/root/et/verl/checkpoints/.../global_step_700"
#   "/root/et/verl/checkpoints/.../global_step_1500/actor"
#   "/root/et/verl/checkpoints/hf_models/qwen2"
# )
MODEL_PATHS=(
    "/root/et/verl/checkpoints/verl_grpo_example_iceberg/qwen2_5_vl_3b_function_rm_iceberg/global_step_100_merged_hf"
    "/root/et/verl/checkpoints/verl_grpo_example_iceberg/qwen2_5_vl_3b_function_rm_iceberg/global_step_700_merged_hf"
    "/root/et/verl/checkpoints/verl_grpo_example_iceberg/qwen2_5_vl_3b_function_rm_iceberg/global_step_1500_merged_hf"
    "/root/et/verl/checkpoints/verl_grpo_example_iceberg/qwen2_5_vl_3b_function_rm_iceberg_simple_example/global_step_700_merged_hf"
    "/root/et/verl/checkpoints/verl_grpo_example_iceberg/qwen2_5_vl_7b_function_rm_iceberg_no_kl/global_step_700/huggingface"
    "/root/et/verl/checkpoints/verl_grpo_example_iceberg/qwen2_5_vl_3b_function_rm_iceberg_freq_reward/global_step_800_merged_hf"
           )
# =================== 配置区结束 ===================

mkdir -p "${OUT_DIR}"
RESULTS_DIR="${OUT_DIR}/results"
LOGS_DIR="${OUT_DIR}/logs"
mkdir -p "${RESULTS_DIR}" "${LOGS_DIR}"

if [[ -z "${PARQUET}" || ! -e "${PARQUET}" ]]; then
  echo "[错误] parquet 不存在: ${PARQUET}" >&2
  exit 1
fi

# 收集待处理的 checkpoint 路径
CHECKPOINTS=()

# 仅从 MODEL_PATHS 收集
if [[ ${#MODEL_PATHS[@]} -gt 0 ]]; then
  for p in "${MODEL_PATHS[@]}"; do
    CHECKPOINTS+=("${p}")
  done
fi

if [[ ${#CHECKPOINTS[@]} -eq 0 ]]; then
  echo "[错误] 未提供任何 checkpoint 路径。请在脚本顶部配置 MODEL_PATHS。" >&2
  exit 1
fi

# 统一转为绝对路径，便于命名
abs_path() {
  python3 - "$1" <<'PY'
import os,sys
p=os.path.abspath(sys.argv[1])
print(p)
PY
}

# 探测 GPU 数量
detect_num_gpus() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --list-gpus 2>/dev/null | wc -l | awk '{print $1}'
  else
    echo 0
  fi
}

# 获取可用端口用于 torchrun
get_free_port() {
  python3 - <<'PY'
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(('127.0.0.1', 0))
print(s.getsockname()[1])
s.close()
PY
}

# 从路径中提取包含关键字的模型名
extract_model_name() {
  local p="$1"
  local IFS='/'
  read -ra parts <<< "$p"
  for seg in "${parts[@]}"; do
    if [[ "$seg" == *"function_rm_iceberg"* ]]; then
      echo "$seg"
      return 0
    fi
  done
  return 1
}

# 生成一个干净的名字用于文件/目录
sanitize_name() {
  local p="$1"
  local candidate
  candidate="$(extract_model_name "${p}")" || true
  if [[ -n "${candidate}" ]]; then
    # 若可提取到模型段，优先用之
    local base_name="${candidate}"
    # 附加 step 标记避免同模型多 step 覆盖
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
  # 如果是 actor 目录，则取其父目录名拼接
  if [[ "${base}" == "actor" ]]; then
    local parent
    parent="$(basename "$(dirname "${p}")")"
    local base_name="${parent}_actor"
    local step
    step="$(extract_step_token "${p}")"
    if [[ -n "${step}" && "${base_name}" != *"${step}"* ]]; then
      echo "${base_name}_${step}"
      return 0
    fi
    echo "${base_name}"
  elif [[ "${base}" == "huggingface" ]]; then
    # 若是 huggingface 子目录，利用上层 step 提取避免重名
    local step
    step="$(extract_step_token "${p}")"
    local parent
    parent="$(basename "$(dirname "${p}")")"
    local base_name
    if [[ -n "${parent}" && "${parent}" != "." ]]; then
      base_name="${parent}_hf"
    else
      base_name="huggingface"
    fi
    if [[ -n "${step}" && "${base_name}" != *"${step}"* ]]; then
      echo "${base_name}_${step}"
      return 0
    fi
    echo "${base_name}"
  else
    # 普通目录名；尝试附加 step
    local step
    step="$(extract_step_token "${p}")"
    if [[ -n "${step}" && "${base}" != *"${step}"* ]]; then
      echo "${base}_${step}"
      return 0
    fi
    echo "${base}"
  fi
}

# 从路径中提取 step 片段（如 global_step_700）
extract_step_token() {
  local p="$1"
  local IFS='/'
  read -ra parts <<< "$p"
  local seg
  for seg in "${parts[@]}"; do
    if [[ "$seg" =~ ^global_step_[0-9]+(_merged_hf)?$ ]]; then
      # 归一化: 去掉后缀 _merged_hf
      echo "${seg%%_merged_hf}"
      return 0
    fi
  done
  # 若未直接命中，检查上一级是否为 huggingface 且上上级为 global_step_xxx
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

echo "[信息] 将处理 ${#CHECKPOINTS[@]} 个 checkpoint"

INDEX_TSV="${OUT_DIR}/index.tsv"
if [[ ! -f "${INDEX_TSV}" ]]; then
  echo -e "name\tsource_path\tmodel_dir" > "${INDEX_TSV}"
fi

for ckpt in "${CHECKPOINTS[@]}"; do
  ckpt_abs="$(abs_path "${ckpt}")"
  if [[ ! -d "${ckpt_abs}" ]]; then
    echo "[警告] 非目录，跳过: ${ckpt_abs}" >&2
    continue
  fi

  name="$(sanitize_name "${ckpt_abs}")"
  src_type="unknown"
  model_dir=""
  actor_dir=""

  if [[ -f "${ckpt_abs}/config.json" ]]; then
    # 已是 HF 目录
    src_type="hf"
    model_dir="${ckpt_abs}"
  elif [[ -f "${ckpt_abs}/fsdp_config.json" ]]; then
    # FSDP actor 目录
    src_type="fsdp_actor"
    actor_dir="${ckpt_abs}"
  elif [[ -d "${ckpt_abs}/actor" && -f "${ckpt_abs}/actor/fsdp_config.json" ]]; then
    # global_step 目录
    src_type="global_step"
    actor_dir="${ckpt_abs}/actor"
  else
    echo "[警告] 无法识别的 checkpoint 结构，跳过: ${ckpt_abs}" >&2
    continue
  fi

  # 推理并保存日志
  result_log="${LOGS_DIR}/${name}.log"
  result_json="${RESULTS_DIR}/${name}.jsonl"
  if [[ "${FORCE}" != "1" && -s "${result_log}" ]]; then
    echo "[信息] 已存在推理结果，跳过: ${result_log}"
  else
    echo "[信息] 开始推理: ${name} -> ${result_log}"
    set +e
    if [[ -z "${model_dir}" ]]; then
      echo "[警告] 非 HF 模型目录，当前脚本不执行合并，跳过: ${ckpt_abs}" >> "${result_log}" 2>&1
      status=1
    else
      # 计算本次并发进程数
      NP="${NUM_GPUS}"
      if [[ "${NP}" == "auto" ]]; then
        NP=$(detect_num_gpus)
      fi
      if [[ -z "${NP}" || "${NP}" -lt 1 ]]; then NP=1; fi
      MPORT=$(get_free_port)

      if [[ "${NP}" -gt 1 ]]; then
        echo "[信息] 使用多卡并发: nproc_per_node=${NP}, master_port=${MPORT}" >> "${result_log}" 2>&1
        TORCH_COMMAND=(torchrun --standalone --nproc_per_node "${NP}" --master_port "${MPORT}" /root/et/verl/scripts/test_hf_inference_from_parquet.py)
        # 断点续传：若目标 json 已存在且非空，则保留并在本次只补增量
        # 注意：torchrun 会写入 .partXXofYY 分片，合并阶段会与旧文件合并
        "${TORCH_COMMAND[@]}" \
          --model_dir "${model_dir}" \
          --parquet "${PARQUET}" \
          --dtype "${DTYPE}" \
          --num_samples "${NUM_SAMPLES}" \
          --max_new_tokens "${MAX_NEW_TOKENS}" \
          --trust_remote_code \
          --resume \
          --json_out "${result_json}" \
          > "${result_log}" 2>&1
        status=$?
        # 合并分片
        if [[ ${status} -eq 0 && "${MERGE_PARTS}" == "1" ]]; then
          parts_merged=0
          echo "[信息] 合并 rank 分片 JSON..." >> "${result_log}" 2>&1
          tmp_out="${result_json}.tmp"
          rm -f "${tmp_out}"
          # 先将旧结果（若存在）并入 tmp，再追加各分片
          if [[ -f "${result_json}" ]]; then
            cat "${result_json}" >> "${tmp_out}"
          fi
          for ((r=0; r<NP; r++)); do
            part=$(printf "%s.part%02dof%02d" "${result_json}" "${r}" "${NP}")
            if [[ -f "${part}" ]]; then
              cat "${part}" >> "${tmp_out}"
              parts_merged=$((parts_merged+1))
            fi
          done
          if [[ -f "${tmp_out}" ]]; then
            mv "${tmp_out}" "${result_json}"
            echo "[信息] 合并完成，共合并 ${parts_merged}/${NP} 个分片 -> ${result_json}" >> "${result_log}" 2>&1
            if [[ "${REMOVE_PARTS_AFTER_MERGE}" == "1" ]]; then
              for ((r=0; r<NP; r++)); do
                part=$(printf "%s.part%02dof%02d" "${result_json}" "${r}" "${NP}")
                [[ -f "${part}" ]] && rm -f "${part}"
              done
            fi
          else
            echo "[警告] 未发现分片文件，跳过合并。" >> "${result_log}" 2>&1
          fi
        fi
      else
        python3 /root/et/verl/scripts/test_hf_inference_from_parquet.py \
          --model_dir "${model_dir}" \
          --parquet "${PARQUET}" \
          --dtype "${DTYPE}" \
          --num_samples "${NUM_SAMPLES}" \
          --max_new_tokens "${MAX_NEW_TOKENS}" \
          --trust_remote_code \
          --resume \
          --json_out "${result_json}" \
          > "${result_log}" 2>&1
        status=$?
      fi
    fi
    status=$?
    set -e
    if [[ $status -ne 0 ]]; then
      echo "[错误] 推理失败: ${name}，请查看日志 ${result_log}" >&2
      continue
    fi
  fi

  # 维护索引
  if ! grep -Fq "${name}\t${ckpt_abs}\t${model_dir}" "${INDEX_TSV}"; then
    echo -e "${name}\t${ckpt_abs}\t${model_dir}" >> "${INDEX_TSV}"
  fi
done

echo "[完成] 所有可用 checkpoint 已处理。结果位于: ${RESULTS_DIR}"



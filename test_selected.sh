#!/usr/bin/env bash
set -euo pipefail

# 批量测试脚本：针对指定的多个 checkpoint 逐一调用 test.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_SCRIPT="${SCRIPT_DIR}/test.sh"

usage() {
  cat <<'EOF'
用法: bash test_selected.sh [options] --ckpt_dir <path> [--ckpt_dir <path> ...]
       bash test_selected.sh [options] --ckpt_list <file>

选项:
  --ckpt_dir PATH      指定一个需要评测的 checkpoint 目录，可重复多次
  --ckpt_list FILE     从文件中读取需要评测的 checkpoint 列表（逐行，一个路径一行）
  --dataset NAME       传递给 test.sh 的 --dataset
  --parquet PATH       传递给 test.sh 的 --parquet
  --out_dir PATH       传递给 test.sh 的 --out_dir
  --dtype DTYPE        传递给 test.sh 的 --dtype
  --num_samples N      传递给 test.sh 的 --num_samples
  --max_new_tokens N   传递给 test.sh 的 --max_new_tokens
  --force FLAG         传递给 test.sh 的 --force
  --help | -h          显示本帮助信息
  -- ARG               之后的参数原样传递给 test.sh（可选）

说明:
  - 可以同时使用 --ckpt_dir 与 --ckpt_list，最终去重后依次执行。
  - 若任一 checkpoint 推理失败，脚本将在全部执行结束后返回非零退出码。
EOF
}

declare -a CKPT_DIRS=()
CKPT_LIST_FILE=""

DATASET=""
PARQUET=""
OUT_DIR=""
DTYPE=""
NUM_SAMPLES=""
MAX_NEW_TOKENS=""
FORCE_FLAG=""

declare -a EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ckpt_dir)
      [[ $# -ge 2 ]] || { echo "[错误] --ckpt_dir 缺少参数" >&2; exit 1; }
      CKPT_DIRS+=("$2"); shift 2 ;;
    --ckpt_list)
      [[ $# -ge 2 ]] || { echo "[错误] --ckpt_list 缺少参数" >&2; exit 1; }
      CKPT_LIST_FILE="$2"; shift 2 ;;
    --dataset)
      [[ $# -ge 2 ]] || { echo "[错误] --dataset 缺少参数" >&2; exit 1; }
      DATASET="$2"; shift 2 ;;
    --parquet)
      [[ $# -ge 2 ]] || { echo "[错误] --parquet 缺少参数" >&2; exit 1; }
      PARQUET="$2"; shift 2 ;;
    --out_dir)
      [[ $# -ge 2 ]] || { echo "[错误] --out_dir 缺少参数" >&2; exit 1; }
      OUT_DIR="$2"; shift 2 ;;
    --dtype)
      [[ $# -ge 2 ]] || { echo "[错误] --dtype 缺少参数" >&2; exit 1; }
      DTYPE="$2"; shift 2 ;;
    --num_samples)
      [[ $# -ge 2 ]] || { echo "[错误] --num_samples 缺少参数" >&2; exit 1; }
      NUM_SAMPLES="$2"; shift 2 ;;
    --max_new_tokens)
      [[ $# -ge 2 ]] || { echo "[错误] --max_new_tokens 缺少参数" >&2; exit 1; }
      MAX_NEW_TOKENS="$2"; shift 2 ;;
    --force)
      [[ $# -ge 2 ]] || { echo "[错误] --force 缺少参数" >&2; exit 1; }
      FORCE_FLAG="$2"; shift 2 ;;
    --help|-h)
      usage; exit 0 ;;
    --)
      shift
      EXTRA_ARGS=("$@")
      break ;;
    *)
      echo "[错误] 未识别的参数: $1" >&2
      usage
      exit 1 ;;
  esac
done

if [[ -n "${CKPT_LIST_FILE}" ]]; then
  if [[ ! -f "${CKPT_LIST_FILE}" ]]; then
    echo "[错误] 指定的 --ckpt_list 文件不存在: ${CKPT_LIST_FILE}" >&2
    exit 1
  fi
  mapfile -t LIST_ITEMS < <(python3 - <<'PY' "${CKPT_LIST_FILE}"
import sys
from pathlib import Path

path = Path(sys.argv[1])
for raw in path.read_text().splitlines():
    stripped = raw.strip()
    if not stripped or stripped.startswith('#'):
        continue
    print(stripped)
PY
)
  if [[ ${#LIST_ITEMS[@]} -gt 0 ]]; then
    CKPT_DIRS+=("${LIST_ITEMS[@]}")
  fi
fi

if [[ ${#CKPT_DIRS[@]} -eq 0 ]]; then
  echo "[错误] 至少需要通过 --ckpt_dir 或 --ckpt_list 提供一个 checkpoint" >&2
  usage
  exit 1
fi

declare -a UNIQUE_CKPTS=()
declare -A CKPT_SEEN=()
for CKPT in "${CKPT_DIRS[@]}"; do
  if [[ -z "${CKPT_SEEN["${CKPT}"]+x}" ]]; then
    CKPT_SEEN["${CKPT}"]=1
    UNIQUE_CKPTS+=("${CKPT}")
  fi
done
CKPT_DIRS=("${UNIQUE_CKPTS[@]}")

if [[ ! -f "${TEST_SCRIPT}" ]]; then
  echo "[错误] 未找到 test.sh 脚本: ${TEST_SCRIPT}" >&2
  exit 1
fi

declare -a PASS_ARGS=()
[[ -n "${DATASET}" ]] && PASS_ARGS+=("--dataset" "${DATASET}")
[[ -n "${PARQUET}" ]] && PASS_ARGS+=("--parquet" "${PARQUET}")
[[ -n "${OUT_DIR}" ]] && PASS_ARGS+=("--out_dir" "${OUT_DIR}")
[[ -n "${DTYPE}" ]] && PASS_ARGS+=("--dtype" "${DTYPE}")
[[ -n "${NUM_SAMPLES}" ]] && PASS_ARGS+=("--num_samples" "${NUM_SAMPLES}")
[[ -n "${MAX_NEW_TOKENS}" ]] && PASS_ARGS+=("--max_new_tokens" "${MAX_NEW_TOKENS}")
[[ -n "${FORCE_FLAG}" ]] && PASS_ARGS+=("--force" "${FORCE_FLAG}")

declare -a SUCCESS_LIST=()
declare -a FAIL_LIST=()

echo "[信息] 将依次测试 ${#CKPT_DIRS[@]} 个 checkpoint"

for CKPT in "${CKPT_DIRS[@]}"; do
  echo "[信息] 开始测试: ${CKPT}"
  set +e
  bash "${TEST_SCRIPT}" --ckpt_dir "${CKPT}" "${PASS_ARGS[@]}" "${EXTRA_ARGS[@]}"
  STATUS=$?
  set -e
  if [[ ${STATUS} -eq 0 ]]; then
    SUCCESS_LIST+=("${CKPT}")
    echo "[信息] 完成: ${CKPT}"
  else
    FAIL_LIST+=("${CKPT} (exit ${STATUS})")
    echo "[警告] 失败: ${CKPT} (exit ${STATUS})"
  fi
done

echo "[信息] 批量测试结束"

if [[ ${#SUCCESS_LIST[@]} -gt 0 ]]; then
  echo "[信息] 成功 (${#SUCCESS_LIST[@]}):"
  for item in "${SUCCESS_LIST[@]}"; do
    echo "  - ${item}"
  done
fi

if [[ ${#FAIL_LIST[@]} -gt 0 ]]; then
  echo "[错误] 失败 (${#FAIL_LIST[@]}):"
  for item in "${FAIL_LIST[@]}"; do
    echo "  - ${item}"
  done
  exit 1
fi

exit 0



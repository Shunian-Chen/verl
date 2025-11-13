#!/usr/bin/env bash
# ============================================================
# H200 (SM90) + CUDA 12.8 环境：PyTorch / FlashAttention / FlashInfer
# 目标：PyTorch 与系统 CUDA 对齐为 cu128；flash-attn/flashinfer 源码本地编译
# tested on: Python 3.10+, nvcc 12.8, gcc 11/12/13
# ============================================================
set -euo pipefail

# ---------- 可调开关 ----------
: "${USE_SGLANG:=1}"           # 是否安装 sglang（按你原脚本，默认装）
: "${USE_VLLM:=1}"             # 是否安装 vLLM
: "${USE_MEGATRON:=0}"         # TransformerEngine/Megatron，默认关闭以避免下载私源失败
: "${USE_FLASHINFER_AOT:=1}"   # 安装 flashinfer-cubin 以加速首启（可选）
: "${USE_FLASHINFER_JITCACHE:=1}"  # 安装 JIT cache（可选）
: "${MAX_JOBS:=16}"

# ---------- 版本锁 ----------
TORCH_VER="2.7.1"              # 你可改为 2.7.0；两者均有 cu128
TORCHVISION_VER="0.22.1"       # 与 2.7.1 搭配；若用 2.7.0 则改 0.22.0
TORCHAUDIO_VER="2.7.1"
SGLANG_VER="0.4.10.post2"
VLLM_VER="0.10.0"
FLASH_ATTN_VER="2.7.4.post1"   # 经验证更稳的版本

# ---------- CUDA / 架构 ----------
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"   # nvcc 12.8 根目录
export TORCH_CUDA_ARCH_LIST="9.0+PTX"              # 供 cpp_extension 使用
export FLASH_ATTN_CUDA_ARCHS="90"                  # flash-attn 专用（H200 = sm_90）
export CMAKE_CUDA_ARCHITECTURES="90"               # CMake 项

# ---------- 可选：PyTorch 安装镜像（默认官方，需外网） ----------
# 若网络不佳，可改用 -f 阿里 PyTorch wheels 页面（示例见下）
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
ALIYUN_TORCH_FIND_LINKS="${ALIYUN_TORCH_FIND_LINKS:-https://mirrors.aliyun.com/pytorch-wheels/cu128/}"

# echo "== [0] 基础工具链 =="
# python -V || true
# nvcc --version || true
# pip install -U --no-cache-dir "pip>=25.0" "setuptools>=75" "wheel>=0.43" cmake ninja build pybind11 packaging

# echo "== [1] 统一安装 PyTorch 到 cu128（CUDA 12.8） =="
# # 先清理旧版本，避免 cu126/组件残留
# pip uninstall -y torch torchvision torchaudio || true

# # 优先官方 cu128 索引；若失败，将注释掉下面一行，改用 -f 镜像命令
# pip install --no-cache-dir --index-url "${TORCH_INDEX_URL}" \
#   "torch==${TORCH_VER}" "torchvision==${TORCHVISION_VER}" "torchaudio==${TORCHAUDIO_VER}" || {
#   echo "[INFO] 官方索引失败，尝试阿里镜像（-f）..."
#   pip install --no-cache-dir -f "${ALIYUN_TORCH_FIND_LINKS}" \
#     "torch==${TORCH_VER}" "torchvision==${TORCHVISION_VER}" "torchaudio==${TORCHAUDIO_VER}"
# }

# python - <<'PY'
# import torch, platform
# print("Torch:", torch.__version__, "CUDA from torch:", torch.version.cuda, "Py:", platform.python_version())
# assert torch.version.cuda and torch.cuda.is_available(), "PyTorch GPU/CUDA 不可用"
# PY

# echo "== [2] 其他通用依赖（与原脚本整体保持一致，略作修订） =="
# # 注意：不再手动安装 nvidia-cudnn-cu12，避免与 torch wheel 自带冲突
# pip install --no-cache-dir \
#   'transformers[hf_xet]>=4.55.4' accelerate datasets peft hf-transfer \
#   'numpy<2.0.0' 'pyarrow>=19.0.1' pandas \
#   'ray[default]' click==8.2.1 codetiming hydra-core pylatexenc qwen-vl-utils wandb dill pybind11 liger-kernel mathruler blobfile \
#   pytest py-spy pyext pre-commit ruff tensorboard matplotlib mbridge 'xgrammar==0.1.25' \
#   'nvidia-ml-py>=12.560.30' 'fastapi[standard]>=0.115.0' 'optree>=0.13.0' 'pydantic>=2.9' 'grpcio>=1.62.1' \
#   -U ms-swift

# if [[ "${USE_SGLANG}" == "1" ]]; then
#   echo "== [2.1] 安装 SGLang（按你原定版本） =="
#   pip install --no-cache-dir "sglang[all]==${SGLANG_VER}"
#   pip install --no-cache-dir torch-memory-saver
# fi

# if [[ "${USE_VLLM}" == "1" ]]; then
#   echo "== [2.2] 安装 vLLM（CUDA 12.8 二进制已官方支持） =="
#   pip install --no-cache-dir "vllm==${VLLM_VER}"
# fi

# echo "== [3] FlashAttention 源码本地编译（强制跳过预编译 wheel 下载） =="
# # 关键：强制本地构建，避免 setup.py 先去 GitHub 抓 wheel 导致失败
# export FLASH_ATTENTION_FORCE_BUILD=TRUE
# export FLASH_ATTENTION_SKIP_CUDA_BUILD=FALSE
# export FLASH_ATTENTION_FORCE_CXX11_ABI=TRUE
# # 清理残留
# pip uninstall -y flash-attn flash_attn || true
# # 使用 PEP517 + 不隔离构建，确保能看到当前 torch/cuda
# MAX_JOBS="${MAX_JOBS}" \
#   pip install --no-cache-dir --no-binary :all: --no-build-isolation --use-pep517 \
#   "flash-attn==${FLASH_ATTN_VER}" -v

echo "== [4] FlashInfer 源码安装 +（可选）AOT/JIT Cache =="
# 若已有目录先清理
# rm -rf flashinfer
# git clone --depth 1 --recursive https://github.com/flashinfer-ai/flashinfer.git
# pushd flashinfer
# # 指定 Hopper 架构
# export CMAKE_CUDA_ARCHITECTURES="90"
# python -m pip install -U build --no-cache-dir
# python -m build -v --no-isolation --wheel
# pip install --no-cache-dir dist/flashinfer_python-*.whl



# if [[ "${USE_FLASHINFER_AOT}" == "1" ]]; then
#   echo "---- 安装 flashinfer-cubin（AOT 预编译内核） ----"
#   pushd flashinfer-cubin
#   python -m build --no-isolation --wheel
#   pip install dist/*.whl
#   popd
# fi

# if [[ "${USE_FLASHINFER_JITCACHE}" == "1" ]]; then
#   echo "---- 安装 flashinfer-jit-cache（cu128 索引） ----"
#   # 官方索引：如需 cu129/cu130，请替换 URL 后缀
#   pip install flashinfer-jit-cache --index-url https://flashinfer.ai/whl/cu128 || true
# fi
# popd

if [[ "${USE_MEGATRON}" == "1" ]]; then
  echo "== [5] TransformerEngine/Megatron（可选） =="
  # 注意：默认关闭；按需启用并自行提供可访问的 TE_WHL/URL
  TE_WHL_NAME=${TE_WHL_NAME:-"transformer_engine-2.4.0.dev0+d3eeda0-cp310-cp310-linux_x86_64.whl"}
  TE_DOWNLOAD_LINK=${TE_DOWNLOAD_LINK:-"http://example.com/path/${TE_WHL_NAME}"}  # 请替换为可访问地址
  rm -f "${TE_WHL_NAME}" || true
  wget -O "${TE_WHL_NAME}" "${TE_DOWNLOAD_LINK}"
  pip install --no-deps "${TE_WHL_NAME}"
  rm -f "${TE_WHL_NAME}"
  # 如需 Megatron-LM：
  # pip install --no-deps git+https://github.com/NVIDIA/Megatron-LM.git@core_v0.12.2
fi

echo "== [6] OpenCV（可选修复） =="
pip install --no-cache-dir opencv-python opencv-fixer
python - <<'PY'
from opencv_fixer import AutoFix; AutoFix()
PY

echo "== [7] 构建验证 =="
python - <<'PY'
import torch
print("Torch:", torch.__version__, "CUDA from torch:", torch.version.cuda)
import flash_attn
print("flash-attn import OK")
try:
    import flashinfer
    print("flashinfer import OK")
except Exception as e:
    print("flashinfer import WARN:", e)
PY

# FlashInfer CLI 自检（若安装了 CLI）
command -v flashinfer >/dev/null && flashinfer show-config || true

echo "==> 完成：Torch=${TORCH_VER}+cu128, vLLM=${VLLM_VER}, sglang=${SGLANG_VER}, FlashAttention=${FLASH_ATTN_VER}(src), FlashInfer(src)"

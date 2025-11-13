import os
from pathlib import Path
from typing import Optional, Tuple

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def sanitize_filename(name: str) -> str:
    """将路径/名称转换为安全的文件名（与 bash 脚本保持一致的替换规则）。"""
    for ch in ['\\', '/', ':', '*', '?', '"', '<', '>', '|', ' ']:
        name = name.replace(ch, '_')
    return name


def derive_model_slug_from_ckpt(ckpt_path: str) -> str:
    """基于 checkpoint 路径/名称生成稳定的别名（slug）。"""
    if not ckpt_path:
        return 'unknown_model'
    try:
        p = Path(ckpt_path)
        for part in reversed(list(p.parts)):
            if isinstance(part, str) and part.startswith('qwen2_5_vl_'):
                return sanitize_filename(part)
    except Exception:
        pass

    base = os.path.basename(os.path.normpath(ckpt_path)) or ckpt_path
    return sanitize_filename(base)


def derive_model_display_name(ckpt_path: str) -> str:
    """从 checkpoint 路径推导用于展示/记录的模型名称。"""
    if not ckpt_path:
        return 'unknown_model'
    p = Path(ckpt_path)
    parts = list(p.parts)

    for part in reversed(parts):
        if isinstance(part, str) and part.startswith('qwen2_5_vl_'):
            return sanitize_filename(part)

    for i, part in enumerate(parts):
        if isinstance(part, str) and part.startswith('global_step_'):
            if i - 1 >= 0:
                return sanitize_filename(parts[i - 1])
            return sanitize_filename(part)

    while parts and parts[-1] in ('huggingface', 'actor'):
        parts.pop()

    if parts:
        return sanitize_filename(parts[-1])
    return 'unknown_model'


def load_model_and_processor(args) -> Tuple[object, object]:
    """加载模型和处理器（支持 vLLM 与 HuggingFace 路径，失败自动回退）。"""
    if getattr(args, 'use_vllm', False):
        if args.cpu_only:
            print("[vLLM] 检测到 --cpu-only，vLLM 不支持，将自动回退到 HuggingFace 推理。")
            args.use_vllm = False
        else:
            print(f"[vLLM] 正在从 '{args.checkpoint_path}' 加载引擎...")
            try:
                from vllm import LLM
            except Exception as e:
                print(f"[vLLM] 未安装或导入失败，将自动回退到 HuggingFace。错误: {e}")
                args.use_vllm = False
            else:
                try:
                    model = LLM(
                        model=args.checkpoint_path,
                        dtype=args.vllm_dtype,
                        tensor_parallel_size=int(args.vllm_tp) if args.vllm_tp else 1,
                        gpu_memory_utilization=float(args.vllm_gpu_mem),
                        trust_remote_code=True,
                    )
                    processor = AutoProcessor.from_pretrained(args.checkpoint_path)
                    print("[vLLM] 引擎与处理器加载完成。")
                    return model, processor
                except Exception as e:
                    if int(getattr(args, 'vllm_tp', 1)) > 1:
                        try:
                            print(f"[vLLM] 多卡初始化失败，将降级为单卡重试。原始错误: {e}")
                            model = LLM(
                                model=args.checkpoint_path,
                                dtype=args.vllm_dtype,
                                tensor_parallel_size=1,
                                gpu_memory_utilization=float(args.vllm_gpu_mem),
                                trust_remote_code=True,
                            )
                            processor = AutoProcessor.from_pretrained(args.checkpoint_path)
                            print("[vLLM] 单卡重试成功。")
                            return model, processor
                        except Exception as e2:
                            print(f"[vLLM] 单卡重试仍失败，将回退到 HuggingFace。错误: {e2}")
                            args.use_vllm = False
                    else:
                        print(f"[vLLM] 加载失败，将自动回退到 HuggingFace。错误: {e}")
                        args.use_vllm = False

    device_map = 'cpu' if args.cpu_only else 'auto'
    print(f"正在从 '{args.checkpoint_path}' 加载模型...")
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": device_map
    }
    if args.flash_attn2:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.checkpoint_path, **model_kwargs)
    processor = AutoProcessor.from_pretrained(args.checkpoint_path)
    print("模型和处理器加载完成。")
    return model, processor



#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Qwen2.5-VL 稳健推理脚本（不依赖 AutoModelForConditionalGeneration）
- 直接使用 Qwen2_5_VLForConditionalGeneration（官方推荐）
- messages -> apply_chat_template -> (qwen_vl_utils)process_vision_info -> processor -> generate
- 兼容 parquet 的 images.bytes / images.path；可选 JSONL 输出；可选 verl 奖励
"""

from __future__ import annotations
# -*- coding: utf-8 -*-

"""
Qwen2.5-VL 稳健推理脚本（不依赖 AutoModelForConditionalGeneration）
- 直接使用 Qwen2_5_VLForConditionalGeneration（官方推荐）
- messages -> apply_chat_template -> (qwen_vl_utils)process_vision_info -> processor -> generate
- 兼容 parquet 的 images.bytes / images.path；可选 JSONL 输出；可选 verl 奖励
"""

from __future__ import annotations
import argparse
import os
import json
import json
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Set
import time
from typing import Any, Dict, List, Optional, Tuple, Set
import time

import torch
import torch.distributed as dist
import torch.distributed as dist
import datasets as hfds
from PIL import Image

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoProcessor,
    AutoModelForCausalLM,   # 仅做纯文本兜底
    AutoModel,              # 最终兜底（若被迫回退则报错）
    Qwen2_5_VLForConditionalGeneration,
)

# 可选：官方多模工具
_QWEN_UTILS_AVAIL = True

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoProcessor,
    AutoModelForCausalLM,   # 仅做纯文本兜底
    AutoModel,              # 最终兜底（若被迫回退则报错）
    Qwen2_5_VLForConditionalGeneration,
)

# 可选：官方多模工具
_QWEN_UTILS_AVAIL = True
try:
    from qwen_vl_utils import process_vision_info  # pip install qwen-vl-utils
except Exception:
    _QWEN_UTILS_AVAIL = False
    process_vision_info = None  # type: ignore
    from qwen_vl_utils import process_vision_info  # pip install qwen-vl-utils
except Exception:
    _QWEN_UTILS_AVAIL = False
    process_vision_info = None  # type: ignore


# ---------------------------- 小工具 ----------------------------

def get_torch_dtype(dtype_str: str) -> Any:
    mapping = {
# ---------------------------- 小工具 ----------------------------

def get_torch_dtype(dtype_str: str) -> Any:
    mapping = {
        "auto": "auto",
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return mapping.get(dtype_str.lower(), "auto")


def is_vlm_config(cfg: AutoConfig) -> bool:
    archs = (getattr(cfg, "architectures", None) or [])
    model_type = (getattr(cfg, "model_type", "") or "").lower()
    any_vl = any(("vl" in (a or "").lower()) for a in archs)
    return any_vl or model_type.endswith("_vl") or "qwen2_5_vl" in model_type


def pil_from_any(x: Any) -> Optional[Image.Image]:
    try:
        if isinstance(x, (bytes, bytearray)):
            return Image.open(BytesIO(x)).convert("RGB")
        if isinstance(x, str) and os.path.exists(x):
            return Image.open(x).convert("RGB")
    except Exception:
        return None
    return None


def extract_text_from_prompt(prompt_list: List[Dict[str, Any]]) -> str:
    for turn in prompt_list or []:
        role = (turn.get("role") or "").lower()
        content = turn.get("content") or ""
        if role == "user" and isinstance(content, str) and content.strip():
            return content
    if prompt_list:
        c = prompt_list[0].get("content") or ""
        return c if isinstance(c, str) else ""
    return ""


def build_messages_and_images(example: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Image.Image], List[str]]:
    """
    将一条样本转换成 Qwen 消息格式与 PIL 图像列表
    支持字段：
      - prompt: list of {role, content}
      - images: list of {bytes?, path?}
    """
    prompt_list = example.get("prompt") or []
    text_content = extract_text_from_prompt(prompt_list)

    pil_images: List[Image.Image] = []
    image_paths: List[str] = []
    for it in (example.get("images") or []):
        if isinstance(it, dict):
            if "bytes" in it and isinstance(it["bytes"], (bytes, bytearray)) and len(it["bytes"]) > 0:
                im = pil_from_any(it["bytes"])
                if im is not None:
                    pil_images.append(im)
            if "path" in it and isinstance(it["path"], str) and it["path"]:
                image_paths.append(it["path"])
                im = pil_from_any(it["path"])
                if im is not None:
                    pil_images.append(im)

    content_items: List[Dict[str, Any]] = []
    if text_content:
        content_items.append({"type": "text", "text": text_content})
    for im in pil_images:
        content_items.append({"type": "image", "image": im})
    if not content_items:
        content_items = [{"type": "text", "text": ""}]

    messages = [{"role": "user", "content": content_items}]
    return messages, pil_images, image_paths


def extract_ground_truth(example: Dict[str, Any]) -> Optional[str]:
    rm = example.get("reward_model") or {}
    gt = rm.get("ground_truth")
    if isinstance(gt, list) and gt:
        return str(gt[0])
    if isinstance(gt, (str, int, float)):
        return str(gt)

    extra = example.get("extra_info") or {}
    ans = extra.get("answer")
    if isinstance(ans, (str, int, float)):
        return str(ans)

    top = example.get("answer")
    if isinstance(top, (str, int, float)):
        return str(top)

    return None


# ---------------------------- 加载与推理 ----------------------------

def get_dist_info() -> Tuple[int, int, int]:
    """从环境变量获取分布式信息。
    返回 (rank, local_rank, world_size)。若未分布式，返回 (0, -1, 1)。
    """
    try:
        rank = int(os.environ.get("RANK", "0"))
    except Exception:
        rank = 0
    try:
        local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    except Exception:
        local_rank = -1
    try:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
    except Exception:
        world_size = 1
    return rank, local_rank, world_size


def setup_distributed_if_needed() -> Tuple[int, int, int]:
    """如需则初始化 torch.distributed，并返回 (rank, local_rank, world_size)。"""
    rank, local_rank, world_size = get_dist_info()
    if world_size > 1 and not (dist.is_available() and dist.is_initialized()):
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    return rank, local_rank, world_size

def load_model_and_processor(
    model_dir: str,
    dtype: str = "auto",
    device: Optional[str] = None,
    trust_remote_code: bool = True,
):
    cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=trust_remote_code)
    torch_dtype = get_torch_dtype(dtype)

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True, trust_remote_code=trust_remote_code)
    return mapping.get(dtype_str.lower(), "auto")


def is_vlm_config(cfg: AutoConfig) -> bool:
    archs = (getattr(cfg, "architectures", None) or [])
    model_type = (getattr(cfg, "model_type", "") or "").lower()
    any_vl = any(("vl" in (a or "").lower()) for a in archs)
    return any_vl or model_type.endswith("_vl") or "qwen2_5_vl" in model_type


def pil_from_any(x: Any) -> Optional[Image.Image]:
    try:
        if isinstance(x, (bytes, bytearray)):
            return Image.open(BytesIO(x)).convert("RGB")
        if isinstance(x, str) and os.path.exists(x):
            return Image.open(x).convert("RGB")
    except Exception:
        return None
    return None


def extract_text_from_prompt(prompt_list: List[Dict[str, Any]]) -> str:
    for turn in prompt_list or []:
        role = (turn.get("role") or "").lower()
        content = turn.get("content") or ""
        if role == "user" and isinstance(content, str) and content.strip():
            return content
    if prompt_list:
        c = prompt_list[0].get("content") or ""
        return c if isinstance(c, str) else ""
    return ""


def build_messages_and_images(example: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Image.Image], List[str]]:
    """
    将一条样本转换成 Qwen 消息格式与 PIL 图像列表
    支持字段：
      - prompt: list of {role, content}
      - images: list of {bytes?, path?}
    """
    prompt_list = example.get("prompt") or []
    text_content = extract_text_from_prompt(prompt_list)

    pil_images: List[Image.Image] = []
    image_paths: List[str] = []
    for it in (example.get("images") or []):
        if isinstance(it, dict):
            if "bytes" in it and isinstance(it["bytes"], (bytes, bytearray)) and len(it["bytes"]) > 0:
                im = pil_from_any(it["bytes"])
                if im is not None:
                    pil_images.append(im)
            if "path" in it and isinstance(it["path"], str) and it["path"]:
                image_paths.append(it["path"])
                im = pil_from_any(it["path"])
                if im is not None:
                    pil_images.append(im)

    content_items: List[Dict[str, Any]] = []
    if text_content:
        content_items.append({"type": "text", "text": text_content})
    for im in pil_images:
        content_items.append({"type": "image", "image": im})
    if not content_items:
        content_items = [{"type": "text", "text": ""}]

    messages = [{"role": "user", "content": content_items}]
    return messages, pil_images, image_paths


def extract_ground_truth(example: Dict[str, Any]) -> Optional[str]:
    rm = example.get("reward_model") or {}
    gt = rm.get("ground_truth")
    if isinstance(gt, list) and gt:
        return str(gt[0])
    if isinstance(gt, (str, int, float)):
        return str(gt)

    extra = example.get("extra_info") or {}
    ans = extra.get("answer")
    if isinstance(ans, (str, int, float)):
        return str(ans)

    top = example.get("answer")
    if isinstance(top, (str, int, float)):
        return str(top)

    return None


# ---------------------------- 加载与推理 ----------------------------

def get_dist_info() -> Tuple[int, int, int]:
    """从环境变量获取分布式信息。
    返回 (rank, local_rank, world_size)。若未分布式，返回 (0, -1, 1)。
    """
    try:
        rank = int(os.environ.get("RANK", "0"))
    except Exception:
        rank = 0
    try:
        local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    except Exception:
        local_rank = -1
    try:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
    except Exception:
        world_size = 1
    return rank, local_rank, world_size


def setup_distributed_if_needed() -> Tuple[int, int, int]:
    """如需则初始化 torch.distributed，并返回 (rank, local_rank, world_size)。"""
    rank, local_rank, world_size = get_dist_info()
    if world_size > 1 and not (dist.is_available() and dist.is_initialized()):
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    return rank, local_rank, world_size

def load_model_and_processor(
    model_dir: str,
    dtype: str = "auto",
    device: Optional[str] = None,
    trust_remote_code: bool = True,
):
    cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=trust_remote_code)
    torch_dtype = get_torch_dtype(dtype)

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True, trust_remote_code=trust_remote_code)
    processor = None
    if os.path.exists(os.path.join(model_dir, "preprocessor_config.json")):
        try:
            processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=trust_remote_code)
            processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=trust_remote_code)
        except Exception:
            processor = None

    if is_vlm_config(cfg):
        # 直接用显式类（官方文档建议）
        try:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_dir,
                torch_dtype=torch_dtype if torch_dtype != "auto" else None,
                device_map=None,
                low_cpu_mem_usage=True,
                trust_remote_code=trust_remote_code,
            )
        except Exception as e:
            # 极端兜底：AutoModel（随后会强制检查 generate）

    if is_vlm_config(cfg):
        # 直接用显式类（官方文档建议）
        try:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_dir,
                torch_dtype=torch_dtype if torch_dtype != "auto" else None,
                device_map=None,
                low_cpu_mem_usage=True,
                trust_remote_code=trust_remote_code,
            )
        except Exception as e:
            # 极端兜底：AutoModel（随后会强制检查 generate）
            model = AutoModel.from_pretrained(
                model_dir,
                torch_dtype=torch_dtype if torch_dtype != "auto" else None,
                device_map=None,
                low_cpu_mem_usage=True,
                trust_remote_code=trust_remote_code,
                model_dir,
                torch_dtype=torch_dtype if torch_dtype != "auto" else None,
                device_map=None,
                low_cpu_mem_usage=True,
                trust_remote_code=trust_remote_code,
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch_dtype if torch_dtype != "auto" else None,
            device_map=None,
            low_cpu_mem_usage=True,
            trust_remote_code=trust_remote_code,
            model_dir,
            torch_dtype=torch_dtype if torch_dtype != "auto" else None,
            device_map=None,
            low_cpu_mem_usage=True,
            trust_remote_code=trust_remote_code,
        )


    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.to(device)
    model.eval()

    if not hasattr(model, "generate"):
        raise RuntimeError(
            f"已加载为 {type(model).__name__}，但缺少 generate()。"
            "请确认这是可生成的 actor/指令模型；或 Transformers 版本足够新；或未错误回退为 base。"
        )

    return model, tokenizer, processor, cfg, device
    if not hasattr(model, "generate"):
        raise RuntimeError(
            f"已加载为 {type(model).__name__}，但缺少 generate()。"
            "请确认这是可生成的 actor/指令模型；或 Transformers 版本足够新；或未错误回退为 base。"
        )

    return model, tokenizer, processor, cfg, device


def load_parquet_dataset(parquet_path: str, num_samples: Optional[int]) -> hfds.Dataset:
def load_parquet_dataset(parquet_path: str, num_samples: Optional[int]) -> hfds.Dataset:
    ds = hfds.load_dataset("parquet", data_files=parquet_path, split="train")
    if num_samples is not None:
        ds = ds.select(range(min(num_samples, len(ds))))
        ds = ds.select(range(min(num_samples, len(ds))))
    return ds


@torch.inference_mode()
def infer_one(
    example: Dict[str, Any],
    model,
    tokenizer,
    processor,
    cfg: AutoConfig,
    device: str,
    max_new_tokens: int = 128,
    repetition_penalty: float = 1.2,
) -> Dict[str, Any]:

    messages, pil_images, image_paths = build_messages_and_images(example)
    is_vlm = is_vlm_config(cfg)

    # A) 官方范式（优先使用）
    if is_vlm and processor is not None and _QWEN_UTILS_AVAIL:
        # 1) 模板文本
        if hasattr(processor, "apply_chat_template"):
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            # 兜底文本
            text = ""
            for c in messages[0]["content"]:
                if c["type"] == "text":
                    text = c["text"]
                    break

        # 2) 打包视觉（官方工具）
        image_inputs, video_inputs = process_vision_info(messages)  # type: ignore

        # 3) 编码
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(device)

        # 4) 生成
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, repetition_penalty=repetition_penalty)

        # 5) 去掉前缀 prompt tokens
        if "input_ids" in inputs:
            trimmed = [out[len(in_ids):] for in_ids, out in zip(inputs.input_ids, output_ids)]
        else:
            trimmed = [output_ids[0]]

        if hasattr(processor, "batch_decode"):
            out_text = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
            out_text = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        else:
            out_text = tokenizer.decode(trimmed[0], skip_special_tokens=True)

        # 提取“问题文本”用于展示
        question_text = ""
        for c in messages[0]["content"]:
            if c["type"] == "text":
                question_text = c["text"]
                break

        return {
            "answer": out_text,
            "question": question_text,
            "image_paths": image_paths,
            "ground_truth": extract_ground_truth(example),
        }

    # B) 降级：无 qwen_vl_utils 但有 processor 且有图像
    if is_vlm and processor is not None and len(pil_images) > 0:
        text = ""
        for c in messages[0]["content"]:
            if c["type"] == "text":
                text = c["text"]
                break
        inputs = processor(text=[text], images=pil_images, padding=True, return_tensors="pt").to(device)
        out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, repetition_penalty=repetition_penalty)
        if hasattr(processor, "batch_decode"):
            out_text = processor.batch_decode(out_ids, skip_special_tokens=True)[0]
        else:
            out_text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
        return {
            "answer": out_text,
            "question": text,
            "image_paths": image_paths,
            "ground_truth": extract_ground_truth(example),
        }

    # C) 纯文本兜底
    text = ""
    for c in messages[0]["content"]:
        if c["type"] == "text":
            text = c["text"]
            break
    enc = tokenizer(text, return_tensors="pt").to(device)
    out_ids = model.generate(**enc, max_new_tokens=max_new_tokens, repetition_penalty=repetition_penalty)
    out_text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    return {
        "answer": out_text,
        "question": text,
        "image_paths": image_paths,
        "ground_truth": extract_ground_truth(example),
    }


def try_compute_rewards(answer: str, gt: Optional[str]) -> Dict[str, Any]:
    """兼容 verl 奖励；缺失时降级为空值"""
    result = {
        "format_reward": 0.0,
        "answer_reward": 0.0,
        "total": 0.0,
        "chain_reward": 0.0,
        "total_with_chain": 0.0,
        "is_correct": False,
        "predicted": None,
        "format_info": {},
    }
    try:
        from verl.utils.reward_score.iceberg import (
            compute_score,
            compute_score_with_chain_bonus,
        )
        base = compute_score(solution_str=answer, ground_truth=str(gt) if gt is not None else "")
        bonus = compute_score_with_chain_bonus(solution_str=answer, ground_truth=str(gt) if gt is not None else "")
        result.update({
            "format_reward": base.get("format_reward", 0.0),
            "answer_reward": base.get("answer_reward", 0.0),
            "total": base.get("score", 0.0),
            "chain_reward": bonus.get("chain_reward", 0.0),
            "total_with_chain": bonus.get("score", 0.0),
            "is_correct": base.get("is_correct", False),
            "predicted": base.get("predicted"),
            "format_info": base.get("format", {}),
        })
    except Exception:
        pass
    return result


# ---------------------------- CLI 主流程 ----------------------------

def main():
    ap = argparse.ArgumentParser(description="Qwen2.5-VL 推理脚本（显式类版本）")
    ap.add_argument("--model_dir", type=str, required=True, help="HuggingFace 模型目录（建议为 *Instruct 或 actor 合并权重）")
    ap.add_argument("--parquet", type=str, required=True, help="parquet 文件路径")
    ap.add_argument("--num_samples", type=int, default=2, help="抽样条数")
    ap.add_argument("--dtype", type=str, default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--trust_remote_code", action="store_true", help="允许远程自定义代码（Qwen 常用）")
    ap.add_argument("--json_out", type=str, default=None, help="可选：将结果以 JSONL 逐条写入该文件")
    ap.add_argument("--resume", dest="resume", action="store_true", help="启用断点续传（默认启用）")
    ap.add_argument("--no_resume", dest="resume", action="store_false", help="禁用断点续传")
    ap.add_argument("--progress_interval", type=int, default=5, help="进度打印间隔（处理样本数）")
    ap.add_argument("--repetition_penalty", type=float, default=1.0, help="重复惩罚因子")
    ap.set_defaults(resume=True)
    args = ap.parse_args()

    # 分布式初始化（如有）
    rank, local_rank, world_size = setup_distributed_if_needed()

    # 设备设置
    device_arg: Optional[str]
    if torch.cuda.is_available():
        if local_rank is not None and local_rank >= 0:
            torch.cuda.set_device(local_rank)
            device_arg = f"cuda:{local_rank}"
        else:
            device_arg = "cuda"
    else:
        device_arg = "cpu"

    # 加载
    model, tokenizer, processor, cfg, device = load_model_and_processor(
        args.model_dir, dtype=args.dtype, device=device_arg, trust_remote_code=bool(args.trust_remote_code)
    )

    # 数据
    ds = load_parquet_dataset(args.parquet, num_samples=args.num_samples)

    def read_processed_indices(base_path: Optional[str]) -> Set[int]:
        processed: Set[int] = set()
        if not base_path:
            return processed
        try:
            base_dir = os.path.dirname(base_path) or "."
            base_file = base_path
            # 1) 读取已合并
            if os.path.isfile(base_file):
                with open(base_file, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            obj = json.loads(line)
                            idx = obj.get("index")
                            if isinstance(idx, int):
                                processed.add(idx)
                        except Exception:
                            continue
            # 2) 读取所有分片
            prefix = os.path.basename(base_path) + ".part"
            for fname in os.listdir(base_dir):
                if fname.startswith(prefix):
                    fpath = os.path.join(base_dir, fname)
                    try:
                        with open(fpath, "r", encoding="utf-8") as f:
                            for line in f:
                                try:
                                    obj = json.loads(line)
                                    idx = obj.get("index")
                                    if isinstance(idx, int):
                                        processed.add(idx)
                                except Exception:
                                    continue
                    except Exception:
                        continue
        except Exception:
            pass
        return processed

    processed_indices: Set[int] = set()
    if args.resume and args.json_out:
        processed_indices = read_processed_indices(args.json_out)

    # 推理
    ds_len = len(ds)
    if rank == 0:
        print(f"[INFO] world_size={world_size} rank={rank} local_rank={local_rank} ds_len={ds_len}")
    # rank 间按步长切分
    assigned_indices = list(range(rank, ds_len, world_size))
    assigned_total = len(assigned_indices)
    already_done = 0
    if args.resume and processed_indices:
        already_done = sum(1 for idx in assigned_indices if idx in processed_indices)
    processed_count = already_done

    t0 = time.time()
    last_print = 0

    for i in assigned_indices:
        if args.resume and i in processed_indices:
            # 跳过已完成样本
            continue
        ex = ds[i]
        try:
            rec = infer_one(
                ex, model, tokenizer, processor, cfg, device,
                max_new_tokens=args.max_new_tokens,
                repetition_penalty=args.repetition_penalty
            )
            ans = rec["answer"]
            q = rec["question"]
            gt = rec.get("ground_truth")
            rewards = try_compute_rewards(ans, gt)

            record = {
                "index": i,
                "model_dir": args.model_dir,
                "question": q,
                "answer": ans,
                "images": rec.get("image_paths", []),
                "ground_truth": gt,
                "reward": rewards,
            }

            if args.json_out:
                # 多卡写入 rank 分片文件，避免竞争
                out_path = args.json_out
                if world_size > 1:
                    out_path = f"{args.json_out}.part{rank:02d}of{world_size:02d}"
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                with open(out_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

            # 更新进度，仅打印摘要进度，不打印问答内容
            processed_count += 1
            if processed_count - last_print >= max(1, int(args.progress_interval)) or processed_count == assigned_total:
                elapsed = time.time() - t0
                speed = processed_count / elapsed if elapsed > 0 else 0.0
                pct = (processed_count / assigned_total * 100.0) if assigned_total > 0 else 100.0
                print(
                    f"[PROGRESS] rank={rank}/{world_size} {processed_count}/{assigned_total} ({pct:.1f}%) "
                    f"elapsed={elapsed:.1f}s speed={speed:.2f} samples/s"
                )
                last_print = processed_count

        except Exception as e:
            print(f"[ERROR] rank={rank} sample {i} failed: {e}")

    # 同步退出，方便外部合并分片
    if dist.is_available() and dist.is_initialized():
        try:
            dist.barrier()
        except Exception:
            pass
        try:
            dist.destroy_process_group()
        except Exception:
            pass
            print(f"[ERROR] rank={rank} sample {i} failed: {e}")

    # 同步退出，方便外部合并分片
    if dist.is_available() and dist.is_initialized():
        try:
            dist.barrier()
        except Exception:
            pass
        try:
            dist.destroy_process_group()
        except Exception:
            pass


if __name__ == "__main__":
    main()

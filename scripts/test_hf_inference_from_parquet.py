#!/usr/bin/env python3
import argparse
import os
from io import BytesIO

import torch
import datasets as hfds
from PIL import Image
from transformers import AutoConfig, AutoProcessor, AutoTokenizer, AutoModelForCausalLM, AutoModel
try:
    # 新版 Transformers 将视觉到序列统一为该接口
    from transformers import AutoModelForVision2Seq  # type: ignore
except Exception:  # pragma: no cover
    AutoModelForVision2Seq = None  # type: ignore
try:
    from transformers import AutoModelForConditionalGeneration  # type: ignore
except Exception:  # pragma: no cover
    AutoModelForConditionalGeneration = None  # type: ignore


def is_vlm(config: AutoConfig) -> bool:
    archs = getattr(config, "architectures", []) or []
    name = archs[0] if archs else ""
    return "VL" in name or getattr(config, "model_type", "").endswith("_vl")


def load_model_and_processor(model_dir: str, dtype: str = "auto", device: str | None = None):
    config = AutoConfig.from_pretrained(model_dir)
    dtype_map = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(dtype, "auto")

    processor = None
    if os.path.exists(os.path.join(model_dir, "preprocessor_config.json")):
        try:
            processor = AutoProcessor.from_pretrained(model_dir)
        except Exception:
            processor = None
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)

    # 多模态模型与纯文本模型使用不同的 Auto 加载器
    if is_vlm(config):
        # 1) 优先 Vision2Seq
        if AutoModelForVision2Seq is not None:
            try:
                model = AutoModelForVision2Seq.from_pretrained(
                    model_dir, torch_dtype=torch_dtype, device_map=None, low_cpu_mem_usage=True
                )
            except Exception:
                model = None  # type: ignore
        else:
            model = None  # type: ignore

        # 2) 退回 ConditionalGeneration
        if model is None and AutoModelForConditionalGeneration is not None:
            try:
                model = AutoModelForConditionalGeneration.from_pretrained(
                    model_dir, torch_dtype=torch_dtype, device_map=None, low_cpu_mem_usage=True
                )
            except Exception:
                model = None  # type: ignore

        # 3) 最后退回基础 AutoModel（多数情况下也具备 generate）
        if model is None:
            model = AutoModel.from_pretrained(
                model_dir, torch_dtype=torch_dtype, device_map=None, low_cpu_mem_usage=True
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype=torch_dtype, device_map=None, low_cpu_mem_usage=True
        )
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    return model, tokenizer, processor, config, device


def load_parquet_dataset(parquet_path: str, num_samples: int | None = None):
    ds = hfds.load_dataset("parquet", data_files=parquet_path, split="train")
    if num_samples is not None:
        num_samples = min(num_samples, len(ds))
        ds = ds.select(range(num_samples))
    return ds


@torch.inference_mode()
def run_inference_on_example(example: dict, model, tokenizer, processor, config, device: str, max_new_tokens: int = 32):
    # prompt: list of {role, content}
    prompt_list = example.get("prompt") or []
    # 取第一条 user 文本作为主要内容
    text_content = ""
    for turn in prompt_list:
        role = (turn.get("role") or "").lower()
        content = turn.get("content") or ""
        if role == "user":
            text_content = content
            break
    if not text_content and prompt_list:
        text_content = prompt_list[0].get("content") or ""

    # images: list of {bytes, path}
    pil_images = []
    for img in example.get("images") or []:
        try:
            data = img.get("bytes") if isinstance(img, dict) else None
            if isinstance(data, (bytes, bytearray)) and len(data) > 0:
                pil_images.append(Image.open(BytesIO(data)).convert("RGB"))
        except Exception:
            continue

    # 构造 chat 消息（若有处理器则使用结构化图文）
    if processor is not None and is_vlm(config) and len(pil_images) > 0:
        content_items = [{"type": "text", "text": text_content}]
        for im in pil_images:
            content_items.append({"type": "image", "image": im})
        messages = [{"role": "user", "content": content_items}]
        if hasattr(processor, "apply_chat_template"):
            prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt_text = text_content
        inputs = processor(text=prompt_text, images=pil_images, return_tensors="pt").to(device)
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        if hasattr(processor, "batch_decode"):
            text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        else:
            text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    else:
        # 纯文本推理兜底
        inputs = tokenizer(text_content, return_tensors="pt").to(device)
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return text


def main():
    parser = argparse.ArgumentParser(description="基于 parquet 数据集测试合并后的 HF 模型推理")
    parser.add_argument("--model_dir", type=str, required=True, help="合并后 HuggingFace 模型目录")
    parser.add_argument("--parquet", type=str, required=True, help="parquet 文件路径")
    parser.add_argument("--num_samples", type=int, default=2, help="抽样条数用于测试")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--max_new_tokens", type=int, default=64)
    args = parser.parse_args()

    model, tokenizer, processor, config, device = load_model_and_processor(args.model_dir, dtype=args.dtype)
    ds = load_parquet_dataset(args.parquet, num_samples=args.num_samples)

    for i in range(len(ds)):
        example = ds[i]
        try:
            out_text = run_inference_on_example(
                example, model, tokenizer, processor, config, device, max_new_tokens=args.max_new_tokens
            )
            print(f"\n===== Sample {i} =====")
            # 打印问题简要
            prompt_list = example.get("prompt") or []
            qtext = ""
            for turn in prompt_list:
                if (turn.get("role") or "").lower() == "user":
                    qtext = (turn.get("content") or "").strip().splitlines()[0][:160]
                    break
            print(f"[QUESTION] {qtext}")
            print(f"[OUTPUT]   {out_text}")
        except Exception as e:
            print(f"[ERROR] sample {i} failed: {e}")


if __name__ == "__main__":
    main()



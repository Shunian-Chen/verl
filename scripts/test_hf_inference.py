#!/usr/bin/env python3
import argparse
import os

import torch
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    AutoModelForCausalLM,
)


def is_vlm(config: AutoConfig) -> bool:
    archs = getattr(config, "architectures", []) or []
    name = archs[0] if archs else ""
    return "VL" in name or getattr(config, "model_type", "").endswith("_vl")


def load_model_and_processor(model_dir: str, dtype: str = "auto", device: str = None):
    config = AutoConfig.from_pretrained(model_dir)
    dtype_map = {"auto": "auto", "float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map.get(dtype, "auto")

    processor = None
    tokenizer = None
    if os.path.exists(os.path.join(model_dir, "preprocessor_config.json")):
        try:
            processor = AutoProcessor.from_pretrained(model_dir)
        except Exception:
            processor = None
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)

    # 对于 VLM，AutoModelForCausalLM 可以加载 Qwen2_5_VLForConditionalGeneration
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch_dtype, device_map=None, low_cpu_mem_usage=True
    )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    return model, tokenizer, processor, config, device


@torch.inference_mode()
def test_text_only(model, tokenizer, device: str, max_new_tokens: int = 32):
    prompt = "You are a helpful assistant. Answer briefly.\nQuestion: What is the capital of France?\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("[TEXT OUTPUT]", text)


@torch.inference_mode()
def test_vision_text(model, processor, tokenizer, device: str, max_new_tokens: int = 32):
    from PIL import Image
    import requests
    from io import BytesIO

    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/image_classification/benz.jpg"
    image = Image.open(BytesIO(requests.get(url, timeout=20).content)).convert("RGB")
    # Qwen2.5-VL 推荐 chat 风格输入
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": "What is in this image?"},
            {"type": "image" if hasattr(processor, "image_processor") else "text", "image": image}
        ]}
    ]
    if hasattr(processor, "apply_chat_template"):
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = "Describe the image briefly."  # 兜底

    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_new_tokens=max_new_tokens)
    # 有些处理器不提供解码，回退到 tokenizer 解码
    if hasattr(processor, "batch_decode"):
        text = processor.batch_decode(output, skip_special_tokens=True)[0]
    else:
        text = tokenizer.decode(output[0], skip_special_tokens=True)
    print("[VLM OUTPUT]", text)


def main():
    parser = argparse.ArgumentParser(description="测试合并后的 HF 模型是否可推理")
    parser.add_argument("--model_dir", type=str, required=True, help="合并后 HuggingFace 模型目录")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--test_vlm", action="store_true", help="若为多模态，尝试图文推理一例")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    args = parser.parse_args()

    model, tokenizer, processor, config, device = load_model_and_processor(args.model_dir, dtype=args.dtype)

    # 纯文本自测
    test_text_only(model, tokenizer, device, max_new_tokens=args.max_new_tokens)

    # 可选：多模态自测（若模型支持且 processor 存在）
    if args.test_vlm and processor is not None and is_vlm(config):
        try:
            test_vision_text(model, processor, tokenizer, device, max_new_tokens=args.max_new_tokens)
        except Exception as e:
            print(f"[WARN] VLM 测试失败: {e}")


if __name__ == "__main__":
    main()



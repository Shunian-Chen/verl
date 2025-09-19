#!/usr/bin/env python3

"""
Convert ShareGPT-style multimodal JSON to the same parquet schema as geo3k.py.

Input JSON example item:
{
  "id": "1152221_I_1",
  "image_path": "/path/to/1152221.jpg",
  "meta": {...},
  "conversations": [
    {"from": "human", "value": "<image>...question..."},
    {"from": "gpt", "value": "<look>...</look>\n<think>...</think>\n<answer>...final...</answer>"}
  ],
  "images": ["/path/to/1152221.jpg"]
}

Output parquet schema matches geo3k.py mapper:
- data_source: str
- prompt: list of {role: str, content: str}
- images: list of str
- ability: str
- reward_model: {style: str, ground_truth: str}
- extra_info: {split: str, index: int, answer: str, question: str}

Usage:
  python sharegpt_to_parquet.py \
    --input_json /root/et/data/sft_demos_gemini-2.5-pro_sharegpt_test.json \
    --local_dir ~/data/geo3k \
    --split test \
    [--hdfs_dir hdfs:///your/dir]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import traceback

import datasets

from verl.utils.hdfs_io import copy, makedirs


instruction = """<image>
-----
You are a specialized vision-language assistant. Your task is to answer the given question about an image by following a strict reasoning process. You must respond using a specific XML-like format consisting of `<look>`, `<think>`, and `<answer>` blocks.

The entire reasoning process must strictly adhere to the following structure and rules:

**Output Structure:**
The response must begin with a `<look>` block, followed by a `<think>` block. You may use multiple, strictly alternating `<look>` and `<think>` blocks if necessary. The entire response must conclude with a single, final `<answer>` block.

**Example Sequence:**

```xml
<look> ... </look>
<think> ... </think>
<look> ... </look>
<think> ... </think>
<answer> ... </answer>
```

**Rules for Each Block:**

1.  **`<look>` Block:**

      * Describe **only** the directly visible evidence in the image that is relevant to answering the question.
      * Identify and name objects, text, people, or any visual entities.
      * **Do not** infer information or provide the final answer here. Keep the description grounded strictly in what is visible.

2.  **`<think>` Block:**

      * Connect the visual evidence identified in the preceding `<look>` block to your background knowledge to form a logical step in your reasoning.
      * Explain how the visual evidence helps you move closer to the answer.
      * **Do not** introduce new visual claims that were not established in a previous `<look>` block.

3.  **`<answer>` Block:**

      * Provide the final, concise answer to the question.
      * For multiple-choice questions, output only the correct option letter and its full text (e.g., "B. 1958").
      * For open-ended questions, provide a direct and brief answer.

-----

**Analyze the provided image and answer the following question using the format defined above.**

**Question:** 
{question}
"""

# A simplified alternative that gives only directional guidance and does not constrain
# detailed content inside <look>/<think>, nor the order between them.
instruction_simple = """<image>
-----
You are a vision-language assistant. Structure your response using `<look>`, `<think>`, and end with `<answer>`.

- You may include one or more `<look>` and `<think>` blocks in any order.
- `<look>`: note observations that are relevant to the question.
- `<think>`: reflect on observations and move toward an answer.
- `<answer>`: provide the final concise answer.

Examples (any order is acceptable):

```xml
<think> ... </think>
<look> ... </look>
<answer> ... </answer>
```

```xml
<look> ... </look>
<think> ... </think>
<answer> ... </answer>
```

```xml
<think> ... </think>
<look> ... </look>
<think> ... </think>
<answer> ... </answer>
```

Question:
{question}
"""

def extract_question_and_answer(conversations: List[Dict[str, Any]]) -> Tuple[str, str]:
    """Extract the first human message as question and parse the <answer>...</answer> from assistant.

    If <answer> tags are missing, use the whole assistant message text as answer.
    Removes leading <image> marker from the question if present.
    """
    question_text: str = ""
    answer_text: str = ""

    for turn in conversations:
        speaker = turn.get("from") or turn.get("role")
        value = turn.get("value") or turn.get("content") or ""
        if speaker in {"human", "user"} and not question_text:
            question_text = value
        if speaker in {"gpt", "assistant"} and not answer_text:
            # try to extract <answer>...</answer>
            lower_value = value.lower()
            start_tag = "<answer>"
            end_tag = "</answer>"
            start = lower_value.find(start_tag)
            end = lower_value.find(end_tag)
            if start != -1 and end != -1 and end > start:
                # slice using original casing
                start_orig = start
                end_orig = end
                answer_text = value[start_orig + len(start_tag):end_orig].strip()
            else:
                answer_text = value.strip()

        if question_text and answer_text:
            break


    return question_text, answer_text


def build_item(
    raw: Dict[str, Any],
    data_source: str,
    split: str,
    index: int,
    instruction_text: str,
) -> Dict[str, Any]:
    question, answer = extract_question_and_answer(raw.get("conversations", []))
    images: List[str] = raw.get("images") or ([] if raw.get("image_path") is None else [raw["image_path"]])


    if "<image>" in question:
        question = question.replace("<image>", "")

    prompt = [
        {
            "role": "user",
            "content": instruction_text.format(question=question),
        }
    ]

    images = [image.replace("/wangbenyou/shunian/workspace/iceberg/data/images", "/root/et/data/images") for image in images]
    # 将图片加载为二进制字节，保证写入 parquet 的结构为 [{'bytes': bytes, 'path': str}]
    image_bytes_list: List[Dict[str, Any]] = []
    for img_path in images:
        # 统一展开与标准化路径，避免相对路径/波浪线等导致的不一致
        resolved_path = os.path.abspath(os.path.expanduser(img_path))
        if not os.path.isfile(resolved_path):
            raise FileNotFoundError(f"Image path does not exist: {resolved_path}")
        try:
            with open(resolved_path, "rb") as f:
                img_bytes = f.read()
            if not img_bytes:
                raise ValueError("empty image bytes")
            image_bytes_list.append({"bytes": img_bytes, "path": resolved_path})
        except Exception as e:
            print(f"[WARN] Skip unreadable image: {resolved_path} -> {e}")
            # 跳过不可读/空图片，避免下游 PIL 打开失败

    item: Dict[str, Any] = {
        "data_source": data_source,
        "prompt": prompt,
        "images": image_bytes_list,
        "ability": "vision",
        "reward_model": {"style": "rule", "ground_truth": answer},
        "extra_info": {
            "split": split,
            "index": index,
            "answer": answer,
            "question": question,
        },
    }
    return item


def _debug_print_item(item: Dict[str, Any], idx: int) -> None:
    print(f"[DEBUG] Item {idx} types:")
    print(f"  data_source: {type(item.get('data_source'))}")
    print(f"  ability: {type(item.get('ability'))}")
    print(f"  prompt type: {type(item.get('prompt'))}, len={len(item.get('prompt', []))}")
    if isinstance(item.get("prompt"), list) and item["prompt"]:
        p0 = item["prompt"][0]
        print(f"    prompt[0] type: {type(p0)}, keys={list(p0.keys()) if isinstance(p0, dict) else None}")
    print(f"  images type: {type(item.get('images'))}, len={len(item.get('images', []))}")
    if isinstance(item.get("images"), list) and item["images"]:
        b0 = item["images"][0]
        b0_desc = f"bytes({len(b0)})" if isinstance(b0, (bytes, bytearray)) else str(type(b0))
        print(f"    images[0]: {b0_desc}")
    rm = item.get("reward_model")
    print(f"  reward_model type: {type(rm)}, keys={list(rm.keys()) if isinstance(rm, dict) else None}")
    ei = item.get("extra_info")
    print(f"  extra_info type: {type(ei)}, keys={list(ei.keys()) if isinstance(ei, dict) else None}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", required=True, help="Path to ShareGPT JSON file")
    parser.add_argument("--local_dir", default="~/data/geo3k")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--data_source", default="iceberg")
    parser.add_argument(
        "--instruction_style",
        choices=["strict", "simple"],
        default="strict",
        help="Choose instruction template: 'strict' for detailed constraints; 'simple' for minimal guidance without ordering constraints.",
    )
    args = parser.parse_args()

    input_json = Path(os.path.expanduser(args.input_json)).resolve()
    local_dir = os.path.expanduser(args.local_dir)
    hdfs_dir = args.hdfs_dir
    split = args.split

    os.makedirs(local_dir, exist_ok=True)

    # Stream read JSON array to avoid high memory usage
    with input_json.open("r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
            records = json.load(f)
        else:
            # JSONL fallback
            records = [json.loads(line) for line in f if line.strip()]

    data: List[Dict[str, Any]] = []
    selected_instruction = instruction if args.instruction_style == "strict" else instruction_simple
    for idx, rec in enumerate(records):
        item = build_item(rec, args.data_source, split, idx, selected_instruction)
        data.append(item)

    # # Debug-print first few items before schema application
    # for i in range(min(3, len(data))):
    #     _debug_print_item(data[i], i)

    # Use inferred schema to avoid encoding mismatches
    dataset = datasets.Dataset.from_list(data)
    print(f"[DEBUG] Inferred features: {dataset.features}")

    # Write parquet; name depends on instruction style
    out_filename = f"{split}.parquet" if args.instruction_style == "strict" else f"{split}_{args.instruction_style}.parquet"
    out_path = os.path.join(local_dir, out_filename)
    dataset.to_parquet(out_path)
    print(f"Wrote parquet: {out_path} ({len(dataset)} rows)")

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
        print(f"Copied {local_dir} to HDFS: {hdfs_dir}")


if __name__ == "__main__":
    main()


'''
python examples/data_preprocess/sharegpt_to_parquet.py --input_json /root/et/data/sft_demos_gemini-2.5-pro_sharegpt.json --local_dir /data/iceberg --split train  --instruction_style simple
'''
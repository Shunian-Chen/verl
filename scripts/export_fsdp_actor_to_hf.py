#!/usr/bin/env python3
import argparse
import json
import os
import sys
import warnings

import torch
import torch.distributed as dist
from accelerate import init_empty_weights
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForTokenClassification,
    AutoModelForVision2Seq,
)


def _infer_auto_model_cls(architectures: list[str]):
    arch = architectures[0] if architectures else ""
    if "ForTokenClassification" in arch:
        return AutoModelForTokenClassification
    if "ForCausalLM" in arch:
        return AutoModelForCausalLM
    if "ForConditionalGeneration" in arch:
        # transformers >= 4.54.0 uses AutoModelForImageTextToText
        try:
            from packaging import version
            import transformers as _tf
            if version.parse(_tf.__version__) >= version.parse("4.54.0"):
                from transformers import AutoModelForImageTextToText  # lazy import to support older versions
                return AutoModelForImageTextToText
        except Exception:
            pass
        return AutoModelForVision2Seq
    raise ValueError(f"无法根据 architectures 推断模型类: {architectures}")


def load_sharded_and_consolidate_full_state(actor_dir: str) -> dict:
    """
    读取 rank 本地分片并在 rank0 聚合为 full state dict。
    约定文件名为 model_world_size_{W}_rank_{R}.pt。
    """
    fsdp_cfg_path = os.path.join(actor_dir, "fsdp_config.json")
    with open(fsdp_cfg_path, "r", encoding="utf-8") as f:
        fsdp_cfg = json.load(f)
    world_size_from_file = int(fsdp_cfg.get("world_size", 1))

    assert dist.is_initialized(), "需要使用 torchrun 启动以初始化分布式环境"
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size_from_file != world_size:
        warnings.warn(
            f"fsdp_config.json world_size={world_size_from_file} 与当前进程组 world_size={world_size} 不一致，继续以进程组为准",
            RuntimeWarning,
        )

    shard_path = os.path.join(actor_dir, f"model_world_size_{world_size_from_file}_rank_{rank}.pt")
    if not os.path.exists(shard_path):
        # 兼容以当前实际 world_size 命名
        shard_path = os.path.join(actor_dir, f"model_world_size_{world_size}_rank_{rank}.pt")
    assert os.path.exists(shard_path), f"未找到当前 rank 的分片: {shard_path}"

    local_sd = torch.load(shard_path, map_location="cpu", weights_only=False)

    # 收集所有 key
    all_keys = sorted(list(local_sd.keys()))
    gathered_keys = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_keys, all_keys)
    key_union = set()
    for ks in gathered_keys:
        key_union.update(ks)
    key_union = sorted(list(key_union))

    # 为每个 key 选取“拥有该 key 的最小 rank”作为唯一广播源，确保所有 rank 一致
    owner_rank = {}
    for k in key_union:
        candidate_ranks = [r for r, ks in enumerate(gathered_keys) if k in ks]
        assert len(candidate_ranks) > 0, f"未找到 key 的拥有者: {k}"
        owner_rank[k] = min(candidate_ranks)

    # rank0 聚合
    full_state = {} if rank == 0 else None
    for k in key_union:
        src = owner_rank[k]
        payload = (k, local_sd[k]) if rank == src else None
        obj_list = [payload]
        dist.broadcast_object_list(obj_list, src=src)
        payload = obj_list[0]
        if rank == 0:
            recv_k, recv_t = payload
            full_state[recv_k] = recv_t.cpu()

    # 释放本地 shard
    del local_sd
    return full_state


def load_all_shards_single_process(actor_dir: str) -> dict:
    """
    单进程（CPU）模式：按 rank 依次读取所有分片并做 key 并集合并。
    要求每个 key 只出现在一个分片中（典型 FSDP 分片情形）。
    """
    fsdp_cfg_path = os.path.join(actor_dir, "fsdp_config.json")
    with open(fsdp_cfg_path, "r", encoding="utf-8") as f:
        fsdp_cfg = json.load(f)
    world_size = int(fsdp_cfg.get("world_size", 1))

    full_state: dict = {}

    def _find_shard_path(rank: int) -> str:
        candidates = [
            os.path.join(actor_dir, f"model_world_size_{world_size}_rank_{rank}.pt"),
            os.path.join(actor_dir, f"model_rank_{rank}.pt"),
        ]
        for p in candidates:
            if os.path.exists(p):
                return p
        # 兜底：在目录中搜索匹配 _rank_{rank}.pt 的文件
        for fname in sorted(os.listdir(actor_dir)):
            if fname.endswith(f"_rank_{rank}.pt") and fname.startswith("model_world_size_"):
                return os.path.join(actor_dir, fname)
        raise FileNotFoundError(f"未找到分片文件：rank={rank}")

    for r in range(world_size):
        shard_path = _find_shard_path(r)
        local_sd = torch.load(shard_path, map_location="cpu", weights_only=False)
        for k, v in local_sd.items():
            if k in full_state:
                raise RuntimeError(f"重复的 key '{k}' 同时出现在多个分片中，无法单进程合并")
            full_state[k] = v.cpu()
        del local_sd

    return full_state


def main():
    parser = argparse.ArgumentParser(description="将 FSDP 分片 actor 权重导出为 HuggingFace 格式")
    parser.add_argument(
        "--actor_dir",
        type=str,
        required=True,
        help="actor 目录（包含 model_world_size_*_rank_*.pt 与 huggingface/ 配置）",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="输出目录（默认写回到 actor_dir/huggingface）",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="保存权重数据类型",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="gloo",
        choices=["gloo", "nccl"],
        help="分布式后端（默认 gloo，避免 NCCL 的 GPU 需求）",
    )
    parser.add_argument(
        "--single_process",
        action="store_true",
        help="单进程在 CPU 上依次读取并合并所有分片（无需分布式/NCCL）",
    )
    parser.add_argument(
        "--model_device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="构建与加载模型权重所用设备；cuda 需要单卡显存足够",
    )
    args = parser.parse_args()

    if not args.single_process and not dist.is_initialized():
        # 尝试使用 env 初始化（torchrun 会设置 LOCAL_RANK 等）
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            from datetime import timedelta
            backend = args.backend
            if backend == "gloo":
                # 强制 CPU 路径，避免 NCCL Duplicate GPU 等问题
                os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
            else:
                # 为 NCCL 正确设置本地 GPU
                try:
                    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
                    if torch.cuda.is_available():
                        torch.cuda.set_device(local_rank)
                except Exception:
                    pass
            dist.init_process_group(backend=backend, timeout=timedelta(seconds=1800))
        else:
            raise RuntimeError("请使用 torchrun 启动，或指定 --single_process 进行单进程合并")

    rank = 0 if args.single_process else dist.get_rank()

    # 读取 HF 配置
    hf_dir = os.path.join(args.actor_dir, "huggingface")
    cfg_path = os.path.join(hf_dir, "config.json")
    assert os.path.exists(cfg_path), f"未找到配置: {cfg_path}"
    config = AutoConfig.from_pretrained(hf_dir)

    # 汇总全量权重到 rank0
    if args.single_process:
        full_state = load_all_shards_single_process(args.actor_dir)
    else:
        full_state = load_sharded_and_consolidate_full_state(args.actor_dir)

    # rank0 构建空模型并保存
    if rank == 0:
        auto_cls = _infer_auto_model_cls(getattr(config, "architectures", []))
        dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
        torch_dtype = dtype_map[args.dtype]

        with init_empty_weights():
            model = auto_cls.from_config(config, torch_dtype=torch_dtype)
        # 允许在单卡上进行模型权重加载（若显存足够）
        if args.model_device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("指定了 --model_device cuda 但当前不可用 CUDA")
            torch.cuda.set_device(0)
            model.to_empty(device="cuda")
        else:
            model.to_empty(device="cpu")

        # 某些模型在保存时需要 generation_config，若存在则一起加载
        try:
            from transformers import GenerationConfig

            gen_cfg_path = os.path.join(hf_dir, "generation_config.json")
            if os.path.exists(gen_cfg_path):
                generation_config = GenerationConfig.from_pretrained(hf_dir)
                model.generation_config = generation_config
        except Exception:
            pass

        model.load_state_dict(full_state, strict=True)

        out_dir = args.out_dir or hf_dir
        os.makedirs(out_dir, exist_ok=True)
        model.save_pretrained(out_dir, safe_serialization=True)

        # 同步 tokenizer / processor 等已存在文件，无需复制（已在 huggingface/ 中）
        print(f"已保存 HuggingFace 权重至: {os.path.abspath(out_dir)}")

    # 广播完成信号
    if not args.single_process and dist.is_initialized():
        dist.barrier()


if __name__ == "__main__":
    main()



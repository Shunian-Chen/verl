#!/usr/bin/env python3
# Copyright 2025
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Offline rubric synthesis script for preprocessing training datasets.

This script takes a raw dataset and enriches it with executable rubrics
for use in agentic rubric RL training.

Usage:
    python rubric_dataset.py \
        --input_path data/raw_tasks.parquet \
        --output_path data/tasks_with_rubrics.parquet \
        --tool_config examples/agentic_rubric/config/tool_config/ \
        --generator_model gpt-4o \
        --batch_size 10 \
        --max_workers 5

The script:
1. Loads the raw dataset
2. Loads tool schemas from config
3. Generates rubrics for each sample using the LLM generator
4. Assigns latency buckets based on estimated execution time
5. Saves the enriched dataset
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

# Add verl to path if running as script
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from verl.experimental.dataset.latency_bucketed_sampler import compute_latency_bucket
from verl.rubric.generator import LLMRubricGenerator, RubricGeneratorConfig
from verl.rubric.generator.base import ToolSchema

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_tool_schemas(tool_config_dir: str) -> list[ToolSchema]:
    """
    Load tool schemas from configuration directory.

    Expects:
    - policy_tools.json: Tools available to the policy
    - verification_tools.json: Additional verification-only tools

    Args:
        tool_config_dir: Path to tool configuration directory.

    Returns:
        Combined list of tool schemas (T_e^rub = T_e^pol ∪ T_e^ver).
    """
    tool_schemas = []
    seen_names = set()

    config_dir = Path(tool_config_dir)

    # Load policy tools first (take precedence)
    policy_tools_path = config_dir / "policy_tools.json"
    if policy_tools_path.exists():
        with open(policy_tools_path) as f:
            policy_tools = json.load(f)
            for tool in policy_tools.get("tools", []):
                schema = ToolSchema(
                    name=tool["name"],
                    description=tool.get("description", ""),
                    parameters=tool.get("parameters", {}),
                    returns=tool.get("returns", {}),
                )
                tool_schemas.append(schema)
                seen_names.add(tool["name"])
        logger.info(f"Loaded {len(seen_names)} policy tools")

    # Load verification tools (deduplicate by name)
    verification_tools_path = config_dir / "verification_tools.json"
    if verification_tools_path.exists():
        with open(verification_tools_path) as f:
            verification_tools = json.load(f)
            added = 0
            for tool in verification_tools.get("tools", []):
                if tool["name"] not in seen_names:
                    schema = ToolSchema(
                        name=tool["name"],
                        description=tool.get("description", ""),
                        parameters=tool.get("parameters", {}),
                        returns=tool.get("returns", {}),
                    )
                    tool_schemas.append(schema)
                    seen_names.add(tool["name"])
                    added += 1
        logger.info(f"Loaded {added} additional verification tools")

    logger.info(f"Total tools available: {len(tool_schemas)}")
    return tool_schemas


def load_dataset(input_path: str) -> pd.DataFrame:
    """
    Load dataset from file.

    Supports:
    - .parquet files
    - .json / .jsonl files
    - .csv files

    Args:
        input_path: Path to input dataset.

    Returns:
        DataFrame with dataset contents.
    """
    path = Path(input_path)

    if path.suffix == ".parquet":
        df = pd.read_parquet(input_path)
    elif path.suffix in [".json", ".jsonl"]:
        df = pd.read_json(input_path, lines=path.suffix == ".jsonl")
    elif path.suffix == ".csv":
        df = pd.read_csv(input_path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    logger.info(f"Loaded {len(df)} samples from {input_path}")
    return df


def save_dataset(df: pd.DataFrame, output_path: str) -> None:
    """
    Save dataset to file.

    Args:
        df: DataFrame to save.
        output_path: Path to output file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix == ".parquet":
        df.to_parquet(output_path, index=False)
    elif path.suffix in [".json", ".jsonl"]:
        df.to_json(output_path, orient="records", lines=path.suffix == ".jsonl")
    elif path.suffix == ".csv":
        df.to_csv(output_path, index=False)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    logger.info(f"Saved {len(df)} samples to {output_path}")


async def generate_rubric_for_sample(
    generator: LLMRubricGenerator,
    tool_schemas: list[ToolSchema],
    sample: dict[str, Any],
    prompt_column: str,
    spec_column: str | None,
    verification_data_column: str | None,
) -> dict[str, Any]:
    """
    Generate a rubric for a single sample.

    Args:
        generator: The rubric generator.
        tool_schemas: Available tool schemas.
        sample: The dataset sample.
        prompt_column: Column name for task input.
        spec_column: Optional column name for specification.
        verification_data_column: Optional column for test cases.

    Returns:
        Dict with rubric and latency_bucket.
    """
    task_input = sample.get(prompt_column, "")
    specification = sample.get(spec_column) if spec_column else None
    verification_data = sample.get(verification_data_column) if verification_data_column else None

    # Parse verification data if it's a string
    if isinstance(verification_data, str):
        try:
            verification_data = json.loads(verification_data)
        except json.JSONDecodeError:
            verification_data = None

    try:
        rubric = await generator.generate(
            task_input=task_input,
            tool_schemas=tool_schemas,
            specification=specification,
            verification_data=verification_data,
        )

        return {
            "rubric": rubric.model_dump(),
            "latency_bucket": compute_latency_bucket(rubric.estimated_latency_ms),
            "error": None,
        }
    except Exception as e:
        logger.warning(f"Failed to generate rubric: {e}")
        return {
            "rubric": None,
            "latency_bucket": 0,
            "error": str(e),
        }


async def process_batch(
    generator: LLMRubricGenerator,
    tool_schemas: list[ToolSchema],
    samples: list[dict[str, Any]],
    prompt_column: str,
    spec_column: str | None,
    verification_data_column: str | None,
    semaphore: asyncio.Semaphore,
) -> list[dict[str, Any]]:
    """
    Process a batch of samples with rate limiting.

    Args:
        generator: The rubric generator.
        tool_schemas: Available tool schemas.
        samples: List of samples to process.
        prompt_column: Column name for task input.
        spec_column: Optional column name for specification.
        verification_data_column: Optional column for test cases.
        semaphore: Semaphore for rate limiting.

    Returns:
        List of results with rubrics and buckets.
    """

    async def process_with_semaphore(sample: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            return await generate_rubric_for_sample(
                generator=generator,
                tool_schemas=tool_schemas,
                sample=sample,
                prompt_column=prompt_column,
                spec_column=spec_column,
                verification_data_column=verification_data_column,
            )

    tasks = [process_with_semaphore(sample) for sample in samples]
    return await asyncio.gather(*tasks)


async def main_async(args: argparse.Namespace) -> None:
    """Main async function for rubric generation."""
    # Load tool schemas
    tool_schemas = load_tool_schemas(args.tool_config)

    # Load dataset
    df = load_dataset(args.input_path)

    # Initialize generator
    config = RubricGeneratorConfig(
        model_name=args.generator_model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        retry_attempts=args.retry_attempts,
    )
    generator = LLMRubricGenerator(
        config=config,
        api_key=args.api_key or os.getenv("OPENAI_API_KEY"),
        api_base=args.api_base,
    )

    # Process in batches with rate limiting
    semaphore = asyncio.Semaphore(args.max_workers)
    samples = df.to_dict(orient="records")

    results = []
    total_batches = (len(samples) + args.batch_size - 1) // args.batch_size

    # Resume from checkpoint if exists
    checkpoint_path = Path(args.output_path).with_suffix(".checkpoint.json")
    start_idx = 0

    if args.resume and checkpoint_path.exists():
        with open(checkpoint_path) as f:
            checkpoint = json.load(f)
            results = checkpoint.get("results", [])
            start_idx = checkpoint.get("processed", 0)
            logger.info(f"Resuming from checkpoint: {start_idx} samples processed")

    for batch_idx in tqdm(
        range(start_idx // args.batch_size, total_batches),
        desc="Generating rubrics",
        initial=start_idx // args.batch_size,
        total=total_batches,
    ):
        batch_start = batch_idx * args.batch_size
        batch_end = min(batch_start + args.batch_size, len(samples))
        batch = samples[batch_start:batch_end]

        batch_results = await process_batch(
            generator=generator,
            tool_schemas=tool_schemas,
            samples=batch,
            prompt_column=args.prompt_column,
            spec_column=args.spec_column,
            verification_data_column=args.verification_data_column,
            semaphore=semaphore,
        )

        results.extend(batch_results)

        # Save checkpoint
        if args.checkpoint_interval > 0 and (batch_idx + 1) % args.checkpoint_interval == 0:
            with open(checkpoint_path, "w") as f:
                json.dump(
                    {"results": results, "processed": batch_end},
                    f,
                )
            logger.info(f"Checkpoint saved: {batch_end} samples processed")

    # Add results to dataframe
    df["extra_info"] = [
        {
            "rubric": r["rubric"],
            "latency_bucket": r["latency_bucket"],
            "rubric_generation_error": r["error"],
        }
        for r in results
    ]

    # Filter out failed generations if requested
    if args.filter_failures:
        original_len = len(df)
        df = df[df["extra_info"].apply(lambda x: x["rubric"] is not None)]
        logger.info(f"Filtered {original_len - len(df)} failed samples")

    # Add data source marker
    df["data_source"] = "agentic_rubric"

    # Save output
    save_dataset(df, args.output_path)

    # Clean up checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    # Print statistics
    total = len(results)
    success = sum(1 for r in results if r["rubric"] is not None)
    logger.info(f"Generation complete: {success}/{total} successful ({100*success/total:.1f}%)")

    # Print bucket distribution
    bucket_counts = {}
    for r in results:
        if r["rubric"] is not None:
            bucket = r["latency_bucket"]
            bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

    logger.info("Latency bucket distribution:")
    for bucket in sorted(bucket_counts.keys()):
        count = bucket_counts[bucket]
        pct = 100 * count / success if success > 0 else 0
        logger.info(f"  Bucket {bucket}: {count} ({pct:.1f}%)")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate executable rubrics for a training dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input/Output
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to input dataset (parquet, json, jsonl, csv)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to output dataset",
    )
    parser.add_argument(
        "--tool_config",
        type=str,
        required=True,
        help="Path to tool configuration directory",
    )

    # Column names
    parser.add_argument(
        "--prompt_column",
        type=str,
        default="prompt",
        help="Column name for task input/prompt",
    )
    parser.add_argument(
        "--spec_column",
        type=str,
        default=None,
        help="Column name for specification (optional)",
    )
    parser.add_argument(
        "--verification_data_column",
        type=str,
        default=None,
        help="Column name for verification data/test cases (optional)",
    )

    # Generator settings
    parser.add_argument(
        "--generator_model",
        type=str,
        default="gpt-4o",
        help="Model to use for rubric generation",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for generation",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=4096,
        help="Maximum tokens for generation",
    )
    parser.add_argument(
        "--retry_attempts",
        type=int,
        default=3,
        help="Number of retry attempts on failure",
    )

    # API settings
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="OpenAI API key (or set OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--api_base",
        type=str,
        default=None,
        help="Custom API base URL",
    )

    # Processing settings
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="Number of samples per batch",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=5,
        help="Maximum concurrent API calls",
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=10,
        help="Save checkpoint every N batches (0 to disable)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint if available",
    )
    parser.add_argument(
        "--filter_failures",
        action="store_true",
        help="Filter out samples where rubric generation failed",
    )

    args = parser.parse_args()

    # Run async main
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()

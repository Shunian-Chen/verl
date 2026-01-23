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
TACO Dataset Pipeline Validation Script.

This script validates the rubric generation pipeline using the BAAI/TACO dataset.
It tests:
1. Data loading from Hugging Face
2. Rubric generation (using MockRubricGenerator or LLM)
3. Latency bucket assignment
4. End-to-end pipeline execution

TACO Dataset (https://huggingface.co/datasets/BAAI/TACO):
- 25,433 training problems, 1,000 test problems
- Fields: question, solutions, input_output, difficulty, tags, etc.
- Competition-level algorithmic programming problems

Usage:
    # Quick validation with mock generator (no API key needed)
    python validate_taco_pipeline.py --mode mock --num_samples 10

    # Full validation with LLM generator
    python validate_taco_pipeline.py --mode llm --num_samples 5 --api_key YOUR_KEY

    # Test specific difficulty levels
    python validate_taco_pipeline.py --difficulty EASY --num_samples 20
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import logging
import sys
from pathlib import Path
from typing import Any

# Get project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
VERL_PKG_ROOT = PROJECT_ROOT / "verl"


def load_module_from_file(module_name: str, file_path: Path):
    """Load a Python module from file without going through package __init__.py"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# Load rubric modules directly to avoid ray import
schemas = load_module_from_file(
    "verl.rubric.schemas",
    VERL_PKG_ROOT / "rubric" / "schemas.py"
)
sampler = load_module_from_file(
    "verl.experimental.dataset.latency_bucketed_sampler",
    VERL_PKG_ROOT / "experimental" / "dataset" / "latency_bucketed_sampler.py"
)

# Import classes from loaded modules
ExecutableRubric = schemas.ExecutableRubric
VerificationItem = schemas.VerificationItem
EvidencePlan = schemas.EvidencePlan
AggregationRule = schemas.AggregationRule
DEFAULT_BUCKET_BOUNDARIES = sampler.DEFAULT_BUCKET_BOUNDARIES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_taco_dataset(
    split: str = "train",
    difficulties: list[str] | None = None,
    num_samples: int | None = None,
    local_dir: str | None = None,
) -> list[dict[str, Any]]:
    """
    Load TACO dataset from Hugging Face or local directory.

    Args:
        split: Dataset split ('train' or 'test').
        difficulties: Filter by difficulty levels.
        num_samples: Limit number of samples to load.
        local_dir: Path to local directory containing parquet files.

    Returns:
        List of dataset samples.
    """
    logger.info(f"Loading TACO dataset (split={split}, difficulties={difficulties})...")

    # Try loading from local directory first
    if local_dir:
        local_path = Path(local_dir)
        if local_path.exists():
            try:
                import pandas as pd
                pattern = f"{split}-*.parquet"
                files = sorted(local_path.glob(pattern))
                if files:
                    logger.info(f"Loading from local directory: {local_path}")
                    samples = []
                    for f in files:
                        df = pd.read_parquet(f)
                        samples.extend(df.to_dict(orient="records"))
                        if num_samples and len(samples) >= num_samples:
                            break

                    # Filter by difficulty if specified
                    if difficulties:
                        samples = [s for s in samples if s.get("difficulty") in difficulties]

                    # Limit samples if requested
                    if num_samples and num_samples < len(samples):
                        samples = samples[:num_samples]

                    logger.info(f"Loaded {len(samples)} samples from local files")
                    return samples
            except Exception as e:
                logger.warning(f"Failed to load from local directory: {e}")

    # Try loading from Hugging Face
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("Please install 'datasets' package: pip install datasets")
        return create_synthetic_taco_samples(num_samples or 10, difficulties)

    try:
        # Try loading the parquet files directly
        # TACO dataset has parquet files in the data/ directory
        dataset = load_dataset(
            "BAAI/TACO",
            split=split,
            data_files=f"data/{split}-*.parquet",
        )
    except Exception as e1:
        logger.warning(f"Failed to load parquet files directly: {e1}")
        try:
            # Try loading without trust_remote_code
            dataset = load_dataset("BAAI/TACO", split=split)
        except Exception as e2:
            logger.warning(f"Failed to load dataset: {e2}")
            # Fallback: Create synthetic test data
            logger.info("Creating synthetic test data for validation...")
            return create_synthetic_taco_samples(num_samples or 10, difficulties)

    # Convert to list of dicts
    samples = [dict(sample) for sample in dataset]

    # Filter by difficulty if specified
    if difficulties:
        samples = [s for s in samples if s.get("difficulty") in difficulties]

    # Limit samples if requested
    if num_samples and num_samples < len(samples):
        samples = samples[:num_samples]

    logger.info(f"Loaded {len(samples)} samples from TACO dataset")
    return samples


def create_synthetic_taco_samples(
    num_samples: int,
    difficulties: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Create synthetic TACO-like samples for testing when dataset loading fails.

    This allows testing the pipeline without requiring the actual dataset.
    """
    import random

    difficulties = difficulties or ["EASY", "MEDIUM", "MEDIUM_HARD", "HARD", "VERY_HARD"]

    # Sample algorithmic problems
    problem_templates = [
        {
            "question": """Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.

You may assume that each input would have exactly one solution, and you may not use the same element twice.

Example:
Input: nums = [2,7,11,15], target = 9
Output: [0,1]
Explanation: Because nums[0] + nums[1] == 9, we return [0, 1].""",
            "name": "two_sum",
            "tags": ["Array", "Hash Table"],
            "skill_types": ["Data structures"],
            "input_output": json.dumps({
                "inputs": ["[2,7,11,15]\n9", "[3,2,4]\n6", "[3,3]\n6"],
                "outputs": ["[0,1]", "[1,2]", "[0,1]"],
            }),
            "solutions": json.dumps([
                "def two_sum(nums, target):\n    seen = {}\n    for i, num in enumerate(nums):\n        if target - num in seen:\n            return [seen[target - num], i]\n        seen[num] = i\n    return []"
            ]),
        },
        {
            "question": """Write a function to find the longest common prefix string amongst an array of strings.

If there is no common prefix, return an empty string "".

Example 1:
Input: strs = ["flower","flow","flight"]
Output: "fl"

Example 2:
Input: strs = ["dog","racecar","car"]
Output: ""
Explanation: There is no common prefix among the input strings.""",
            "name": "longest_common_prefix",
            "tags": ["String"],
            "skill_types": ["Complete search"],
            "input_output": json.dumps({
                "inputs": ['["flower","flow","flight"]', '["dog","racecar","car"]'],
                "outputs": ['"fl"', '""'],
            }),
            "solutions": json.dumps([
                'def longest_common_prefix(strs):\n    if not strs:\n        return ""\n    prefix = strs[0]\n    for s in strs[1:]:\n        while not s.startswith(prefix):\n            prefix = prefix[:-1]\n            if not prefix:\n                return ""\n    return prefix'
            ]),
        },
        {
            "question": """Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.

An input string is valid if:
1. Open brackets must be closed by the same type of brackets.
2. Open brackets must be closed in the correct order.
3. Every close bracket has a corresponding open bracket of the same type.

Example 1:
Input: s = "()"
Output: true

Example 2:
Input: s = "([)]"
Output: false""",
            "name": "valid_parentheses",
            "tags": ["String", "Stack"],
            "skill_types": ["Data structures"],
            "input_output": json.dumps({
                "inputs": ['"()"', '"()[]{}"', '"(]"', '"([)]"'],
                "outputs": ["true", "true", "false", "false"],
            }),
            "solutions": json.dumps([
                "def is_valid(s):\n    stack = []\n    mapping = {')': '(', '}': '{', ']': '['}\n    for char in s:\n        if char in mapping:\n            if not stack or stack.pop() != mapping[char]:\n                return False\n        else:\n            stack.append(char)\n    return not stack"
            ]),
        },
        {
            "question": """You are climbing a staircase. It takes n steps to reach the top.

Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?

Example 1:
Input: n = 2
Output: 2
Explanation: There are two ways to climb to the top.
1. 1 step + 1 step
2. 2 steps

Example 2:
Input: n = 3
Output: 3""",
            "name": "climbing_stairs",
            "tags": ["Math", "Dynamic Programming"],
            "skill_types": ["Dynamic programming"],
            "input_output": json.dumps({
                "inputs": ["2", "3", "4", "5"],
                "outputs": ["2", "3", "5", "8"],
            }),
            "solutions": json.dumps([
                "def climb_stairs(n):\n    if n <= 2:\n        return n\n    a, b = 1, 2\n    for _ in range(3, n + 1):\n        a, b = b, a + b\n    return b"
            ]),
        },
        {
            "question": """Given an integer array nums, find the subarray with the largest sum, and return its sum.

A subarray is a contiguous non-empty sequence of elements within an array.

Example 1:
Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
Output: 6
Explanation: The subarray [4,-1,2,1] has the largest sum 6.

Example 2:
Input: nums = [5,4,-1,7,8]
Output: 23""",
            "name": "maximum_subarray",
            "tags": ["Array", "Dynamic Programming", "Divide and Conquer"],
            "skill_types": ["Dynamic programming", "Greedy algorithms"],
            "input_output": json.dumps({
                "inputs": ["[-2,1,-3,4,-1,2,1,-5,4]", "[1]", "[5,4,-1,7,8]"],
                "outputs": ["6", "1", "23"],
            }),
            "solutions": json.dumps([
                "def max_subarray(nums):\n    max_sum = current_sum = nums[0]\n    for num in nums[1:]:\n        current_sum = max(num, current_sum + num)\n        max_sum = max(max_sum, current_sum)\n    return max_sum"
            ]),
        },
    ]

    samples = []
    for i in range(num_samples):
        template = problem_templates[i % len(problem_templates)]
        difficulty = random.choice(difficulties)
        sample = {
            **template,
            "difficulty": difficulty,
            "source": "synthetic",
        }
        samples.append(sample)

    logger.info(f"Created {len(samples)} synthetic TACO-like samples")
    return samples


def parse_taco_sample(sample: dict[str, Any]) -> dict[str, Any]:
    """
    Parse a TACO sample into the format expected by the rubric pipeline.

    TACO fields:
    - question: Problem description
    - solutions: List of solution codes (JSON string)
    - input_output: Test cases as JSON string
    - difficulty: Problem difficulty
    - tags: Problem tags/categories

    Returns:
        Parsed sample with:
        - prompt: Task input
        - verification_data: Parsed test cases
        - metadata: Additional info
    """
    # Extract question as the prompt
    prompt = sample.get("question", "")

    # Parse input_output as verification data
    verification_data = None
    input_output_str = sample.get("input_output", "")
    if input_output_str:
        try:
            io_data = json.loads(input_output_str)
            # TACO format: {"inputs": [...], "outputs": [...]}
            if isinstance(io_data, dict):
                inputs = io_data.get("inputs", [])
                outputs = io_data.get("outputs", [])
                verification_data = [
                    {"input": inp, "expected_output": out}
                    for inp, out in zip(inputs, outputs)
                ]
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse input_output for sample")

    # Parse solutions
    solutions = []
    solutions_str = sample.get("solutions", "")
    if solutions_str:
        try:
            solutions = json.loads(solutions_str)
        except json.JSONDecodeError:
            pass

    return {
        "prompt": prompt,
        "verification_data": verification_data,
        "solutions": solutions,
        "metadata": {
            "difficulty": sample.get("difficulty", ""),
            "tags": sample.get("tags", []),
            "skill_types": sample.get("skill_types", []),
            "source": sample.get("source", ""),
            "name": sample.get("name", ""),
        },
    }


def create_mock_rubric(sample: dict[str, Any], sample_idx: int) -> dict[str, Any]:
    """
    Create a mock rubric for testing the pipeline without LLM calls.

    This generates a realistic rubric structure based on the sample metadata.
    """
    # Use globally imported classes
    metadata = sample.get("metadata", {})
    verification_data = sample.get("verification_data", [])

    # Create verification items
    checklist = []

    # Always add a code execution check
    checklist.append(
        VerificationItem(
            id=f"check-exec-{sample_idx}",
            description="Code executes without errors",
            category="correctness",
            weight=1.0,
            is_gate=True,
            scoring_primitive="code_execution",
        )
    )

    # Add test case checks based on verification data
    num_tests = len(verification_data) if verification_data else 3
    for i in range(min(num_tests, 5)):  # Limit to 5 test cases
        checklist.append(
            VerificationItem(
                id=f"check-test-{sample_idx}-{i}",
                description=f"Passes test case {i + 1}",
                category="correctness",
                weight=2.0,
                scoring_primitive="code_execution",
            )
        )

    # Add style check for harder problems
    difficulty = metadata.get("difficulty", "")
    if difficulty in ["MEDIUM_HARD", "HARD", "VERY_HARD"]:
        checklist.append(
            VerificationItem(
                id=f"check-style-{sample_idx}",
                description="Code follows good style practices",
                category="style",
                weight=0.5,
                scoring_primitive="contains",
                scoring_config={"expected_value": "def "},
            )
        )

    # Create evidence plans
    evidence_plans = []

    # Add sandbox execution plan
    evidence_plans.append(
        EvidencePlan(
            check_id=f"check-exec-{sample_idx}",
            tool_name="sandbox_fusion",
            tool_arguments={
                "language": "python",
                "timeout": 10,
            },
            argument_extractor="{{ policy_output }}",
            timeout_ms=30000,
        )
    )

    # Add test runner plans
    for i, item in enumerate(checklist[1:], 1):
        if item.category == "correctness":
            test_data = verification_data[i - 1] if verification_data and i <= len(verification_data) else {}
            evidence_plans.append(
                EvidencePlan(
                    check_id=item.id,
                    tool_name="test_runner",
                    tool_arguments={
                        "test_input": test_data.get("input", ""),
                        "expected_output": test_data.get("expected_output", ""),
                    },
                    timeout_ms=10000,
                )
            )

    # Estimate latency based on number of evidence plans and difficulty
    base_latency = len(evidence_plans) * 5000  # 5s per plan
    difficulty_multiplier = {
        "EASY": 1.0,
        "MEDIUM": 1.5,
        "MEDIUM_HARD": 2.0,
        "HARD": 3.0,
        "VERY_HARD": 4.0,
    }.get(difficulty, 1.5)
    estimated_latency = int(base_latency * difficulty_multiplier)

    # Create rubric
    rubric = ExecutableRubric(
        id=f"taco-rubric-{sample_idx}",
        task_intent=f"Solve algorithmic problem: {metadata.get('name', 'unknown')}",
        verification_checklist=checklist,
        evidence_plans=evidence_plans,
        aggregation=AggregationRule(
            method="weighted_sum",
            normalize=True,
            gate_threshold=0.5,
        ),
        estimated_latency_ms=estimated_latency,
    )

    return rubric.model_dump()


async def generate_rubric_with_llm(
    sample: dict[str, Any],
    sample_idx: int,
    generator,
    tool_schemas: list,
) -> dict[str, Any]:
    """
    Generate a rubric using the LLM generator.
    """
    try:
        rubric = await generator.generate(
            task_input=sample["prompt"],
            tool_schemas=tool_schemas,
            verification_data=sample.get("verification_data"),
        )
        return rubric.model_dump()
    except Exception as e:
        logger.error(f"Failed to generate rubric for sample {sample_idx}: {e}")
        return None


def compute_latency_bucket(latency_ms: int) -> int:
    """Compute latency bucket for a given latency."""
    # Use globally imported DEFAULT_BUCKET_BOUNDARIES
    for i, boundary in enumerate(DEFAULT_BUCKET_BOUNDARIES):
        if latency_ms < boundary:
            return i
    return len(DEFAULT_BUCKET_BOUNDARIES)


def print_sample_info(sample: dict[str, Any], parsed: dict[str, Any], idx: int):
    """Print information about a sample."""
    metadata = parsed.get("metadata", {})
    verification_data = parsed.get("verification_data", [])

    print(f"\n{'='*60}")
    print(f"Sample {idx + 1}")
    print(f"{'='*60}")
    print(f"Name: {metadata.get('name', 'N/A')}")
    print(f"Difficulty: {metadata.get('difficulty', 'N/A')}")
    print(f"Source: {metadata.get('source', 'N/A')}")
    print(f"Tags: {metadata.get('tags', [])}")
    print(f"Skill Types: {metadata.get('skill_types', [])}")
    print(f"Number of test cases: {len(verification_data) if verification_data else 0}")
    print(f"Prompt preview: {parsed['prompt'][:200]}...")


def print_rubric_info(rubric: dict[str, Any], idx: int):
    """Print information about a generated rubric."""
    if rubric is None:
        print(f"  Rubric: FAILED TO GENERATE")
        return

    print(f"\n  Rubric ID: {rubric['id']}")
    print(f"  Task Intent: {rubric['task_intent'][:100]}...")
    print(f"  Verification Items: {len(rubric['verification_checklist'])}")
    print(f"  Evidence Plans: {len(rubric['evidence_plans'])}")
    print(f"  Aggregation Method: {rubric['aggregation']['method']}")
    print(f"  Estimated Latency: {rubric['estimated_latency_ms']}ms")
    print(f"  Latency Bucket: {compute_latency_bucket(rubric['estimated_latency_ms'])}")

    # Print verification items
    print(f"\n  Verification Checklist:")
    for item in rubric["verification_checklist"][:5]:  # Limit to first 5
        gate_marker = " [GATE]" if item.get("is_gate") else ""
        print(f"    - {item['id']}: {item['description'][:50]}... (weight={item['weight']}{gate_marker})")

    if len(rubric["verification_checklist"]) > 5:
        print(f"    ... and {len(rubric['verification_checklist']) - 5} more items")


async def run_validation(args: argparse.Namespace):
    """Run the validation pipeline."""
    # Load TACO dataset
    difficulties = [args.difficulty] if args.difficulty != "ALL" else None
    samples = load_taco_dataset(
        split=args.split,
        difficulties=difficulties,
        num_samples=args.num_samples,
        local_dir=args.local_dir,
    )

    if not samples:
        logger.error("No samples loaded!")
        return

    # Parse samples
    parsed_samples = [parse_taco_sample(sample) for sample in samples]

    # Initialize generator if using LLM mode
    generator = None
    tool_schemas = []

    if args.mode == "llm":
        try:
            from verl.rubric.generator import LLMRubricGenerator, RubricGeneratorConfig
            from verl.rubric.generator.base import ToolSchema

            config = RubricGeneratorConfig(
                model_name=args.model,
                temperature=0.0,
            )
            generator = LLMRubricGenerator(
                config=config,
                api_key=args.api_key,
                api_base=args.api_base,
            )

            # Create basic tool schemas
            tool_schemas = [
                ToolSchema(
                    name="sandbox_fusion",
                    description="Execute code in a sandboxed environment",
                    parameters={
                        "type": "object",
                        "properties": {
                            "code": {"type": "string", "description": "Code to execute"},
                            "language": {"type": "string", "description": "Programming language"},
                            "timeout": {"type": "integer", "description": "Timeout in seconds"},
                        },
                        "required": ["code"],
                    },
                ),
                ToolSchema(
                    name="test_runner",
                    description="Run test cases against code",
                    parameters={
                        "type": "object",
                        "properties": {
                            "code": {"type": "string"},
                            "test_input": {"type": "string"},
                            "expected_output": {"type": "string"},
                        },
                        "required": ["code", "test_input", "expected_output"],
                    },
                ),
            ]
            logger.info(f"Initialized LLM generator with model: {args.model}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM generator: {e}")
            logger.info("Falling back to mock mode")
            args.mode = "mock"

    # Process samples
    results = []
    bucket_distribution = {i: 0 for i in range(5)}

    print("\n" + "=" * 60)
    print("TACO Pipeline Validation")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Samples: {len(parsed_samples)}")
    print(f"Difficulty filter: {args.difficulty}")

    for idx, (sample, parsed) in enumerate(zip(samples, parsed_samples)):
        if args.verbose:
            print_sample_info(sample, parsed, idx)

        # Generate rubric
        if args.mode == "mock":
            rubric = create_mock_rubric(parsed, idx)
        else:
            rubric = await generate_rubric_with_llm(
                parsed, idx, generator, tool_schemas
            )

        if rubric:
            bucket = compute_latency_bucket(rubric["estimated_latency_ms"])
            bucket_distribution[bucket] += 1
            results.append({
                "sample_idx": idx,
                "rubric": rubric,
                "latency_bucket": bucket,
                "metadata": parsed.get("metadata", {}),
            })

            if args.verbose:
                print_rubric_info(rubric, idx)
        else:
            logger.warning(f"Failed to generate rubric for sample {idx}")

    # Print summary
    print("\n" + "=" * 60)
    print("Validation Summary")
    print("=" * 60)

    success_rate = len(results) / len(parsed_samples) * 100 if parsed_samples else 0
    print(f"Total samples: {len(parsed_samples)}")
    print(f"Successful rubrics: {len(results)} ({success_rate:.1f}%)")

    print("\nLatency Bucket Distribution:")
    bucket_labels = ["<1s", "1-5s", "5-15s", "15-30s", ">30s"]
    for bucket, label in enumerate(bucket_labels):
        count = bucket_distribution[bucket]
        pct = count / len(results) * 100 if results else 0
        bar = "#" * int(pct / 2)
        print(f"  Bucket {bucket} ({label:>7}): {count:4} ({pct:5.1f}%) {bar}")

    # Difficulty distribution
    print("\nDifficulty Distribution:")
    difficulty_counts = {}
    for r in results:
        diff = r["metadata"].get("difficulty", "UNKNOWN")
        difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1
    for diff, count in sorted(difficulty_counts.items()):
        pct = count / len(results) * 100 if results else 0
        print(f"  {diff:12}: {count:4} ({pct:5.1f}%)")

    # Save results if output path specified
    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {args.output_path}")

    print("\nValidation complete!")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Validate rubric generation pipeline with TACO dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset options
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Dataset split to use",
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        default="EASY",
        choices=["EASY", "MEDIUM", "MEDIUM_HARD", "HARD", "VERY_HARD", "ALL"],
        help="Difficulty level filter",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of samples to process",
    )
    parser.add_argument(
        "--local_dir",
        type=str,
        default=None,
        help="Path to local directory containing TACO parquet files",
    )

    # Mode options
    parser.add_argument(
        "--mode",
        type=str,
        default="mock",
        choices=["mock", "llm"],
        help="Generation mode: 'mock' for testing, 'llm' for real generation",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="LLM model to use (for llm mode)",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="API key for LLM (or set OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--api_base",
        type=str,
        default=None,
        help="Custom API base URL",
    )

    # Output options
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save validation results (JSON)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed information for each sample",
    )

    args = parser.parse_args()
    asyncio.run(run_validation(args))


if __name__ == "__main__":
    main()

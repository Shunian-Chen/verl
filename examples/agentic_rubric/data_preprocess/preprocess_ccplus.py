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
Preprocess ccplus-1x-optimized dataset for agentic rubric RL training.

Converts ccplus parquet data into verl training format with executable rubrics.
No LLM calls required - rubrics are deterministically constructed from the
dataset's test cases and reference solutions.

Each problem produces 3 verification items:
  - check_correctness (gate): test case pass rate
  - check_time: execution time ratio vs time_opt reference
  - check_memory: memory usage ratio vs memory_opt reference

Usage:
    python preprocess_ccplus.py \\
        --input_path /path/to/ccplus/parquet/dir \\
        --output_dir examples/agentic_rubric/data/ccplus/ \\
        --train_ratio 0.9
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

# Add verl to path if running as script
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from verl.experimental.dataset.latency_bucketed_sampler import compute_latency_bucket
from verl.rubric.schemas import (
    AggregationMethod,
    AggregationRule,
    EvidencePlan,
    ExecutableRubric,
    ScoringPrimitive,
    VerificationCategory,
    VerificationItem,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a competitive programmer. Write a complete Python solution that "
    "reads from stdin and writes to stdout. Your solution should be correct, "
    "efficient in both time and memory. Output ONLY the code, no explanations."
)


def build_rubric_id(title: str) -> str:
    """Generate a deterministic rubric ID from the problem title."""
    h = hashlib.sha256(title.encode("utf-8")).hexdigest()[:12]
    return f"rubric_ccplus_{h}"


def build_prompt(title: str, description: str, time_limit: int, memory_limit: int) -> list[dict[str, str]]:
    """Build chat-format prompt for the problem."""
    user_content = (
        f"## {title}\n\n"
        f"{description}\n\n"
        f"Time Limit: {time_limit}ms\n"
        f"Memory Limit: {memory_limit}MB"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def map_test_cases(test_cases: list[dict[str, str]]) -> list[dict[str, str]]:
    """Map ccplus test case format to test_runner format.

    ccplus uses 'output' while test_runner expects 'expected_output'.
    """
    mapped = []
    for tc in test_cases:
        mapped.append({
            "input": tc.get("input", ""),
            "expected_output": tc.get("output", tc.get("expected_output", "")),
        })
    return mapped


def estimate_latency(time_limit: int, num_tests: int, num_evidence_plans: int, latency_factor: float) -> int:
    """Estimate total rubric execution latency in ms.

    Plans 2-5 run in parallel (after correctness gate), so the total is
    roughly: correctness_time + max(parallel_plans_time).
    """
    per_test_ms = time_limit * latency_factor
    correctness_ms = per_test_ms * num_tests
    # Parallel plans: each runs all tests, take the max (= one plan's time)
    parallel_ms = per_test_ms * num_tests if num_evidence_plans > 1 else 0
    return int(correctness_ms + parallel_ms + 5000)  # 5s overhead


def build_rubric(
    title: str,
    description: str,
    time_limit: int,
    memory_limit: int,
    test_cases: list[dict[str, str]],
    time_opt: str | None,
    memory_opt: str | None,
    latency_factor: float,
) -> ExecutableRubric:
    """Build an ExecutableRubric for a ccplus problem.

    Args:
        title: Problem title.
        description: Problem description.
        time_limit: Time limit per test case in ms.
        memory_limit: Memory limit in MB.
        test_cases: List of test cases with input/output.
        time_opt: Time-optimized reference solution (or None).
        memory_opt: Memory-optimized reference solution (or None).
        latency_factor: Factor for latency estimation.

    Returns:
        ExecutableRubric with correctness gate and optional efficiency checks.
    """
    mapped_cases = map_test_cases(test_cases)
    num_tests = len(mapped_cases)
    rubric_id = build_rubric_id(title)

    verification_checklist: list[VerificationItem] = []
    evidence_plans: list[EvidencePlan] = []

    # --- 1. Correctness check (gate) ---
    verification_checklist.append(
        VerificationItem(
            id="check_correctness",
            description=f"All {num_tests} test cases pass",
            category=VerificationCategory.CORRECTNESS,
            weight=0.0,
            is_gate=True,
            scoring_primitive=ScoringPrimitive.CODE_EXECUTION,
            scoring_config={"partial_credit": True},
        )
    )
    evidence_plans.append(
        EvidencePlan(
            check_id="check_correctness",
            tool_name="test_runner",
            tool_arguments={
                "test_cases": mapped_cases,
                "language": "python",
                "timeout_per_test_ms": time_limit,
            },
            argument_extractor='{"code": {{ policy_output | tojson }}}',
            timeout_ms=time_limit * num_tests + 5000,
        )
    )

    # --- 2. Time efficiency check (if time_opt available) ---
    if time_opt:
        verification_checklist.append(
            VerificationItem(
                id="check_time",
                description="Time efficiency vs time-optimized reference",
                category=VerificationCategory.EFFICIENCY,
                weight=1.0,
                is_gate=False,
                scoring_primitive=ScoringPrimitive.PERFORMANCE_RATIO,
                scoring_config={
                    "metric_field": "execution_time_ms",
                    "direction": "lower_is_better",
                },
            )
        )
        # Plan: run model code for time metrics
        evidence_plans.append(
            EvidencePlan(
                check_id="check_time",
                tool_name="test_runner",
                tool_arguments={
                    "test_cases": mapped_cases,
                    "language": "python",
                    "timeout_per_test_ms": time_limit,
                    "role": "model",
                },
                argument_extractor='{"code": {{ policy_output | tojson }}}',
                depends_on=["check_correctness"],
                timeout_ms=time_limit * num_tests + 5000,
            )
        )
        # Plan: run time_opt reference code
        evidence_plans.append(
            EvidencePlan(
                check_id="check_time",
                tool_name="test_runner",
                tool_arguments={
                    "test_cases": mapped_cases,
                    "language": "python",
                    "timeout_per_test_ms": time_limit,
                    "code": time_opt,
                    "role": "reference",
                },
                depends_on=["check_correctness"],
                timeout_ms=time_limit * num_tests + 5000,
            )
        )

    # --- 3. Memory efficiency check (if memory_opt available) ---
    if memory_opt:
        verification_checklist.append(
            VerificationItem(
                id="check_memory",
                description="Memory efficiency vs memory-optimized reference",
                category=VerificationCategory.EFFICIENCY,
                weight=1.0,
                is_gate=False,
                scoring_primitive=ScoringPrimitive.PERFORMANCE_RATIO,
                scoring_config={
                    "metric_field": "max_memory_mb",
                    "direction": "lower_is_better",
                },
            )
        )
        # Plan: run model code for memory metrics
        evidence_plans.append(
            EvidencePlan(
                check_id="check_memory",
                tool_name="test_runner",
                tool_arguments={
                    "test_cases": mapped_cases,
                    "language": "python",
                    "timeout_per_test_ms": time_limit,
                    "role": "model",
                },
                argument_extractor='{"code": {{ policy_output | tojson }}}',
                depends_on=["check_correctness"],
                timeout_ms=time_limit * num_tests + 5000,
            )
        )
        # Plan: run memory_opt reference code
        evidence_plans.append(
            EvidencePlan(
                check_id="check_memory",
                tool_name="test_runner",
                tool_arguments={
                    "test_cases": mapped_cases,
                    "language": "python",
                    "timeout_per_test_ms": time_limit,
                    "code": memory_opt,
                    "role": "reference",
                },
                depends_on=["check_correctness"],
                timeout_ms=time_limit * num_tests + 5000,
            )
        )

    estimated_latency = estimate_latency(
        time_limit, num_tests, len(evidence_plans), latency_factor,
    )

    return ExecutableRubric(
        id=rubric_id,
        task_intent=f"Solve competitive programming problem: {title}",
        verification_checklist=verification_checklist,
        evidence_plans=evidence_plans,
        aggregation=AggregationRule(
            method=AggregationMethod.WEIGHTED_SUM,
            normalize=True,
            gate_threshold=0.5,
        ),
        estimated_latency_ms=estimated_latency,
        metadata={
            "source": "ccplus-1x-optimized",
            "title": title,
            "time_limit": time_limit,
            "memory_limit": memory_limit,
            "num_tests": num_tests,
            "has_time_opt": bool(time_opt),
            "has_memory_opt": bool(memory_opt),
        },
    )


def parse_test_cases(raw: Any) -> list[dict[str, str]]:
    """Parse test_cases from various formats (str, list, etc.)."""
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return []
    if isinstance(raw, list):
        return raw
    return []


def parse_code_field(raw: Any) -> str | None:
    """Parse a code field, returning None if empty or invalid."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s.lower() in ("none", "nan", "null", ""):
        return None
    return s


def find_parquet_files(input_path: str) -> list[Path]:
    """Find parquet files in the input path (file or directory)."""
    p = Path(input_path)
    if p.is_file() and p.suffix == ".parquet":
        return [p]
    if p.is_dir():
        files = sorted(p.glob("**/*.parquet"))
        if files:
            return files
    raise FileNotFoundError(f"No parquet files found at: {input_path}")


def process_parquet_files(
    parquet_files: list[Path],
    latency_factor: float,
) -> list[dict[str, Any]]:
    """Process parquet files and build verl-format rows.

    Uses streaming reads to handle large files without OOM.
    """
    rows: list[dict[str, Any]] = []
    skipped = 0

    for pf in parquet_files:
        logger.info(f"Reading {pf}...")
        parquet_file = pq.ParquetFile(pf)

        for batch in parquet_file.iter_batches(batch_size=100):
            table = batch.to_pydict()
            batch_size = len(table.get("title", []))

            for i in range(batch_size):
                title = str(table.get("title", [""])[i]).strip()
                description = str(table.get("description", [""])[i]).strip()

                if not title or not description:
                    skipped += 1
                    continue

                # Parse numeric fields with defaults
                try:
                    time_limit = int(table.get("time_limit", [2000])[i])
                except (ValueError, TypeError):
                    time_limit = 2000
                try:
                    memory_limit = int(table.get("memory_limit", [256])[i])
                except (ValueError, TypeError):
                    memory_limit = 256

                # Parse test cases
                test_cases_raw = table.get("test_cases", [None])[i]
                test_cases = parse_test_cases(test_cases_raw)
                if not test_cases:
                    skipped += 1
                    continue

                # Parse reference solutions
                time_opt = parse_code_field(table.get("time_opt", [None])[i])
                memory_opt = parse_code_field(table.get("memory_opt", [None])[i])

                # Build prompt
                prompt = build_prompt(title, description, time_limit, memory_limit)

                # Build rubric
                rubric = build_rubric(
                    title=title,
                    description=description,
                    time_limit=time_limit,
                    memory_limit=memory_limit,
                    test_cases=test_cases,
                    time_opt=time_opt,
                    memory_opt=memory_opt,
                    latency_factor=latency_factor,
                )

                # Validate round-trip
                rubric_dict = rubric.model_dump()
                ExecutableRubric.model_validate(rubric_dict)

                latency_bucket = compute_latency_bucket(rubric.estimated_latency_ms)

                rows.append({
                    "prompt": prompt,
                    "extra_info": {
                        "rubric": rubric_dict,
                        "latency_bucket": latency_bucket,
                    },
                    "data_source": "ccplus",
                })

    if skipped:
        logger.warning(f"Skipped {skipped} rows (missing title/description/test_cases)")

    return rows


def split_and_save(
    rows: list[dict[str, Any]],
    output_dir: str,
    train_ratio: float,
    seed: int,
) -> None:
    """Split rows into train/val and save as parquet."""
    import random

    import pandas as pd

    random.seed(seed)
    indices = list(range(len(rows)))
    random.shuffle(indices)

    split_idx = int(len(rows) * train_ratio)
    train_indices = sorted(indices[:split_idx])
    val_indices = sorted(indices[split_idx:])

    train_rows = [rows[i] for i in train_indices]
    val_rows = [rows[i] for i in val_indices]

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for name, data in [("train", train_rows), ("val", val_rows)]:
        if not data:
            logger.warning(f"No data for {name} split, skipping")
            continue
        df = pd.DataFrame(data)
        path = out / f"{name}.parquet"
        df.to_parquet(path, index=False)
        logger.info(f"Saved {len(df)} rows to {path}")


def print_statistics(rows: list[dict[str, Any]]) -> None:
    """Print dataset statistics."""
    total = len(rows)
    logger.info(f"Total processed rows: {total}")

    # Count check types
    has_time = sum(
        1 for r in rows
        if any(v["id"] == "check_time" for v in r["extra_info"]["rubric"]["verification_checklist"])
    )
    has_memory = sum(
        1 for r in rows
        if any(v["id"] == "check_memory" for v in r["extra_info"]["rubric"]["verification_checklist"])
    )
    has_both = sum(
        1 for r in rows
        if (
            any(v["id"] == "check_time" for v in r["extra_info"]["rubric"]["verification_checklist"])
            and any(v["id"] == "check_memory" for v in r["extra_info"]["rubric"]["verification_checklist"])
        )
    )

    logger.info(f"  With time_opt: {has_time} ({100*has_time/total:.1f}%)")
    logger.info(f"  With memory_opt: {has_memory} ({100*has_memory/total:.1f}%)")
    logger.info(f"  With both: {has_both} ({100*has_both/total:.1f}%)")

    # Latency bucket distribution
    bucket_counts: dict[int, int] = {}
    for r in rows:
        b = r["extra_info"]["latency_bucket"]
        bucket_counts[b] = bucket_counts.get(b, 0) + 1

    logger.info("Latency bucket distribution:")
    for bucket in sorted(bucket_counts.keys()):
        count = bucket_counts[bucket]
        pct = 100 * count / total
        logger.info(f"  Bucket {bucket}: {count} ({pct:.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess ccplus-1x-optimized dataset for rubric RL training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to ccplus parquet file or directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="examples/agentic_rubric/data/ccplus/",
        help="Output directory for train/val parquet files",
    )
    parser.add_argument(
        "--latency_factor",
        type=float,
        default=0.3,
        help="Factor for latency estimation (time_limit * factor * num_tests)",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.9,
        help="Fraction of data for training split",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val split",
    )

    args = parser.parse_args()

    # Find and process parquet files
    parquet_files = find_parquet_files(args.input_path)
    logger.info(f"Found {len(parquet_files)} parquet file(s)")

    rows = process_parquet_files(parquet_files, args.latency_factor)

    if not rows:
        logger.error("No valid rows produced. Check input data.")
        sys.exit(1)

    # Print statistics
    print_statistics(rows)

    # Split and save
    split_and_save(rows, args.output_dir, args.train_ratio, args.seed)

    logger.info("Preprocessing complete.")


if __name__ == "__main__":
    main()

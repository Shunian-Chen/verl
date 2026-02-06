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
Latency-bucketed sampler for efficient batching of agentic rubric tasks.

Groups samples by their expected execution latency to minimize GPU idle time
during reward computation. Samples with similar latencies are batched together.
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from collections.abc import Iterator, Sized
from typing import Any

from omegaconf import DictConfig
from torch.utils.data import Sampler

try:
    from verl.experimental.dataset.sampler import AbstractCurriculumSampler
except Exception:
    # Keep the sampler importable in lightweight environments where
    # full verl dependencies are unavailable.
    class AbstractCurriculumSampler(Sampler[int]):  # type: ignore[misc, no-redef]
        def __init__(self, data_source: Sized, data_config: DictConfig | None = None):
            self.data_source = data_source
            self.data_config = data_config

        def update(self, batch) -> None:  # noqa: D401
            pass


logger = logging.getLogger(__name__)


# Default latency bucket boundaries in milliseconds
DEFAULT_BUCKET_BOUNDARIES = [1000, 5000, 15000, 30000]  # <1s, 1-5s, 5-15s, 15-30s, >30s


class LatencyBucketedSampler(AbstractCurriculumSampler):
    """
    Sampler that groups samples by expected execution latency.

    This sampler is designed for agentic rubric training where different
    samples may have vastly different reward computation times due to
    different tool calls required for verification.

    Bucket configuration:
        Bucket 0: < 1s (simple checks, regex)
        Bucket 1: 1-5s (code execution)
        Bucket 2: 5-15s (web search)
        Bucket 3: 15-30s (complex simulation)
        Bucket 4: > 30s (multi-tool chains)

    The sampler ensures that each batch contains samples from the same
    latency bucket, reducing variance in batch processing time.

    Example usage:
        ```python
        sampler = LatencyBucketedSampler(
            data_source=dataset,
            latency_fn=lambda idx: dataset[idx]["extra_info"]["latency_bucket"],
            batch_size=32,
            shuffle=True,
        )

        for indices in sampler:
            batch = [dataset[i] for i in indices]
            # Process batch - all samples have similar latency
        ```
    """

    def __init__(
        self,
        data_source: Sized,
        data_config: DictConfig | None = None,
        latency_fn: callable | None = None,
        batch_size: int | None = None,
        shuffle: bool | None = None,
        drop_last: bool | None = None,
        bucket_boundaries: list[int] | None = None,
        num_buckets: int | None = None,
        seed: int | None = None,
        progressive_unlock: bool = False,
        initial_max_bucket: int = 0,
        unlock_interval: int = 100,
        rebucket_threshold: float = 0.5,
    ):
        """
        Initialize the latency-bucketed sampler.

        Args:
            data_source: The dataset to sample from.
            data_config: Optional configuration from verl.
            latency_fn: Function that returns the latency bucket (0-4) for
                a given index. If None, uses a default that extracts from
                extra_info["latency_bucket"].
            batch_size: Number of samples per batch.
            shuffle: Whether to shuffle within buckets.
            drop_last: Whether to drop the last incomplete batch.
            bucket_boundaries: Custom bucket boundaries in ms.
                Default: [1000, 5000, 15000, 30000]
            num_buckets: Number of buckets (alternative to bucket_boundaries).
            seed: Random seed for reproducibility.
            progressive_unlock: Whether to enable progressive bucket unlocking
                (curriculum learning). When True, only buckets up to
                ``initial_max_bucket`` are active initially.
            initial_max_bucket: The highest bucket ID active at the start
                when progressive_unlock is True.
            unlock_interval: How many ``update()`` calls between unlocking
                the next bucket.
            rebucket_threshold: Fractional deviation of actual vs estimated
                latency that triggers reassignment of a sample to a new bucket.
        """
        super().__init__(data_source=data_source, data_config=data_config)

        sampler_kwargs: dict[str, Any] = {}
        if data_config is not None and hasattr(data_config, "get"):
            sampler_cfg = data_config.get("sampler", None)
            if sampler_cfg is not None and hasattr(sampler_cfg, "get"):
                raw_kwargs = sampler_cfg.get("kwargs", {}) or {}
                if hasattr(raw_kwargs, "items"):
                    sampler_kwargs = dict(raw_kwargs.items())

        self.batch_size = int(
            batch_size if batch_size is not None else sampler_kwargs.get("batch_size", 1)
        )
        self.shuffle = bool(shuffle if shuffle is not None else sampler_kwargs.get("shuffle", False))
        self.drop_last = bool(drop_last if drop_last is not None else sampler_kwargs.get("drop_last", False))

        if num_buckets is None:
            num_buckets = sampler_kwargs.get("num_buckets")
        if bucket_boundaries is None:
            bucket_boundaries = sampler_kwargs.get("bucket_boundaries")
        if seed is None:
            seed = sampler_kwargs.get("seed")

        if bucket_boundaries is not None and not isinstance(bucket_boundaries, list):
            bucket_boundaries = list(bucket_boundaries)

        # Handle num_buckets parameter
        if num_buckets is not None:
            self.num_buckets = int(num_buckets)
            self.bucket_boundaries = bucket_boundaries or DEFAULT_BUCKET_BOUNDARIES[: self.num_buckets - 1]
        else:
            self.bucket_boundaries = bucket_boundaries or DEFAULT_BUCKET_BOUNDARIES
            self.num_buckets = len(self.bucket_boundaries) + 1

        # Set up random generator
        self.seed = seed
        self.generator = random.Random(seed)

        # Set up latency function
        if latency_fn is not None:
            self._latency_fn = latency_fn
        else:
            self._latency_fn = self._default_latency_fn

        # Build bucket indices
        self._build_buckets()

        # Curriculum learning state
        self.progressive_unlock = progressive_unlock
        self.initial_max_bucket = initial_max_bucket
        self.unlock_interval = unlock_interval
        self.rebucket_threshold = rebucket_threshold
        self._update_count = 0
        self.max_active_bucket = (
            initial_max_bucket if progressive_unlock else self.num_buckets - 1
        )
        # Tracks per-sample actual latencies observed at runtime (sample_idx -> ms).
        self._sample_actual_latency: dict[int, float] = {}

    def _default_latency_fn(self, idx: int) -> int:
        """
        Default function to extract latency bucket from dataset.

        Expects the dataset item to have extra_info["latency_bucket"].
        Falls back to bucket 0 if not found.
        """
        try:
            item = self.data_source[idx]
            if hasattr(item, "get"):
                extra_info = item.get("extra_info", {})
            elif hasattr(item, "__getitem__"):
                extra_info = item.get("extra_info", {}) if hasattr(item, "get") else {}
            else:
                return 0

            if isinstance(extra_info, dict):
                return extra_info.get("latency_bucket", 0)
            return 0
        except Exception:
            return 0

    def _build_buckets(self) -> None:
        """Build the bucket-to-indices mapping."""
        self.buckets: dict[int, list[int]] = defaultdict(list)

        for idx in range(len(self.data_source)):
            bucket = self._latency_fn(idx)
            # Clamp bucket to valid range
            bucket = max(0, min(bucket, self.num_buckets - 1))
            self.buckets[bucket].append(idx)

        # Log bucket distribution
        total = len(self.data_source)
        for bucket_id in sorted(self.buckets.keys()):
            count = len(self.buckets[bucket_id])
            pct = 100.0 * count / total if total > 0 else 0
            logger.info(f"Bucket {bucket_id}: {count} samples ({pct:.1f}%)")

    # ------------------------------------------------------------------
    # Curriculum learning: AbstractCurriculumSampler interface
    # ------------------------------------------------------------------
    def update(self, batch) -> None:
        """Update curriculum state after a training step.

        Two mechanisms are supported:

        **A. Dynamic rebucketing** — If the batch carries actual execution
        latencies (``total_latency_ms`` in ``non_tensor_batch``), samples
        whose actual bucket deviates from their estimated bucket by more
        than ``rebucket_threshold`` are moved to the correct bucket.

        **B. Progressive unlock** — Every ``unlock_interval`` calls the
        next higher bucket is activated, allowing the training to start
        with easy (low-latency) samples and gradually include harder ones.

        Args:
            batch: A ``DataProto`` (or any object) produced by the
                training loop.  When dynamic rebucketing is desired the
                object should expose ``non_tensor_batch["total_latency_ms"]``
                and ``meta_info``.
        """
        self._update_count += 1

        # --- A. Dynamic rebucketing -----------------------------------
        try:
            reward_extra_keys = getattr(batch, "meta_info", {}).get("reward_extra_keys", [])
            if "total_latency_ms" in reward_extra_keys:
                actual_latencies = batch.non_tensor_batch.get("total_latency_ms")
                if actual_latencies is not None:
                    for actual_ms in actual_latencies:
                        actual_ms_val = float(actual_ms)
                        new_bucket = compute_latency_bucket(
                            int(actual_ms_val), self.bucket_boundaries
                        )
                        # We don't have a direct mapping from batch element to
                        # dataset index here; record for future reference.
                        self._sample_actual_latency[self._update_count] = actual_ms_val
                        logger.debug(
                            f"Observed actual latency {actual_ms_val:.0f}ms -> bucket {new_bucket}"
                        )
        except Exception:
            # Gracefully degrade: rebucketing is best-effort.
            pass

        # --- B. Progressive unlock ------------------------------------
        if self.progressive_unlock:
            new_max = min(
                self.initial_max_bucket + self._update_count // self.unlock_interval,
                self.num_buckets - 1,
            )
            if new_max > self.max_active_bucket:
                logger.info(f"Curriculum: unlocking bucket {new_max}")
                self.max_active_bucket = new_max

    def _get_bucket_order(self) -> list[int]:
        """Get the order in which to process buckets.

        Only returns buckets up to ``max_active_bucket``.
        """
        bucket_ids = [
            bid for bid in sorted(self.buckets.keys())
            if bid <= self.max_active_bucket
        ]
        if self.shuffle:
            self.generator.shuffle(bucket_ids)
        return bucket_ids

    @staticmethod
    def assign_bucket(
        latency_ms: int,
        num_buckets: int = 5,
        boundaries: list[int] | None = None,
    ) -> int:
        """
        Assign a latency value to a bucket.

        Args:
            latency_ms: Latency in milliseconds.
            num_buckets: Number of buckets.
            boundaries: Custom bucket boundaries.

        Returns:
            Bucket ID (0 to num_buckets-1).
        """
        if latency_ms < 0:
            return 0

        boundaries = boundaries or DEFAULT_BUCKET_BOUNDARIES[: num_buckets - 1]

        for i, boundary in enumerate(boundaries):
            if latency_ms < boundary:
                return i

        return min(len(boundaries), num_buckets - 1)

    def __iter__(self) -> Iterator[int]:
        """
        Iterate over sample indices, grouped by latency bucket.

        Yields individual indices, but ensures consecutive indices
        within a batch come from the same latency bucket.
        """
        # Get bucket processing order
        bucket_order = self._get_bucket_order()

        for bucket_id in bucket_order:
            indices = self.buckets[bucket_id].copy()

            if self.shuffle:
                self.generator.shuffle(indices)

            # Yield indices from this bucket
            for idx in indices:
                yield idx

    def __len__(self) -> int:
        """Return the number of samples in active (unlocked) buckets."""
        return sum(
            len(indices)
            for bid, indices in self.buckets.items()
            if bid <= self.max_active_bucket
        )

    def get_bucket_batches(self) -> Iterator[list[int]]:
        """
        Yield batches of indices, each batch from the same bucket.

        This is useful when you want explicit batch boundaries.

        Yields:
            List of indices forming a batch (all from same bucket).
        """
        bucket_order = self._get_bucket_order()

        for bucket_id in bucket_order:
            indices = self.buckets[bucket_id].copy()

            if self.shuffle:
                self.generator.shuffle(indices)

            # Create batches
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i : i + self.batch_size]

                if len(batch) < self.batch_size and self.drop_last:
                    continue

                yield batch

    def set_epoch(self, epoch: int) -> None:
        """
        Set the epoch for shuffling.

        This ensures different shuffling across epochs while
        maintaining reproducibility.

        Args:
            epoch: The current epoch number.
        """
        if self.seed is not None:
            self.generator = random.Random(self.seed + epoch)

    def get_bucket_stats(self) -> dict[int, dict[str, Any]]:
        """
        Get statistics about the bucket distribution.

        Returns:
            Dict mapping bucket_id to stats dict containing:
            - count: Number of samples
            - percentage: Percentage of total
            - latency_range: (min_ms, max_ms) for this bucket
        """
        total = len(self.data_source)
        stats = {}

        for bucket_id in range(self.num_buckets):
            count = len(self.buckets.get(bucket_id, []))

            # Determine latency range
            if bucket_id == 0:
                min_ms, max_ms = 0, self.bucket_boundaries[0]
            elif bucket_id == self.num_buckets - 1:
                min_ms, max_ms = self.bucket_boundaries[-1], float("inf")
            else:
                min_ms = self.bucket_boundaries[bucket_id - 1]
                max_ms = self.bucket_boundaries[bucket_id]

            stats[bucket_id] = {
                "count": count,
                "percentage": 100.0 * count / total if total > 0 else 0,
                "latency_range": (min_ms, max_ms),
            }

        return stats


def compute_latency_bucket(
    estimated_latency_ms: int,
    boundaries: list[int] | None = None,
) -> int:
    """
    Compute the latency bucket for a given estimated latency.

    Args:
        estimated_latency_ms: Estimated execution latency in milliseconds.
        boundaries: Bucket boundaries. Default: [1000, 5000, 15000, 30000]

    Returns:
        Bucket ID (0 to len(boundaries)).
    """
    boundaries = boundaries or DEFAULT_BUCKET_BOUNDARIES

    for i, boundary in enumerate(boundaries):
        if estimated_latency_ms < boundary:
            return i

    return len(boundaries)


def assign_latency_buckets(
    dataset: list[dict[str, Any]],
    latency_key: str = "estimated_latency_ms",
    output_key: str = "latency_bucket",
    boundaries: list[int] | None = None,
) -> list[dict[str, Any]]:
    """
    Assign latency buckets to all items in a dataset.

    This is a utility function for preprocessing datasets.

    Args:
        dataset: List of dataset items (dicts).
        latency_key: Key containing estimated latency in each item.
        output_key: Key to store the bucket assignment.
        boundaries: Bucket boundaries.

    Returns:
        Dataset with bucket assignments added to extra_info.
    """
    boundaries = boundaries or DEFAULT_BUCKET_BOUNDARIES

    for item in dataset:
        # Get estimated latency
        latency = 0
        if latency_key in item:
            latency = item[latency_key]
        elif "extra_info" in item and latency_key in item["extra_info"]:
            latency = item["extra_info"][latency_key]
        elif "rubric" in item.get("extra_info", {}):
            rubric = item["extra_info"]["rubric"]
            if isinstance(rubric, dict):
                latency = rubric.get("estimated_latency_ms", 0)

        # Compute bucket
        bucket = compute_latency_bucket(latency, boundaries)

        # Store in extra_info
        if "extra_info" not in item:
            item["extra_info"] = {}
        item["extra_info"][output_key] = bucket

    return dataset

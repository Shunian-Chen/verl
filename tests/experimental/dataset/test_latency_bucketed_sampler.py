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

import pytest
from omegaconf import OmegaConf

from verl.experimental.dataset.latency_bucketed_sampler import LatencyBucketedSampler


class DummyDataset:
    def __init__(self, buckets):
        self.items = [{"extra_info": {"latency_bucket": bucket}} for bucket in buckets]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def test_latency_bucketed_sampler_orders_by_bucket():
    dataset = DummyDataset([1, 0, 1, 2])
    data_config = OmegaConf.create({})
    sampler = LatencyBucketedSampler(dataset, data_config, num_buckets=3)

    indices = list(iter(sampler))
    assert indices == [1, 0, 2, 3]


@pytest.mark.parametrize(
    "latency_ms, expected_bucket",
    [
        (-1, 0),
        (0, 0),
        (999, 0),
        (1000, 1),
        (4999, 1),
        (5000, 2),
        (14999, 2),
        (15000, 3),
        (29999, 3),
        (30000, 4),
        (999999, 4),
    ],
)
def test_assign_bucket_boundaries(latency_ms, expected_bucket):
    bucket = LatencyBucketedSampler.assign_bucket(latency_ms, num_buckets=5)
    assert bucket == expected_bucket


# ---- Curriculum / progressive unlock tests ----

def test_progressive_unlock_initial_state():
    """Only bucket 0 is active when progressive_unlock starts at bucket 0."""
    dataset = DummyDataset([0, 1, 2, 0, 1])
    sampler = LatencyBucketedSampler(
        dataset, OmegaConf.create({}),
        num_buckets=3,
        progressive_unlock=True,
        initial_max_bucket=0,
        unlock_interval=2,
    )

    # Only bucket-0 items should appear
    indices = list(iter(sampler))
    assert set(indices) == {0, 3}  # items at positions 0 and 3 are bucket 0
    assert len(sampler) == 2


def test_progressive_unlock_after_updates():
    """Buckets unlock after sufficient update() calls."""
    dataset = DummyDataset([0, 1, 2, 0, 1])
    sampler = LatencyBucketedSampler(
        dataset, OmegaConf.create({}),
        num_buckets=3,
        progressive_unlock=True,
        initial_max_bucket=0,
        unlock_interval=2,
    )

    # Before any update: only bucket 0
    assert sampler.max_active_bucket == 0

    # After 2 updates: bucket 1 should unlock (0 + 2//2 = 1)
    sampler.update(None)
    sampler.update(None)
    assert sampler.max_active_bucket == 1

    indices = list(iter(sampler))
    # Bucket 0 items: idx 0, 3; Bucket 1 items: idx 1, 4
    assert set(indices) == {0, 1, 3, 4}
    assert len(sampler) == 4

    # After 2 more updates (total 4): bucket 2 should unlock (0 + 4//2 = 2)
    sampler.update(None)
    sampler.update(None)
    assert sampler.max_active_bucket == 2

    indices = list(iter(sampler))
    assert set(indices) == {0, 1, 2, 3, 4}
    assert len(sampler) == 5


def test_update_without_progressive_unlock_is_noop():
    """When progressive_unlock is False, update() does not change bucket access."""
    dataset = DummyDataset([0, 1, 2])
    sampler = LatencyBucketedSampler(
        dataset, OmegaConf.create({}), num_buckets=3,
    )

    assert sampler.max_active_bucket == 2
    initial_len = len(sampler)

    for _ in range(10):
        sampler.update(None)

    assert sampler.max_active_bucket == 2
    assert len(sampler) == initial_len


def test_inherits_abstract_curriculum_sampler():
    """LatencyBucketedSampler inherits from AbstractCurriculumSampler."""
    try:
        from verl.experimental.dataset.sampler import AbstractCurriculumSampler
        assert issubclass(LatencyBucketedSampler, AbstractCurriculumSampler)
    except ImportError:
        pytest.skip("AbstractCurriculumSampler not available")

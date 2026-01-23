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

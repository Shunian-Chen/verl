#!/usr/bin/env python3
"""
Category-based sampling strategies for balanced dataset construction
"""

import json
import random
from typing import List, Dict, Optional, Tuple
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np


class CategorySampler:
    """
    Implements various category-based sampling strategies
    """

    def __init__(self, data: List[Dict], seed: int = 42):
        """
        Initialize sampler with data

        Args:
            data: List of data items, each with 'categories' field
            seed: Random seed for reproducibility
        """
        self.data = data
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

        # Build category index
        self.items_by_category = defaultdict(list)
        self.category_counts = Counter()

        for idx, item in enumerate(data):
            categories = item.get('categories', [])
            if categories:
                first_cat = categories[0]
                self.items_by_category[first_cat].append(idx)
                self.category_counts[first_cat] += 1

        self.categories = list(self.items_by_category.keys())
        print(f"Initialized sampler with {len(data)} items across {len(self.categories)} categories")

    def stratified_sample(
        self,
        sample_size: int,
        min_per_category: int = 1,
        max_per_category: Optional[int] = None
    ) -> List[Dict]:
        """
        Stratified sampling: ensure representation from all categories

        Strategy:
        1. First, sample min_per_category from each category (if available)
        2. Distribute remaining samples proportionally

        Args:
            sample_size: Total number of samples to return
            min_per_category: Minimum samples per category
            max_per_category: Maximum samples per category (None = no limit)

        Returns:
            List of sampled data items
        """
        if sample_size >= len(self.data):
            return self.data.copy()

        selected_indices = set()

        # Phase 1: Ensure minimum representation
        for category in self.categories:
            indices = self.items_by_category[category]
            n_to_sample = min(min_per_category, len(indices))
            sampled = random.sample(indices, n_to_sample)
            selected_indices.update(sampled)

        # Phase 2: Distribute remaining samples proportionally
        remaining = sample_size - len(selected_indices)
        if remaining > 0:
            # Calculate proportional allocation
            total_items = sum(self.category_counts.values())
            category_allocation = {}

            for category, count in self.category_counts.items():
                proportion = count / total_items
                allocated = int(remaining * proportion)

                if max_per_category:
                    already_sampled = len([i for i in selected_indices if i in self.items_by_category[category]])
                    allocated = min(allocated, max_per_category - already_sampled)

                category_allocation[category] = allocated

            # Sample according to allocation
            for category, n_samples in category_allocation.items():
                if n_samples > 0:
                    available = [i for i in self.items_by_category[category] if i not in selected_indices]
                    if available:
                        n_to_sample = min(n_samples, len(available))
                        sampled = random.sample(available, n_to_sample)
                        selected_indices.update(sampled)

        # Convert to list and return data items
        sampled_data = [self.data[i] for i in sorted(selected_indices)]

        print(f"Stratified sampling: selected {len(sampled_data)} items from {len(set([self.data[i].get('categories', [''])[0] for i in selected_indices if self.data[i].get('categories')]))} categories")

        return sampled_data

    def balanced_sample(
        self,
        sample_size: int,
        category_limit: Optional[int] = None
    ) -> List[Dict]:
        """
        Balanced sampling: try to equalize samples across categories

        Strategy:
        1. Calculate target samples per category
        2. Sample equally from each category up to the limit
        3. If some categories have fewer samples, redistribute to others

        Args:
            sample_size: Total number of samples to return
            category_limit: Maximum samples per category (auto if None)

        Returns:
            List of sampled data items
        """
        if sample_size >= len(self.data):
            return self.data.copy()

        n_categories = len(self.categories)

        # Calculate target per category
        if category_limit is None:
            target_per_category = max(1, sample_size // n_categories)
        else:
            target_per_category = min(category_limit, sample_size // n_categories)

        selected_indices = set()

        # Phase 1: Sample target amount from each category
        undersampled = []
        for category in self.categories:
            indices = self.items_by_category[category]
            n_to_sample = min(target_per_category, len(indices))
            sampled = random.sample(indices, n_to_sample)
            selected_indices.update(sampled)

            if len(indices) < target_per_category:
                undersampled.append((category, target_per_category - len(indices)))

        # Phase 2: Redistribute if we haven't reached sample_size
        remaining = sample_size - len(selected_indices)
        if remaining > 0:
            # Redistribute to categories that have more samples available
            oversamplable = [(cat, len(self.items_by_category[cat]) - target_per_category)
                           for cat in self.categories
                           if len(self.items_by_category[cat]) > target_per_category]

            oversamplable.sort(key=lambda x: x[1], reverse=True)

            for category, available_extra in oversamplable:
                if remaining <= 0:
                    break

                current_sampled = [i for i in selected_indices if i in self.items_by_category[category]]
                available = [i for i in self.items_by_category[category] if i not in selected_indices]

                n_to_sample = min(remaining, len(available))
                if n_to_sample > 0:
                    sampled = random.sample(available, n_to_sample)
                    selected_indices.update(sampled)
                    remaining -= n_to_sample

        sampled_data = [self.data[i] for i in sorted(selected_indices)]

        # Analyze distribution
        category_dist = Counter()
        for idx in selected_indices:
            item = self.data[idx]
            if item.get('categories'):
                category_dist[item['categories'][0]] += 1

        print(f"Balanced sampling: selected {len(sampled_data)} items")
        print(f"  Categories represented: {len(category_dist)}")
        print(f"  Min samples per category: {min(category_dist.values()) if category_dist else 0}")
        print(f"  Max samples per category: {max(category_dist.values()) if category_dist else 0}")
        print(f"  Mean samples per category: {sum(category_dist.values()) / len(category_dist) if category_dist else 0:.2f}")

        return sampled_data

    def priority_sample(
        self,
        sample_size: int,
        priority_categories: Optional[List[str]] = None,
        priority_ratio: float = 0.5
    ) -> List[Dict]:
        """
        Priority sampling: oversample certain categories

        Args:
            sample_size: Total number of samples
            priority_categories: Categories to prioritize (None = auto-select underrepresented)
            priority_ratio: Fraction of samples to allocate to priority categories

        Returns:
            List of sampled data items
        """
        if sample_size >= len(self.data):
            return self.data.copy()

        # Auto-select priority categories if not provided
        if priority_categories is None:
            # Prioritize underrepresented categories (fewer samples)
            median_count = sorted(self.category_counts.values())[len(self.category_counts) // 2]
            priority_categories = [cat for cat, count in self.category_counts.items() if count <= median_count]
            print(f"Auto-selected {len(priority_categories)} priority categories (count <= {median_count})")

        priority_set = set(priority_categories)
        priority_indices = []
        regular_indices = []

        for category, indices in self.items_by_category.items():
            if category in priority_set:
                priority_indices.extend(indices)
            else:
                regular_indices.extend(indices)

        # Allocate samples
        n_priority = int(sample_size * priority_ratio)
        n_regular = sample_size - n_priority

        selected_indices = set()

        # Sample priority
        if priority_indices:
            n_to_sample = min(n_priority, len(priority_indices))
            selected_indices.update(random.sample(priority_indices, n_to_sample))

        # Sample regular
        if regular_indices:
            n_to_sample = min(n_regular, len(regular_indices))
            selected_indices.update(random.sample(regular_indices, n_to_sample))

        # If we haven't reached sample_size, sample from either
        remaining = sample_size - len(selected_indices)
        if remaining > 0:
            all_available = [i for i in range(len(self.data)) if i not in selected_indices]
            if all_available:
                n_to_sample = min(remaining, len(all_available))
                selected_indices.update(random.sample(all_available, n_to_sample))

        sampled_data = [self.data[i] for i in sorted(selected_indices)]

        print(f"Priority sampling: selected {len(sampled_data)} items")
        print(f"  Priority categories: {len(priority_set)}")
        print(f"  Samples from priority categories: {len([i for i in selected_indices if i in priority_indices])}")

        return sampled_data

    def cluster_sample(
        self,
        sample_size: int,
        n_clusters: int = 100
    ) -> List[Dict]:
        """
        Cluster-based sampling: group similar categories and sample from clusters

        This is useful when you have too many categories for pure stratification

        Args:
            sample_size: Total number of samples
            n_clusters: Number of category clusters to create

        Returns:
            List of sampled data items
        """
        if sample_size >= len(self.data):
            return self.data.copy()

        # Group categories by size
        sorted_categories = sorted(self.category_counts.items(), key=lambda x: x[1], reverse=True)

        # Divide into clusters
        cluster_size = max(1, len(sorted_categories) // n_clusters)
        clusters = []
        for i in range(0, len(sorted_categories), cluster_size):
            cluster_categories = [cat for cat, _ in sorted_categories[i:i + cluster_size]]
            cluster_indices = []
            for cat in cluster_categories:
                cluster_indices.extend(self.items_by_category[cat])
            clusters.append(cluster_indices)

        # Sample equally from each cluster
        target_per_cluster = sample_size // len(clusters)
        selected_indices = set()

        for cluster_indices in clusters:
            n_to_sample = min(target_per_cluster, len(cluster_indices))
            if n_to_sample > 0:
                sampled = random.sample(cluster_indices, n_to_sample)
                selected_indices.update(sampled)

        # Distribute remaining
        remaining = sample_size - len(selected_indices)
        if remaining > 0:
            all_available = [i for i in range(len(self.data)) if i not in selected_indices]
            if all_available:
                n_to_sample = min(remaining, len(all_available))
                selected_indices.update(random.sample(all_available, n_to_sample))

        sampled_data = [self.data[i] for i in sorted(selected_indices)]

        print(f"Cluster sampling: selected {len(sampled_data)} items from {len(clusters)} clusters")

        return sampled_data


def compare_sampling_strategies(
    data: List[Dict],
    sample_size: int,
    seed: int = 42
) -> Dict:
    """
    Compare different sampling strategies and their category distributions

    Returns:
        Dictionary with strategy names and their category distribution stats
    """
    sampler = CategorySampler(data, seed=seed)

    strategies = {
        'random': random.sample(data, min(sample_size, len(data))),
        'stratified': sampler.stratified_sample(sample_size),
        'balanced': sampler.balanced_sample(sample_size),
        'priority': sampler.priority_sample(sample_size),
        'cluster': sampler.cluster_sample(sample_size)
    }

    results = {}
    for strategy_name, sampled_data in strategies.items():
        # Analyze category distribution
        category_dist = Counter()
        for item in sampled_data:
            if item.get('categories'):
                category_dist[item['categories'][0]] += 1

        results[strategy_name] = {
            'total_items': len(sampled_data),
            'unique_categories': len(category_dist),
            'min_per_category': min(category_dist.values()) if category_dist else 0,
            'max_per_category': max(category_dist.values()) if category_dist else 0,
            'mean_per_category': sum(category_dist.values()) / len(category_dist) if category_dist else 0,
            'category_distribution': dict(category_dist.most_common(10))
        }

    return results


if __name__ == "__main__":
    import sys

    # Load data
    data_path = sys.argv[1] if len(sys.argv) > 1 else "/data_ali/shunian/data/iceberg/scripts/data_clean.json"
    sample_size = int(sys.argv[2]) if len(sys.argv) > 2 else 1000

    print(f"Loading data from {data_path}...")
    with open(data_path, 'r') as f:
        data = json.load(f)

    print(f"\nComparing sampling strategies for {sample_size} samples...\n")
    results = compare_sampling_strategies(data, sample_size)

    print("\n" + "=" * 80)
    print("STRATEGY COMPARISON")
    print("=" * 80)

    for strategy, stats in results.items():
        print(f"\n{strategy.upper()}:")
        print(f"  Total items: {stats['total_items']}")
        print(f"  Unique categories: {stats['unique_categories']}")
        print(f"  Min/Max/Mean per category: {stats['min_per_category']}/{stats['max_per_category']}/{stats['mean_per_category']:.2f}")
        print(f"  Top 5 categories:")
        for cat, count in list(stats['category_distribution'].items())[:5]:
            print(f"    {cat}: {count}")

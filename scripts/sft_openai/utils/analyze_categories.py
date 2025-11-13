#!/usr/bin/env python3
"""
Analyze category distribution in the dataset
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

def analyze_category_distribution(data_path: str):
    """Analyze how categories are distributed in the dataset"""

    print(f"Loading data from {data_path}...")
    with open(data_path, 'r') as f:
        data = json.load(f)

    print(f"Total items: {len(data)}\n")

    # Analyze categories
    all_categories = []
    first_categories = []
    items_by_first_category = defaultdict(list)

    for idx, item in enumerate(data):
        categories = item.get('categories', [])
        if categories:
            # Extract first category
            first_cat = categories[0]
            first_categories.append(first_cat)
            items_by_first_category[first_cat].append(idx)

            # Collect all categories
            all_categories.extend(categories)

    # Count occurrences
    first_cat_counter = Counter(first_categories)
    all_cat_counter = Counter(all_categories)

    print("=" * 80)
    print("CATEGORY DISTRIBUTION ANALYSIS")
    print("=" * 80)

    print(f"\nTotal unique first categories: {len(first_cat_counter)}")
    print(f"Total unique all categories: {len(all_cat_counter)}")

    print("\n" + "=" * 80)
    print("TOP 20 FIRST CATEGORIES (most common)")
    print("=" * 80)
    for cat, count in first_cat_counter.most_common(20):
        percentage = (count / len(data)) * 100
        print(f"{cat:60s} {count:6d} ({percentage:5.2f}%)")

    print("\n" + "=" * 80)
    print("BOTTOM 20 FIRST CATEGORIES (least common)")
    print("=" * 80)
    for cat, count in list(first_cat_counter.most_common())[-20:]:
        percentage = (count / len(data)) * 100
        print(f"{cat:60s} {count:6d} ({percentage:5.2f}%)")

    print("\n" + "=" * 80)
    print("DISTRIBUTION STATISTICS")
    print("=" * 80)
    counts = list(first_cat_counter.values())
    print(f"Mean items per category: {sum(counts) / len(counts):.2f}")
    print(f"Median items per category: {sorted(counts)[len(counts)//2]}")
    print(f"Max items in a category: {max(counts)}")
    print(f"Min items in a category: {min(counts)}")

    # Calculate Gini coefficient (measure of inequality)
    sorted_counts = sorted(counts)
    n = len(sorted_counts)
    cumsum = sum((i+1) * val for i, val in enumerate(sorted_counts))
    gini = (2 * cumsum) / (n * sum(sorted_counts)) - (n + 1) / n
    print(f"Gini coefficient (0=perfect equality, 1=perfect inequality): {gini:.3f}")

    # Save detailed distribution
    output_path = Path(data_path).parent / "category_distribution.json"
    distribution_data = {
        "total_items": len(data),
        "unique_first_categories": len(first_cat_counter),
        "category_counts": dict(first_cat_counter),
        "items_by_category": {k: len(v) for k, v in items_by_first_category.items()},
        "statistics": {
            "mean": sum(counts) / len(counts),
            "median": sorted(counts)[len(counts)//2],
            "max": max(counts),
            "min": min(counts),
            "gini": gini
        }
    }

    with open(output_path, 'w') as f:
        json.dump(distribution_data, f, indent=2)

    print(f"\nDetailed distribution saved to: {output_path}")

    return items_by_first_category, first_cat_counter


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        data_path = "/data_ali/shunian/data/iceberg/scripts/data_clean.json"

    analyze_category_distribution(data_path)

#!/bin/bash
#
# Test category-based sampling strategies
#

set -e

SOURCE_DATA="/data_ali/shunian/data/iceberg/scripts/data_clean.json"
SCRIPT_DIR="/data_ali/shunian/verl/scripts/sft_openai"
OUTPUT_BASE="/data_ali/shunian/verl/data_output"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=========================================="
echo "Category Sampling Strategy Test"
echo "=========================================="
echo ""

# Test 1: Compare sampling strategies
echo "Test 1: Comparing different sampling strategies (1000 samples)"
echo "--------------------------------------------------------------"
python3 ${SCRIPT_DIR}/category_sampling.py ${SOURCE_DATA} 1000
echo ""

# Test 2: Verify category distribution for cluster sampling
echo ""
echo "Test 2: Detailed cluster sampling analysis (10000 samples)"
echo "------------------------------------------------------------"
python3 -c "
import json
import sys
sys.path.insert(0, '${SCRIPT_DIR}')
from category_sampling import CategorySampler
from collections import Counter

# Load data
with open('${SOURCE_DATA}', 'r') as f:
    data = json.load(f)

# Create sampler
sampler = CategorySampler(data, seed=42)

# Test cluster sampling
sample_size = 10000
sampled = sampler.cluster_sample(sample_size)

# Analyze category distribution
category_dist = Counter()
for item in sampled:
    if item.get('categories'):
        category_dist[item['categories'][0]] += 1

print(f'\\nSampled {len(sampled)} items')
print(f'Unique categories: {len(category_dist)}')
print(f'Coverage: {len(category_dist) / len(sampler.categories) * 100:.2f}% of all categories')
print(f'\\nDistribution statistics:')
print(f'  Min per category: {min(category_dist.values())}')
print(f'  Max per category: {max(category_dist.values())}')
print(f'  Mean per category: {sum(category_dist.values()) / len(category_dist):.2f}')
print(f'  Median per category: {sorted(category_dist.values())[len(category_dist)//2]}')

# Show top 20 categories
print(f'\\nTop 20 categories:')
for i, (cat, count) in enumerate(category_dist.most_common(20), 1):
    print(f'  {i:2d}. {cat[:60]:60s} {count:3d}')

# Save distribution for verification
output = {
    'total_sampled': len(sampled),
    'unique_categories': len(category_dist),
    'coverage_percentage': len(category_dist) / len(sampler.categories) * 100,
    'category_distribution': dict(category_dist)
}

output_file = '${OUTPUT_BASE}/cluster_sampling_distribution_${TIMESTAMP}.json'
import os
os.makedirs('${OUTPUT_BASE}', exist_ok=True)
with open(output_file, 'w') as f:
    json.dump(output, f, indent=2)

print(f'\\nDistribution saved to: {output_file}')
"

echo ""
echo "=========================================="
echo "Category Sampling Tests Complete!"
echo "=========================================="

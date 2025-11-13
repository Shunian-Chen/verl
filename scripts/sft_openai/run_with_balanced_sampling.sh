#!/bin/bash
#
# Quick start script for running GPT pipeline with balanced category sampling
#

set -e

# Check for API key
if [ -z "$OPENAI_API_KEY" ]; then
    if [ -f .env ]; then
        export $(cat .env | grep OPENAI_API_KEY | xargs)
    fi
fi

if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY not set!"
    echo "Please set it via:"
    echo "  export OPENAI_API_KEY='your-key-here'"
    echo "Or create a .env file with:"
    echo "  OPENAI_API_KEY=your-key-here"
    exit 1
fi

# Configuration
SOURCE_DATA="/data_ali/shunian/data/iceberg/scripts/data_clean.json"
OUTPUT_BASE="/data_ali/shunian/verl/data_output"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${OUTPUT_BASE}/gpt_balanced_${TIMESTAMP}"

# Parse command line arguments
SAMPLE_SIZE=${1:-100}
EXAMPLES_PER_ITEM=${2:-2}
SAMPLING_STRATEGY=${3:-cluster}

echo "=========================================="
echo "GPT Data Construction with Balanced Sampling"
echo "=========================================="
echo "Source: ${SOURCE_DATA}"
echo "Output: ${OUTPUT_DIR}"
echo "Sample size: ${SAMPLE_SIZE}"
echo "Examples per item: ${EXAMPLES_PER_ITEM}"
echo "Sampling strategy: ${SAMPLING_STRATEGY}"
echo "=========================================="
echo ""

# Show expected cost
EXPECTED_EXAMPLES=$((SAMPLE_SIZE * EXAMPLES_PER_ITEM))
COST_PER_EXAMPLE=0.091
EXPECTED_COST=$(python3 -c "print(f'{${EXPECTED_EXAMPLES} * ${COST_PER_EXAMPLE}:.2f}')")

echo "Cost Estimation:"
echo "  Expected examples: ~${EXPECTED_EXAMPLES}"
echo "  Cost per example: \$${COST_PER_EXAMPLE}"
echo "  Total estimated cost: \$${EXPECTED_COST}"
echo ""

read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "Starting pipeline..."
echo ""

# Set environment variables if not already set
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}"
export GENERATION_MODEL="${GENERATION_MODEL:-gpt-4o-mini}"
export VALIDATION_MODEL="${VALIDATION_MODEL:-gpt-4o-mini}"

# Run pipeline
python3 data_construction_gpt_pipeline.py \
  --source "${SOURCE_DATA}" \
  --output "${OUTPUT_DIR}" \
  --sample ${SAMPLE_SIZE} \
  --examples-per-item ${EXAMPLES_PER_ITEM} \
  --sampling-strategy ${SAMPLING_STRATEGY} \
  --seed 42 \
  --max-concurrent 10

echo ""
echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo "Output: ${OUTPUT_DIR}"
echo ""

# Show quick stats
if [ -f "${OUTPUT_DIR}/generated_examples.jsonl" ]; then
    EXAMPLE_COUNT=$(wc -l < "${OUTPUT_DIR}/generated_examples.jsonl")
    echo "Generated examples: ${EXAMPLE_COUNT}"

    # Analyze category distribution
    echo ""
    echo "Category distribution:"
    python3 -c "
import json
from collections import Counter

with open('${OUTPUT_DIR}/generated_examples.jsonl', 'r') as f:
    examples = [json.loads(line) for line in f]

category_dist = Counter()
for ex in examples:
    metadata = ex.get('metadata', {})
    cats = metadata.get('source_categories', [])
    if cats:
        category_dist[cats[0]] += 1

print(f'  Unique categories: {len(category_dist)}')
if category_dist:
    print(f'  Min per category: {min(category_dist.values())}')
    print(f'  Max per category: {max(category_dist.values())}')
    print(f'  Mean per category: {sum(category_dist.values()) / len(category_dist):.2f}')
    print(f'\\n  Top 10 categories:')
    for i, (cat, count) in enumerate(category_dist.most_common(10), 1):
        print(f'    {i:2d}. {cat[:60]:60s} {count:3d}')
"
fi

echo ""
echo "Next steps:"
echo "1. Review examples: ${OUTPUT_DIR}/generated_examples.jsonl"
echo "2. Check quality report: ${OUTPUT_DIR}/pipeline_report.json"
echo "3. Analyze with: python3 gpt_pipeline_utils.py analyze-quality --input ${OUTPUT_DIR}/generated_examples.jsonl"
echo ""

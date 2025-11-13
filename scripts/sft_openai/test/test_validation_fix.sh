#!/bin/bash
#
# Test validation fix with small sample
#

set -e

cd /data_ali/shunian/verl/scripts/sft_openai

# Check environment
if [ -z "$OPENAI_API_KEY" ]; then
    if [ -f .env ]; then
        export $(cat .env | grep OPENAI_API_KEY | xargs)
    fi
fi

if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY not set!"
    exit 1
fi

export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}"
export GENERATION_MODEL="${GENERATION_MODEL:-gpt-4o-mini}"
export VALIDATION_MODEL="${VALIDATION_MODEL:-gpt-4o-mini}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="./test_validation_fix_${TIMESTAMP}"

echo "=========================================="
echo "Testing Validation Fix"
echo "=========================================="
echo "Output: ${OUTPUT_DIR}"
echo ""

# Run with 5 items
python3 data_construction_gpt_pipeline.py \
  --source /data_ali/shunian/data/iceberg/scripts/data_clean.json \
  --output "${OUTPUT_DIR}" \
  --sample 5 \
  --examples-per-item 2 \
  --sampling-strategy cluster \
  --seed 42 \
  --max-concurrent 5

echo ""
echo "=========================================="
echo "Test Complete!"
echo "=========================================="

# Check results
if [ -f "${OUTPUT_DIR}/generated_examples.jsonl" ]; then
    EXAMPLE_COUNT=$(wc -l < "${OUTPUT_DIR}/generated_examples.jsonl")
    echo "✓ Generated ${EXAMPLE_COUNT} examples"

    # Show validation methods used
    echo ""
    echo "Validation methods:"
    python3 -c "
import json
from collections import Counter

with open('${OUTPUT_DIR}/generated_examples.jsonl', 'r') as f:
    examples = [json.loads(line) for line in f]

methods = Counter()
for ex in examples:
    metadata = ex.get('metadata', {})
    validation = metadata.get('validation', {})
    method = validation.get('validation_method', 'unknown')
    methods[method] += 1

for method, count in methods.items():
    print(f'  {method}: {count}')
"

    # Show sample
    echo ""
    echo "First example (truncated):"
    head -n 1 "${OUTPUT_DIR}/generated_examples.jsonl" | python3 -m json.tool | head -50

else
    echo "✗ No examples generated!"
    echo ""
    echo "Check report:"
    cat "${OUTPUT_DIR}/pipeline_report.json"
fi

echo ""
echo "Full report: ${OUTPUT_DIR}/pipeline_report.json"
echo "Log: /data_ali/shunian/verl/scripts/sft_openai/gpt_pipeline.log"

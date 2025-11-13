#!/bin/bash
#
# Test failed examples saving feature
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
OUTPUT_DIR="./test_failed_${TIMESTAMP}"

echo "=========================================="
echo "Testing Failed Examples Feature"
echo "=========================================="
echo "Output: ${OUTPUT_DIR}"
echo ""
echo "This test will generate some examples and"
echo "save both successful and failed ones."
echo ""

# Run with 5 items, higher examples per item to increase chance of some failures
python3 data_construction_gpt_pipeline.py \
  --source /data_ali/shunian/data/iceberg/scripts/data_clean.json \
  --output "${OUTPUT_DIR}" \
  --sample 5 \
  --examples-per-item 4 \
  --sampling-strategy cluster \
  --seed 42 \
  --max-concurrent 5 \
  --multiple-choice-ratio 0.2

echo ""
echo "=========================================="
echo "Test Complete!"
echo "=========================================="

# Check results
if [ -f "${OUTPUT_DIR}/generated_examples.jsonl" ]; then
    VALID_COUNT=$(wc -l < "${OUTPUT_DIR}/generated_examples.jsonl")
    echo "✓ Generated ${VALID_COUNT} valid examples"
else
    VALID_COUNT=0
    echo "✗ No valid examples file found"
fi

if [ -f "${OUTPUT_DIR}/failed_examples.jsonl" ]; then
    FAILED_COUNT=$(wc -l < "${OUTPUT_DIR}/failed_examples.jsonl")
    echo "✓ Saved ${FAILED_COUNT} failed examples"

    if [ ${FAILED_COUNT} -gt 0 ]; then
        echo ""
        echo "Failed example summary:"
        python3 -c "
import json
from collections import Counter

with open('${OUTPUT_DIR}/failed_examples.jsonl', 'r') as f:
    failed = [json.loads(line) for line in f]

# Most common issues
all_issues = []
for ex in failed:
    all_issues.extend(ex.get('failure_reason', []))

print('Most common failure reasons:')
for reason, count in Counter(all_issues).most_common(5):
    print(f'  - {reason}: {count}')

# Score distribution
scores = [ex.get('overall_score', 0) for ex in failed]
print(f'\nScore range: {min(scores):.1f} - {max(scores):.1f}')
print(f'Average score: {sum(scores)/len(scores):.1f}')
"
    fi
else
    echo "✓ No failed examples (all passed validation!)"
    FAILED_COUNT=0
fi

TOTAL=$((VALID_COUNT + FAILED_COUNT))
if [ ${TOTAL} -gt 0 ]; then
    SUCCESS_RATE=$(python3 -c "print(f'{${VALID_COUNT}/${TOTAL}*100:.1f}')")
    echo ""
    echo "Overall statistics:"
    echo "  Total examples: ${TOTAL}"
    echo "  Success rate: ${SUCCESS_RATE}%"
fi

# Run analysis if there are failures
if [ ${FAILED_COUNT} -gt 0 ]; then
    echo ""
    echo "=========================================="
    echo "Running Failure Analysis"
    echo "=========================================="
    python3 analyze_failures.py "${OUTPUT_DIR}"
fi

echo ""
echo "Files:"
echo "  Valid examples: ${OUTPUT_DIR}/generated_examples.jsonl"
echo "  Failed examples: ${OUTPUT_DIR}/failed_examples.jsonl"
echo "  Report: ${OUTPUT_DIR}/pipeline_report.json"
echo "  Log: /data_ali/shunian/verl/scripts/sft_openai/gpt_pipeline.log"

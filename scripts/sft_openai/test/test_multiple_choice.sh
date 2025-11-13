#!/bin/bash
#
# Test multiple choice question generation
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
OUTPUT_DIR="./test_mc_${TIMESTAMP}"

echo "=========================================="
echo "Testing Multiple Choice Question Generation"
echo "=========================================="
echo "Output: ${OUTPUT_DIR}"
echo ""

# Run with 5 items, 20% multiple choice
python3 data_construction_gpt_pipeline.py \
  --source /data_ali/shunian/data/iceberg/scripts/data_clean.json \
  --output "${OUTPUT_DIR}" \
  --sample 5 \
  --examples-per-item 3 \
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
    EXAMPLE_COUNT=$(wc -l < "${OUTPUT_DIR}/generated_examples.jsonl")
    echo "✓ Generated ${EXAMPLE_COUNT} examples"

    # Show question type distribution
    echo ""
    echo "Question type distribution:"
    python3 -c "
import json
from collections import Counter

with open('${OUTPUT_DIR}/generated_examples.jsonl', 'r') as f:
    examples = [json.loads(line) for line in f]

mc_count = sum(1 for ex in examples if ex.get('is_multiple_choice', False))
open_count = len(examples) - mc_count

print(f'  Multiple choice: {mc_count} ({mc_count/len(examples)*100:.1f}%)')
print(f'  Open-ended: {open_count} ({open_count/len(examples)*100:.1f}%)')

# Show a sample MC question if any
mc_examples = [ex for ex in examples if ex.get('is_multiple_choice', False)]
if mc_examples:
    print()
    print('Sample multiple choice question:')
    print('-' * 60)
    ex = mc_examples[0]
    print(f'Question: {ex[\"question\"]}')
    print()
    if ex.get('options'):
        for opt in ex['options']:
            print(f'  {opt}')
    print()
    print(f'Correct Answer: {ex.get(\"correct_answer\", \"N/A\")}')
    print()
    print('Response (first 300 chars):')
    print(ex['response'][:300] + '...')
"

else
    echo "✗ No examples generated!"
    echo ""
    echo "Check report:"
    cat "${OUTPUT_DIR}/pipeline_report.json"
fi

echo ""
echo "Full report: ${OUTPUT_DIR}/pipeline_report.json"
echo "Log: /data_ali/shunian/verl/scripts/sft_openai/gpt_pipeline.log"

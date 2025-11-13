# Quick Start Guide: GPT-Based Data Pipeline

This guide will help you get started with the GPT-based data construction pipeline in under 30 minutes.

## Prerequisites

```bash
# 1. Install dependencies
pip install openai>=1.0.0 aiohttp backoff tqdm numpy

# 2. Set up your OpenAI API key
export OPENAI_API_KEY="sk-your-key-here"

# 3. Verify source data exists
ls -lh /data_ali/shunian/data/iceberg/scripts/data_clean.json
```

## Step 1: Estimate Costs (2 minutes)

Before running anything, estimate costs for your desired scale:

```bash
# Estimate cost for 1,000 items (testing)
python gpt_pipeline_utils.py estimate-cost \
  --items 1000 \
  --examples-per-item 2 \
  --generation-model gpt-4-turbo-preview \
  --validation-model gpt-3.5-turbo

# Estimate cost for full dataset (395K items)
python gpt_pipeline_utils.py estimate-cost \
  --items 395290 \
  --examples-per-item 2
```

Expected output for 1,000 items:
```
GPT PIPELINE COST ESTIMATE
================================================================================
Input Parameters:
  Source items: 1,000
  Examples per item: 2

Expected Output:
  Total generation attempts: 2,000
  Expected valid examples: 1,400

Cost Breakdown:
  Cost per valid example: $0.0914
  Total estimated cost: $127.96

Processing Time (with max_concurrent=10):
  Hours: 4.4
  Days: 0.2
================================================================================
```

## Step 2: Test with Small Sample (15 minutes)

Always start with a small sample to verify everything works:

```bash
# Test with 10 items (should complete in 2-3 minutes)
python data_construction_gpt_pipeline.py \
  --api-key $OPENAI_API_KEY \
  --source /data_ali/shunian/data/iceberg/scripts/data_clean.json \
  --output ./test_output_10 \
  --sample 10 \
  --examples-per-item 2 \
  --max-concurrent 5

# Expected cost: ~$1.30
# Expected time: 2-3 minutes
# Expected valid examples: ~14 (70% pass rate)
```

### Monitor Progress

While it's running, watch the progress in another terminal:

```bash
# Watch log file
tail -f gpt_pipeline.log

# Check output file growth
watch -n 5 'wc -l test_output_10/generated_examples.jsonl'

# Check current costs
watch -n 10 'cat test_output_10/checkpoint.json | jq .usage_stats.total_cost_usd'
```

### Verify Output

```bash
# Inspect the generated examples
python gpt_pipeline_utils.py inspect-examples \
  --input test_output_10/generated_examples.jsonl \
  --max-lines 3

# Analyze quality
python gpt_pipeline_utils.py analyze-quality \
  --input test_output_10/generated_examples.jsonl \
  --sample 2
```

## Step 3: Scale to 100 Items (30-60 minutes)

If the test looks good, scale to 100 items:

```bash
python data_construction_gpt_pipeline.py \
  --api-key $OPENAI_API_KEY \
  --source /data_ali/shunian/data/iceberg/scripts/data_clean.json \
  --output ./pilot_output_100 \
  --sample 100 \
  --examples-per-item 2 \
  --max-concurrent 10

# Expected cost: ~$13
# Expected time: 30-45 minutes
# Expected valid examples: ~140
```

### Review Quality

After completion:

```bash
# View detailed quality analysis
python gpt_pipeline_utils.py analyze-quality \
  --input pilot_output_100/generated_examples.jsonl

# Sample 5 random examples
python gpt_pipeline_utils.py analyze-quality \
  --input pilot_output_100/generated_examples.jsonl \
  --sample 5
```

Expected quality metrics:
```
QUALITY ANALYSIS REPORT
================================================================================
1. BASIC STATISTICS
  Total examples: 140
  Unique images: 100
  Avg examples per image: 1.40

2. VALIDATION SCORES
  Mean: 8.20
  Median: 8.30
  Std: 0.85
  Range: 7.00 - 10.00

3. DIMENSION SCORES
  content_quality: 8.30 ± 0.80
  coherence: 8.50 ± 0.70
  diversity: 7.80 ± 1.00
  educational_value: 8.40 ± 0.75
```

If quality looks good, proceed to larger scale!

## Step 4: Production Scale Options

Choose based on your budget and timeline:

### Option A: Conservative Scale (10K items)

Good for initial training and experimentation.

```bash
python data_construction_gpt_pipeline.py \
  --api-key $OPENAI_API_KEY \
  --source /data_ali/shunian/data/iceberg/scripts/data_clean.json \
  --output ./production_10k \
  --sample 10000 \
  --examples-per-item 2 \
  --max-concurrent 10 \
  --batch-size 100 \
  --checkpoint-interval 500

# Expected cost: ~$1,300
# Expected time: 20-30 hours (1-2 days)
# Expected valid examples: ~14,000
```

### Option B: Medium Scale (50K items)

Balanced cost-quality trade-off.

```bash
python data_construction_gpt_pipeline.py \
  --api-key $OPENAI_API_KEY \
  --source /data_ali/shunian/data/iceberg/scripts/data_clean.json \
  --output ./production_50k \
  --sample 50000 \
  --examples-per-item 2 \
  --max-concurrent 15 \
  --batch-size 200 \
  --checkpoint-interval 1000

# Expected cost: ~$6,500
# Expected time: 4-5 days
# Expected valid examples: ~70,000
```

### Option C: Full Scale (395K items)

Maximum dataset size and diversity.

```bash
python data_construction_gpt_pipeline.py \
  --api-key $OPENAI_API_KEY \
  --source /data_ali/shunian/data/iceberg/scripts/data_clean.json \
  --output ./production_full \
  --examples-per-item 2 \
  --max-concurrent 20 \
  --batch-size 200 \
  --checkpoint-interval 2000

# Expected cost: ~$51,000
# Expected time: 5-7 days
# Expected valid examples: ~553,000
```

### Budget-Optimized: Use GPT-3.5 for Everything

```bash
python data_construction_gpt_pipeline.py \
  --api-key $OPENAI_API_KEY \
  --source /data_ali/shunian/data/iceberg/scripts/data_clean.json \
  --output ./production_budget \
  --generation-model gpt-3.5-turbo \
  --validation-model gpt-3.5-turbo \
  --sample 100000 \
  --examples-per-item 2 \
  --max-concurrent 20

# Expected cost: ~$600 (95% cost reduction!)
# Expected time: 2-3 days
# Expected valid examples: ~140,000
# Quality: Good but less sophisticated than GPT-4
```

## Step 5: Monitor Long-Running Jobs

For production runs, use these monitoring commands:

```bash
# 1. Check progress
python gpt_pipeline_utils.py inspect-checkpoint \
  --checkpoint ./production_10k/checkpoint.json

# 2. Calculate remaining time
PROCESSED=$(jq .processed_items ./production_10k/checkpoint.json)
TOTAL=10000
REMAINING=$((TOTAL - PROCESSED))
RATE=50  # items per minute (approximate)
REMAINING_MINUTES=$((REMAINING / RATE))
echo "Estimated remaining time: $((REMAINING_MINUTES / 60)) hours"

# 3. Monitor costs in real-time
watch -n 60 'python gpt_pipeline_utils.py inspect-checkpoint --checkpoint ./production_10k/checkpoint.json'

# 4. Sample generated examples periodically
python gpt_pipeline_utils.py analyze-quality \
  --input ./production_10k/generated_examples.jsonl \
  --sample 3
```

## Step 6: Handle Interruptions

The pipeline automatically saves checkpoints. If interrupted:

```bash
# Simply re-run the exact same command
# It will automatically resume from the last checkpoint

python data_construction_gpt_pipeline.py \
  --api-key $OPENAI_API_KEY \
  --source /data_ali/shunian/data/iceberg/scripts/data_clean.json \
  --output ./production_10k \
  --sample 10000 \
  --examples-per-item 2 \
  --max-concurrent 10 \
  --batch-size 100 \
  --checkpoint-interval 500

# The pipeline will detect existing checkpoint and resume
```

## Step 7: Post-Processing

After generation completes, run post-processing:

```bash
# Deduplicate and create train/val split
python data_quality_control.py \
  --input ./production_10k/generated_examples.jsonl \
  --output ./production_10k/final \
  --max-per-image 3 \
  --target-size 12000

# This creates:
#   ./production_10k/final/train.jsonl
#   ./production_10k/final/val.jsonl
#   ./production_10k/final/quality_report.txt
```

## Common Issues and Solutions

### Issue 1: Rate Limit Errors

```
Error: Rate limit reached for requests
```

**Solution**: Reduce `--max-concurrent`:

```bash
# From 10 to 5
--max-concurrent 5
```

### Issue 2: High Failure Rate (>40%)

```bash
# Check what's failing
grep '"pass": false' production_10k/generated_examples.jsonl | \
  jq '.validation_metadata.issues' | \
  head -20

# If many "format errors", check source data quality
# If many "repetitive cycles", adjust prompts
# If many "low educational value", try GPT-4 instead of GPT-3.5
```

### Issue 3: Slow Processing

```bash
# Check your OpenAI tier limits
# Tier 1: max_concurrent should be ~5
# Tier 2: max_concurrent can be ~10
# Tier 3+: max_concurrent can be 20+

# Also check if you're hitting TPM (tokens per minute) limits
# Solution: Reduce max_concurrent or upgrade OpenAI tier
```

### Issue 4: Out of Budget Mid-Run

```bash
# Check current costs
cat production_10k/checkpoint.json | jq .usage_stats

# To stop gracefully: Ctrl+C
# The checkpoint will be saved automatically

# To continue with smaller scope:
# Reduce --examples-per-item from 2 to 1
# Or switch to GPT-3.5-turbo for remaining items
```

## Cost Optimization Strategies

### Strategy 1: Pre-filter Source Data

Only process high-quality items:

```python
# Create a filtered dataset first
python -c "
import json
with open('/data_ali/shunian/data/iceberg/scripts/data_clean.json') as f:
    data = json.load(f)

# Filter for rich content (>300 words) and multiple categories (>=3)
filtered = []
for item in data:
    content = str(item.get('pred_response', ''))
    if len(content.split()) >= 300 and len(item.get('categories', [])) >= 3:
        filtered.append(item)

print(f'Filtered: {len(filtered)} items from {len(data)}')

with open('data_filtered_high_quality.json', 'w') as f:
    json.dump(filtered, f)
"

# Then run pipeline on filtered data
python data_construction_gpt_pipeline.py \
  --source ./data_filtered_high_quality.json \
  ...
```

### Strategy 2: Adaptive Generation

Generate 1 example for simple items, 2 for complex:

```bash
# Run two separate pipelines
# Simple items (1 example each)
--examples-per-item 1

# Complex items (2 examples each)
--examples-per-item 2
```

### Strategy 3: Mixed Model Strategy

Use GPT-4 for generation, GPT-3.5 for validation (already the default):

```bash
--generation-model gpt-4-turbo-preview \
--validation-model gpt-3.5-turbo
```

## Next Steps

1. **Review outputs**: Manually check 50-100 examples
2. **Train a test model**: Use 10K examples for quick SFT training
3. **Evaluate quality**: Check model performance on held-out validation
4. **Scale production**: Generate full dataset if test results are good
5. **Iterate on prompts**: Refine prompts based on observed issues

## Getting Help

If you encounter issues:

1. Check `gpt_pipeline.log` for detailed error messages
2. Inspect checkpoint: `python gpt_pipeline_utils.py inspect-checkpoint --checkpoint ./output/checkpoint.json`
3. Validate examples: `python gpt_pipeline_utils.py analyze-quality --input ./output/generated_examples.jsonl`
4. Review the full documentation: `GPT_PIPELINE_README.md`

## Example Complete Workflow

```bash
#!/bin/bash
# Complete workflow from start to finish

# 1. Estimate costs
python gpt_pipeline_utils.py estimate-cost --items 10000 --examples-per-item 2

# 2. Test with small sample
python data_construction_gpt_pipeline.py \
  --api-key $OPENAI_API_KEY \
  --source /data_ali/shunian/data/iceberg/scripts/data_clean.json \
  --output ./test_10 \
  --sample 10 \
  --examples-per-item 2

# 3. Verify test quality
python gpt_pipeline_utils.py analyze-quality \
  --input ./test_10/generated_examples.jsonl \
  --sample 3

# 4. Run production (10K items)
python data_construction_gpt_pipeline.py \
  --api-key $OPENAI_API_KEY \
  --source /data_ali/shunian/data/iceberg/scripts/data_clean.json \
  --output ./production_10k \
  --sample 10000 \
  --examples-per-item 2 \
  --max-concurrent 10

# 5. Post-process
python data_quality_control.py \
  --input ./production_10k/generated_examples.jsonl \
  --output ./production_10k/final \
  --max-per-image 3

# 6. Final analysis
python gpt_pipeline_utils.py analyze-quality \
  --input ./production_10k/final/train.jsonl

echo "Pipeline complete! Train set: ./production_10k/final/train.jsonl"
```

This completes the quick start guide. You should now be able to generate high-quality SFT training data using GPT models!

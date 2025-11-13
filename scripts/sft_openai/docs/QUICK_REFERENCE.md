# Statistics Pipeline Quick Reference

## Essential Commands

### Generate Statistics
```bash
python3 statistics_pipeline.py <output_dir>
```

### Generate Visualizations
```bash
python3 visualize_stats.py <output_dir>/data_statistics.json
```

### Complete Workflow
```bash
python3 statistics_pipeline.py <output_dir> && \
python3 visualize_stats.py <output_dir>/data_statistics.json
```

---

## Key Metrics at a Glance

### Overall Health
```bash
# Check success rate
cat data_statistics.json | jq '.meta.overall_success_rate'

# Check average quality score
cat data_statistics.json | jq '.quality_metrics.validation_scores.mean'

# Check total valid examples
cat data_statistics.json | jq '.meta.total_valid_examples'
```

### Distribution Balance
```bash
# Strategy distribution
cat data_statistics.json | jq '.distribution_analysis.question_strategies.distribution'

# Complexity distribution
cat data_statistics.json | jq '.distribution_analysis.complexity.distribution'

# Question type split
cat data_statistics.json | jq '.distribution_analysis.question_types'
```

### Quality Assessment
```bash
# Get recommendations
cat data_statistics.json | jq '.recommendations'

# Check imbalance warnings
cat data_statistics.json | jq '.data_quality_insights.imbalance_warnings'

# View failure breakdown
cat data_statistics.json | jq '.failure_analysis.failure_types'
```

---

## Quality Thresholds

### Minimum for Training
- **Examples**: >= 1,000 (ideally 10,000+)
- **Success Rate**: >= 70%
- **Avg Quality Score**: >= 7.5
- **Strategy Entropy**: >= 2.0
- **No HIGH Warnings**: 0

### Production Quality
- **Examples**: >= 10,000
- **Success Rate**: >= 85%
- **Avg Quality Score**: >= 8.0
- **Strategy Balance**: Max deviation < 15%
- **Category Coverage**: Gini < 0.6

---

## Common Issues & Quick Fixes

### Low Success Rate (<60%)
```bash
# 1. Check top failure reasons
cat data_statistics.json | jq '.failure_analysis.quality_issues.top_20_issues[0:5]'

# 2. Review by strategy
cat data_statistics.json | jq '.quality_metrics.success_by_strategy.by_category[0:5]'

# 3. Fix prompts for worst-performing strategies
```

### Imbalanced Data
```bash
# Check strategy imbalance
cat data_statistics.json | jq '.distribution_analysis.question_strategies.entropy'
# < 2.0 = imbalanced

# Adjust sampling weights in generation pipeline
```

### Many Near-Miss Failures
```bash
# Check near-miss count
cat data_statistics.json | jq '.failure_analysis.near_misses'

# Consider relaxing validation thresholds or implementing retry logic
```

---

## Visualization Files

All saved to `<output_dir>/visualizations/`:

1. **dashboard.png** - Start here! Comprehensive overview
2. **strategy_distribution.png** - Strategy balance
3. **score_distribution.png** - Quality distribution
4. **success_by_strategy.png** - Which strategies work best
5. **failure_analysis.png** - What's failing and why

---

## Programmatic Access

### Python
```python
import json

# Load statistics
with open('data_statistics.json') as f:
    stats = json.load(f)

# Check if ready for training
ready = (
    stats['meta']['overall_success_rate'] >= 70 and
    stats['quality_metrics']['validation_scores']['mean'] >= 7.5 and
    stats['meta']['total_valid_examples'] >= 1000
)

if ready:
    print("Ready for training!")
else:
    print("Issues to address:")
    for rec in stats['recommendations']:
        if rec['priority'] == 'HIGH':
            print(f"  - {rec['issue']}")
```

### Bash
```bash
#!/bin/bash

# Extract key metrics
SUCCESS_RATE=$(cat data_statistics.json | jq -r '.meta.overall_success_rate')
AVG_SCORE=$(cat data_statistics.json | jq -r '.quality_metrics.validation_scores.mean')
TOTAL_EXAMPLES=$(cat data_statistics.json | jq -r '.meta.total_valid_examples')

# Check thresholds
if (( $(echo "$SUCCESS_RATE >= 70" | bc -l) )) && \
   (( $(echo "$AVG_SCORE >= 7.5" | bc -l) )) && \
   (( TOTAL_EXAMPLES >= 1000 )); then
    echo "Data quality check: PASSED"
    exit 0
else
    echo "Data quality check: FAILED"
    echo "  Success rate: $SUCCESS_RATE% (need >= 70%)"
    echo "  Avg score: $AVG_SCORE (need >= 7.5)"
    echo "  Examples: $TOTAL_EXAMPLES (need >= 1000)"
    exit 1
fi
```

---

## Metric Interpretation

### Success Rate
- **>= 85%**: Excellent
- **70-85%**: Good
- **50-70%**: Needs improvement
- **< 50%**: Critical - review pipeline

### Validation Scores
- **>= 8.5**: Excellent quality
- **7.5-8.5**: Good quality
- **6.5-7.5**: Acceptable
- **< 6.5**: Quality issues

### Strategy Entropy
- **>= 2.5**: Very diverse
- **2.0-2.5**: Good diversity
- **1.5-2.0**: Moderate diversity
- **< 1.5**: Imbalanced

### Gini Coefficient (Category Coverage)
- **< 0.5**: Excellent coverage
- **0.5-0.7**: Good coverage
- **0.7-0.9**: Moderate coverage
- **>= 0.9**: Poor coverage

---

## Files Overview

| File | Purpose | Size |
|------|---------|------|
| `statistics_pipeline.py` | Main statistics computation | 850+ lines |
| `visualize_stats.py` | Visualization generation | 600+ lines |
| `DATA_STATISTICS_README.md` | Complete documentation | 1100+ lines |
| `STATISTICS_PIPELINE_SUMMARY.md` | Implementation summary | This document |
| `QUICK_REFERENCE.md` | Quick reference guide | This document |
| `example_data_statistics.json` | Sample output structure | Reference |

---

## Comparison with analyze_failures.py

| Feature | statistics_pipeline.py | analyze_failures.py |
|---------|------------------------|---------------------|
| **Focus** | Quantitative metrics | Qualitative analysis |
| **Output** | JSON + visualizations | Console text |
| **Scope** | 50+ metrics | Failure deep-dive |
| **Use Case** | Monitoring, tracking | Debugging |
| **Best For** | Overview, trends | Understanding failures |

**Use Both**: Run statistics_pipeline.py for metrics, then analyze_failures.py for detailed failure investigation.

---

## Integration Examples

### Daily Monitoring
```bash
#!/bin/bash
# cron job: run daily

python3 data_construction_pipeline.py --output ./daily_$(date +%Y%m%d)
python3 statistics_pipeline.py ./daily_$(date +%Y%m%d)

# Alert if quality drops
SUCCESS=$(cat ./daily_$(date +%Y%m%d)/data_statistics.json | jq -r '.meta.overall_success_rate')
if (( $(echo "$SUCCESS < 70" | bc -l) )); then
    echo "ALERT: Success rate dropped to $SUCCESS%" | mail -s "Data Quality Alert" team@example.com
fi
```

### Pre-Training Validation
```bash
#!/bin/bash
# Before starting expensive training

python3 statistics_pipeline.py ./training_data

# Check all criteria
python3 << 'EOF'
import json, sys
stats = json.load(open('./training_data/data_statistics.json'))

checks = {
    'Total examples >= 1000': stats['meta']['total_valid_examples'] >= 1000,
    'Success rate >= 70%': stats['meta']['overall_success_rate'] >= 70,
    'Avg score >= 7.5': stats['quality_metrics']['validation_scores']['mean'] >= 7.5,
    'No HIGH warnings': sum(1 for r in stats['recommendations'] if r['priority'] == 'HIGH') == 0
}

failed = [k for k, v in checks.items() if not v]
if failed:
    print("Pre-training validation FAILED:")
    for f in failed:
        print(f"  ✗ {f}")
    sys.exit(1)
else:
    print("Pre-training validation PASSED")
    print("Proceeding with training...")
EOF
```

### A/B Testing
```bash
# Compare two approaches
python3 statistics_pipeline.py ./approach_a --output stats_a.json
python3 statistics_pipeline.py ./approach_b --output stats_b.json

python3 << 'EOF'
import json

a = json.load(open('stats_a.json'))
b = json.load(open('stats_b.json'))

print("Comparison:")
print(f"  Success Rate: A={a['meta']['overall_success_rate']:.1f}% vs B={b['meta']['overall_success_rate']:.1f}%")
print(f"  Avg Score: A={a['quality_metrics']['validation_scores']['mean']:.2f} vs B={b['quality_metrics']['validation_scores']['mean']:.2f}")
print(f"  Examples: A={a['meta']['total_valid_examples']} vs B={b['meta']['total_valid_examples']}")

if b['quality_metrics']['validation_scores']['mean'] > a['quality_metrics']['validation_scores']['mean']:
    print("\nRecommendation: Use approach B")
else:
    print("\nRecommendation: Use approach A")
EOF
```

---

## Troubleshooting Quick Fixes

### "No such file or directory"
```bash
# Verify files exist
ls -la <output_dir>/*.jsonl

# Use absolute path
python3 statistics_pipeline.py /absolute/path/to/output_dir
```

### "ImportError: matplotlib"
```bash
# Install visualization dependencies
pip install matplotlib seaborn

# Or run without visualization
python3 statistics_pipeline.py <output_dir>  # Still works!
```

### "Failed to parse line X"
```bash
# Find bad lines
python3 -c "
import json
with open('generated_examples.jsonl') as f:
    for i, line in enumerate(f, 1):
        try:
            json.loads(line)
        except:
            print(f'Error at line {i}')
"

# Remove bad lines and retry
```

---

## Getting Help

1. **Read the docs**: `DATA_STATISTICS_README.md` (comprehensive)
2. **Check examples**: `example_data_statistics.json` (sample output)
3. **Run with verbose**: `python3 statistics_pipeline.py <dir> --verbose`
4. **Review summary**: `STATISTICS_PIPELINE_SUMMARY.md` (overview)
5. **This guide**: `QUICK_REFERENCE.md` (you are here!)

---

## Next Steps After Running Statistics

Based on your results:

### If Success Rate < 60%
1. Run `analyze_failures.py` for detailed failure analysis
2. Review top failure patterns in `failure_analysis.quality_issues`
3. Fix prompts for worst-performing strategies
4. Re-run and compare statistics

### If Data Imbalanced
1. Check `distribution_analysis.question_strategies.distribution`
2. Adjust sampling weights in generation pipeline
3. Generate targeted batches for underrepresented categories
4. Verify balance improved in next run

### If Quality Scores Low
1. Review `quality_metrics.score_dimensions` to find weak areas
2. Improve prompts for low-scoring dimensions
3. Consider using higher-quality model
4. Add better few-shot examples

### If Dataset Too Small
1. Scale up generation pipeline
2. Target minimum 1,000 examples
3. Run multiple batches if needed
4. Monitor quality as you scale

### If Ready for Training
1. Export to training format
2. Configure training based on statistics
3. Use class weights for imbalanced strategies
4. Monitor training metrics vs data statistics

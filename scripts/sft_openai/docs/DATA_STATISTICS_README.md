# Data Statistics Pipeline Documentation

## Overview

The Data Statistics Pipeline provides comprehensive analysis and visualization tools for GPT-based data construction systems. It computes detailed metrics, identifies quality issues, and generates actionable insights to improve ML training data.

## Components

### 1. statistics_pipeline.py
Main statistics computation engine that analyzes generated training data and produces detailed reports.

### 2. visualize_stats.py
Visualization tool that creates charts, graphs, and dashboards from statistics reports.

### 3. Output Files
- `data_statistics.json` - Comprehensive statistics report
- `visualizations/` - Directory containing all generated charts and graphs

---

## Installation

### Basic Requirements
```bash
# Core functionality (statistics only)
pip install numpy

# Visualization support (optional)
pip install matplotlib seaborn numpy
```

### Verify Installation
```bash
python3 -c "import numpy; print('NumPy OK')"
python3 -c "import matplotlib; import seaborn; print('Visualization libraries OK')"
```

---

## Quick Start

### Step 1: Generate Statistics

```bash
# Basic usage
python3 statistics_pipeline.py ./test_output_10

# With verbose output
python3 statistics_pipeline.py ./test_output_10 --verbose

# Custom output location
python3 statistics_pipeline.py ./test_output_10 --output ./my_stats.json
```

**Expected output:**
- Console summary with key metrics
- `data_statistics.json` file with detailed analysis

### Step 2: Generate Visualizations (Optional)

```bash
# Basic usage
python3 visualize_stats.py ./test_output_10/data_statistics.json

# Custom output directory
python3 visualize_stats.py ./test_output_10/data_statistics.json --output-dir ./my_visualizations
```

**Expected output:**
- 14 PNG files in `visualizations/` directory
- Dashboard with comprehensive overview

---

## Metrics Computed

### 1. Distribution Analysis

#### Question Strategies Distribution
- **What it measures**: Frequency and balance of different question generation strategies
- **Key metrics**:
  - Count and percentage for each strategy
  - Entropy score (higher = more diverse)
  - Top strategy dominance
- **Interpretation**:
  - Entropy > 2.5: Good diversity
  - Top strategy > 60%: Potential imbalance
- **ML Impact**: Imbalanced strategies can lead to biased model behavior

#### Category Distribution
- **What it measures**: Wikipedia category coverage and co-occurrence patterns
- **Key metrics**:
  - Unique categories count
  - Average categories per example
  - Top 20 most common categories
  - Category co-occurrence pairs
- **Interpretation**:
  - More unique categories = broader knowledge coverage
  - High Gini coefficient (>0.7) indicates uneven distribution
- **ML Impact**: Category diversity affects model's domain generalization

#### Question Type Distribution
- **What it measures**: Balance between multiple-choice and open-ended questions
- **Key metrics**:
  - Count and percentage of each type
- **Interpretation**:
  - Balanced (40-60% each): Good for diverse evaluation
  - Highly skewed: May limit assessment capabilities
- **ML Impact**: Question type balance affects model's response generation strategies

#### Complexity Distribution
- **What it measures**: Distribution across difficulty levels (low, medium, high)
- **Key metrics**:
  - Count and percentage per level
- **Interpretation**:
  - Balanced distribution helps model learn varied reasoning depths
  - Over-representation of one level may limit capabilities
- **ML Impact**: Complexity balance is crucial for robust reasoning abilities

#### Response Length Distribution
- **What it measures**: Word count statistics for generated responses
- **Key metrics**:
  - Mean, standard deviation, min, max
  - Percentiles: 25th, 50th, 75th, 90th, 95th, 99th
  - Histogram bins
- **Interpretation**:
  - Mean 400-800 words: Typical for detailed responses
  - High variance: Good diversity in response complexity
  - Outliers: May indicate generation issues
- **ML Impact**: Response length affects token efficiency and training dynamics

#### Cycle Count Distribution
- **What it measures**: Number of look-think iterations in responses
- **Key metrics**:
  - Mean, range, percentiles
  - Distribution histogram
- **Interpretation**:
  - Mean 2-3 cycles: Standard analytical depth
  - Range indicates reasoning complexity variation
- **ML Impact**: Cycle patterns affect model's reasoning structure learning

### 2. Quality Metrics

#### Validation Score Distribution
- **What it measures**: Overall quality scores (0-10 scale) for valid examples
- **Key metrics**:
  - Mean, standard deviation
  - Full percentile distribution
  - Histogram
- **Interpretation**:
  - Mean > 8.0: High quality dataset
  - Mean 7.0-8.0: Good quality
  - Mean < 7.0: Quality improvements needed
- **ML Impact**: Higher average scores correlate with better model performance

#### Score Dimensions
- **What it measures**: Individual quality aspects
- **Dimensions**:
  - `format_score`: Tag structure and formatting compliance
  - `content_score`: Factual accuracy and relevance
  - `coherence_score`: Logical flow and consistency
  - `diversity_score`: Variation and creativity
  - `educational_score`: Learning value and depth
- **Interpretation**:
  - Consistently low dimension: Systematic generation issue
  - Balanced dimensions: Well-rounded quality
- **ML Impact**: Weak dimensions indicate areas for prompt improvement

#### Success Rate by Strategy
- **What it measures**: Percentage of valid examples per question strategy
- **Key metrics**:
  - Success rate for each strategy
  - Valid/failed/total counts
- **Interpretation**:
  - Strategy with <50% success: Needs prompt refinement
  - Strategy with >90% success: Well-tuned
- **ML Impact**: Low-success strategies may need more examples or better prompts

#### Success Rate by Complexity
- **What it measures**: Validation success across difficulty levels
- **Key metrics**:
  - Success rate per complexity level
- **Interpretation**:
  - Lower success at high complexity: Expected, but monitor gap
  - Very low success (<40%): Complexity definition may be off
- **ML Impact**: Helps calibrate difficulty and generation parameters

#### Success Rate by Question Type
- **What it measures**: Validation success for MC vs open-ended
- **Key metrics**:
  - Success rate for each type
- **Interpretation**:
  - Large gap (>20%): One type needs improvement
- **ML Impact**: Informs which question type needs better generation

#### Success Rate by Category
- **What it measures**: Validation success across Wikipedia categories
- **Key metrics**:
  - Top 20 by frequency
  - Top/bottom 10 by success rate (min 3 examples)
- **Interpretation**:
  - Categories with low success: May lack good source images
  - Identifies domains that need attention
- **ML Impact**: Category success patterns reveal domain-specific challenges

### 3. Failure Analysis

#### Failure Type Breakdown
- **What it measures**: Structure vs quality validation failures
- **Key metrics**:
  - Structure failure count and percentage
  - Quality failure count and percentage
- **Interpretation**:
  - High structure failures: Generation template issues
  - High quality failures: Content or reasoning problems
- **ML Impact**: Helps prioritize fixing generation vs validation

#### Structure Error Patterns
- **What it measures**: Types of tag structure violations
- **Common errors**:
  - `not_alternating`: Look/think tags not properly interleaved
  - `text_between_tags`: Unexpected text between sections
  - `missing_tags`: Required tags absent
  - `empty_sections`: Tags present but no content
- **Interpretation**:
  - Recurring error type: Systematic prompt issue
- **ML Impact**: Structure errors corrupt training signal

#### Quality Issue Patterns
- **What it measures**: Semantic and content quality problems
- **Common issues**:
  - "Does not address the question"
  - "Missing required information"
  - "Factual inaccuracies"
  - "Insufficient visual analysis"
  - "Poor reasoning quality"
- **Interpretation**:
  - Top issues indicate prompt weaknesses
- **ML Impact**: Quality issues degrade model learning

#### Near-Miss Analysis
- **What it measures**: Examples that almost passed (score > 6 but failed)
- **Key metrics**:
  - Count and percentage of near-misses
  - Average score of near-misses
- **Interpretation**:
  - High near-miss rate (>30%): Consider threshold adjustment
  - Low near-miss rate: Clear quality separation
- **ML Impact**: Near-misses may be recoverable with minor fixes

### 4. Data Quality Insights

#### Category Coverage
- **What it measures**: How evenly categories are represented
- **Key metrics**:
  - Unique categories count
  - Gini coefficient (0 = perfect equality, 1 = maximum inequality)
  - Coverage quality rating
- **Interpretation**:
  - Gini < 0.6: Good coverage
  - Gini 0.6-0.8: Moderate coverage
  - Gini > 0.8: Poor coverage (dominated by few categories)
- **ML Impact**: Poor coverage = limited domain generalization

#### Strategy Balance
- **What it measures**: How evenly question strategies are distributed
- **Key metrics**:
  - Expected percentage (100 / num_strategies)
  - Max deviation from expected
  - Entropy
  - Balance quality rating
- **Interpretation**:
  - Max deviation < 20%: Well balanced
  - Max deviation > 30%: Significant imbalance
- **ML Impact**: Imbalance can cause strategy overfitting

#### Quality Trends
- **What it measures**: Score distribution patterns and outliers
- **Key metrics**:
  - Score range (min, max, span)
  - Outlier counts (using IQR method)
  - Quality tiers (excellent/good/acceptable/poor)
- **Interpretation**:
  - Many low outliers: Review generation pipeline
  - Quality tier distribution shows dataset maturity
- **ML Impact**: Tier distribution affects training stability

#### Look-Think-Answer Patterns
- **What it measures**: Analytical reasoning structure
- **Key metrics**:
  - Cycle count distribution
  - Correlation between cycles and word count
  - Average words per cycle by cycle count
- **Interpretation**:
  - Positive correlation: More cycles → deeper analysis
  - Consistent words/cycle: Structured reasoning
- **ML Impact**: Patterns influence model's reasoning habits

#### Multiple Choice Quality
- **What it measures**: MC question characteristics
- **Key metrics**:
  - Total MC questions
  - Option count statistics
  - Correct answer distribution (A/B/C/D balance)
  - Answer balance quality
- **Interpretation**:
  - Unbalanced correct answers (one >40%): Potential bias
  - Balanced distribution: Good MC design
- **ML Impact**: Answer bias can lead to shortcut learning

#### Imbalance Warnings
- **What it measures**: Automatic detection of data quality issues
- **Warning types**:
  - Strategy imbalance
  - Complexity imbalance
  - Question type imbalance
  - Small dataset size
- **Severity levels**: HIGH, MEDIUM, LOW
- **ML Impact**: Each warning identifies potential training problems

### 5. Recommendations

Automatically generated, prioritized recommendations based on computed metrics:

- **Priority Levels**: HIGH, MEDIUM, LOW
- **Categories**:
  - Overall Quality
  - Data Balance
  - Coverage
  - Quality Scores
  - Question Type Balance
  - Validation Tuning

---

## Understanding the Output

### Console Output Structure

```
================================================================================
LOADING DATA
================================================================================
Loaded 150 successful examples
Loaded 48 failed examples

Total examples: 198

================================================================================
COMPUTING STATISTICS
================================================================================

[1/5] Computing distribution analysis...
[2/5] Computing quality metrics...
[3/5] Computing failure analysis...
[4/5] Computing data quality insights...
[5/5] Generating recommendations...

Statistics computation complete!

================================================================================
DATA STATISTICS SUMMARY
================================================================================
[Detailed metrics printed here...]

================================================================================
RECOMMENDATIONS
================================================================================
[Actionable recommendations listed here...]

================================================================================
Statistics saved to: ./test_output_10/data_statistics.json
================================================================================
```

### JSON Output Structure

```json
{
  "meta": {
    "output_directory": "...",
    "timestamp": "...",
    "total_valid_examples": 150,
    "total_failed_examples": 48,
    "overall_success_rate": 75.76
  },
  "distribution_analysis": {
    "question_strategies": {...},
    "categories": {...},
    "question_types": {...},
    "complexity": {...},
    "response_length": {...},
    "cycle_count": {...}
  },
  "quality_metrics": {
    "validation_scores": {...},
    "score_dimensions": {...},
    "success_by_strategy": {...},
    "success_by_complexity": {...},
    "success_by_question_type": {...},
    "success_by_category": {...}
  },
  "failure_analysis": {
    "failure_types": {...},
    "structure_errors": {...},
    "quality_issues": {...},
    "failure_score_distribution": {...},
    "near_misses": {...}
  },
  "data_quality_insights": {
    "category_coverage": {...},
    "strategy_balance": {...},
    "quality_trends": {...},
    "lta_patterns": {...},
    "mc_quality": {...},
    "imbalance_warnings": [...]
  },
  "recommendations": [
    {
      "priority": "HIGH",
      "category": "Overall Quality",
      "issue": "...",
      "recommendation": "..."
    }
  ]
}
```

---

## Visualization Gallery

### Generated Visualizations

1. **strategy_distribution.png**
   - Horizontal bar chart of question strategies
   - Shows count for each strategy type

2. **complexity_distribution.png**
   - Bar chart and pie chart of complexity levels
   - Shows both counts and percentages

3. **question_type_distribution.png**
   - MC vs open-ended comparison
   - Bar and pie chart views

4. **response_length.png**
   - Histogram of word counts
   - Box plot with statistics overlay
   - Mean and median indicators

5. **cycle_distribution.png**
   - Histogram of look-think cycles
   - Mean indicator

6. **score_distribution.png**
   - Validation score histogram
   - Statistics summary panel
   - Mean and median lines

7. **score_dimensions.png**
   - Bar chart comparing score dimensions
   - Error bars showing standard deviation

8. **success_by_strategy.png**
   - Horizontal bar chart with color coding
   - Green (>80%), Yellow (60-80%), Red (<60%)
   - Sample size annotations

9. **success_by_complexity.png**
   - Success rate bars
   - Stacked bar showing valid vs failed

10. **category_coverage.png**
    - Top 15 Wikipedia categories
    - Horizontal bar chart

11. **quality_tiers.png**
    - Distribution across quality levels
    - Bar and pie chart views

12. **failure_analysis.png**
    - Multi-panel overview
    - Failure types, structure errors, quality issues

13. **score_correlation.png**
    - Heatmap showing dimension similarity
    - Color-coded matrix

14. **dashboard.png**
    - Comprehensive overview panel
    - 7 subplots with key metrics
    - Publication-ready summary

---

## Use Cases

### Use Case 1: Daily Quality Monitoring

**Scenario**: You run data generation daily and want to track quality trends.

**Workflow**:
```bash
# Generate data
python3 data_construction_pipeline.py --input wiki_images.parquet --output ./daily_run_$(date +%Y%m%d)

# Compute statistics
python3 statistics_pipeline.py ./daily_run_$(date +%Y%m%d)

# Check console output for issues
# Review data_statistics.json for detailed metrics
# Compare with previous days' statistics
```

**Key Metrics to Monitor**:
- Overall success rate
- Average validation score
- Top failure reasons
- Strategy balance

### Use Case 2: A/B Testing Generation Prompts

**Scenario**: You want to compare two different prompt templates.

**Workflow**:
```bash
# Generate with prompt A
python3 data_construction_pipeline.py --prompt-template prompt_a.txt --output ./test_prompt_a

# Generate with prompt B
python3 data_construction_pipeline.py --prompt-template prompt_b.txt --output ./test_prompt_b

# Compare statistics
python3 statistics_pipeline.py ./test_prompt_a --output ./stats_a.json
python3 statistics_pipeline.py ./test_prompt_b --output ./stats_b.json

# Compare key metrics programmatically
python3 << EOF
import json
with open('./stats_a.json') as f: stats_a = json.load(f)
with open('./stats_b.json') as f: stats_b = json.load(f)

print(f"Prompt A success rate: {stats_a['meta']['overall_success_rate']:.1f}%")
print(f"Prompt B success rate: {stats_b['meta']['overall_success_rate']:.1f}%")

print(f"Prompt A avg score: {stats_a['quality_metrics']['validation_scores']['mean']:.2f}")
print(f"Prompt B avg score: {stats_b['quality_metrics']['validation_scores']['mean']:.2f}")
EOF
```

**Comparison Metrics**:
- Success rate difference
- Average validation score
- Failure type distribution
- Response length consistency

### Use Case 3: Identifying Data Imbalances

**Scenario**: Before training, you want to ensure balanced data.

**Workflow**:
```bash
# Run statistics
python3 statistics_pipeline.py ./training_data

# Review the imbalance_warnings section in output
# Check recommendations for HIGH priority items
```

**What to Look For**:
```json
"imbalance_warnings": [
  {
    "type": "strategy_imbalance",
    "severity": "medium",
    "message": "Strategy 'visual_perception' dominates with 65.3%",
    "recommendation": "Increase diversity in question strategy sampling"
  }
]
```

**Action Items**:
- Adjust sampling weights in generation pipeline
- Generate more examples for underrepresented strategies
- Consider upsampling/downsampling during training

### Use Case 4: Debugging Low Success Rates

**Scenario**: Your pipeline has <60% success rate.

**Workflow**:
```bash
# Generate detailed statistics
python3 statistics_pipeline.py ./problematic_run --verbose

# Generate visualizations
python3 visualize_stats.py ./problematic_run/data_statistics.json
```

**Diagnostic Steps**:

1. **Check failure_types**:
   ```json
   "failure_types": {
     "structure_failures": {"count": 120, "percentage": 75.0},
     "quality_failures": {"count": 40, "percentage": 25.0}
   }
   ```
   - If structure failures dominate: Fix prompt templates
   - If quality failures dominate: Improve content generation

2. **Review top structure errors**:
   ```json
   "structure_errors": {
     "distribution": [
       {"error_type": "not_alternating", "count": 80, "percentage": 66.7}
     ]
   }
   ```
   - Focus on most common error type
   - Update prompt to enforce correct structure

3. **Examine quality issues**:
   ```json
   "quality_issues": {
     "top_20_issues": [
       {"issue": "Does not address the question", "count": 25}
     ]
   }
   ```
   - Common issues indicate systematic problems
   - Refine prompts to address specific issues

4. **Check success_by_strategy**:
   - Identify which strategies fail most
   - Focus debugging efforts on problematic strategies

### Use Case 5: Pre-Training Data Audit

**Scenario**: Final validation before starting expensive training run.

**Checklist**:

```bash
# Generate comprehensive report
python3 statistics_pipeline.py ./final_training_data
python3 visualize_stats.py ./final_training_data/data_statistics.json
```

**Validation Criteria**:

- [ ] **Minimum size**: Total valid examples >= 1,000 (preferably 10,000+)
- [ ] **Success rate**: >= 70%
- [ ] **Average quality**: Mean validation score >= 7.5
- [ ] **Strategy balance**: No single strategy > 50%
- [ ] **Category coverage**: Unique categories >= 50
- [ ] **Question type balance**: Both MC and open-ended >= 25%
- [ ] **No HIGH priority warnings**: Review all HIGH severity imbalance warnings
- [ ] **Quality tiers**: Excellent + Good >= 70% of examples

**Red Flags**:
- Small dataset (<500 examples)
- Low success rate (<50%)
- Poor quality scores (mean <6.5)
- Severe imbalances (one category >80%)
- Many structure failures (>30%)

### Use Case 6: Continuous Improvement Tracking

**Scenario**: You're iteratively improving your pipeline over weeks.

**Workflow**:

```bash
# Week 1 baseline
python3 statistics_pipeline.py ./week1_data --output ./metrics/week1_stats.json

# Week 2 after improvements
python3 statistics_pipeline.py ./week2_data --output ./metrics/week2_stats.json

# Compare trends
python3 << EOF
import json
import glob

stats_files = sorted(glob.glob('./metrics/week*_stats.json'))
for f in stats_files:
    with open(f) as fp:
        stats = json.load(fp)
        week = f.split('/')[-1].replace('_stats.json', '')
        print(f"{week}:")
        print(f"  Success: {stats['meta']['overall_success_rate']:.1f}%")
        print(f"  Avg Score: {stats['quality_metrics']['validation_scores']['mean']:.2f}")
        print(f"  Examples: {stats['meta']['total_valid_examples']}")
        print()
EOF
```

**Track Over Time**:
- Success rate trend (should increase)
- Quality score trend (should increase)
- Failure type distribution (structure failures should decrease)
- Strategy balance (entropy should stabilize)

---

## Interpreting Recommendations

The pipeline generates prioritized recommendations automatically. Here's how to act on them:

### HIGH Priority

**Example**: "Low success rate (45.2%)"

**Actions**:
1. Run analyze_failures.py for detailed failure analysis
2. Review top failure patterns
3. Update prompt templates
4. Test on small batch
5. Re-run statistics to verify improvement

**Example**: "Low average validation score (6.45)"

**Actions**:
1. Review score_dimensions to identify weak areas
2. Add few-shot examples for low-scoring dimensions
3. Consider using higher-quality model (e.g., GPT-4 vs GPT-3.5)
4. Improve validation criteria if too strict

**Example**: "Only 87 valid examples - too small for robust training"

**Actions**:
1. Scale up generation pipeline
2. Target minimum 1,000 examples
3. Run multiple batches if needed

### MEDIUM Priority

**Example**: "Imbalanced strategy distribution"

**Actions**:
1. Adjust sampling weights in strategy selection
2. Generate targeted batches for underrepresented strategies
3. Use balanced sampling during training

**Example**: "Many near-miss failures (35 examples)"

**Actions**:
1. Review near-miss examples manually
2. Consider adjusting validation thresholds
3. Implement retry mechanism for borderline cases
4. Add validation with different criteria

### LOW Priority

**Example**: "Limited category diversity (18 unique categories)"

**Actions**:
1. Expand source data to include more diverse topics
2. Not critical if current categories align with use case
3. Monitor for domain overfitting during training

**Example**: "Question type imbalance (MC: 72.3%)"

**Actions**:
1. Adjust MC vs open-ended ratio in generation
2. Consider use case: Some applications prefer one type
3. Balance if training general-purpose model

---

## Advanced Usage

### Programmatic Access

```python
from statistics_pipeline import DataStatistics

# Initialize
stats = DataStatistics('./test_output_10', verbose=True)

# Load data
stats.load_data()

# Compute statistics
stats.compute_all_statistics()

# Access metrics
success_rate = stats.stats['meta']['overall_success_rate']
avg_score = stats.stats['quality_metrics']['validation_scores']['mean']

# Custom analysis
if success_rate < 70:
    print("WARNING: Low success rate!")
    # Trigger alerts, retry generation, etc.

# Save
stats.save_report()
```

### Batch Processing

```bash
#!/bin/bash
# Process multiple output directories

for dir in ./output_batch_*/; do
    echo "Processing $dir..."
    python3 statistics_pipeline.py "$dir"
    python3 visualize_stats.py "$dir/data_statistics.json"
done

echo "All batches processed!"
```

### Custom Metrics

Extend the `DataStatistics` class to add domain-specific metrics:

```python
from statistics_pipeline import DataStatistics

class CustomStatistics(DataStatistics):
    def _compute_quality_insights(self):
        super()._compute_quality_insights()

        # Add custom metric
        self.stats['data_quality_insights']['custom_metric'] = self._my_custom_analysis()

    def _my_custom_analysis(self):
        # Your custom analysis logic
        return {
            'custom_score': 42,
            'custom_distribution': [...]
        }
```

---

## Performance Considerations

### Memory Usage

- **Small datasets (<1K examples)**: <100MB RAM
- **Medium datasets (1K-10K)**: ~500MB RAM
- **Large datasets (10K-100K)**: 1-2GB RAM

For very large datasets (>100K examples):
```python
# Use chunked processing
def load_jsonl_chunked(filepath, chunk_size=10000):
    chunk = []
    with open(filepath) as f:
        for line in f:
            chunk.append(json.loads(line))
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
    if chunk:
        yield chunk
```

### Execution Time

- **Statistics computation**: ~1-5 seconds per 1K examples
- **Visualization generation**: ~10-30 seconds total

### Optimization Tips

1. **Skip visualizations** if only need metrics:
   ```bash
   python3 statistics_pipeline.py ./data --output ./stats.json
   # Skip visualize_stats.py
   ```

2. **Parallel batch processing**:
   ```bash
   ls -d output_*/ | xargs -P 4 -I {} python3 statistics_pipeline.py {}
   ```

3. **Incremental statistics** for continuous generation:
   - Store running statistics
   - Update incrementally as new data arrives
   - Avoid recomputing from scratch

---

## Troubleshooting

### Issue: "No such file or directory"

**Cause**: Incorrect output directory path

**Solution**:
```bash
# Verify directory exists and contains required files
ls -la ./test_output_10/
# Should show: generated_examples.jsonl, failed_examples.jsonl, pipeline_report.json

# Use absolute path if needed
python3 statistics_pipeline.py /absolute/path/to/output_dir
```

### Issue: "Failed to parse line X"

**Cause**: Corrupted JSONL file

**Solution**:
```bash
# Find problematic lines
python3 << EOF
import json
with open('./generated_examples.jsonl') as f:
    for i, line in enumerate(f, 1):
        try:
            json.loads(line)
        except:
            print(f"Error at line {i}: {line[:100]}")
EOF

# Fix or remove problematic lines
# Re-run statistics
```

### Issue: ImportError for visualization libraries

**Cause**: matplotlib/seaborn not installed

**Solution**:
```bash
# Install visualization dependencies
pip install matplotlib seaborn

# Or use statistics without visualization
python3 statistics_pipeline.py ./data  # This still works
```

### Issue: Empty or missing statistics sections

**Cause**: No valid examples or missing fields in data

**Solution**:
```bash
# Verify data structure
python3 << EOF
import json
with open('./generated_examples.jsonl') as f:
    ex = json.loads(f.readline())
    print(json.dumps(ex, indent=2))

# Check for required fields
required = ['question_strategy', 'complexity', 'word_count', 'num_cycles']
for field in required:
    print(f"{field}: {field in ex}")
EOF
```

### Issue: Visualizations look incorrect

**Cause**: Missing or sparse data

**Solution**:
- Check that you have sufficient examples (>10)
- Verify data is not all identical values
- Review console warnings during visualization

---

## Best Practices

### 1. Run Statistics After Every Generation

Make it part of your pipeline:
```bash
#!/bin/bash
set -e

# Generate data
python3 data_construction_pipeline.py --input data.parquet --output ./output_$(date +%Y%m%d)

# Compute statistics
python3 statistics_pipeline.py ./output_$(date +%Y%m%d)

# Alert if success rate too low
SUCCESS_RATE=$(python3 -c "import json; print(json.load(open('./output_$(date +%Y%m%d)/data_statistics.json'))['meta']['overall_success_rate'])")
if (( $(echo "$SUCCESS_RATE < 70" | bc -l) )); then
    echo "WARNING: Low success rate: $SUCCESS_RATE%"
    # Send alert, abort pipeline, etc.
fi
```

### 2. Archive Statistics for Comparison

```bash
# Create statistics archive
mkdir -p ./statistics_archive/
cp ./output_dir/data_statistics.json ./statistics_archive/stats_$(date +%Y%m%d_%H%M%S).json

# Track trends over time
python3 trend_analysis.py ./statistics_archive/
```

### 3. Use Recommendations as Checklist

Before marking data as "production-ready":
- Address all HIGH priority recommendations
- Review and plan for MEDIUM priority items
- Document LOW priority items for future work

### 4. Combine with Manual Review

Statistics are powerful but not complete:
- Randomly sample 10-20 examples manually
- Check for subtle quality issues statistics miss
- Validate that high scores match human judgment

### 5. Version Control Your Metrics

```bash
# Add statistics to git for reproducibility
git add ./output_dir/data_statistics.json
git commit -m "Statistics for run $(date +%Y%m%d): ${SUCCESS_RATE}% success, ${AVG_SCORE} avg score"
```

---

## FAQ

**Q: How many examples do I need before running statistics?**

A: Minimum 10 for basic stats, 100+ for reliable distributions, 1000+ for robust ML training assessment.

**Q: What's a "good" success rate?**

A: 70-80% is typical for complex generation tasks. 85%+ is excellent. <60% needs investigation.

**Q: Should I always aim for perfectly balanced strategies?**

A: Not necessarily. Balance is good for general models, but domain-specific use cases may intentionally skew toward certain strategies.

**Q: How do I know if my validation criteria are too strict?**

A: High near-miss rate (>40%) + manual review showing good quality = criteria may be too strict.

**Q: Can I run statistics on subsets of data?**

A: Yes! Create subset JSONL files and run statistics_pipeline.py on them. Useful for category-specific or strategy-specific analysis.

**Q: What if I have custom fields in my data?**

A: The pipeline handles missing fields gracefully. Extend the DataStatistics class to add custom metrics for your fields.

**Q: How often should I regenerate visualizations?**

A: After significant changes to data or when preparing reports. Not needed for routine monitoring (use console output).

---

## Examples

### Example Output Directory Structure

```
test_output_10/
├── generated_examples.jsonl       # Valid examples
├── failed_examples.jsonl          # Failed examples
├── pipeline_report.json           # Generation stats
├── data_statistics.json           # Computed statistics (NEW)
└── visualizations/                # Generated charts (NEW)
    ├── dashboard.png
    ├── strategy_distribution.png
    ├── complexity_distribution.png
    ├── question_type_distribution.png
    ├── response_length.png
    ├── cycle_distribution.png
    ├── score_distribution.png
    ├── score_dimensions.png
    ├── success_by_strategy.png
    ├── success_by_complexity.png
    ├── category_coverage.png
    ├── quality_tiers.png
    ├── failure_analysis.png
    └── score_correlation.png
```

### Example Command Sequence

```bash
# Complete workflow
cd /data_ali/shunian/verl/scripts/sft_openai

# 1. Generate training data
python3 data_construction_pipeline.py \
    --input /path/to/wiki_images.parquet \
    --output ./production_run_v1 \
    --num-workers 4 \
    --batch-size 100

# 2. Compute statistics
python3 statistics_pipeline.py ./production_run_v1 --verbose

# 3. Generate visualizations
python3 visualize_stats.py ./production_run_v1/data_statistics.json

# 4. Review outputs
cat ./production_run_v1/data_statistics.json | jq '.recommendations'
open ./production_run_v1/visualizations/dashboard.png

# 5. If quality is good, proceed to training
# If not, review failures and iterate
python3 analyze_failures.py ./production_run_v1
```

---

## Integration with ML Training

### Using Statistics to Configure Training

```python
import json

# Load statistics
with open('./training_data/data_statistics.json') as f:
    stats = json.load(f)

# Extract key metrics
total_examples = stats['meta']['total_valid_examples']
strategies = stats['distribution_analysis']['question_strategies']['distribution']
complexity_dist = stats['distribution_analysis']['complexity']['distribution']

# Configure training based on data characteristics
training_config = {
    'batch_size': min(32, total_examples // 100),
    'epochs': 3 if total_examples > 10000 else 5,
    'class_weights': {
        item['value']: 1.0 / item['percentage'] * 100
        for item in strategies
    }
}

print(json.dumps(training_config, indent=2))
```

### Data Augmentation Priorities

```python
# Identify underrepresented categories for targeted augmentation
stats = json.load(open('./data_statistics.json'))

strategies = stats['quality_metrics']['success_by_strategy']['by_category']
underrep = [
    s['value'] for s in strategies
    if s['total'] < 100  # Fewer than 100 examples
]

print("Generate more examples for:", underrep)
```

---

## Support and Contributing

### Getting Help

1. Check this README thoroughly
2. Review troubleshooting section
3. Examine example output files
4. Run with `--verbose` flag for detailed output

### Reporting Issues

Include:
- Command run
- Error message or unexpected output
- Sample of input data (anonymized)
- Statistics output (if generated)

### Extending the Pipeline

The code is designed to be extensible:
- Add new metrics in `_compute_*` methods
- Add new visualizations in `visualize_stats.py`
- Contribute improvements via pull requests

---

## Summary

The Data Statistics Pipeline provides comprehensive, production-ready analysis for ML training data:

- **Comprehensive**: 50+ metrics across 5 categories
- **Actionable**: Automatic recommendations with priorities
- **Visual**: 14 publication-ready visualizations
- **Fast**: Processes 10K examples in seconds
- **Extensible**: Easy to add custom metrics
- **ML-focused**: Metrics that matter for model training

Use it to:
- Monitor data quality continuously
- Debug generation issues
- Validate data before training
- Track improvements over time
- Make data-driven decisions

**Quick Commands Reference:**
```bash
# Generate statistics
python3 statistics_pipeline.py <output_dir>

# Generate visualizations
python3 visualize_stats.py <output_dir>/data_statistics.json

# Complete analysis
python3 statistics_pipeline.py <output_dir> && \
python3 visualize_stats.py <output_dir>/data_statistics.json
```

#!/usr/bin/env python3
"""
分析失败样本的脚本

用于对比成功和失败的样本，帮助理解质量问题
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Dict


def load_jsonl(filepath: str) -> List[Dict]:
    """Load JSONL file"""
    examples = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def analyze_validation_failures(failed_examples: List[Dict]):
    """Analyze why examples failed validation"""
    print("\n" + "=" * 80)
    print("VALIDATION FAILURE ANALYSIS")
    print("=" * 80)

    # Separate structure failures from quality failures
    structure_failures = []
    quality_failures = []

    for ex in failed_examples:
        validation = ex.get('validation_result', {})
        if validation.get('validation_method') == 'strict_tag_structure':
            structure_failures.append(ex)
        else:
            quality_failures.append(ex)

    print(f"\nTotal failed examples: {len(failed_examples)}")
    print(f"  - Tag structure failures: {len(structure_failures)}")
    print(f"  - Quality validation failures: {len(quality_failures)}")

    # Analyze structure failures
    if structure_failures:
        print(f"\nTag Structure Errors:")
        structure_errors = Counter([
            ex['validation_result'].get('structure_error', 'unknown')
            for ex in structure_failures
        ])
        for error, count in structure_errors.most_common():
            print(f"  - {error}: {count}")

    # Analyze quality failures
    if quality_failures:
        print(f"\nQuality Validation Issues:")
        all_issues = []
        for ex in quality_failures:
            validation = ex.get('validation_result', {})
            issues = validation.get('issues', [])
            all_issues.extend(issues)

        issue_counts = Counter(all_issues)
        for issue, count in issue_counts.most_common(10):
            print(f"  - {issue}: {count}")

        # Score distribution (only for quality failures)
        score_distribution = defaultdict(int)
        for ex in quality_failures:
            score = ex.get('overall_score', 0)
            score_bin = int(score)  # 0-10
            score_distribution[score_bin] += 1

        print(f"\nQuality score distribution:")
        for score in sorted(score_distribution.keys()):
            count = score_distribution[score]
            bar = "█" * (count * 50 // len(quality_failures))
            print(f"  {score:2d}: {bar} ({count})")


def compare_success_vs_failure(valid_examples: List[Dict], failed_examples: List[Dict]):
    """Compare characteristics of successful vs failed examples"""
    print("\n" + "=" * 80)
    print("SUCCESS vs FAILURE COMPARISON")
    print("=" * 80)

    # Question strategies
    valid_strategies = Counter(ex['question_strategy'] for ex in valid_examples)
    failed_strategies = Counter(ex['question_strategy'] for ex in failed_examples)

    print("\nQuestion strategies:")
    all_strategies = set(valid_strategies.keys()) | set(failed_strategies.keys())
    for strategy in sorted(all_strategies):
        valid_count = valid_strategies.get(strategy, 0)
        failed_count = failed_strategies.get(strategy, 0)
        total = valid_count + failed_count
        success_rate = valid_count / total * 100 if total > 0 else 0
        print(f"  {strategy:25s}: {valid_count:3d} valid, {failed_count:3d} failed ({success_rate:.1f}% success)")

    # Question types
    valid_mc = sum(1 for ex in valid_examples if ex.get('is_multiple_choice', False))
    failed_mc = sum(1 for ex in failed_examples if ex.get('is_multiple_choice', False))

    print(f"\nQuestion types:")
    print(f"  Multiple choice: {valid_mc} valid, {failed_mc} failed")
    print(f"  Open-ended:      {len(valid_examples) - valid_mc} valid, {len(failed_examples) - failed_mc} failed")

    # Word count
    valid_word_counts = [ex.get('word_count', 0) for ex in valid_examples if 'word_count' in ex]
    failed_word_counts = [ex.get('response', '').split() for ex in failed_examples]
    failed_word_counts = [len(words) for words in failed_word_counts]

    if valid_word_counts and failed_word_counts:
        print(f"\nResponse length:")
        print(f"  Valid:  avg {sum(valid_word_counts)/len(valid_word_counts):.0f} words")
        print(f"  Failed: avg {sum(failed_word_counts)/len(failed_word_counts):.0f} words")


def show_sample_comparisons(valid_examples: List[Dict], failed_examples: List[Dict], n: int = 2):
    """Show side-by-side comparison of samples"""
    print("\n" + "=" * 80)
    print("SAMPLE COMPARISONS")
    print("=" * 80)

    print(f"\n{'='*40}")
    print("SUCCESSFUL EXAMPLES")
    print(f"{'='*40}")

    for i, ex in enumerate(valid_examples[:n], 1):
        print(f"\n--- Sample {i} ---")
        print(f"Strategy: {ex['question_strategy']}")
        print(f"Type: {'Multiple choice' if ex.get('is_multiple_choice') else 'Open-ended'}")
        print(f"Question: {ex['question'][:150]}...")
        if ex.get('is_multiple_choice') and ex.get('options'):
            print("Options:")
            for opt in ex['options'][:2]:
                print(f"  {opt}")
            print("  ...")
        print(f"Response (first 200 chars): {ex['response'][:200]}...")
        validation = ex.get('validation_metadata', {})
        print(f"Validation score: {validation.get('overall_score', 'N/A')}")

    print(f"\n{'='*40}")
    print("FAILED EXAMPLES")
    print(f"{'='*40}")

    for i, ex in enumerate(failed_examples[:n], 1):
        print(f"\n--- Sample {i} ---")
        print(f"Strategy: {ex['question_strategy']}")
        print(f"Type: {'Multiple choice' if ex.get('is_multiple_choice') else 'Open-ended'}")
        print(f"Question: {ex['question'][:150]}...")
        if ex.get('is_multiple_choice') and ex.get('options'):
            print("Options:")
            for opt in ex['options'][:2]:
                print(f"  {opt}")
            print("  ...")
        print(f"Response (first 200 chars): {ex['response'][:200]}...")
        print(f"Validation score: {ex.get('overall_score', 0)}")
        print(f"Failure reasons: {', '.join(ex.get('failure_reason', []))}")


def generate_report(output_dir: str):
    """Generate comprehensive failure analysis report"""
    output_path = Path(output_dir)

    valid_file = output_path / 'generated_examples.jsonl'
    failed_file = output_path / 'failed_examples.jsonl'

    # Check files exist
    if not valid_file.exists():
        print(f"Error: {valid_file} not found")
        return

    if not failed_file.exists():
        print(f"Warning: {failed_file} not found - no failures to analyze")
        valid_examples = load_jsonl(str(valid_file))
        print(f"\nLoaded {len(valid_examples)} successful examples")
        print("All examples passed validation!")
        return

    # Load data
    print("Loading data...")
    valid_examples = load_jsonl(str(valid_file))
    failed_examples = load_jsonl(str(failed_file))

    print(f"Loaded {len(valid_examples)} successful examples")
    print(f"Loaded {len(failed_examples)} failed examples")

    if len(failed_examples) == 0:
        print("\nNo failed examples to analyze!")
        return

    # Overall statistics
    total = len(valid_examples) + len(failed_examples)
    success_rate = len(valid_examples) / total * 100

    print("\n" + "=" * 80)
    print("OVERALL STATISTICS")
    print("=" * 80)
    print(f"Total examples generated: {total}")
    print(f"Successful: {len(valid_examples)} ({success_rate:.1f}%)")
    print(f"Failed: {len(failed_examples)} ({100-success_rate:.1f}%)")

    # Detailed analysis
    analyze_validation_failures(failed_examples)
    compare_success_vs_failure(valid_examples, failed_examples)
    show_sample_comparisons(valid_examples, failed_examples)

    # Save detailed report
    report_file = output_path / 'failure_analysis_report.txt'
    print(f"\n{'='*80}")
    print(f"Detailed report would be saved to: {report_file}")
    print("=" * 80)


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_failures.py <output_dir>")
        print("\nExample:")
        print("  python3 analyze_failures.py ./test_output_10")
        sys.exit(1)

    output_dir = sys.argv[1]
    generate_report(output_dir)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Comprehensive Data Statistics Pipeline for GPT-based Data Construction System

This script computes detailed statistics and quality metrics for generated training data,
providing actionable insights for data quality assessment and ML training preparation.

Usage:
    python3 statistics_pipeline.py <output_dir> [--viz] [--verbose]

Example:
    python3 statistics_pipeline.py ./test_output_10 --viz
"""

import json
import sys
import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
import numpy as np


class DataStatistics:
    """Comprehensive statistics computation for ML training data"""

    def __init__(self, output_dir: str, verbose: bool = False):
        self.output_dir = Path(output_dir)
        self.verbose = verbose

        # File paths
        self.valid_file = self.output_dir / 'generated_examples.jsonl'
        self.failed_file = self.output_dir / 'failed_examples.jsonl'
        self.report_file = self.output_dir / 'pipeline_report.json'

        # Data containers
        self.valid_examples = []
        self.failed_examples = []
        self.pipeline_report = {}

        # Statistics output
        self.stats = {
            'meta': {},
            'distribution_analysis': {},
            'quality_metrics': {},
            'failure_analysis': {},
            'data_quality_insights': {},
            'recommendations': []
        }

    def load_data(self) -> bool:
        """Load all data files"""
        print("\n" + "="*80)
        print("LOADING DATA")
        print("="*80)

        # Load valid examples
        if not self.valid_file.exists():
            print(f"ERROR: {self.valid_file} not found")
            return False

        self.valid_examples = self._load_jsonl(self.valid_file)
        print(f"Loaded {len(self.valid_examples):,} successful examples")

        # Load failed examples (optional)
        if self.failed_file.exists():
            self.failed_examples = self._load_jsonl(self.failed_file)
            print(f"Loaded {len(self.failed_examples):,} failed examples")
        else:
            print("No failed examples file found")

        # Load pipeline report (optional)
        if self.report_file.exists():
            with open(self.report_file, 'r') as f:
                self.pipeline_report = json.load(f)
            print(f"Loaded pipeline report")

        total = len(self.valid_examples) + len(self.failed_examples)
        print(f"\nTotal examples: {total:,}")

        return True

    def _load_jsonl(self, filepath: Path) -> List[Dict]:
        """Load JSONL file efficiently"""
        examples = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    examples.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"WARNING: Failed to parse line {line_num}: {e}")
        return examples

    def compute_all_statistics(self):
        """Compute all statistics categories"""
        print("\n" + "="*80)
        print("COMPUTING STATISTICS")
        print("="*80)

        # Meta information
        self._compute_meta()

        # Distribution analysis
        print("\n[1/5] Computing distribution analysis...")
        self._compute_distributions()

        # Quality metrics
        print("[2/5] Computing quality metrics...")
        self._compute_quality_metrics()

        # Failure analysis
        print("[3/5] Computing failure analysis...")
        self._compute_failure_analysis()

        # Data quality insights
        print("[4/5] Computing data quality insights...")
        self._compute_quality_insights()

        # Recommendations
        print("[5/5] Generating recommendations...")
        self._generate_recommendations()

        print("\nStatistics computation complete!")

    def _compute_meta(self):
        """Compute metadata"""
        self.stats['meta'] = {
            'output_directory': str(self.output_dir),
            'timestamp': datetime.now().isoformat(),
            'total_valid_examples': len(self.valid_examples),
            'total_failed_examples': len(self.failed_examples),
            'total_examples': len(self.valid_examples) + len(self.failed_examples),
            'overall_success_rate': self._safe_percentage(
                len(self.valid_examples),
                len(self.valid_examples) + len(self.failed_examples)
            ),
            'pipeline_report': self.pipeline_report.get('pipeline_stats', {})
        }

    def _compute_distributions(self):
        """Compute distribution statistics"""
        dist = {}

        # Question strategies distribution
        dist['question_strategies'] = self._analyze_categorical(
            self.valid_examples, 'question_strategy',
            title="Question Strategies"
        )

        # Category distribution
        dist['categories'] = self._analyze_categories()

        # Question type distribution
        dist['question_types'] = self._analyze_question_types()

        # Complexity distribution
        dist['complexity'] = self._analyze_categorical(
            self.valid_examples, 'complexity',
            title="Complexity Levels"
        )

        # Response length distribution
        dist['response_length'] = self._analyze_continuous(
            [ex.get('word_count', 0) for ex in self.valid_examples],
            title="Response Word Count"
        )

        # Cycle count distribution
        dist['cycle_count'] = self._analyze_continuous(
            [ex.get('num_cycles', 0) for ex in self.valid_examples],
            title="Look-Think Cycles",
            is_integer=True
        )

        self.stats['distribution_analysis'] = dist

    def _compute_quality_metrics(self):
        """Compute quality metrics"""
        metrics = {}

        # Overall validation scores
        valid_scores = [
            ex.get('validation_metadata', {}).get('overall_score', 0)
            for ex in self.valid_examples
        ]

        metrics['validation_scores'] = self._analyze_continuous(
            valid_scores, title="Validation Scores"
        )

        # Per-dimension quality scores with full distribution
        # Check what dimensions are actually available in the data
        available_dimensions = set()
        for ex in self.valid_examples:
            validation = ex.get('validation_metadata', {})
            for key, value in validation.items():
                # Only include numeric fields (int or float)
                if key not in ['overall_score', 'pass', 'issues', 'validation_method'] and \
                   isinstance(value, (int, float)):
                    available_dimensions.add(key)

        metrics['score_dimensions'] = {}
        for dim in sorted(available_dimensions):
            scores = [
                ex.get('validation_metadata', {}).get(dim, 0)
                for ex in self.valid_examples
                if dim in ex.get('validation_metadata', {}) and \
                   isinstance(ex.get('validation_metadata', {}).get(dim), (int, float))
            ]
            if scores:
                metrics['score_dimensions'][dim] = self._analyze_continuous(
                    scores, title=dim.replace('_', ' ').title()
                )

        # Success rate by strategy
        metrics['success_by_strategy'] = self._success_by_category('question_strategy')

        # Success rate by complexity
        metrics['success_by_complexity'] = self._success_by_category('complexity')

        # Success rate by question type
        metrics['success_by_question_type'] = self._success_by_question_type()

        # Success rate by category
        metrics['success_by_category'] = self._success_by_categories()

        self.stats['quality_metrics'] = metrics

    def _compute_failure_analysis(self):
        """Compute failure analysis statistics"""
        if not self.failed_examples:
            self.stats['failure_analysis'] = {
                'no_failures': True,
                'message': 'No failed examples to analyze'
            }
            return

        analysis = {}

        # Failure type breakdown
        analysis['failure_types'] = self._analyze_failure_types()

        # Structure error patterns
        analysis['structure_errors'] = self._analyze_structure_errors()

        # Quality issue patterns
        analysis['quality_issues'] = self._analyze_quality_issues()

        # Score distribution for failures
        failed_scores = [ex.get('overall_score', 0) for ex in self.failed_examples]
        analysis['failure_score_distribution'] = self._analyze_continuous(
            failed_scores, title="Failure Scores"
        )

        # Near-miss analysis (failures with score > 6)
        near_misses = [ex for ex in self.failed_examples if ex.get('overall_score', 0) > 6]
        analysis['near_misses'] = {
            'count': len(near_misses),
            'percentage': self._safe_percentage(len(near_misses), len(self.failed_examples)),
            'avg_score': np.mean([ex.get('overall_score', 0) for ex in near_misses]) if near_misses else 0
        }

        self.stats['failure_analysis'] = analysis

    def _compute_quality_insights(self):
        """Compute data quality insights"""
        insights = {}

        # Category coverage
        insights['category_coverage'] = self._analyze_category_coverage()

        # Strategy balance
        insights['strategy_balance'] = self._analyze_strategy_balance()

        # Response quality trends
        insights['quality_trends'] = self._analyze_quality_trends()

        # Look-think-answer patterns
        insights['lta_patterns'] = self._analyze_lta_patterns()

        # Multiple choice quality
        insights['mc_quality'] = self._analyze_mc_quality()

        # Data imbalance detection
        insights['imbalance_warnings'] = self._detect_imbalances()

        self.stats['data_quality_insights'] = insights

    def _generate_recommendations(self):
        """Generate actionable recommendations"""
        recommendations = []

        total = len(self.valid_examples) + len(self.failed_examples)
        success_rate = self._safe_percentage(len(self.valid_examples), total)

        # Success rate recommendations
        if success_rate < 50:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'Overall Quality',
                'issue': f'Low success rate ({success_rate:.1f}%)',
                'recommendation': 'Review prompt templates and validation criteria. Consider adjusting thresholds or improving generation prompts.'
            })
        elif success_rate < 80:
            recommendations.append({
                'priority': 'MEDIUM',
                'category': 'Overall Quality',
                'issue': f'Moderate success rate ({success_rate:.1f}%)',
                'recommendation': 'Identify top failure patterns and address them systematically.'
            })

        # Strategy imbalance
        strategy_dist = self.stats['distribution_analysis']['question_strategies']
        if strategy_dist['entropy'] < 2.0 and len(strategy_dist['distribution']) > 3:
            recommendations.append({
                'priority': 'MEDIUM',
                'category': 'Data Balance',
                'issue': 'Imbalanced strategy distribution',
                'recommendation': 'Increase sampling diversity across question strategies for better model generalization.'
            })

        # Category coverage
        category_coverage = self.stats['data_quality_insights']['category_coverage']
        if category_coverage['unique_categories'] < 20:
            recommendations.append({
                'priority': 'LOW',
                'category': 'Coverage',
                'issue': f'Limited category diversity ({category_coverage["unique_categories"]} unique categories)',
                'recommendation': 'Expand data sources to cover more diverse categories.'
            })

        # Quality score trends
        avg_score = self.stats['quality_metrics']['validation_scores']['mean']
        if avg_score < 7.0:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'Quality',
                'issue': f'Low average validation score ({avg_score:.2f})',
                'recommendation': 'Review generation prompts and consider using higher quality models or better few-shot examples.'
            })

        # Multiple choice imbalance
        mc_stats = self.stats['distribution_analysis']['question_types']
        mc_pct = mc_stats['multiple_choice']['percentage']
        if mc_pct < 30 or mc_pct > 70:
            recommendations.append({
                'priority': 'LOW',
                'category': 'Question Type Balance',
                'issue': f'Question type imbalance (MC: {mc_pct:.1f}%)',
                'recommendation': 'Balance multiple choice and open-ended questions for diverse evaluation capabilities.'
            })

        # Near-miss failures
        if not self.stats['failure_analysis'].get('no_failures'):
            near_miss = self.stats['failure_analysis'].get('near_misses', {})
            if near_miss.get('count', 0) > len(self.failed_examples) * 0.3:
                recommendations.append({
                    'priority': 'MEDIUM',
                    'category': 'Validation Tuning',
                    'issue': f'Many near-miss failures ({near_miss["count"]} examples)',
                    'recommendation': 'Consider adjusting validation thresholds or implementing validation retries for borderline cases.'
                })

        self.stats['recommendations'] = recommendations

    # Helper methods for categorical analysis
    def _analyze_categorical(self, examples: List[Dict], field: str, title: str = "") -> Dict:
        """Analyze categorical field distribution"""
        values = [ex.get(field, 'unknown') for ex in examples]
        counter = Counter(values)
        total = len(values)

        distribution = [
            {
                'value': val,
                'count': count,
                'percentage': self._safe_percentage(count, total)
            }
            for val, count in counter.most_common()
        ]

        # Calculate entropy for diversity measure
        probabilities = [item['percentage'] / 100.0 for item in distribution]
        entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in probabilities)

        return {
            'total_examples': total,
            'unique_values': len(counter),
            'distribution': distribution,
            'entropy': round(entropy, 3),
            'top_value': distribution[0]['value'] if distribution else None,
            'top_value_percentage': distribution[0]['percentage'] if distribution else 0
        }

    def _analyze_categories(self) -> Dict:
        """Analyze Wikipedia category distribution"""
        all_categories = []
        for ex in self.valid_examples:
            cats = ex.get('categories', [])
            all_categories.extend(cats)

        counter = Counter(all_categories)
        total_examples = len(self.valid_examples)

        # Top categories
        top_categories = [
            {
                'category': cat,
                'count': count,
                'percentage': self._safe_percentage(count, total_examples)
            }
            for cat, count in counter.most_common(20)
        ]

        # Category co-occurrence
        cooccurrence = self._compute_category_cooccurrence()

        return {
            'unique_categories': len(counter),
            'total_category_mentions': len(all_categories),
            'avg_categories_per_example': round(len(all_categories) / total_examples, 2) if total_examples > 0 else 0,
            'top_categories': top_categories,
            'cooccurrence_patterns': cooccurrence
        }

    def _compute_category_cooccurrence(self, top_n: int = 5) -> List[Dict]:
        """Find frequently co-occurring category pairs"""
        from itertools import combinations

        pair_counter = Counter()
        for ex in self.valid_examples:
            cats = ex.get('categories', [])
            if len(cats) > 1:
                for pair in combinations(sorted(cats), 2):
                    pair_counter[pair] += 1

        return [
            {
                'category_1': cat1,
                'category_2': cat2,
                'count': count
            }
            for (cat1, cat2), count in pair_counter.most_common(top_n)
        ]

    def _analyze_question_types(self) -> Dict:
        """Analyze question type distribution"""
        mc_count = sum(1 for ex in self.valid_examples if ex.get('is_multiple_choice', False))
        open_count = len(self.valid_examples) - mc_count
        total = len(self.valid_examples)

        return {
            'multiple_choice': {
                'count': mc_count,
                'percentage': self._safe_percentage(mc_count, total)
            },
            'open_ended': {
                'count': open_count,
                'percentage': self._safe_percentage(open_count, total)
            }
        }

    def _analyze_continuous(self, values: List[float], title: str = "", is_integer: bool = False) -> Dict:
        """Analyze continuous/numerical distribution"""
        if not values:
            return {
                'count': 0,
                'mean': 0,
                'std': 0,
                'min': 0,
                'max': 0,
                'percentiles': {}
            }

        arr = np.array(values)

        percentiles = {
            'p25': float(np.percentile(arr, 25)),
            'p50': float(np.percentile(arr, 50)),
            'p75': float(np.percentile(arr, 75)),
            'p90': float(np.percentile(arr, 90)),
            'p95': float(np.percentile(arr, 95)),
            'p99': float(np.percentile(arr, 99)) if len(arr) > 100 else float(np.max(arr))
        }

        if is_integer:
            percentiles = {k: int(v) for k, v in percentiles.items()}

        result = {
            'count': len(values),
            'mean': round(float(np.mean(arr)), 2),
            'std': round(float(np.std(arr)), 2),
            'min': int(np.min(arr)) if is_integer else round(float(np.min(arr)), 2),
            'max': int(np.max(arr)) if is_integer else round(float(np.max(arr)), 2),
            'percentiles': percentiles
        }

        # Add histogram bins
        if is_integer:
            hist, bin_edges = np.histogram(arr, bins=range(int(np.min(arr)), int(np.max(arr)) + 2))
        else:
            hist, bin_edges = np.histogram(arr, bins=20)

        result['histogram'] = {
            'counts': hist.tolist(),
            'bin_edges': [round(float(x), 2) for x in bin_edges]
        }

        return result

    def _success_by_category(self, field: str) -> Dict:
        """Calculate success rate by categorical field"""
        valid_counter = Counter(ex.get(field, 'unknown') for ex in self.valid_examples)
        failed_counter = Counter(ex.get(field, 'unknown') for ex in self.failed_examples)

        all_values = set(valid_counter.keys()) | set(failed_counter.keys())

        results = []
        for value in sorted(all_values):
            valid_count = valid_counter.get(value, 0)
            failed_count = failed_counter.get(value, 0)
            total = valid_count + failed_count

            results.append({
                'value': value,
                'valid_count': valid_count,
                'failed_count': failed_count,
                'total': total,
                'success_rate': self._safe_percentage(valid_count, total)
            })

        # Sort by total count descending
        results.sort(key=lambda x: x['total'], reverse=True)

        return {
            'by_category': results,
            'highest_success': max(results, key=lambda x: x['success_rate'])['value'] if results else None,
            'lowest_success': min(results, key=lambda x: x['success_rate'])['value'] if results else None
        }

    def _success_by_question_type(self) -> Dict:
        """Calculate success rate by question type"""
        valid_mc = sum(1 for ex in self.valid_examples if ex.get('is_multiple_choice', False))
        failed_mc = sum(1 for ex in self.failed_examples if ex.get('is_multiple_choice', False))

        valid_open = len(self.valid_examples) - valid_mc
        failed_open = len(self.failed_examples) - failed_mc

        return {
            'multiple_choice': {
                'valid_count': valid_mc,
                'failed_count': failed_mc,
                'total': valid_mc + failed_mc,
                'success_rate': self._safe_percentage(valid_mc, valid_mc + failed_mc)
            },
            'open_ended': {
                'valid_count': valid_open,
                'failed_count': failed_open,
                'total': valid_open + failed_open,
                'success_rate': self._safe_percentage(valid_open, valid_open + failed_open)
            }
        }

    def _success_by_categories(self) -> Dict:
        """Calculate success rate by Wikipedia categories"""
        category_stats = defaultdict(lambda: {'valid': 0, 'failed': 0})

        for ex in self.valid_examples:
            for cat in ex.get('categories', []):
                category_stats[cat]['valid'] += 1

        for ex in self.failed_examples:
            for cat in ex.get('categories', []):
                category_stats[cat]['failed'] += 1

        results = []
        for cat, counts in category_stats.items():
            total = counts['valid'] + counts['failed']
            results.append({
                'category': cat,
                'valid_count': counts['valid'],
                'failed_count': counts['failed'],
                'total': total,
                'success_rate': self._safe_percentage(counts['valid'], total)
            })

        # Sort by total descending
        results.sort(key=lambda x: x['total'], reverse=True)

        return {
            'top_20_by_frequency': results[:20],
            'top_10_by_success': sorted(
                [r for r in results if r['total'] >= 3],
                key=lambda x: x['success_rate'],
                reverse=True
            )[:10],
            'bottom_10_by_success': sorted(
                [r for r in results if r['total'] >= 3],
                key=lambda x: x['success_rate']
            )[:10]
        }

    def _analyze_failure_types(self) -> Dict:
        """Analyze types of failures"""
        structure_failures = []
        quality_failures = []

        for ex in self.failed_examples:
            validation = ex.get('validation_result', {})
            method = validation.get('validation_method', 'unknown')

            if method == 'strict_tag_structure' or 'structure_error' in validation:
                structure_failures.append(ex)
            else:
                quality_failures.append(ex)

        return {
            'total_failures': len(self.failed_examples),
            'structure_failures': {
                'count': len(structure_failures),
                'percentage': self._safe_percentage(len(structure_failures), len(self.failed_examples))
            },
            'quality_failures': {
                'count': len(quality_failures),
                'percentage': self._safe_percentage(len(quality_failures), len(self.failed_examples))
            }
        }

    def _analyze_structure_errors(self) -> Dict:
        """Analyze structure error patterns"""
        structure_errors = []

        for ex in self.failed_examples:
            validation = ex.get('validation_result', {})
            if 'structure_error' in validation:
                structure_errors.append(validation['structure_error'])

        if not structure_errors:
            return {'no_structure_errors': True}

        counter = Counter(structure_errors)

        return {
            'total_count': len(structure_errors),
            'unique_error_types': len(counter),
            'distribution': [
                {
                    'error_type': error,
                    'count': count,
                    'percentage': self._safe_percentage(count, len(structure_errors))
                }
                for error, count in counter.most_common()
            ]
        }

    def _analyze_quality_issues(self) -> Dict:
        """Analyze quality issue patterns"""
        all_issues = []

        for ex in self.failed_examples:
            validation = ex.get('validation_result', {})
            issues = validation.get('issues', [])
            all_issues.extend(issues)

        if not all_issues:
            return {'no_quality_issues': True}

        counter = Counter(all_issues)

        return {
            'total_issues': len(all_issues),
            'unique_issue_types': len(counter),
            'top_20_issues': [
                {
                    'issue': issue,
                    'count': count,
                    'percentage': self._safe_percentage(count, len(all_issues))
                }
                for issue, count in counter.most_common(20)
            ]
        }

    def _analyze_category_coverage(self) -> Dict:
        """Analyze category coverage and distribution"""
        all_categories = []
        for ex in self.valid_examples:
            all_categories.extend(ex.get('categories', []))

        counter = Counter(all_categories)

        # Calculate Gini coefficient for inequality measure
        counts = sorted(counter.values(), reverse=True)
        n = len(counts)
        if n == 0:
            gini = 0
        else:
            cumsum = np.cumsum(counts)
            gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n if cumsum[-1] > 0 else 0

        return {
            'unique_categories': len(counter),
            'total_mentions': len(all_categories),
            'gini_coefficient': round(gini, 3),
            'coverage_quality': 'good' if gini < 0.6 else 'moderate' if gini < 0.8 else 'poor'
        }

    def _analyze_strategy_balance(self) -> Dict:
        """Analyze question strategy balance"""
        strategy_dist = self.stats['distribution_analysis']['question_strategies']

        # Check if distribution is balanced
        if strategy_dist['unique_values'] == 0:
            return {'balanced': True, 'note': 'No data'}

        expected_percentage = 100.0 / strategy_dist['unique_values']
        max_deviation = max(
            abs(item['percentage'] - expected_percentage)
            for item in strategy_dist['distribution']
        )

        is_balanced = max_deviation < 20.0  # Within 20% of expected

        return {
            'unique_strategies': strategy_dist['unique_values'],
            'expected_percentage': round(expected_percentage, 2),
            'max_deviation': round(max_deviation, 2),
            'is_balanced': is_balanced,
            'entropy': strategy_dist['entropy'],
            'balance_quality': 'good' if is_balanced else 'needs_improvement'
        }

    def _analyze_quality_trends(self) -> Dict:
        """Analyze response quality trends"""
        scores = [
            ex.get('validation_metadata', {}).get('overall_score', 0)
            for ex in self.valid_examples
        ]

        if not scores:
            return {}

        # Identify outliers (using IQR method)
        q1, q3 = np.percentile(scores, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers_low = [s for s in scores if s < lower_bound]
        outliers_high = [s for s in scores if s > upper_bound]

        return {
            'score_range': {
                'min': round(min(scores), 2),
                'max': round(max(scores), 2),
                'range': round(max(scores) - min(scores), 2)
            },
            'outliers': {
                'low_count': len(outliers_low),
                'high_count': len(outliers_high),
                'low_threshold': round(lower_bound, 2),
                'high_threshold': round(upper_bound, 2)
            },
            'quality_tiers': {
                'excellent': sum(1 for s in scores if s >= 9),
                'good': sum(1 for s in scores if 7 <= s < 9),
                'acceptable': sum(1 for s in scores if 5 <= s < 7),
                'poor': sum(1 for s in scores if s < 5)
            }
        }

    def _analyze_lta_patterns(self) -> Dict:
        """Analyze look-think-answer patterns"""
        cycles = [ex.get('num_cycles', 0) for ex in self.valid_examples]
        word_counts = [ex.get('word_count', 0) for ex in self.valid_examples]

        if not cycles:
            return {}

        # Correlation between cycles and word count
        correlation = np.corrcoef(cycles, word_counts)[0, 1] if len(cycles) > 1 else 0

        # Average words per cycle
        avg_words_per_cycle = {}
        for ex in self.valid_examples:
            num_cycles = ex.get('num_cycles', 0)
            if num_cycles > 0:
                words = ex.get('word_count', 0)
                if num_cycles not in avg_words_per_cycle:
                    avg_words_per_cycle[num_cycles] = []
                avg_words_per_cycle[num_cycles].append(words / num_cycles)

        words_per_cycle_stats = {
            cycles: round(np.mean(words), 2)
            for cycles, words in avg_words_per_cycle.items()
        }

        return {
            'cycle_distribution': self._analyze_continuous(cycles, is_integer=True),
            'cycle_word_correlation': round(correlation, 3),
            'avg_words_per_cycle_by_cycle_count': words_per_cycle_stats
        }

    def _analyze_mc_quality(self) -> Dict:
        """Analyze multiple choice question quality"""
        mc_examples = [ex for ex in self.valid_examples if ex.get('is_multiple_choice', False)]

        if not mc_examples:
            return {'no_mc_questions': True}

        # Option count distribution
        option_counts = [len(ex.get('options', [])) for ex in mc_examples]

        # Correct answer distribution
        correct_answers = [ex.get('correct_answer', '') for ex in mc_examples]
        answer_counter = Counter(correct_answers)

        # Quality scores for MC questions
        mc_scores = [
            ex.get('validation_metadata', {}).get('overall_score', 0)
            for ex in mc_examples
        ]

        return {
            'total_mc_questions': len(mc_examples),
            'option_count_stats': self._analyze_continuous(option_counts, is_integer=True),
            'correct_answer_distribution': [
                {
                    'answer': ans,
                    'count': count,
                    'percentage': self._safe_percentage(count, len(mc_examples))
                }
                for ans, count in answer_counter.most_common()
            ],
            'mc_quality_scores': self._analyze_continuous(mc_scores),
            'answer_balance_quality': 'good' if max(answer_counter.values()) / len(mc_examples) < 0.4 else 'imbalanced'
        }

    def _detect_imbalances(self) -> List[Dict]:
        """Detect data imbalance issues"""
        warnings = []

        # Strategy imbalance
        strategy_dist = self.stats['distribution_analysis']['question_strategies']
        if strategy_dist['top_value_percentage'] > 60:
            warnings.append({
                'type': 'strategy_imbalance',
                'severity': 'medium',
                'message': f'Strategy "{strategy_dist["top_value"]}" dominates with {strategy_dist["top_value_percentage"]:.1f}%',
                'recommendation': 'Increase diversity in question strategy sampling'
            })

        # Complexity imbalance
        complexity_dist = self.stats['distribution_analysis']['complexity']
        if complexity_dist['top_value_percentage'] > 70:
            warnings.append({
                'type': 'complexity_imbalance',
                'severity': 'low',
                'message': f'Complexity "{complexity_dist["top_value"]}" is over-represented at {complexity_dist["top_value_percentage"]:.1f}%',
                'recommendation': 'Balance complexity levels for robust model training'
            })

        # Question type imbalance
        qt_dist = self.stats['distribution_analysis']['question_types']
        mc_pct = qt_dist['multiple_choice']['percentage']
        if mc_pct > 80 or mc_pct < 20:
            warnings.append({
                'type': 'question_type_imbalance',
                'severity': 'medium',
                'message': f'Question type distribution is skewed (MC: {mc_pct:.1f}%)',
                'recommendation': 'Balance multiple choice and open-ended questions'
            })

        # Low sample size
        if len(self.valid_examples) < 100:
            warnings.append({
                'type': 'small_dataset',
                'severity': 'high',
                'message': f'Only {len(self.valid_examples)} valid examples - too small for robust training',
                'recommendation': 'Generate more data (target: 1000+ examples minimum)'
            })

        return warnings

    def _safe_percentage(self, numerator: float, denominator: float) -> float:
        """Safely calculate percentage"""
        if denominator == 0:
            return 0.0
        return round((numerator / denominator) * 100, 2)

    def print_summary(self):
        """Print comprehensive summary to console"""
        print("\n" + "="*80)
        print("DATA STATISTICS SUMMARY")
        print("="*80)

        # Meta info
        meta = self.stats['meta']
        print(f"\nDataset: {meta['output_directory']}")
        print(f"Generated: {meta['timestamp']}")
        print(f"\nTotal Examples: {meta['total_examples']:,}")
        print(f"  Valid:  {meta['total_valid_examples']:,} ({meta['overall_success_rate']:.1f}%)")
        print(f"  Failed: {meta['total_failed_examples']:,} ({100-meta['overall_success_rate']:.1f}%)")

        # Distribution summary
        print("\n" + "-"*80)
        print("DISTRIBUTION SUMMARY")
        print("-"*80)

        dist = self.stats['distribution_analysis']

        print(f"\nQuestion Strategies: {dist['question_strategies']['unique_values']} types")
        for item in dist['question_strategies']['distribution'][:5]:
            print(f"  {item['value']:30s}: {item['count']:4d} ({item['percentage']:5.1f}%)")

        print(f"\nComplexity Levels:")
        for item in dist['complexity']['distribution']:
            print(f"  {item['value']:30s}: {item['count']:4d} ({item['percentage']:5.1f}%)")

        print(f"\nQuestion Types:")
        qt = dist['question_types']
        print(f"  Multiple Choice: {qt['multiple_choice']['count']:4d} ({qt['multiple_choice']['percentage']:5.1f}%)")
        print(f"  Open-ended:      {qt['open_ended']['count']:4d} ({qt['open_ended']['percentage']:5.1f}%)")

        print(f"\nResponse Length (words):")
        rl = dist['response_length']
        print(f"  Mean: {rl['mean']:.1f}, Std: {rl['std']:.1f}")
        print(f"  Range: [{rl['min']}, {rl['max']}]")
        print(f"  Percentiles: P50={rl['percentiles']['p50']:.0f}, P95={rl['percentiles']['p95']:.0f}")

        print(f"\nLook-Think Cycles:")
        cycles = dist['cycle_count']
        print(f"  Mean: {cycles['mean']:.1f}, Range: [{cycles['min']}, {cycles['max']}]")
        print(f"  Percentiles: P50={cycles['percentiles']['p50']}, P95={cycles['percentiles']['p95']}")

        print(f"\nCategories:")
        cat = dist['categories']
        print(f"  Unique categories: {cat['unique_categories']:,}")
        print(f"  Avg per example: {cat['avg_categories_per_example']:.1f}")
        print(f"  Top 5 categories:")
        for item in cat['top_categories'][:5]:
            print(f"    {item['category']:60s}: {item['count']:3d}")

        # Quality metrics summary
        print("\n" + "-"*80)
        print("QUALITY METRICS")
        print("-"*80)

        qm = self.stats['quality_metrics']

        print(f"\nValidation Scores:")
        vs = qm['validation_scores']
        print(f"  Mean: {vs['mean']:.2f}, Std: {vs['std']:.2f}")
        print(f"  Range: [{vs['min']:.2f}, {vs['max']:.2f}]")
        print(f"  Percentiles: P25={vs['percentiles']['p25']:.2f}, P50={vs['percentiles']['p50']:.2f}, P75={vs['percentiles']['p75']:.2f}")

        if qm['score_dimensions']:
            print(f"\nScore Dimensions:")
            for dim, stats in qm['score_dimensions'].items():
                print(f"  {dim:25s}: mean={stats['mean']:.2f}, std={stats['std']:.2f}")

        print(f"\nSuccess Rate by Strategy (Top 5):")
        for item in qm['success_by_strategy']['by_category'][:5]:
            print(f"  {item['value']:30s}: {item['success_rate']:5.1f}% ({item['valid_count']}/{item['total']})")

        print(f"\nSuccess Rate by Complexity:")
        for item in qm['success_by_complexity']['by_category']:
            print(f"  {item['value']:30s}: {item['success_rate']:5.1f}% ({item['valid_count']}/{item['total']})")

        print(f"\nSuccess Rate by Question Type:")
        qt_success = qm['success_by_question_type']
        print(f"  Multiple Choice: {qt_success['multiple_choice']['success_rate']:5.1f}% ({qt_success['multiple_choice']['valid_count']}/{qt_success['multiple_choice']['total']})")
        print(f"  Open-ended:      {qt_success['open_ended']['success_rate']:5.1f}% ({qt_success['open_ended']['valid_count']}/{qt_success['open_ended']['total']})")

        # Failure analysis
        if not self.stats['failure_analysis'].get('no_failures'):
            print("\n" + "-"*80)
            print("FAILURE ANALYSIS")
            print("-"*80)

            fa = self.stats['failure_analysis']

            print(f"\nFailure Types:")
            print(f"  Structure failures: {fa['failure_types']['structure_failures']['count']:4d} ({fa['failure_types']['structure_failures']['percentage']:5.1f}%)")
            print(f"  Quality failures:   {fa['failure_types']['quality_failures']['count']:4d} ({fa['failure_types']['quality_failures']['percentage']:5.1f}%)")

            if not fa['structure_errors'].get('no_structure_errors'):
                print(f"\nTop Structure Errors:")
                for item in fa['structure_errors']['distribution'][:5]:
                    print(f"  {item['error_type']:40s}: {item['count']:3d} ({item['percentage']:5.1f}%)")

            if not fa['quality_issues'].get('no_quality_issues'):
                print(f"\nTop Quality Issues:")
                for item in fa['quality_issues']['top_20_issues'][:5]:
                    print(f"  {item['issue']:60s}: {item['count']:3d}")

            nm = fa['near_misses']
            print(f"\nNear-miss Failures (score > 6): {nm['count']} ({nm['percentage']:.1f}%)")

        # Insights
        print("\n" + "-"*80)
        print("DATA QUALITY INSIGHTS")
        print("-"*80)

        insights = self.stats['data_quality_insights']

        print(f"\nCategory Coverage:")
        cc = insights['category_coverage']
        print(f"  Unique categories: {cc['unique_categories']:,}")
        print(f"  Gini coefficient: {cc['gini_coefficient']:.3f} ({cc['coverage_quality']})")

        print(f"\nStrategy Balance:")
        sb = insights['strategy_balance']
        print(f"  Balance quality: {sb['balance_quality']}")
        print(f"  Entropy: {sb['entropy']:.3f}")
        print(f"  Max deviation from expected: {sb['max_deviation']:.1f}%")

        if 'quality_trends' in insights and insights['quality_trends']:
            print(f"\nQuality Trends:")
            qt = insights['quality_trends']
            print(f"  Score range: [{qt['score_range']['min']:.2f}, {qt['score_range']['max']:.2f}]")
            print(f"  Quality tiers:")
            for tier, count in qt['quality_tiers'].items():
                print(f"    {tier:12s}: {count:4d}")

        if not insights['mc_quality'].get('no_mc_questions'):
            print(f"\nMultiple Choice Quality:")
            mc = insights['mc_quality']
            print(f"  Total MC questions: {mc['total_mc_questions']}")
            print(f"  Answer balance: {mc['answer_balance_quality']}")
            print(f"  Correct answer distribution:")
            for item in mc['correct_answer_distribution']:
                print(f"    {item['answer']}: {item['count']:3d} ({item['percentage']:5.1f}%)")

        if insights['imbalance_warnings']:
            print(f"\nData Imbalance Warnings:")
            for warning in insights['imbalance_warnings']:
                print(f"  [{warning['severity'].upper()}] {warning['type']}")
                print(f"    {warning['message']}")

        # Recommendations
        if self.stats['recommendations']:
            print("\n" + "-"*80)
            print("RECOMMENDATIONS")
            print("-"*80)

            for i, rec in enumerate(self.stats['recommendations'], 1):
                print(f"\n{i}. [{rec['priority']}] {rec['category']}")
                print(f"   Issue: {rec['issue']}")
                print(f"   Recommendation: {rec['recommendation']}")

    def save_report(self, output_path: Optional[Path] = None):
        """Save statistics to JSON file"""
        if output_path is None:
            output_path = self.output_dir / 'data_statistics.json'

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*80}")
        print(f"Statistics saved to: {output_path}")
        print("="*80)

        return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive statistics pipeline for GPT-based data construction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 statistics_pipeline.py ./test_output_10
  python3 statistics_pipeline.py ./test_output_10 --verbose
  python3 statistics_pipeline.py ./test_output_10 --output ./custom_stats.json
        """
    )

    parser.add_argument(
        'output_dir',
        type=str,
        help='Directory containing generated_examples.jsonl and failed_examples.jsonl'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Custom output path for statistics JSON (default: <output_dir>/data_statistics.json)'
    )

    args = parser.parse_args()

    # Initialize statistics pipeline
    stats = DataStatistics(args.output_dir, verbose=args.verbose)

    # Load data
    if not stats.load_data():
        print("\nERROR: Failed to load data files")
        sys.exit(1)

    # Compute statistics
    stats.compute_all_statistics()

    # Print summary
    stats.print_summary()

    # Save report
    output_path = Path(args.output) if args.output else None
    stats.save_report(output_path)

    print("\nDone!")


if __name__ == '__main__':
    main()

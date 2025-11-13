"""
Utility scripts for GPT-based data construction pipeline
Includes cost estimation, quality analysis, and dataset inspection tools
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List
from collections import Counter, defaultdict
import numpy as np


class CostEstimator:
    """Estimate costs for GPT-based generation"""

    # Pricing (per 1M tokens, as of Nov 2025)
    PRICING = {
        'gpt-4-turbo-preview': {'input': 10.0, 'output': 30.0},
        'gpt-4': {'input': 30.0, 'output': 60.0},
        'gpt-3.5-turbo': {'input': 0.5, 'output': 1.5},
    }

    # Average token counts per operation
    AVG_TOKENS = {
        'question_generation': {'prompt': 1500, 'completion': 100},
        'response_generation': {'prompt': 2000, 'completion': 800},
        'validation': {'prompt': 3000, 'completion': 200}
    }

    def __init__(self,
                 generation_model: str = 'gpt-4-turbo-preview',
                 validation_model: str = 'gpt-3.5-turbo'):
        self.generation_model = generation_model
        self.validation_model = validation_model

    def estimate_per_example_cost(self, validation_pass_rate: float = 0.7) -> Dict:
        """
        Estimate cost per valid example

        Args:
            validation_pass_rate: Expected pass rate (0.0 to 1.0)

        Returns:
            Dictionary with cost breakdown
        """
        gen_pricing = self.PRICING[self.generation_model]
        val_pricing = self.PRICING[self.validation_model]

        # Question generation cost
        q_cost = (
            self.AVG_TOKENS['question_generation']['prompt'] * gen_pricing['input'] +
            self.AVG_TOKENS['question_generation']['completion'] * gen_pricing['output']
        ) / 1_000_000

        # Response generation cost
        r_cost = (
            self.AVG_TOKENS['response_generation']['prompt'] * gen_pricing['input'] +
            self.AVG_TOKENS['response_generation']['completion'] * gen_pricing['output']
        ) / 1_000_000

        # Validation cost
        v_cost = (
            self.AVG_TOKENS['validation']['prompt'] * val_pricing['input'] +
            self.AVG_TOKENS['validation']['completion'] * val_pricing['output']
        ) / 1_000_000

        # Per-attempt cost (question + response + validation)
        per_attempt = q_cost + r_cost + v_cost

        # Cost per valid example (accounting for failures)
        per_valid = per_attempt / validation_pass_rate

        return {
            'question_generation': q_cost,
            'response_generation': r_cost,
            'validation': v_cost,
            'per_attempt': per_attempt,
            'per_valid_example': per_valid,
            'validation_pass_rate': validation_pass_rate
        }

    def estimate_total_cost(self,
                           num_items: int,
                           examples_per_item: int = 2,
                           validation_pass_rate: float = 0.7) -> Dict:
        """
        Estimate total cost for dataset

        Returns:
            Dictionary with total cost estimates
        """
        per_example = self.estimate_per_example_cost(validation_pass_rate)

        total_attempts = num_items * examples_per_item
        total_valid = int(total_attempts * validation_pass_rate)
        total_cost = total_attempts * per_example['per_attempt']

        # Token estimates
        total_tokens = total_attempts * sum([
            self.AVG_TOKENS['question_generation']['prompt'],
            self.AVG_TOKENS['question_generation']['completion'],
            self.AVG_TOKENS['response_generation']['prompt'],
            self.AVG_TOKENS['response_generation']['completion'],
            self.AVG_TOKENS['validation']['prompt'],
            self.AVG_TOKENS['validation']['completion']
        ])

        # Time estimate (assuming 8 seconds per example with concurrency=10)
        processing_time_hours = (total_attempts * 8) / (3600 * 10)

        return {
            'num_source_items': num_items,
            'examples_per_item': examples_per_item,
            'total_attempts': total_attempts,
            'expected_valid_examples': total_valid,
            'total_cost_usd': total_cost,
            'cost_per_valid_example': per_example['per_valid_example'],
            'total_tokens': total_tokens,
            'estimated_processing_hours': processing_time_hours,
            'estimated_processing_days': processing_time_hours / 24,
            'models': {
                'generation': self.generation_model,
                'validation': self.validation_model
            }
        }

    def print_estimate(self, num_items: int, examples_per_item: int = 2):
        """Print formatted cost estimate"""
        estimate = self.estimate_total_cost(num_items, examples_per_item)

        print("=" * 80)
        print("GPT PIPELINE COST ESTIMATE")
        print("=" * 80)
        print(f"\nInput Parameters:")
        print(f"  Source items: {num_items:,}")
        print(f"  Examples per item: {examples_per_item}")
        print(f"  Generation model: {self.generation_model}")
        print(f"  Validation model: {self.validation_model}")
        print(f"\nExpected Output:")
        print(f"  Total generation attempts: {estimate['total_attempts']:,}")
        print(f"  Expected valid examples: {estimate['expected_valid_examples']:,}")
        print(f"  Validation pass rate: 70%")
        print(f"\nCost Breakdown:")
        print(f"  Cost per valid example: ${estimate['cost_per_valid_example']:.4f}")
        print(f"  Total estimated cost: ${estimate['total_cost_usd']:,.2f}")
        print(f"\nToken Usage:")
        print(f"  Total tokens (approx): {estimate['total_tokens']:,}")
        print(f"\nProcessing Time (with max_concurrent=10):")
        print(f"  Hours: {estimate['estimated_processing_hours']:.1f}")
        print(f"  Days: {estimate['estimated_processing_days']:.1f}")
        print("=" * 80)


class QualityAnalyzer:
    """Analyze quality of generated examples"""

    def __init__(self):
        self.examples = []

    def load_examples(self, filepath: str):
        """Load examples from JSONL file"""
        self.examples = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                self.examples.append(json.loads(line))
        print(f"Loaded {len(self.examples)} examples")

    def analyze(self) -> Dict:
        """Perform comprehensive quality analysis"""
        if not self.examples:
            print("No examples loaded. Call load_examples() first.")
            return {}

        analysis = {}

        # Basic stats
        analysis['total_examples'] = len(self.examples)
        analysis['total_unique_images'] = len(set(ex['image'] for ex in self.examples))

        # Validation scores
        scores = [ex['validation_metadata'].get('overall_score', 0) for ex in self.examples]
        analysis['validation_scores'] = {
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'min': float(np.min(scores)),
            'max': float(np.max(scores)),
            'median': float(np.median(scores)),
            'percentile_25': float(np.percentile(scores, 25)),
            'percentile_75': float(np.percentile(scores, 75))
        }

        # Dimension-wise scores
        for dimension in ['content_quality', 'coherence',
                         'diversity', 'educational_value']:
            dim_scores = [ex['validation_metadata'].get(dimension, 0) for ex in self.examples]
            analysis[f'{dimension}_scores'] = {
                'mean': float(np.mean(dim_scores)),
                'std': float(np.std(dim_scores))
            }

        # Strategy distribution
        strategy_counts = Counter(ex['question_strategy'] for ex in self.examples)
        analysis['strategy_distribution'] = dict(strategy_counts)

        # Complexity distribution
        complexity_counts = Counter(ex['complexity'] for ex in self.examples)
        analysis['complexity_distribution'] = dict(complexity_counts)

        # Word count stats
        word_counts = [ex['word_count'] for ex in self.examples]
        analysis['word_counts'] = {
            'mean': float(np.mean(word_counts)),
            'std': float(np.std(word_counts)),
            'min': int(np.min(word_counts)),
            'max': int(np.max(word_counts))
        }

        # Cycle stats
        num_cycles = [ex['num_cycles'] for ex in self.examples]
        analysis['cycle_distribution'] = dict(Counter(num_cycles))

        # Issue frequency
        all_issues = []
        for ex in self.examples:
            all_issues.extend(ex['validation_metadata'].get('issues', []))
        analysis['common_issues'] = dict(Counter(all_issues).most_common(10))

        # Strength frequency
        all_strengths = []
        for ex in self.examples:
            all_strengths.extend(ex['validation_metadata'].get('strengths', []))
        analysis['common_strengths'] = dict(Counter(all_strengths).most_common(10))

        return analysis

    def print_analysis(self, analysis: Dict = None):
        """Print formatted analysis"""
        if analysis is None:
            analysis = self.analyze()

        print("\n" + "=" * 80)
        print("QUALITY ANALYSIS REPORT")
        print("=" * 80)

        print(f"\n1. BASIC STATISTICS")
        print(f"  Total examples: {analysis['total_examples']:,}")
        print(f"  Unique images: {analysis['total_unique_images']:,}")
        print(f"  Avg examples per image: {analysis['total_examples']/analysis['total_unique_images']:.2f}")

        print(f"\n2. VALIDATION SCORES")
        vs = analysis['validation_scores']
        print(f"  Mean: {vs['mean']:.2f}")
        print(f"  Median: {vs['median']:.2f}")
        print(f"  Std: {vs['std']:.2f}")
        print(f"  Range: {vs['min']:.2f} - {vs['max']:.2f}")
        print(f"  25th percentile: {vs['percentile_25']:.2f}")
        print(f"  75th percentile: {vs['percentile_75']:.2f}")

        print(f"\n3. DIMENSION SCORES")
        for dim in ['content_quality', 'coherence',
                   'diversity', 'educational_value']:
            scores = analysis[f'{dim}_scores']
            print(f"  {dim}: {scores['mean']:.2f} ± {scores['std']:.2f}")

        print(f"\n4. STRATEGY DISTRIBUTION")
        for strategy, count in sorted(analysis['strategy_distribution'].items()):
            pct = 100 * count / analysis['total_examples']
            print(f"  {strategy}: {count:,} ({pct:.1f}%)")

        print(f"\n5. COMPLEXITY DISTRIBUTION")
        for complexity, count in sorted(analysis['complexity_distribution'].items()):
            pct = 100 * count / analysis['total_examples']
            print(f"  {complexity}: {count:,} ({pct:.1f}%)")

        print(f"\n6. RESPONSE LENGTH")
        wc = analysis['word_counts']
        print(f"  Mean: {wc['mean']:.0f} words")
        print(f"  Range: {wc['min']} - {wc['max']} words")
        print(f"  Std: {wc['std']:.0f}")

        print(f"\n7. CYCLE DISTRIBUTION")
        for cycles, count in sorted(analysis['cycle_distribution'].items()):
            pct = 100 * count / analysis['total_examples']
            print(f"  {cycles} cycles: {count:,} ({pct:.1f}%)")

        if analysis['common_issues']:
            print(f"\n8. MOST COMMON ISSUES")
            for issue, count in list(analysis['common_issues'].items())[:5]:
                print(f"  {issue}: {count}")

        if analysis['common_strengths']:
            print(f"\n9. MOST COMMON STRENGTHS")
            for strength, count in list(analysis['common_strengths'].items())[:5]:
                print(f"  {strength}: {count}")

        print("=" * 80 + "\n")

    def sample_examples(self, n: int = 5, strategy: str = None):
        """Print sample examples"""
        examples = self.examples

        if strategy:
            examples = [ex for ex in examples if ex['question_strategy'] == strategy]

        if len(examples) < n:
            n = len(examples)

        samples = np.random.choice(examples, n, replace=False)

        print("\n" + "=" * 80)
        print(f"SAMPLE EXAMPLES (n={n})")
        if strategy:
            print(f"Filtered by strategy: {strategy}")
        print("=" * 80)

        for i, ex in enumerate(samples, 1):
            print(f"\n--- Example {i} ---")
            print(f"ID: {ex['id']}")
            print(f"Image: {ex['image']}")
            print(f"Wiki Title: {ex['wiki_title']}")
            print(f"Strategy: {ex['question_strategy']}")
            print(f"Complexity: {ex['complexity']}")
            print(f"Validation Score: {ex['validation_metadata'].get('overall_score', 'N/A')}")
            print(f"\nQuestion:")
            print(f"  {ex['question']}")
            print(f"\nResponse Preview (first 500 chars):")
            print(f"  {ex['response'][:500]}...")
            print()

        print("=" * 80 + "\n")


class DatasetInspector:
    """Inspect and validate dataset files"""

    @staticmethod
    def inspect_checkpoint(checkpoint_path: str):
        """Print checkpoint information"""
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)

        print("\n" + "=" * 80)
        print("CHECKPOINT INSPECTION")
        print("=" * 80)
        print(f"Processed items: {checkpoint['processed_items']:,}")
        print(f"Total examples: {checkpoint['total_examples']:,}")
        print(f"Timestamp: {checkpoint['timestamp']}")

        if 'usage_stats' in checkpoint:
            stats = checkpoint['usage_stats']
            print(f"\nAPI Usage:")
            print(f"  Total requests: {stats['total_requests']:,}")
            print(f"  Successful: {stats['successful_requests']:,}")
            print(f"  Failed: {stats['failed_requests']:,}")
            print(f"  Total tokens: {stats['total_tokens']:,}")
            print(f"  Total cost: ${stats['total_cost_usd']:.2f}")
            if stats['total_examples']:
                print(f"  Cost per example: ${stats['total_cost_usd']/stats.get('valid_examples', 1):.4f}")

        print("=" * 80 + "\n")

    @staticmethod
    def inspect_examples_file(examples_path: str, max_lines: int = 10):
        """Inspect examples JSONL file"""
        examples = []
        with open(examples_path, 'r') as f:
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                examples.append(json.loads(line))

        total_lines = sum(1 for _ in open(examples_path))

        print("\n" + "=" * 80)
        print("EXAMPLES FILE INSPECTION")
        print("=" * 80)
        print(f"File: {examples_path}")
        print(f"Total examples: {total_lines:,}")
        print(f"Showing first {len(examples)} examples:")

        for i, ex in enumerate(examples, 1):
            print(f"\n--- Example {i} ---")
            print(f"ID: {ex['id']}")
            print(f"Strategy: {ex['question_strategy']}")
            print(f"Complexity: {ex['complexity']}")
            print(f"Word count: {ex['word_count']}")
            print(f"Cycles: {ex['num_cycles']}")
            print(f"Validation score: {ex['validation_metadata'].get('overall_score', 'N/A'):.2f}")
            print(f"Question: {ex['question'][:100]}...")

        print("\n" + "=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Utility tools for GPT pipeline'
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Cost estimation
    cost_parser = subparsers.add_parser('estimate-cost', help='Estimate pipeline costs')
    cost_parser.add_argument('--items', type=int, required=True,
                            help='Number of source items')
    cost_parser.add_argument('--examples-per-item', type=int, default=2,
                            help='Examples per item')
    cost_parser.add_argument('--generation-model', type=str,
                            default='gpt-4-turbo-preview',
                            help='Generation model')
    cost_parser.add_argument('--validation-model', type=str,
                            default='gpt-3.5-turbo',
                            help='Validation model')

    # Quality analysis
    quality_parser = subparsers.add_parser('analyze-quality',
                                          help='Analyze generated examples quality')
    quality_parser.add_argument('--input', type=str, required=True,
                               help='Path to generated_examples.jsonl')
    quality_parser.add_argument('--sample', type=int, default=0,
                               help='Number of examples to sample and print')

    # Checkpoint inspection
    checkpoint_parser = subparsers.add_parser('inspect-checkpoint',
                                             help='Inspect checkpoint file')
    checkpoint_parser.add_argument('--checkpoint', type=str, required=True,
                                  help='Path to checkpoint.json')

    # Examples inspection
    inspect_parser = subparsers.add_parser('inspect-examples',
                                          help='Inspect examples file')
    inspect_parser.add_argument('--input', type=str, required=True,
                               help='Path to generated_examples.jsonl')
    inspect_parser.add_argument('--max-lines', type=int, default=10,
                               help='Max examples to show')

    args = parser.parse_args()

    if args.command == 'estimate-cost':
        estimator = CostEstimator(
            generation_model=args.generation_model,
            validation_model=args.validation_model
        )
        estimator.print_estimate(args.items, args.examples_per_item)

    elif args.command == 'analyze-quality':
        analyzer = QualityAnalyzer()
        analyzer.load_examples(args.input)
        analyzer.print_analysis()

        if args.sample > 0:
            analyzer.sample_examples(args.sample)

    elif args.command == 'inspect-checkpoint':
        DatasetInspector.inspect_checkpoint(args.checkpoint)

    elif args.command == 'inspect-examples':
        DatasetInspector.inspect_examples_file(args.input, args.max_lines)

    else:
        parser.print_help()


if __name__ == '__main__':
    main()

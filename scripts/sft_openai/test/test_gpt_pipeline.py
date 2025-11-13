#!/usr/bin/env python3
"""
Quick test script for GPT pipeline
Tests the pipeline on a small sample and validates all components work correctly
"""

import os
import sys
import json
import argparse
from pathlib import Path
import subprocess
import time


def check_dependencies():
    """Check if required packages are installed"""
    print("Checking dependencies...")
    required = ['openai', 'aiohttp', 'backoff', 'tqdm', 'numpy']
    missing = []

    for package in required:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            missing.append(package)
            print(f"  ✗ {package} - MISSING")

    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False
    return True


def check_api_key():
    """Check if OpenAI API key is set"""
    print("\nChecking OpenAI API key...")
    api_key = os.environ.get('OPENAI_API_KEY')

    if not api_key:
        print("  ✗ OPENAI_API_KEY not set")
        print("  Set it with: export OPENAI_API_KEY='sk-your-key-here'")
        return None

    if not api_key.startswith('sk-'):
        print("  ✗ API key format looks invalid (should start with 'sk-')")
        return None

    print(f"  ✓ API key found (starts with {api_key[:20]}...)")
    return api_key


def check_source_data(source_path):
    """Check if source data file exists and is valid"""
    print(f"\nChecking source data: {source_path}")

    if not Path(source_path).exists():
        print(f"  ✗ File not found: {source_path}")
        return False

    try:
        with open(source_path, 'r') as f:
            data = json.load(f)
        print(f"  ✓ Valid JSON with {len(data):,} items")

        # Check first item structure
        if data:
            item = data[0]
            required_fields = ['wiki_title', 'pred_response', 'categories', 'image']
            missing = [f for f in required_fields if f not in item]
            if missing:
                print(f"  ✗ Missing required fields: {missing}")
                return False
            print(f"  ✓ Data structure looks good")
        return True
    except json.JSONDecodeError:
        print(f"  ✗ Invalid JSON format")
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def run_test(api_key, source_path, output_dir, num_items=5):
    """Run a test with small sample"""
    print(f"\n{'='*80}")
    print(f"Running test with {num_items} items...")
    print(f"{'='*80}\n")

    cmd = [
        'python', 'data_construction_gpt_pipeline.py',
        '--api-key', api_key,
        '--source', source_path,
        '--output', output_dir,
        '--sample', str(num_items),
        '--examples-per-item', '2',
        '--max-concurrent', '3'
    ]

    print(f"Command: {' '.join(cmd)}\n")

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=False,
            text=True,
            check=True
        )

        elapsed = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"Test completed successfully in {elapsed:.1f} seconds!")
        print(f"{'='*80}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"\n{'='*80}")
        print(f"Test failed with error code {e.returncode}")
        print(f"{'='*80}")
        return False
    except KeyboardInterrupt:
        print(f"\n\nTest interrupted by user")
        return False


def analyze_results(output_dir):
    """Analyze test results"""
    print(f"\n{'='*80}")
    print("Analyzing results...")
    print(f"{'='*80}\n")

    output_path = Path(output_dir)

    # Check checkpoint
    checkpoint_file = output_path / 'checkpoint.json'
    if checkpoint_file.exists():
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
        print("Checkpoint Info:")
        print(f"  Items processed: {checkpoint['processed_items']}")
        print(f"  Examples generated: {checkpoint['total_examples']}")

        if 'usage_stats' in checkpoint:
            stats = checkpoint['usage_stats']
            print(f"\nAPI Usage:")
            print(f"  Total requests: {stats['total_requests']}")
            print(f"  Successful: {stats['successful_requests']}")
            print(f"  Failed: {stats['failed_requests']}")
            print(f"  Total tokens: {stats['total_tokens']:,}")
            print(f"  Total cost: ${stats['total_cost_usd']:.2f}")

    # Check examples
    examples_file = output_path / 'generated_examples.jsonl'
    if examples_file.exists():
        examples = []
        with open(examples_file, 'r') as f:
            for line in f:
                examples.append(json.loads(line))

        print(f"\nGenerated Examples: {len(examples)}")

        if examples:
            scores = [ex['validation_metadata'].get('overall_score', 0) for ex in examples]
            strategies = [ex['question_strategy'] for ex in examples]

            print(f"  Avg validation score: {sum(scores)/len(scores):.2f}/10")
            print(f"  Score range: {min(scores):.1f} - {max(scores):.1f}")
            print(f"\n  Strategy distribution:")
            for strategy, count in sorted(dict(zip(*zip(*[(s, strategies.count(s)) for s in set(strategies)]))).items()):
                print(f"    {strategy}: {count}")

            # Show one example
            print(f"\n{'='*80}")
            print("Sample Generated Example:")
            print(f"{'='*80}")
            ex = examples[0]
            print(f"\nWiki Title: {ex['wiki_title']}")
            print(f"Strategy: {ex['question_strategy']}")
            print(f"Validation Score: {ex['validation_metadata'].get('overall_score', 'N/A'):.1f}/10")
            print(f"\nQuestion:")
            print(f"  {ex['question']}")
            print(f"\nResponse Preview (first 400 chars):")
            print(f"  {ex['response'][:400]}...")
            print(f"\n{'='*80}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Test GPT pipeline with small sample'
    )
    parser.add_argument('--source', type=str,
                       default='/data_ali/shunian/data/iceberg/scripts/data_clean.json',
                       help='Source data path')
    parser.add_argument('--output', type=str,
                       default='./test_gpt_output',
                       help='Output directory')
    parser.add_argument('--num-items', type=int, default=5,
                       help='Number of items to test (default: 5)')
    parser.add_argument('--skip-checks', action='store_true',
                       help='Skip dependency and data checks')

    args = parser.parse_args()

    print("="*80)
    print("GPT PIPELINE TEST SCRIPT")
    print("="*80)

    # Run checks
    if not args.skip_checks:
        if not check_dependencies():
            return 1

        api_key = check_api_key()
        if not api_key:
            return 1

        if not check_source_data(args.source):
            return 1
    else:
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            print("Error: OPENAI_API_KEY not set")
            return 1

    # Estimate cost
    cost_per_example = 0.091
    estimated_cost = args.num_items * 2 * cost_per_example
    estimated_time = args.num_items * 0.5  # ~30 seconds per item

    print(f"\nTest Configuration:")
    print(f"  Items: {args.num_items}")
    print(f"  Expected examples: ~{int(args.num_items * 2 * 0.7)} (70% pass rate)")
    print(f"  Estimated cost: ${estimated_cost:.2f}")
    print(f"  Estimated time: {estimated_time:.1f} minutes")

    proceed = input("\nProceed with test? (yes/no): ")
    if proceed.lower() != 'yes':
        print("Test cancelled.")
        return 0

    # Run test
    success = run_test(api_key, args.source, args.output, args.num_items)

    if success:
        # Analyze results
        analyze_results(args.output)
        print("\n✓ Test completed successfully!")
        print(f"  Output directory: {args.output}")
        print(f"\nNext steps:")
        print(f"  1. Review the generated examples")
        print(f"  2. Run quality analysis: python gpt_pipeline_utils.py analyze-quality --input {args.output}/generated_examples.jsonl")
        print(f"  3. If satisfied, scale up with run_gpt_pipeline_examples.sh")
        return 0
    else:
        print("\n✗ Test failed")
        print(f"  Check gpt_pipeline.log for details")
        return 1


if __name__ == '__main__':
    sys.exit(main())

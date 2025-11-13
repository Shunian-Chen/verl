"""
OpenAI GPT-Based Data Construction Pipeline for Vision-Language SFT Training
Implements intelligent question generation and quality validation using GPT models

Author: Data ML Architect
Date: 2025-11-03
Version: 2.0 (GPT-powered)
"""

import json
import os
import time
import hashlib
import asyncio
import aiohttp
import random
import base64
import mimetypes
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import logging
from utils import QuestionStrategy, APIUsageStats, GeneratedExample
from prompts import PromptLibrary
import numpy as np
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm
from dotenv import load_dotenv
try:
    from generator import GPTDataGenerator
except Exception:
    try:
        from generator import GPTDataGenerator
    except Exception as _e:
        raise ImportError("GPTDataGenerator not found; ensure generator module is importable") from _e

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
GENERATION_MODEL = os.getenv("GENERATION_MODEL")
VALIDATION_MODEL = os.getenv("VALIDATION_MODEL")
# Ensure MAX_TOKENS is an int
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2000"))



# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gpt_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)



class DataLoader:
    """
    Handles loading and preprocessing of source data
    """

    def __init__(self, min_content_length: int = 100):
        self.min_content_length = min_content_length
        self.stats = defaultdict(int)

    def load_data(
        self,
        filepath: str,
        limit: Optional[int] = None,
        sampling_strategy: str = 'random',
        seed: int = 42
    ) -> List[Dict]:
        """
        Load and preprocess data from JSON file

        Args:
            filepath: Path to JSON file
            limit: Maximum number of items to load
            sampling_strategy: Strategy for sampling when limit is set
                - 'random': Random sampling (default)
                - 'sequential': Take first N items
                - 'balanced': Balance across categories
                - 'cluster': Cluster-based sampling for maximum category coverage
            seed: Random seed for reproducibility

        Returns:
            List of data items
        """
        logger.info(f"Loading data from {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        logger.info(f"Loaded {len(data)} items")

        if limit and limit < len(data):
            if sampling_strategy == 'sequential':
                data = data[:limit]
                logger.info(f"Sequential sampling: limited to first {limit} items")

            elif sampling_strategy == 'random':
                random.seed(seed)
                data = random.sample(data, limit)
                logger.info(f"Random sampling: selected {limit} items")

            elif sampling_strategy in ['balanced', 'cluster']:
                # Import sampling module
                try:
                    from category_sampling import CategorySampler
                    sampler = CategorySampler(data, seed=seed)

                    if sampling_strategy == 'balanced':
                        data = sampler.balanced_sample(limit)
                        logger.info(f"Balanced sampling: selected {len(data)} items")
                    else:  # cluster
                        data = sampler.cluster_sample(limit)
                        logger.info(f"Cluster sampling: selected {len(data)} items with maximum category coverage")

                except ImportError:
                    logger.warning("category_sampling module not found, falling back to random sampling")
                    random.seed(seed)
                    data = random.sample(data, limit)

            else:
                logger.warning(f"Unknown sampling strategy '{sampling_strategy}', using random")
                random.seed(seed)
                data = random.sample(data, limit)

        return data

    def parse_pred_response(self, pred_response: str) -> str:
        """Parse pred_response string to extract content"""
        try:
            # Try to parse as dict string
            if isinstance(pred_response, str):
                # Use ast.literal_eval for safer parsing
                import ast
                parsed = ast.literal_eval(pred_response)
                if isinstance(parsed, dict):
                    return parsed.get('image_description_and_background', '')
            return str(pred_response)
        except:
            return str(pred_response)

    def preprocess_item(self, item: Dict) -> Optional[Dict]:
        """
        Preprocess a single item

        Returns:
            Processed item or None if invalid
        """
        # Check required fields
        if not item.get('image') or not item.get('pred_response'):
            self.stats['missing_required_fields'] += 1
            return None

        # Parse content
        content = self.parse_pred_response(item['pred_response'])

        # Check content length
        if len(content) < self.min_content_length:
            self.stats['content_too_short'] += 1
            return None

        # Check categories
        if not item.get('categories') or len(item['categories']) == 0:
            self.stats['no_categories'] += 1
            return None

        processed = {
            'wiki_title': item.get('wiki_title', ''),
            'title': item.get('title', ''),
            'image': item['image'],
            'categories': item['categories'],
            'first_category': item.get('first_category', ''),
            'content': content
        }

        self.stats['valid'] += 1
        return processed

    def preprocess_batch(self, items: List[Dict]) -> List[Dict]:
        """Preprocess batch of items"""
        processed = []
        for item in tqdm(items, desc="Preprocessing"):
            result = self.preprocess_item(item)
            if result:
                processed.append(result)

        logger.info(f"Preprocessed {len(processed)} valid items from {len(items)} total")
        logger.info(f"Preprocessing stats: {dict(self.stats)}")
        return processed


class GPTDataConstructionPipeline:
    """
    Main pipeline orchestrator using GPT for generation and validation
    """

    def __init__(self,
                 api_key: str,
                 base_url: str,
                 source_path: str,
                 output_dir: str,
                 generation_model: str = "gpt-5",
                 validation_model: str = "gpt-5",
                 examples_per_item: int = 2,
                 max_concurrent_requests: int = 10,
                 batch_size: int = 100,
                 checkpoint_interval: int = 500,
                 sample_size: Optional[int] = None,
                 sampling_strategy: str = 'cluster',
                 seed: int = 42,
                 multiple_choice_ratio: float = 0.2):

        self.source_path = source_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.examples_per_item = examples_per_item
        self.batch_size = batch_size
        # 规范化并保护 checkpoint 间隔，避免非法值导致不可预期行为
        try:
            self.checkpoint_interval = max(1, int(checkpoint_interval))
        except Exception:
            self.checkpoint_interval = 500
            logger.warning("Invalid checkpoint_interval provided; falling back to 500")
        self.sample_size = sample_size
        self.sampling_strategy = sampling_strategy
        self.seed = seed
        self.multiple_choice_ratio = multiple_choice_ratio

        # Initialize components
        self.loader = DataLoader()
        self.generator = GPTDataGenerator(
            api_key=api_key,
            base_url=base_url,
            generation_model=generation_model,
            validation_model=validation_model,
            max_concurrent_requests=max_concurrent_requests
        )

        # Strategy distribution
        self.strategies = list(QuestionStrategy)

        # Checkpoint file
        self.checkpoint_file = self.output_dir / 'checkpoint.json'
        self.examples_file = self.output_dir / 'generated_examples.jsonl'
        self.failed_examples_file = self.output_dir / 'failed_examples.jsonl'

        # 断点续传：已处理条目ID索引文件与内存集合
        self.progress_file = self.output_dir / 'processed_item_ids.txt'
        self.processed_ids = set()
        self._load_processed_ids()

        # Stats
        self.stats = defaultdict(int)

        # 从已存在的生成文件中预热正确选项分布（便于跨多次运行保持全局均衡）
        try:
            self.generator.seed_mc_counts_from_file(str(self.examples_file))
        except Exception as e:
            logger.warning(f"Failed to seed MC counts from existing file: {e}")

        logger.info("=" * 80)
        logger.info("GPT Data Construction Pipeline Initialized")
        logger.info("=" * 80)
        logger.info(f"Source: {source_path}")
        logger.info(f"Output: {output_dir}")
        logger.info(f"Generation model: {generation_model}")
        logger.info(f"Validation model: {validation_model}")
        logger.info(f"Examples per item: {examples_per_item}")
        logger.info(f"Max concurrent requests: {max_concurrent_requests}")
        logger.info(f"Multiple choice ratio: {multiple_choice_ratio:.0%}")


    def load_checkpoint(self) -> Dict:
        """Load checkpoint if exists"""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            logger.info(f"Loaded checkpoint: processed {checkpoint['processed_items']} items")
            # Restore multiple-choice balancing counts if available
            if 'mc_correct_counts' in checkpoint and isinstance(checkpoint['mc_correct_counts'], dict):
                try:
                    self.generator.set_mc_counts(checkpoint['mc_correct_counts'])
                except Exception:
                    logger.warning("Failed to restore mc_correct_counts from checkpoint; using zeros")
            return checkpoint
        return {'processed_items': 0, 'total_examples': 0}

    def compute_item_id(self, item: Dict) -> str:
        """为每条输入构造稳定ID用于断点续传与去重。"""
        try:
            key = {
                'image': item.get('image', ''),
                'wiki_title': item.get('wiki_title', ''),
                'title': item.get('title', ''),
                'first_category': item.get('first_category', '')
            }
            payload = json.dumps(key, ensure_ascii=False, sort_keys=True)
        except Exception:
            payload = f"{item.get('image','')}|{item.get('title','')}|{item.get('wiki_title','')}"
        return hashlib.md5(payload.encode('utf-8')).hexdigest()

    def _load_processed_ids(self):
        """加载已处理条目的ID集合。"""
        try:
            if self.progress_file.exists():
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    self.processed_ids = set(line.strip() for line in f if line.strip())
                logger.info(f"Loaded {len(self.processed_ids)} processed item IDs from {self.progress_file}")
        except Exception as e:
            logger.warning(f"Failed to load processed IDs: {e}")

    def _append_processed_id(self, item_id: str):
        """将已完成条目的ID追加持久化，确保中途中断可恢复。"""
        try:
            if item_id in self.processed_ids:
                return
            with open(self.progress_file, 'a', encoding='utf-8') as f:
                f.write(item_id + '\n')
            self.processed_ids.add(item_id)
        except Exception as e:
            logger.error(f"Failed to write processed ID: {e}")

    def save_checkpoint(self, processed_items: int, total_examples: int):
        """Save checkpoint"""
        checkpoint = {
            'processed_items': processed_items,
            'total_examples': total_examples,
            'timestamp': datetime.now().isoformat(),
            'usage_stats': self.generator.get_usage_stats(),
            'mc_correct_counts': self.generator.get_mc_counts()
        }
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        logger.info(f"Checkpoint saved: {processed_items} items, {total_examples} examples")

    def append_examples(self, examples: List[GeneratedExample]):
        """Append examples to output file"""
        with open(self.examples_file, 'a', encoding='utf-8') as f:
            for example in examples:
                f.write(json.dumps(example.to_dict(), ensure_ascii=False) + '\n')

    def append_failed_examples(self, examples: List[Dict]):
        """Append failed examples to separate file for analysis"""
        with open(self.failed_examples_file, 'a', encoding='utf-8') as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')

    async def process_item(self, item: Dict) -> Tuple[List[GeneratedExample], List[Dict]]:
        """
        Process a single item to generate multiple examples

        Returns:
            Tuple of (valid_examples, failed_examples)
        """
        valid_examples = []
        failed_examples = []

        # Select strategies for this item
        selected_strategies = np.random.choice(
            self.strategies,
            size=min(self.examples_per_item, len(self.strategies)),
            replace=False
        )

        # Generate examples for each strategy
        tasks = []
        for strategy in selected_strategies:
            task = self.generator.generate_full_example(
                item,
                strategy,
                multiple_choice_ratio=self.multiple_choice_ratio,
                balance_mc=True
            )
            tasks.append(task)

        # Run all generations concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Generation failed: {str(result)}")
                self.stats['generation_errors'] += 1
            elif isinstance(result, tuple) and len(result) == 3:
                example, failed_data, is_valid = result
                if is_valid and example is not None:
                    valid_examples.append(example)
                    self.stats['valid_examples'] += 1
                    # Track question types
                    if example.is_multiple_choice:
                        self.stats['multiple_choice_questions'] += 1
                    else:
                        self.stats['open_ended_questions'] += 1
                else:
                    # Save failed example
                    if failed_data:
                        failed_examples.append(failed_data)
                        # Track tag structure failures separately
                        validation = failed_data.get('validation_result', {})
                        if validation.get('validation_method') == 'strict_tag_structure':
                            self.stats['tag_structure_failures'] += 1
                        else:
                            self.stats['quality_validation_failures'] += 1
                    self.stats['failed_validation'] += 1
            else:
                logger.warning(f"Unexpected result format: {type(result)}")

        return valid_examples, failed_examples

    async def _process_item_with_id(self, item: Dict, item_id: str) -> Tuple[str, List[GeneratedExample], List[Dict]]:
        """包装单条处理以携带item_id，便于完成后即时记录进度。"""
        valid_examples, failed_examples = await self.process_item(item)
        return item_id, valid_examples, failed_examples

    async def process_batch(self, items: List[Dict], start_idx: int,
                            next_checkpoint: int,
                            start_total_examples: int,
                            last_saved_items: Optional[int]) -> Tuple[List[GeneratedExample], List[Dict], int, Optional[int], int]:
        """Process a batch of items，支持带item_id的条目并在完成后持久化进度，并实时按阈值保存checkpoint。

        Returns:
            (valid_results, failed_results, next_checkpoint, last_saved_items, persisted_valid_count)
        """
        # 兼容传入 List[Dict] 或 List[Tuple[item_id, Dict]] 两种形式
        if items and isinstance(items[0], tuple):
            items_with_ids = items  # type: ignore
        else:
            items_with_ids = [(self.compute_item_id(item), item) for item in items]

        tasks = [self._process_item_with_id(item, item_id) for (item_id, item) in items_with_ids]

        valid_results = []
        failed_results = []
        # 累计本批次已经写盘的有效样本数量
        persisted_valid_count = 0
        for coro in tqdm_asyncio.as_completed(tasks, desc=f"Batch {start_idx}-{start_idx+len(items_with_ids)}"):
            item_id, valid_examples, failed_examples = await coro
            valid_results.extend(valid_examples)
            failed_results.extend(failed_examples)
            # 处理完单条后立即记录ID，确保可中断恢复
            self._append_processed_id(item_id)
            # 实时将该条目的结果写盘，避免中途断电导致结果丢失
            if valid_examples:
                self.append_examples(valid_examples)
                persisted_valid_count += len(valid_examples)
            if failed_examples:
                self.append_failed_examples(failed_examples)
            # 达到阈值则实时保存 checkpoint（使用批内已生成数量估算总examples）
            current_processed = len(self.processed_ids)
            while current_processed >= next_checkpoint:
                estimated_total_examples = start_total_examples + persisted_valid_count
                self.save_checkpoint(next_checkpoint, estimated_total_examples)
                last_saved_items = next_checkpoint
                next_checkpoint += self.checkpoint_interval

        return valid_results, failed_results, next_checkpoint, last_saved_items, persisted_valid_count

    def split_train_test(self, test_ratio: float = 0.1, seed: Optional[int] = None):
        """根据已生成的 examples 文件，随机划分训练/测试集并写入新文件。

        Args:
            test_ratio: 测试集占比 (0-1)
            seed: 随机种子，复现实验
        """
        try:
            if not self.examples_file.exists():
                logger.warning(f"Examples file not found: {self.examples_file}")
                return

            with open(self.examples_file, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]

            if not lines:
                logger.warning("No examples to split.")
                return

            rnd = random.Random(seed if seed is not None else self.seed)
            rnd.shuffle(lines)

            n_total = len(lines)
            n_test = int(n_total * max(0.0, min(1.0, test_ratio)))
            n_test = max(1, n_test) if n_total > 1 and test_ratio > 0 else n_test

            test_lines = lines[:n_test]
            train_lines = lines[n_test:]

            train_path = self.output_dir / 'train_examples.jsonl'
            test_path = self.output_dir / 'test_examples.jsonl'

            with open(train_path, 'w', encoding='utf-8') as ft:
                for l in train_lines:
                    ft.write(l + '\n')
            with open(test_path, 'w', encoding='utf-8') as fv:
                for l in test_lines:
                    fv.write(l + '\n')

            logger.info(f"Train/Test split done. Train: {len(train_lines)}, Test: {len(test_lines)}")
        except Exception as e:
            logger.error(f"Failed to split train/test: {str(e)}")

    async def run_async(self):
        """Run the async pipeline"""
        logger.info("Starting GPT-based data generation")

        # Load data with category-based sampling
        raw_data = self.loader.load_data(
            self.source_path,
            limit=self.sample_size,
            sampling_strategy=self.sampling_strategy,
            seed=self.seed
        )
        processed_data = self.loader.preprocess_batch(raw_data)

        # Load checkpoint
        checkpoint = self.load_checkpoint()
        start_idx = checkpoint['processed_items']

        # 基于已处理ID过滤，避免重复处理（优先于按索引切片）
        items_with_ids = []
        skipped = 0
        for item in processed_data:
            iid = self.compute_item_id(item)
            if iid in self.processed_ids:
                skipped += 1
                continue
            items_with_ids.append((iid, item))
        if skipped:
            logger.info(f"Skip {skipped} already-processed items via ID index; remaining {len(items_with_ids)} to process")

        # 当设置了 sample_size 时，基于当前输出目录下已处理的条目数进行全局上限裁剪
        # 这样可保证多次运行同一输出目录时，不会超过 --sample 设定的目标数量
        if self.sample_size is not None:
            already_processed = len(self.processed_ids)
            remaining_quota = max(0, self.sample_size - already_processed)
            if remaining_quota <= 0:
                logger.info(
                    f"sample_size cap reached (sample={self.sample_size}, already={already_processed}); nothing to process."
                )
                # 确保 checkpoint 与统计信息一致后直接返回
                self.save_checkpoint(already_processed, checkpoint['total_examples'])
                self.stats['total_items_processed'] = already_processed
                self.stats['total_examples_generated'] = checkpoint['total_examples']
                return self.stats
            if len(items_with_ids) > remaining_quota:
                items_with_ids = items_with_ids[:remaining_quota]
                logger.info(f"Truncated to remaining quota: {remaining_quota} items")

        # Process in batches
        total_examples = checkpoint['total_examples']
        # 预计算下一次需要保存 checkpoint 的目标阈值，确保跨批次也能按间隔落盘
        current_count = len(self.processed_ids)
        next_checkpoint = ((current_count // self.checkpoint_interval) + 1) * self.checkpoint_interval
        # 记录最近一次按阈值保存时使用的 processed_items，避免最终重复保存相同计数
        last_saved_items: Optional[int] = None

        for i in range(0, len(items_with_ids), self.batch_size):
            batch = items_with_ids[i:i+self.batch_size]
            current_idx = start_idx + i

            logger.info(f"\nProcessing batch: items {current_idx} to {current_idx + len(batch)}")

            # Process batch（内部逐条写盘，并在阈值触发时保存 checkpoint）
            valid_examples, failed_examples, next_checkpoint, last_saved_items, persisted_valid_count = await self.process_batch(
                batch, current_idx, next_checkpoint, total_examples, last_saved_items
            )

            # 本批次有效样本已在条目完成时写盘，这里只更新累计计数与日志
            if persisted_valid_count:
                total_examples += persisted_valid_count
                logger.info(f"Generated {persisted_valid_count} valid examples in this batch")

            # 失败样本也已在条目完成时写盘，这里仅日志
            if failed_examples:
                logger.info(f"Saved {len(failed_examples)} failed examples for analysis")

            # 按条目内已实时保存，无需在批次末尾再按阈值循环保存

            # Log usage stats
            usage = self.generator.get_usage_stats()
            logger.info(f"API Usage - Requests: {usage['total_requests']}, "
                       f"Tokens: {usage['total_tokens']:,}, "
                       f"Cost: ${usage['total_cost_usd']:.2f}")

        # Final checkpoint：若最后一次阈值保存已等于当前计数，则不再重复保存
        if last_saved_items != len(self.processed_ids):
            self.save_checkpoint(len(self.processed_ids), total_examples)

        # Final stats
        self.stats['total_items_processed'] = len(self.processed_ids)
        self.stats['total_examples_generated'] = total_examples

        return self.stats

    def run(self):
        """Run the pipeline"""
        return asyncio.run(self.run_async())

    def generate_final_report(self):
        """Generate final statistics report"""
        usage_stats = self.generator.get_usage_stats()

        report = {
            'pipeline_stats': dict(self.stats),
            'api_usage': usage_stats,
            'timestamp': datetime.now().isoformat()
        }

        report_path = self.output_dir / 'pipeline_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        # Human-readable report
        text_report_path = self.output_dir / 'pipeline_report.txt'
        with open(text_report_path, 'w') as f:
            f.write("GPT Data Construction Pipeline Report\n")
            f.write("=" * 80 + "\n\n")

            f.write("PIPELINE STATISTICS\n")
            f.write("-" * 80 + "\n")
            for key, value in self.stats.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")

            f.write("API USAGE STATISTICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total Requests: {usage_stats['total_requests']}\n")
            f.write(f"Successful: {usage_stats['successful_requests']}\n")
            f.write(f"Failed: {usage_stats['failed_requests']}\n")
            f.write(f"Generation Requests: {usage_stats['generation_requests']}\n")
            f.write(f"Validation Requests: {usage_stats['validation_requests']}\n")
            f.write(f"\nTotal Tokens: {usage_stats['total_tokens']:,}\n")
            f.write(f"Prompt Tokens: {usage_stats['total_prompt_tokens']:,}\n")
            f.write(f"Completion Tokens: {usage_stats['total_completion_tokens']:,}\n")
            f.write(f"\nTotal Cost: ${usage_stats['total_cost_usd']:.2f}\n")
            f.write(f"Avg Cost per Example: ${usage_stats['total_cost_usd']/max(self.stats.get('valid_examples', 1), 1):.4f}\n")

        logger.info(f"Reports saved to {self.output_dir}")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description='GPT-based SFT data construction pipeline'
    )

    parser.add_argument('--source', type=str, required=True,
                       help='Path to source data JSON file')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory')
    parser.add_argument('--generation-model', type=str,
                       default='gpt-5',
                       help='Model for generation (default: gpt-5)')
    parser.add_argument('--validation-model', type=str,
                       default='gpt-5',
                       help='Model for validation (default: gpt-5)')
    parser.add_argument('--examples-per-item', type=int, default=2,
                       help='Examples to generate per item (default: 2)')
    parser.add_argument('--max-concurrent', type=int, default=10,
                       help='Max concurrent API requests (default: 10)')
    parser.add_argument('--batch-size', type=int, default=100,
                       help='Batch size for processing (default: 100)')
    parser.add_argument('--checkpoint-interval', type=int, default=500,
                       help='Checkpoint interval (default: 500)')
    parser.add_argument('--sample', type=int, default=None,
                       help='Sample size for testing (default: None = all data)')
    parser.add_argument('--sampling-strategy', type=str, default='cluster',
                       choices=['random', 'sequential', 'balanced', 'cluster'],
                       help='Sampling strategy when sample size is set (default: cluster for max category coverage)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--multiple-choice-ratio', type=float, default=0.5,
                       help='Ratio of multiple choice questions (default: 0.2 = 20%%)')
    parser.add_argument('--test-ratio', type=float, default=0.1,
                       help='Test set ratio for automatic split (default: 0.1)')
    parser.add_argument('--no-split', action='store_true',
                       help='Disable automatic train/test split')

    args = parser.parse_args()

    # Create pipeline
    pipeline = GPTDataConstructionPipeline(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        source_path=args.source,
        output_dir=args.output,
        generation_model=GENERATION_MODEL,
        validation_model=VALIDATION_MODEL,
        examples_per_item=args.examples_per_item,
        max_concurrent_requests=args.max_concurrent,
        batch_size=args.batch_size,
        checkpoint_interval=args.checkpoint_interval,
        sample_size=args.sample,
        sampling_strategy=args.sampling_strategy,
        seed=args.seed,
        multiple_choice_ratio=args.multiple_choice_ratio
    )

    # Run pipeline
    try:
        stats = pipeline.run()

        # Generate final report
        pipeline.generate_final_report()

        # Automatic train/test split unless disabled
        if not args.no_split and args.test_ratio and args.test_ratio > 0:
            pipeline.split_train_test(test_ratio=args.test_ratio, seed=args.seed)

        logger.info("\n" + "=" * 80)
        logger.info("Pipeline Complete!")
        logger.info("=" * 80)
        logger.info(f"Generated {stats['total_examples_generated']} examples")
        logger.info(f"Output directory: {args.output}")

    except KeyboardInterrupt:
        logger.info("\nPipeline interrupted. Progress saved to checkpoint.")
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        raise


if __name__ == '__main__':
    main()

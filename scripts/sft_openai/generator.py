import asyncio
import aiohttp
import base64
import mimetypes
import os
import random
import time
import hashlib
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from utils import APIUsageStats, QuestionStrategy, GeneratedExample
from prompts import PromptLibrary
import json
import backoff
from openai import AsyncOpenAI, OpenAI
import logging
from collections import Counter

logger = logging.getLogger(__name__)
# OpenAI API
try:
    from openai import AsyncOpenAI, OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("ERROR: openai package not available. Install with: pip install openai")

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
GENERATION_MODEL = os.getenv("GENERATION_MODEL")
VALIDATION_MODEL = os.getenv("VALIDATION_MODEL")
# Ensure MAX_TOKENS is an int
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2000"))

class GPTDataGenerator:
    """
    Handles interaction with OpenAI GPT API for data generation
    """

    def __init__(self,
                 api_key: str,
                 base_url: str,
                 generation_model: str = "gpt-5",
                 validation_model: str = "gpt-5",
                 max_concurrent_requests: int = 10,
                 temperature: float = 0.8,
                 max_retries: int = 3):

        if not OPENAI_AVAILABLE:
            raise ImportError("openai package required. Install with: pip install openai")

        self.api_key = api_key
        self.base_url = base_url
        self.generation_model = generation_model
        self.validation_model = validation_model
        self.max_concurrent_requests = max_concurrent_requests
        self.temperature = temperature
        self.max_retries = max_retries

        # Initialize async client
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.sync_client = OpenAI(api_key=api_key, base_url=base_url)

        # Semaphore for rate limiting
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)

        # Usage statistics
        self.usage_stats = APIUsageStats()

        # Multiple-choice balancing state
        self.mc_correct_counts = Counter({'A': 0, 'B': 0, 'C': 0, 'D': 0})
        self.mc_lock = asyncio.Lock()

        # Prompt library
        self.prompts = PromptLibrary()

        logger.info(f"Initialized GPT Data Generator")
        logger.info(f"Base URL: {base_url}")
        logger.info(f"Generation model: {generation_model}")
        logger.info(f"Validation model: {validation_model}")
        logger.info(f"Max concurrent requests: {max_concurrent_requests}")

    def seed_mc_counts_from_file(self, examples_file_path: str):
        """从已存在的 generated_examples.jsonl 文件预热多选题正确选项分布计数。"""
        try:
            if not examples_file_path or not os.path.exists(examples_file_path):
                return
            counter = Counter({'A': 0, 'B': 0, 'C': 0, 'D': 0})
            with open(examples_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ex = json.loads(line)
                    except Exception:
                        continue
                    if ex.get('is_multiple_choice', False):
                        ans = str(ex.get('correct_answer', '')).strip().upper()
                        if ans in counter:
                            counter[ans] += 1
            if sum(counter.values()) > 0:
                self.mc_correct_counts.update(counter)
        except Exception:
            # 预热失败不影响主流程
            pass

    def get_mc_counts(self) -> Dict[str, int]:
        return dict(self.mc_correct_counts)

    def set_mc_counts(self, counts: Dict[str, int]):
        try:
            parsed = {k.upper(): int(v) for k, v in counts.items() if k.upper() in {'A', 'B', 'C', 'D'}}
            self.mc_correct_counts = Counter(parsed)
        except Exception:
            # 恢复失败时忽略，使用现有计数
            pass

    @staticmethod
    def _parse_options_to_map(options: List[str]) -> Optional[Dict[str, str]]:
        """将选项列表解析为 {'A': text, 'B': text, 'C': text, 'D': text}。"""
        import re
        letter_to_text: Dict[str, str] = {}
        for opt in options:
            s = str(opt).strip()
            m = re.match(r'^([A-D])\.\s*(.*)$', s)
            if m:
                letter_to_text[m.group(1)] = m.group(2)
        if len(letter_to_text) == 4:
            return letter_to_text
        # 顺序回退
        letters = ['A', 'B', 'C', 'D']
        if len(options) >= 4:
            mapped: Dict[str, str] = {}
            for i, ltr in enumerate(letters):
                text = str(options[i])
                if '.' in text:
                    text = text.split('.', 1)[1].strip()
                mapped[ltr] = text
            return mapped
        return None

    async def _rebalance_correct_option_with_reservation(
        self,
        options: List[str],
        correct_answer: str
    ) -> Tuple[List[str], str, Optional[str]]:
        """在生成回答之前对多选题正确选项进行均衡并进行配额预留。"""
        letter_map = self._parse_options_to_map(options)
        balanced_options = list(options)
        balanced_correct = str(correct_answer).strip().upper() if correct_answer else ''
        reserved_letter: Optional[str] = None

        if not letter_map or balanced_correct not in {'A', 'B', 'C', 'D'}:
            return balanced_options, balanced_correct, reserved_letter

        async with self.mc_lock:
            target_letter = min(['A', 'B', 'C', 'D'], key=lambda l: (self.mc_correct_counts[l], l))
            if target_letter != balanced_correct:
                letter_map[balanced_correct], letter_map[target_letter] = (
                    letter_map[target_letter], letter_map[balanced_correct]
                )
                balanced_correct = target_letter
            balanced_options = [
                f"A. {letter_map['A']}",
                f"B. {letter_map['B']}",
                f"C. {letter_map['C']}",
                f"D. {letter_map['D']}"
            ]
            self.mc_correct_counts[balanced_correct] += 1
            reserved_letter = balanced_correct

        return balanced_options, balanced_correct, reserved_letter

    @backoff.on_exception(
        backoff.expo,
        (aiohttp.ClientError, asyncio.TimeoutError),
        max_tries=3,
        max_time=60
    )
    async def _call_gpt_async(self,
                              messages: List[Dict[str, str]],
                              model: str,
                              temperature: float = None,
                              max_completion_tokens: int = 2000) -> Tuple[str, Dict]:
        """
        Make async call to GPT API with retry logic

        Returns:
            (response_text, usage_dict)
        """
        if temperature is None:
            temperature = self.temperature

        async with self.semaphore:
            try:
                response = await self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_completion_tokens=MAX_TOKENS
                )

                content = response.choices[0].message.content

                # Handle None or empty content
                if content is None:
                    logger.warning(f"API returned None content for model {model}")
                    content = ""

                usage = {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                }

                return content, usage

            except Exception as e:
                logger.error(f"API call failed: {str(e)}")
                self.usage_stats.failed_requests += 1
                raise

    def _build_multimodal_user_content(self, user_prompt: str, item: Dict) -> Any:
        """Build OpenAI chat message content with text and image if available.

        Supports:
        - HTTP(S) image URLs passed directly
        - Local image files encoded as data URLs
        If no valid image found, returns text-only content.
        """
        image_path = item.get('image')

        # Always include the textual prompt part
        content_parts: List[Dict[str, Any]] = [
            {"type": "text", "text": user_prompt}
        ]

        if not image_path or not isinstance(image_path, str):
            return content_parts

        try:
            # Remote URL
            if image_path.startswith('http://') or image_path.startswith('https://'):
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": image_path}
                })
                return content_parts

            # Local file → data URL
            if os.path.exists(image_path):
                mime, _ = mimetypes.guess_type(image_path)
                mime = mime or 'image/jpeg'
                with open(image_path, 'rb') as f:
                    b64 = base64.b64encode(f.read()).decode('utf-8')
                data_url = f"data:{mime};base64,{b64}"
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": data_url}
                })
                return content_parts
        except Exception as e:
            logger.warning(f"Failed to attach image '{image_path}': {e}")

        return content_parts

    async def generate_question(self, item: Dict, strategy: QuestionStrategy) -> Tuple[str, Dict]:
        """
        Generate a question for the given item and strategy

        Returns:
            (question_text, metadata)
        """
        system_prompt = self.prompts.get_question_system_prompt()
        user_prompt = self.prompts.get_question_generation_prompt(item, strategy)

        user_content = self._build_multimodal_user_content(user_prompt, item)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        try:
            question, usage = await self._call_gpt_async(
                messages,
                self.generation_model,
                temperature=self.temperature,
                max_completion_tokens=MAX_TOKENS
            )

            # Update stats
            self.usage_stats.add_usage(
                usage['prompt_tokens'],
                usage['completion_tokens'],
                self.generation_model,
                "generation"
            )

            # Clean up question
            question = question.strip()
            if question.startswith("Question:"):
                question = question[9:].strip()

            metadata = {
                'strategy': strategy.value,
                'model': self.generation_model,
                'temperature': self.temperature,
                'usage': usage,
                'timestamp': datetime.now().isoformat()
            }

            return question, metadata

        except Exception as e:
            logger.error(f"Failed to generate question: {str(e)}")
            raise

    async def generate_response(self,
                               item: Dict,
                               question: str,
                               strategy: QuestionStrategy) -> Tuple[str, Dict]:
        """
        Generate a structured response for the question

        Returns:
            (response_text, metadata)
        """
        system_prompt = self.prompts.get_response_system_prompt()
        user_prompt = self.prompts.get_response_generation_prompt(item, question, strategy)

        user_content = self._build_multimodal_user_content(user_prompt, item)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        try:
            response, usage = await self._call_gpt_async(
                messages,
                self.generation_model,
                temperature=self.temperature,
                max_completion_tokens=MAX_TOKENS
            )

            # Update stats
            self.usage_stats.add_usage(
                usage['prompt_tokens'],
                usage['completion_tokens'],
                self.generation_model,
                "generation"
            )

            metadata = {
                'model': self.generation_model,
                'temperature': self.temperature,
                'usage': usage,
                'timestamp': datetime.now().isoformat()
            }

            return response, metadata

        except Exception as e:
            logger.error(f"Failed to generate response: {str(e)}")
            raise

    async def generate_multiple_choice_question(self, item: Dict, strategy: QuestionStrategy) -> Tuple[str, List[str], str, Dict]:
        """
        Generate a multiple choice question with 4 options

        Returns:
            (question_text, options_list, correct_answer, metadata)
        """
        system_prompt = self.prompts.get_question_system_prompt()
        user_prompt = self.prompts.get_multiple_choice_question_prompt(item, strategy)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        try:
            response_text, usage = await self._call_gpt_async(
                messages,
                self.generation_model,
                temperature=self.temperature,
                max_completion_tokens=MAX_TOKENS
            )

            # Update stats
            self.usage_stats.add_usage(
                usage['prompt_tokens'],
                usage['completion_tokens'],
                self.generation_model,
                "generation"
            )

            # Parse the response to extract question, options, and correct answer
            lines = response_text.strip().split('\n')
            question = ""
            options = []
            correct_answer = ""

            for line in lines:
                line = line.strip()
                if line.startswith("Question:"):
                    question = line[9:].strip()
                elif line.startswith("A.") or line.startswith("B.") or line.startswith("C.") or line.startswith("D."):
                    options.append(line)
                elif line.startswith("Correct Answer:"):
                    correct_answer = line[15:].strip()

            # Fallback parsing if structured format not found
            if not question:
                # Try to extract first meaningful line as question
                for line in lines:
                    if line and not line.startswith(("A.", "B.", "C.", "D.", "Correct")):
                        question = line.strip()
                        break

            if not options:
                # Try to extract options
                import re
                option_pattern = r'^[A-D]\.\s*(.+)$'
                for line in lines:
                    match = re.match(option_pattern, line.strip())
                    if match:
                        options.append(line.strip())

            if not correct_answer:
                # Try to find correct answer
                import re
                answer_match = re.search(r'[Cc]orrect\s+[Aa]nswer:\s*([A-D])', response_text)
                if answer_match:
                    correct_answer = answer_match.group(1)

            metadata = {
                'strategy': strategy.value,
                'model': self.generation_model,
                'temperature': self.temperature,
                'usage': usage,
                'timestamp': datetime.now().isoformat(),
                'is_multiple_choice': True
            }

            return question, options, correct_answer, metadata

        except Exception as e:
            logger.error(f"Failed to generate multiple choice question: {str(e)}")
            raise

    async def generate_multiple_choice_response(self,
                                               item: Dict,
                                               question: str,
                                               options: List[str],
                                               correct_answer: str,
                                               strategy: QuestionStrategy) -> Tuple[str, Dict]:
        """
        Generate a structured response for multiple choice question

        Returns:
            (response_text, metadata)
        """
        system_prompt = self.prompts.get_response_system_prompt()
        user_prompt = self.prompts.get_multiple_choice_response_prompt(
            item, question, options, correct_answer, strategy
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        try:
            response, usage = await self._call_gpt_async(
                messages,
                self.generation_model,
                temperature=self.temperature,
                max_completion_tokens=MAX_TOKENS
            )

            # Update stats
            self.usage_stats.add_usage(
                usage['prompt_tokens'],
                usage['completion_tokens'],
                self.generation_model,
                "generation"
            )

            metadata = {
                'model': self.generation_model,
                'temperature': self.temperature,
                'usage': usage,
                'timestamp': datetime.now().isoformat(),
                'is_multiple_choice': True
            }

            return response, metadata

        except Exception as e:
            logger.error(f"Failed to generate multiple choice response: {str(e)}")
            raise

    def validate_tag_structure(self, response: str) -> Tuple[bool, Dict]:
        """
        严格验证response的tag结构

        要求：
        1. tag由若干 <look>/<think> 严格交替的区块组成
        2. 可从 <look> 或 <think> 开始
        3. 以且仅以一个 <answer> 区块结尾
        4. 不应存在区块外的非空文本

        Returns:
            (is_valid, validation_dict)
        """
        import re

        # Remove leading/trailing whitespace
        response = response.strip()

        # Extract all tags with their positions
        tag_pattern = r'<(look|think|answer)>(.*?)</\1>'
        matches = list(re.finditer(tag_pattern, response, re.DOTALL))

        if not matches:
            return False, {
                'error': 'no_valid_tags',
                'message': f'No valid tag blocks found: {response}',
                'pass': False
            }

        # Check if there's text outside of tags
        covered_ranges = []
        for match in matches:
            covered_ranges.append((match.start(), match.end()))

        # Sort ranges and check for gaps
        covered_ranges.sort()

        # Check text before first tag
        if covered_ranges[0][0] > 0:
            text_before = response[:covered_ranges[0][0]].strip()
            if text_before:
                return False, {
                    'error': 'text_before_tags',
                    'message': f'Found non-empty text before first tag: "{text_before[:50]}"',
                    'pass': False
                }

        # Check text between tags
        for i in range(len(covered_ranges) - 1):
            gap_start = covered_ranges[i][1]
            gap_end = covered_ranges[i + 1][0]
            gap_text = response[gap_start:gap_end].strip()
            if gap_text:
                return False, {
                    'error': 'text_between_tags',
                    'message': f'Found non-empty text between tags: "{gap_text}"',
                    'pass': False
                }

        # Check text after last tag
        if covered_ranges[-1][1] < len(response):
            text_after = response[covered_ranges[-1][1]:].strip()
            if text_after:
                return False, {
                    'error': 'text_after_tags',
                    'message': f'Found non-empty text after last tag: "{text_after}"',
                    'pass': False
                }

        # Extract tag sequence
        tag_sequence = [match.group(1) for match in matches]

        # Check: must end with exactly one <answer>
        if tag_sequence[-1] != 'answer':
            return False, {
                'error': 'no_final_answer',
                'message': f'Response must end with <answer>, found: <{tag_sequence[-1]}>',
                'pass': False
            }

        if tag_sequence.count('answer') > 1:
            return False, {
                'error': 'multiple_answers',
                'message': f'Found {tag_sequence.count("answer")} <answer> tags, should have exactly 1',
                'pass': False
            }

        # Check: before answer, should be alternating look/think
        before_answer = tag_sequence[:-1]

        if len(before_answer) == 0:
            return False, {
                'error': 'no_look_think',
                'message': 'Must have at least one <look> or <think> before <answer>',
                'pass': False
            }

        # Check first tag is look or think
        if before_answer[0] not in ['look', 'think']:
            return False, {
                'error': 'invalid_first_tag',
                'message': f'First tag must be <look> or <think>, found: <{before_answer[0]}>',
                'pass': False
            }

        # Check strict alternation
        for i in range(len(before_answer) - 1):
            current = before_answer[i]
            next_tag = before_answer[i + 1]

            # Must alternate between look and think
            if current == 'look' and next_tag != 'think':
                return False, {
                    'error': 'not_alternating',
                    'message': f'After <look> must be <think>, found <{next_tag}> at position {i+1}',
                    'pass': False
                }
            elif current == 'think' and next_tag != 'look':
                return False, {
                    'error': 'not_alternating',
                    'message': f'After <think> must be <look>, found <{next_tag}> at position {i+1}',
                    'pass': False
                }

        # Check tag content is not empty
        for i, match in enumerate(matches):
            tag_name = match.group(1)
            tag_content = match.group(2).strip()
            if not tag_content:
                return False, {
                    'error': 'empty_tag_content',
                    'message': f'Tag <{tag_name}> at position {i} has empty content',
                    'pass': False
                }

        # All checks passed
        return True, {
            'pass': True,
            'tag_sequence': tag_sequence,
            'num_look': tag_sequence.count('look'),
            'num_think': tag_sequence.count('think'),
            'validation_method': 'strict_tag_structure'
        }

    async def validate_example(self,
                              question: str,
                              response: str,
                              item: Dict) -> Tuple[bool, Dict]:
        """
        Validate quality of generated example using GPT
        First performs strict tag structure validation, then GPT quality validation

        Returns:
            (is_valid, validation_dict)
        """
        # Step 1: Strict tag structure validation
        structure_valid, structure_result = self.validate_tag_structure(response)

        if not structure_valid:
            logger.warning(f"Tag structure validation failed: {structure_result.get('message', 'Unknown error')}")
            # Return immediately without calling GPT validation API
            return False, {
                'content_quality': 0,
                'coherence': 0,
                'diversity': 0,
                'educational_value': 0,
                'overall_score': 0,
                'pass': False,
                'issues': [structure_result.get('message', 'Tag structure validation failed')],
                'validation_method': 'strict_tag_structure',
                'structure_error': structure_result.get('error', 'unknown'),
                'timestamp': datetime.now().isoformat()
            }

        # Step 2: GPT quality validation (only if structure is valid)
        logger.debug(f"Tag structure valid: {structure_result.get('tag_sequence', [])}")

        validation_prompt = self.prompts.get_validation_prompt(question, response, item)

        messages = [
            {"role": "system", "content": "You are a multimodal quality validator. Consider the provided image. Respond only with valid JSON."},
            {"role": "user", "content": self._build_multimodal_user_content(validation_prompt, item)}
        ]

        try:
            validation_text, usage = await self._call_gpt_async(
                messages,
                self.validation_model,
                temperature=0.3,  # Lower temperature for validation
                max_completion_tokens=MAX_TOKENS
            )

            # Update stats
            self.usage_stats.add_usage(
                usage['prompt_tokens'],
                usage['completion_tokens'],
                self.validation_model,
                "validation"
            )

            # Handle empty response from API
            if not validation_text or validation_text.strip() == "":
                logger.warning(f"Validation API returned empty response, using basic validation")
                # Perform basic format validation instead
                has_look = '<look>' in response and '</look>' in response
                has_think = '<think>' in response and '</think>' in response
                has_answer = '<answer>' in response and '</answer>' in response

                basic_score = 7.0  # Default passing score
                if has_look and has_think and has_answer:
                    return True, {
                        'format_score': 8.0,
                        'content_score': 7.0,
                        'coherence_score': 7.0,
                        'diversity_score': 7.0,
                        'educational_score': 7.0,
                        'image_consistency_score': 7.0,
                        'overall_score': 7.2,
                        'pass': True,
                        'issues': [],
                        'validation_method': 'basic_fallback',
                        'usage': usage,
                        'timestamp': datetime.now().isoformat()
                    }
                else:
                    return False, {
                        'error': 'missing_tags',
                        'overall_score': 3.0,
                        'pass': False,
                        'issues': ['Missing required tags'],
                        'validation_method': 'basic_fallback',
                        'usage': usage
                    }

            # Parse JSON response
            # Remove markdown code blocks if present
            validation_text = validation_text.strip()
            if validation_text.startswith("```json"):
                validation_text = validation_text[7:]
            if validation_text.startswith("```"):
                validation_text = validation_text[3:]
            if validation_text.endswith("```"):
                validation_text = validation_text[:-3]
            validation_text = validation_text.strip()

            validation_result = json.loads(validation_text)
            validation_result['usage'] = usage
            validation_result['timestamp'] = datetime.now().isoformat()
            validation_result['validation_method'] = 'gpt'

            is_valid = validation_result.get('pass', False)

            return is_valid, validation_result

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse validation response: {str(e)}")
            logger.error(f"Response: {validation_text}")
            # Use basic validation as fallback
            has_look = '<look>' in response and '</look>' in response
            has_think = '<think>' in response and '</think>' in response
            has_answer = '<answer>' in response and '</answer>' in response

            if has_look and has_think and has_answer:
                logger.info("Using basic tag validation as fallback")
                return True, {
                    'format_score': 7.0,
                    'content_score': 6.0,
                    'coherence_score': 6.0,
                    'diversity_score': 6.0,
                    'educational_score': 6.0,
                    'image_consistency_score': 6.0,
                    'overall_score': 6.2,
                    'pass': True,
                    'issues': ['Validation API failed, used basic checks'],
                    'validation_method': 'basic_fallback',
                    'error': str(e)
                }
            else:
                return False, {
                    'error': 'parse_error_and_missing_tags',
                    'overall_score': 0,
                    'pass': False,
                    'issues': ['Failed to parse validation response', 'Missing required tags'],
                    'validation_method': 'basic_fallback'
                }
        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            raise

    async def generate_full_example(self,
                                   item: Dict,
                                   strategy: QuestionStrategy,
                                   multiple_choice_ratio: float = 0.2,
                                   balance_mc: bool = False) -> Tuple[Optional[GeneratedExample], Dict, bool]:
        """
        Generate complete example: question + response + validation
        Generates multiple choice questions with probability of multiple_choice_ratio (default 20%)

        Returns:
            Tuple of (example, failed_data, is_valid)
            - example: GeneratedExample if valid, None if failed
            - failed_data: Dict with failure info if failed, empty dict if valid
            - is_valid: True if validation passed, False otherwise
        """
        try:
            # Decide if this should be a multiple choice question (20% probability)
            is_multiple_choice = random.random() < multiple_choice_ratio

            if is_multiple_choice:
                # Generate multiple choice question
                question, options, correct_answer, q_metadata = await self.generate_multiple_choice_question(item, strategy)

                # 可选：在生成回答前做正确选项均衡（并预留配额）
                balanced_options = options
                balanced_correct = correct_answer
                reserved_letter: Optional[str] = None
                if balance_mc:
                    try:
                        balanced_options, balanced_correct, reserved_letter = await self._rebalance_correct_option_with_reservation(
                            options, correct_answer
                        )
                    except Exception:
                        # 均衡失败不阻断流程，退化为原选项
                        balanced_options = options
                        balanced_correct = correct_answer

                # Generate response for multiple choice（使用均衡后的选项）
                try:
                    response, r_metadata = await self.generate_multiple_choice_response(
                        item, question, balanced_options, balanced_correct, strategy
                    )
                except Exception as e:
                    # 生成失败需回滚预留
                    if reserved_letter in {'A', 'B', 'C', 'D'}:
                        async with self.mc_lock:
                            self.mc_correct_counts[reserved_letter] -= 1
                    raise

                # Validate
                is_valid, validation = await self.validate_example(question, response, item)

                # Parse response metadata
                num_cycles = response.count('<look>')
                word_count = len(response.split())

                # Determine complexity
                if word_count < 500:
                    complexity = "simple"
                elif word_count < 800:
                    complexity = "medium"
                else:
                    complexity = "complex"

                # Create example ID
                item_hash = hashlib.md5(item['wiki_title'].encode()).hexdigest()[:8]
                example_id = f"gpt_{item_hash}_{strategy.value[:4]}_mc_{int(time.time())}"

                example = GeneratedExample(
                    id=example_id,
                    image=item['image'],
                    wiki_title=item['wiki_title'],
                    categories=item['categories'],
                    question=question,
                    question_strategy=strategy.value,
                    complexity=complexity,
                    response=response,
                    num_cycles=num_cycles,
                    word_count=word_count,
                    gpt_generation_metadata={
                        'question_metadata': q_metadata,
                        'response_metadata': r_metadata
                    },
                    validation_metadata=validation,
                    is_multiple_choice=True,
                    options=balanced_options,
                    correct_answer=balanced_correct
                )
                if not is_valid and reserved_letter in {'A', 'B', 'C', 'D'}:
                    # 验证失败回滚预留
                    async with self.mc_lock:
                        self.mc_correct_counts[reserved_letter] -= 1

            else:
                # Generate regular open-ended question
                question, q_metadata = await self.generate_question(item, strategy)

                # Generate response
                response, r_metadata = await self.generate_response(item, question, strategy)

                # Validate
                is_valid, validation = await self.validate_example(question, response, item)

                # Parse response metadata
                num_cycles = response.count('<look>')
                word_count = len(response.split())

                # Determine complexity
                if word_count < 500:
                    complexity = "simple"
                elif word_count < 800:
                    complexity = "medium"
                else:
                    complexity = "complex"

                # Create example ID
                item_hash = hashlib.md5(item['wiki_title'].encode()).hexdigest()[:8]
                example_id = f"gpt_{item_hash}_{strategy.value[:4]}_{int(time.time())}"

                example = GeneratedExample(
                    id=example_id,
                    image=item['image'],
                    wiki_title=item['wiki_title'],
                    categories=item['categories'],
                    question=question,
                    question_strategy=strategy.value,
                    complexity=complexity,
                    response=response,
                    num_cycles=num_cycles,
                    word_count=word_count,
                    gpt_generation_metadata={
                        'question_metadata': q_metadata,
                        'response_metadata': r_metadata
                    },
                    validation_metadata=validation,
                    is_multiple_choice=False,
                    options=None,
                    correct_answer=None
                )

            # Return both valid and invalid examples
            if is_valid:
                return example, {}, True
            else:
                logger.debug(f"Example failed validation: {validation.get('issues', [])}")
                # Create failed example data with all information
                failed_data = {
                    'id': example.id,
                    'image': example.image,
                    'wiki_title': example.wiki_title,
                    'categories': example.categories,
                    'question': example.question,
                    'question_strategy': example.question_strategy,
                    'response': example.response,
                    'is_multiple_choice': example.is_multiple_choice,
                    'options': example.options,
                    'correct_answer': example.correct_answer,
                    'validation_result': validation,
                    'failure_reason': validation.get('issues', []),
                    'overall_score': validation.get('overall_score', 0),
                    'timestamp': datetime.now().isoformat()
                }
                return None, failed_data, False

        except Exception as e:
            logger.error(f"Failed to generate full example: {str(e)}")
            # Return error information
            error_data = {
                'wiki_title': item.get('wiki_title', 'unknown'),
                'image': item.get('image', 'unknown'),
                'error': str(e),
                'strategy': strategy.value,
                'timestamp': datetime.now().isoformat()
            }
            return None, error_data, False

    def get_usage_stats(self) -> Dict:
        """Get current usage statistics"""
        return self.usage_stats.to_dict()

    def reset_stats(self):
        """Reset usage statistics"""
        self.usage_stats = APIUsageStats()
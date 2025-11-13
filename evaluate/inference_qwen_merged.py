import os
# -*- coding: utf-8 -*-
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import statistics
import re

from prompts import (
    instruction_simple,
    instruction_extra_simple,
    instruction_thinking,
    instruction_normal,
    instruction_strong_tag,
)

# --- 工具函数导入 ---
from utils import (
    sanitize_filename as _sanitize_filename,
    derive_model_slug_from_ckpt as _derive_model_slug_from_ckpt,
    derive_model_display_name as _derive_model_display_name,
    load_model_and_processor,
)

# --- [新] 引入本地模型所需的库 ---
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from argparse import ArgumentParser
from PIL import Image
import os
try:
    from tqdm import tqdm
except Exception:
    # 简易降级：若未安装 tqdm，则不显示进度条
    def tqdm(iterable=None, total=None, desc=None):
        return iterable if iterable is not None else range(total or 0)

# --- [修改] 配置项：从API配置改为本地模型配置 ---
# 默认的模型路径，指向你全参微调后的模型文件夹
DEFAULT_CKPT_PATH = '/data_ali/yurui/result_ckpt/comparison_datascale/iceberg20w_llava2w_first14layers_weight003_tem015_bs128_lr_5e-6'
# DEFAULT_CKPT_PATH = '/wangbenyou/yurui/model_ckpt/qwen2.5vl72b'
PROCESS_LIMIT = -1   # -1表示处理所有数据，建议初始测试设置一个较小的值

DEFAULT_MULTIPLE_CHOICE_INSTRUCTION = """You are an expert AI assistant taking a multiple-choice test. Your goal is to be as accurate as possible. Please follow these instructions carefully:

**Instructions for Answering:**
1.  **Independent Questions:** Treat each question as a separate, new task. Your answer to one question should not be influenced by any previous questions.
2.  **Select the Best Answer:** Based on the information in the question (and the image, if provided), combined with your general knowledge, select the most accurate and logical answer from the options A, B, C, or D.
3.  **Handling Uncertainty (Option E):** If you carefully determine that none of the options from A to D are correct, or if the question is genuinely unanswerable, select option E (if available). Do not choose E simply because a question is difficult; reserve it for clearly incorrect or unanswerable scenarios.
4.  **Strict Output Format:** You MUST respond with only the single, uppercase letter of your chosen answer (e.g., A, B, C, D, or E). Do not add any extra text, explanations, or punctuation.
"""

# --- 输入输出文件路径 ---
# 请确保此路径相对于您运行脚本的位置是正确的
INPUT_FILE = "/data_ali/yurui/benchmark/iceberg/question_new/merged_questions_1104.json"
# 输出文件路径在 main 中根据 checkpoint 生成（包含模型别名与时间戳）
OUTPUT_FILE = None  # 将在运行时设置为: <output_dir>/evaluation_results_merged_<model_slug>_<ts>.json

# --- 配置日志 ---
# (日志配置保持不变)
Path("/data_ali/shunian/verl/evaluate/logs/outputs_qwen/logs_call_llm").mkdir(exist_ok=True, parents=True)
Path("/data_ali/shunian/verl/evaluate/logs/outputs_qwen/outputs_call_llm").mkdir(exist_ok=True, parents=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'/data_ali/shunian/verl/evaluate/logs/outputs_qwen/logs_call_llm/evaluation_merged_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# --- [新] 从我们之前的脚本中引入模型加载和参数解析功能 ---
def parse_args():
    """解析命令行参数"""
    parser = ArgumentParser()
    parser.add_argument('-c', '--checkpoint-path', type=str, default=DEFAULT_CKPT_PATH,
                        help='模型检查点名称或本地路径')
    parser.add_argument('--batch-size', type=int, default=1, help='推理时的批大小')
    parser.add_argument('--cpu-only', action='store_true', help='仅使用 CPU 运行')
    parser.add_argument('--flash-attn2', action='store_true', default=False,
                        help='加载模型时启用 flash_attention_2')
    parser.add_argument('--output-dir', type=str, default=None, help='结果输出目录，默认取环境变量 EVAL_RESULTS_DIR 或脚本默认路径')
    # vLLM 相关
    parser.add_argument('--use-vllm', action='store_true', help='使用 vLLM 引擎进行推理')
    parser.add_argument('--vllm-dtype', type=str, default='bfloat16', help='vLLM 推理精度: auto/float16/bfloat16')
    parser.add_argument('--vllm-tp', type=int, default=1, help='vLLM tensor parallel size')
    parser.add_argument('--vllm-gpu-mem', type=float, default=0.9, help='vLLM GPU 显存利用率 (0-1)')
    # 设备/环境相关
    parser.add_argument('--cuda-visible-devices', type=str, default=None,
                        help='设置 CUDA_VISIBLE_DEVICES，例如 "0" 或 "0,1"')
    args = parser.parse_args()
    return args


# --- 数据结构类 ---
@dataclass
class QuestionResult:
    """单个问题的评测结果"""
    question_id: str
    question_type: str
    question_text: str
    correct_answer: str
    model_answer: str  # 这是处理后的答案 (e.g., "A")
    is_correct: bool
    response_time: float
    model_input_text: Optional[str] = None  # [新增] 用于记录模型实际收到的完整输入文本
    model_raw_output: Optional[str] = None  # [新增] 用于记录模型未经处理的原始输出
    image_used: Optional[str] = None
    error: Optional[str] = None

@dataclass
class EvaluationSummary:
    """评测总结"""
    total_questions: int
    knowledge_questions: Dict[str, Any] = field(default_factory=dict)
    bridge_questions: Dict[str, Any] = field(default_factory=dict)
    multimodal_questions: Dict[str, Any] = field(default_factory=dict)
    overall_stats: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0


IMAGE_BASE_PATH = '/data_ali/yurui/benchmark/iceberg/'

class MultiModalQuestionEvaluator:
    def __init__(self, model, processor, image_base_path: str, use_vllm: bool = False, checkpoint_identifier: Optional[str] = None):
        self.model = model
        self.processor = processor
        self.use_vllm = use_vllm
        # 模型名兼容 vLLM/HF
        model_name = None
        if self.use_vllm and hasattr(model, "llm_engine") and hasattr(model.llm_engine, "model_config"):
            model_name = getattr(model.llm_engine.model_config, "model", None)
            if model_name is None:
                model_name = getattr(model.llm_engine.model_config, "model_path", None)
        if not model_name and hasattr(model, "config"):
            model_name = getattr(model.config, "name_or_path", None)
        self.model_name = model_name or ("vLLMEngine" if self.use_vllm else "HFModel")
        self.image_base_path = Path(image_base_path)
        self.results: List[QuestionResult] = []
        self._checkpoint_prompt_identifiers: List[str] = []
        if checkpoint_identifier:
            self.register_checkpoint_prompt_identifier(checkpoint_identifier)

    def register_checkpoint_prompt_identifier(self, identifier: Optional[str]):
        if not identifier:
            return
        identifier_lower = str(identifier).lower()
        if identifier_lower not in self._checkpoint_prompt_identifiers:
            self._checkpoint_prompt_identifiers.append(identifier_lower)

    def _resolve_instruction_template(self) -> str:
        reference_parts: List[str] = []
        if self._checkpoint_prompt_identifiers:
            reference_parts.append(" ".join(self._checkpoint_prompt_identifiers))
        if self.model_name:
            reference_parts.append(str(self.model_name).lower())
        reference_text = " ".join(reference_parts)

        if 'strong_tag' in reference_text:
            return instruction_strong_tag
        if 'extra_simple' in reference_text:
            return instruction_extra_simple
        if 'simple' in reference_text:
            return instruction_simple
        if 'think' in reference_text:
            return instruction_thinking
        if 'normal' in reference_text:
            return instruction_normal
        return DEFAULT_MULTIPLE_CHOICE_INSTRUCTION

    def _parse_model_answer(self, raw_response: Optional[str], options: Optional[Dict[str, Any]]) -> str:
        if not raw_response:
            return ""

        options_keys = {str(k).upper() for k in (options or {}).keys() if k is not None}
        response_text = raw_response.strip()
        answer_segment = response_text

        match = re.search(r"<answer>(.*?)(</answer>|$)", response_text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            extracted = match.group(1).strip()
            if extracted:
                answer_segment = extracted

        candidate_sources: List[str] = []
        if answer_segment:
            candidate_sources.append(answer_segment)
        if response_text and response_text != answer_segment:
            candidate_sources.append(response_text)

        candidate_tokens: List[str] = []
        for source in candidate_sources:
            upper_source = source.strip().upper()
            if not upper_source:
                continue
            candidate_tokens.append(upper_source)
            candidate_tokens.extend(tok for tok in re.split(r"[^A-Z0-9]+", upper_source) if tok)

        for token in candidate_tokens:
            token_upper = token.strip().upper()
            if not token_upper:
                continue
            if token_upper in options_keys:
                return token_upper
            first_char = token_upper[0]
            if first_char in options_keys:
                return first_char
            if len(token_upper) == 1 and first_char in {'A', 'B', 'C', 'D', 'E'}:
                return first_char

        if candidate_tokens:
            fallback = candidate_tokens[0].strip().upper()
            if fallback:
                first_char = fallback[0]
                if first_char in options_keys or first_char in {'A', 'B', 'C', 'D', 'E'}:
                    return first_char
                return fallback

        if candidate_sources:
            fallback_text = candidate_sources[0].strip().upper()
            if fallback_text:
                first_char = fallback_text[0]
                if first_char in options_keys or first_char in {'A', 'B', 'C', 'D', 'E'}:
                    return first_char
                return fallback_text

        return ""

    def create_prompt(self, question_data: Dict[str, Any], question_type: str) -> str:
        """创建适合不同问题类型的、带有详细约束的prompt模板"""
        base_instruction = DEFAULT_MULTIPLE_CHOICE_INSTRUCTION
        if question_type in ["bridge", "multimodal"]:
            base_instruction = self._resolve_instruction_template()

        question_text = question_data.get('question_text', '')
        options = question_data.get('options', {})
        options_text = "\n".join([f"{key}. {value}" for key, value in options.items()])

        prompt_text = f"""{base_instruction}
Question: {question_text}

Options:
{options_text}

Your Answer:"""
        return prompt_text

    def evaluate_question(self, question_data: Dict[str, Any], question_type: str, question_id: str, image_path: Optional[str] = None) -> QuestionResult:
        """使用本地模型评估单个问题。"""
        start_time = time.time()

        images = []
        image_used_path = None
        model_input_text_to_log = None  # 初始化变量，用于记录模型输入
        raw_model_output = None         # 初始化变量，用于记录原始输出

        # 加载图片
        if question_type in ['bridge', 'multimodal'] and image_path:
            image_file = Path(image_path)
            if image_file.is_file():
                try:
                    images.append(Image.open(image_file).convert("RGB"))
                    image_used_path = str(image_file.resolve())
                    logger.info(f"正在为问题 {question_id} 加载图片 '{image_path}'。")
                except Exception as e:
                    logger.error(f"无法为问题 {question_id} 打开或处理图片 '{image_path}': {e}")
            else:
                logger.warning(f"为问题 {question_id} 指定了图片路径但文件未找到: '{image_path}'")

        try:
            # 1. 获取纯文本的 prompt
            prompt_text = self.create_prompt(question_data, question_type)

            # 2. 构建 messages 结构
            content = []
            if images:
                content.append({"type": "image"})
            content.append({"type": "text", "text": prompt_text})

            messages = [{"role": "user", "content": content}]

            # 3. 使用 apply_chat_template 生成模型可理解的文本
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_input_text_to_log = text  # 捕获这个文本用于记录

            if self.use_vllm:
                # vLLM 推理
                from vllm import SamplingParams
                sampling_params = SamplingParams(max_tokens=8192, temperature=0.0, top_p=1.0, top_k=-1, n=1)
                if images:
                    prompts_in = [{"prompt": text, "multi_modal_data": {"image": images}}]
                else:
                    prompts_in = [text]
                outputs = self.model.generate(prompts=prompts_in, sampling_params=sampling_params, use_tqdm=False)
                response = outputs[0].outputs[0].text
                raw_model_output = response
            else:
                # HuggingFace 本地模型推理
                processor_inputs = {"text": [text], "return_tensors": "pt", "padding": True}
                if images:
                    processor_inputs["images"] = images

                inputs = self.processor(**processor_inputs)

                # 确保数据类型正确
                inputs = inputs.to(self.model.device)
                if 'pixel_values' in inputs and inputs['pixel_values'] is not None:
                    inputs['pixel_values'] = inputs['pixel_values'].to(self.model.dtype)

                gen_kwargs = {"max_new_tokens": 8192, "do_sample": False}
                generated_ids = self.model.generate(**inputs, **gen_kwargs)

                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
                ]
                response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                raw_model_output = response

            response_time = time.time() - start_time
            options = question_data.get('options', {}) or {}
            option_keys = {str(k).upper() for k in options.keys()}
            parsed_answer = self._parse_model_answer(response, options)
            if parsed_answer:
                model_answer = parsed_answer.strip().upper()
            else:
                model_answer = response.strip().upper()
            if model_answer and model_answer[0] in option_keys:
                model_answer = model_answer[0]
            correct_answer = question_data.get('correct_answer', '').upper()

            is_correct = model_answer == correct_answer

            # 在返回结果时，包含捕获到的模型输入和原始输出
            return QuestionResult(
                question_id=question_id, question_type=question_type,
                question_text=question_data.get('question_text', ''),
                correct_answer=correct_answer, model_answer=model_answer,
                is_correct=is_correct, response_time=response_time,
                model_input_text=model_input_text_to_log,
                model_raw_output=raw_model_output,
                image_used=image_used_path
            )

        except Exception as e:
            response_time = time.time() - start_time
            logger.error(f"本地模型推理时发生错误 (问题 {question_id}): {e}", exc_info=True)
            # 在返回错误结果时，也包含已有的记录
            return QuestionResult(
                question_id=question_id, question_type=question_type,
                question_text=question_data.get('question_text', ''),
                correct_answer=question_data.get('correct_answer', ''),
                model_answer="MODEL_RUNTIME_ERROR", is_correct=False,
                response_time=response_time, error=str(e),
                model_input_text=model_input_text_to_log,
                model_raw_output=raw_model_output,
                image_used=image_used_path
            )

    def process_triplet(self, item: Dict[str, Any]) -> List[QuestionResult]:
        """处理一个triplet中的所有问题

        注意：merged_questions.json的结构是每个item包含一个triplet字段
        """
        item_id = str(item.get('id', 'unknown'))
        triplet = item.get('triplet', {})

        relative_image_path = item.get('sample_image_path')
        full_image_path = None
        if relative_image_path:
            full_image_path = self.image_base_path / relative_image_path.lstrip('./')

        results = []

        # 处理 knowledge_question
        if triplet.get('knowledge_question'):
            q_data = triplet['knowledge_question']
            result = self.evaluate_question(q_data, 'knowledge', f"{item_id}_knowledge")
            results.append(result)

        # 处理 bridge_question
        if triplet.get('bridge_question'):
            q_data = triplet['bridge_question']
            result = self.evaluate_question(q_data, 'bridge', f"{item_id}_bridge", image_path=full_image_path)
            results.append(result)

        # 处理 final_multimodal_question
        if triplet.get('final_multimodal_question'):
            q_data = triplet['final_multimodal_question']
            result = self.evaluate_question(q_data, 'multimodal', f"{item_id}_multimodal", image_path=full_image_path)
            results.append(result)

        return results

    def calculate_summary(self) -> EvaluationSummary:
        """计算评测总结"""
        knowledge_results = [r for r in self.results if r.question_type == 'knowledge']
        bridge_results = [r for r in self.results if r.question_type == 'bridge']
        multimodal_results = [r for r in self.results if r.question_type == 'multimodal']

        def calculate_stats(results: List[QuestionResult]) -> Dict[str, Any]:
            if not results:
                return {"total": 0, "correct": 0, "accuracy": 0.0, "avg_response_time": 0.0, "error_count": 0}

            correct_count = sum(1 for r in results if r.is_correct)
            total_count = len(results)
            accuracy = correct_count / total_count if total_count > 0 else 0.0

            valid_times = [r.response_time for r in results if r.error is None]
            avg_time = statistics.mean(valid_times) if valid_times else 0.0

            return {
                "total": total_count, "correct": correct_count,
                "accuracy": accuracy, "avg_response_time": avg_time,
                "error_count": sum(1 for r in results if r.error is not None)
            }

        return EvaluationSummary(
            total_questions=len(self.results),
            knowledge_questions=calculate_stats(knowledge_results),
            bridge_questions=calculate_stats(bridge_results),
            multimodal_questions=calculate_stats(multimodal_results),
            overall_stats=calculate_stats(self.results)
        )

    def run_evaluation(self, input_file: str, process_limit: int = -1, batch_size: int = 1):
        """运行完整的评测流程"""
        start_time = time.time()

        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            logger.error(f"输入文件未找到: {input_file}")
            return None
        except json.JSONDecodeError:
            logger.error(f"无法解析JSON文件: {input_file}")
            return None

        # merged_questions.json 是一个数组，每个元素是一个包含triplet的字典
        items = data
        if not items:
            logger.warning(f"文件中未找到数据。")
            return None

        if process_limit > 0:
            items = items[:process_limit]

        # 构建所有问题的任务列表（每个triplet最多3个问题）
        tasks = []
        for item in items:
            item_id = str(item.get('id', 'unknown'))
            triplet = item.get('triplet', {})
            relative_image_path = item.get('sample_image_path')
            full_image_path = None
            if relative_image_path:
                full_image_path = self.image_base_path / relative_image_path.lstrip('./')

            if triplet.get('knowledge_question'):
                tasks.append({
                    'question_id': f"{item_id}_knowledge",
                    'question_type': 'knowledge',
                    'question_data': triplet['knowledge_question'],
                    'image_path': None,
                })
            if triplet.get('bridge_question'):
                tasks.append({
                    'question_id': f"{item_id}_bridge",
                    'question_type': 'bridge',
                    'question_data': triplet['bridge_question'],
                    'image_path': full_image_path,
                })
            if triplet.get('final_multimodal_question'):
                tasks.append({
                    'question_id': f"{item_id}_multimodal",
                    'question_type': 'multimodal',
                    'question_data': triplet['final_multimodal_question'],
                    'image_path': full_image_path,
                })

        if not tasks:
            logger.warning("未生成任何评测任务。")
            return None

        logger.info(f"开始评测 {len(tasks)} 个问题（批大小={batch_size}）...")

        # 进度条
        pbar = tqdm(total=len(tasks), desc="评测进度")

        # 批处理问题
        for start in range(0, len(tasks), max(1, int(batch_size))):
            batch = tasks[start:start + max(1, int(batch_size))]
            batch_start_time = time.time()

            # 准备 prompts 与图像
            prepared = []  # 每项：{id,type,correct_answer,text,image(list或None),image_used_path}
            for t in batch:
                q_data = t['question_data']
                q_type = t['question_type']
                q_id = t['question_id']

                images = []
                image_used_path = None
                if q_type in ['bridge', 'multimodal'] and t.get('image_path'):
                    image_file = Path(t['image_path'])
                    if image_file.is_file():
                        try:
                            images.append(Image.open(image_file).convert("RGB"))
                            image_used_path = str(image_file.resolve())
                        except Exception as e:
                            logger.error(f"无法为问题 {q_id} 打开或处理图片 '{t['image_path']}': {e}")
                    else:
                        logger.warning(f"为问题 {q_id} 指定了图片路径但文件未找到: '{t['image_path']}'")

                prompt_text = self.create_prompt(q_data, q_type)

                content = []
                if images:
                    content.append({"type": "image"})
                content.append({"type": "text", "text": prompt_text})
                messages = [{"role": "user", "content": content}]

                text = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

                prepared.append({
                    'question_id': q_id,
                    'question_type': q_type,
                    'question_text': q_data.get('question_text', ''),
                    'correct_answer': q_data.get('correct_answer', '').upper(),
                    'options': q_data.get('options', {}),
                    'text': text,
                    'images': images if images else None,
                    'image_used_path': image_used_path,
                })

            # 执行推理
            responses = []  # 与 prepared 对齐
            if self.use_vllm:
                from vllm import SamplingParams
                sampling_params = SamplingParams(max_tokens=8192, temperature=0.0, top_p=1.0, top_k=-1, n=1)
                prompts_in = []
                for item in prepared:
                    if item['images']:
                        prompts_in.append({"prompt": item['text'], "multi_modal_data": {"image": item['images']}})
                    else:
                        prompts_in.append(item['text'])
                outputs = self.model.generate(prompts=prompts_in, sampling_params=sampling_params, use_tqdm=False)
                for out in outputs:
                    responses.append(out.outputs[0].text)
            else:
                # HF 分别处理有图/无图两组以适配处理器接口
                idx_noimg = [i for i, it in enumerate(prepared) if not it['images']]
                idx_img = [i for i, it in enumerate(prepared) if it['images']]
                responses = [None] * len(prepared)

                if idx_noimg:
                    texts = [prepared[i]['text'] for i in idx_noimg]
                    inputs = self.processor(text=texts, return_tensors="pt", padding=True)
                    inputs = inputs.to(self.model.device)
                    gen_kwargs = {"max_new_tokens": 8192, "do_sample": False}
                    generated_ids = self.model.generate(**inputs, **gen_kwargs)
                    # 去掉提示部分
                    trimmed = [
                        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    decoded = self.processor.batch_decode(trimmed, skip_special_tokens=True)
                    for j, i in enumerate(idx_noimg):
                        responses[i] = decoded[j]

                if idx_img:
                    texts = [prepared[i]['text'] for i in idx_img]
                    images_list = [prepared[i]['images'] for i in idx_img]
                    inputs = self.processor(text=texts, images=images_list, return_tensors="pt", padding=True)
                    inputs = inputs.to(self.model.device)
                    if 'pixel_values' in inputs and inputs['pixel_values'] is not None:
                        inputs['pixel_values'] = inputs['pixel_values'].to(self.model.dtype)
                    gen_kwargs = {"max_new_tokens": 8192, "do_sample": False}
                    generated_ids = self.model.generate(**inputs, **gen_kwargs)
                    trimmed = [
                        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    decoded = self.processor.batch_decode(trimmed, skip_special_tokens=True)
                    for j, i in enumerate(idx_img):
                        responses[i] = decoded[j]

            # 写入结果
            batch_elapsed = time.time() - batch_start_time
            per_item_time = batch_elapsed / max(1, len(prepared))
            for item, resp in zip(prepared, responses):
                options = item.get('options', {}) or {}
                option_keys = {str(k).upper() for k in options.keys()}
                parsed_answer = self._parse_model_answer(resp or "", options)
                if parsed_answer:
                    model_answer = parsed_answer.strip().upper()
                else:
                    model_answer = (resp or '').strip().upper()
                correct_answer = item['correct_answer']
                if model_answer and (model_answer[0] in option_keys):
                    model_answer = model_answer[0]
                is_correct = (model_answer == correct_answer)

                self.results.append(QuestionResult(
                    question_id=item['question_id'],
                    question_type=item['question_type'],
                    question_text=item['question_text'],
                    correct_answer=correct_answer,
                    model_answer=model_answer,
                    is_correct=is_correct,
                    response_time=per_item_time,
                    model_input_text=item['text'],
                    model_raw_output=resp,
                    image_used=item['image_used_path']
                ))

            pbar.update(len(batch))

        try:
            pbar.close()
        except Exception:
            pass

        summary = self.calculate_summary()
        summary.processing_time = time.time() - start_time

        self.save_results(summary)
        return summary

    def save_results(self, summary: EvaluationSummary):
        """保存评测结果"""
        output_data = {
            "metadata": {
                "evaluation_date": datetime.now().isoformat(),
                "model_name": self.model_name,
                "total_processing_time": summary.processing_time,
                "input_file": str(Path(INPUT_FILE).resolve()),
                "process_limit": PROCESS_LIMIT
            },
            "summary": summary.__dict__,
            "detailed_results": [r.__dict__ for r in self.results]
        }

        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        logger.info(f"结果已保存到: {OUTPUT_FILE}")
        # 确保批处理脚本可从标准输出可靠解析到结果路径
        print(f"结果已保存到: {OUTPUT_FILE}", flush=True)


def main():
    """主函数"""
    args = parse_args()

    # 根据参数设置环境变量（尽量在创建任何 CUDA 上下文之前）
    if getattr(args, 'cuda_visible_devices', None):
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_visible_devices
        print(f"[ENV] CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

    # 计算输出目录与文件名（包含模型别名）
    ckpt_slug = _derive_model_slug_from_ckpt(getattr(args, 'checkpoint_path', ''))
    output_dir = args.output_dir or os.environ.get('EVAL_RESULTS_DIR', "/data_ali/shunian/verl/evaluate/results")
    try:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"无法创建输出目录 {output_dir}: {e}")
        return
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    global OUTPUT_FILE
    OUTPUT_FILE = str(Path(output_dir) / f"evaluation_results_merged_{ckpt_slug}_{ts}.json")


    try:
        model, processor = load_model_and_processor(args)
    except Exception as e:
        logger.error(f"加载本地模型失败: {e}", exc_info=True)
        return

    evaluator = MultiModalQuestionEvaluator(
        model,
        processor,
        IMAGE_BASE_PATH,
        use_vllm=getattr(args, 'use_vllm', False),
        checkpoint_identifier=getattr(args, 'checkpoint_path', None),
    )
    # 用 checkpoint 路径推导更友好的模型名，覆盖默认的 name_or_path/huggingface 尾部
    try:
        derived_display_name = _derive_model_display_name(getattr(args, 'checkpoint_path', ''))
        evaluator.model_name = derived_display_name
        evaluator.register_checkpoint_prompt_identifier(derived_display_name)
        evaluator.register_checkpoint_prompt_identifier(_derive_model_slug_from_ckpt(getattr(args, 'checkpoint_path', '')))
    except Exception:
        pass

    try:
        summary = evaluator.run_evaluation(INPUT_FILE, PROCESS_LIMIT, batch_size=getattr(args, 'batch_size', 1))

        if summary:
            print("\n" + "="*50 + "\n评测结果摘要\n" + "="*50)
            print(f"评测模型: {evaluator.model_name}")
            print(f"总处理时间: {summary.processing_time:.2f}秒")
            print(f"总问题数: {summary.total_questions}")
            print()

            for q_type in ["knowledge", "bridge", "multimodal"]:
                stats = getattr(summary, f"{q_type}_questions")
                title = q_type.replace('_', ' ').title()
                print(f"{title} Questions:")
                print(f"  - 总数: {stats['total']}, 正确: {stats['correct']}, 准确率: {stats['accuracy']:.2%}")
                print(f"  - 平均响应时间: {stats['avg_response_time']:.2f}秒, 错误数: {stats['error_count']}")
                print()

            print("整体统计:")
            print(f"  - 总准确率: {summary.overall_stats['accuracy']:.2%}")
            print(f"  - 总错误数: {summary.overall_stats['error_count']}")
        else:
            logger.error("评测未能生成任何结果。请检查日志以获取详细信息。")

    except Exception as e:
        logger.error(f"评测主流程发生严重错误: {e}", exc_info=True)

if __name__ == "__main__":
    main()

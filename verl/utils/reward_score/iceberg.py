# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from typing import Dict, Any, List, Tuple, Optional

_SOLUTION_CLIP_CHARS = 300

def extract_mcq_option_letter(text: str) -> Optional[str]:
    """从文本中解析选项字母（A-D）。优先匹配 <answer> 标签内的选项，其次回退到通用模式。

    返回大写字母 A/B/C/D，若未匹配则返回 None。
    """
    patterns = [
        r"<answer>\s*([A-Da-d])\b",  # <answer>B. ... 或 <answer>B ...
        r"\bAnswer\s*[:：]?\s*([A-Da-d])\b",
        r"\b选项\s*[:：]?\s*([A-Da-d])\b",
        r"\b([A-Da-d])\s*[\.)]\s*$",  # 末尾如 "B." 或 "B)"
        r"^\s*([A-Da-d])\s*[\.)]\b",  # 开头如 "A." 或 "C)"
    ]
    for pattern in patterns:
        m = re.search(pattern, text.strip(), flags=re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).upper()
    # 宽松匹配：抓取第一个独立的 A-D
    m = re.search(r"\b([A-Da-d])\b", text)
    if m:
        return m.group(1).upper()
    return None
    
def extract_solution(solution_str) -> str:
    """Extract the first human message as question and parse the <answer>...</answer> from assistant.

    If <answer> tags are missing, use the whole assistant message text as answer.
    Removes leading <image> marker from the question if present.
    """
    answer_text: str = ""

    lower_value = solution_str.lower()
    start_tag = "<answer>"
    end_tag = "</answer>"
    start = lower_value.find(start_tag)
    end = lower_value.find(end_tag)
    if start != -1 and end != -1 and end > start:
        # slice using original casing
        start_orig = start
        end_orig = end
        answer_text = solution_str[start_orig + len(start_tag):end_orig].strip()
    else:
        answer_text = solution_str.strip()

    return answer_text

def validate_assistant_format(output_text: str) -> Dict[str, Any]:
    """校验 assistant 输出是否符合格式协议：
    - 由若干 <look>/<think> 严格交替的区块组成（可从 <look> 或 <think> 开始），
    - 以且仅以一个 <answer> 区块结尾，
    - <answer> 内需给出单个选项字母与其文本（如："B. ..." 或 "C) ..."），
    - 不得出现 "BACKGROUND NOTE" 字样，
    - 不应存在区块外的非空文本。

    返回一个包含校验细节的字典：
    {
      "valid": bool,
      "errors": List[str],
      "tags": List[str],
      "num_look": int,
      "num_think": int,
      "num_answer": int,
      "answer": {"raw": str, "letter": Optional[str], "text": Optional[str]}
    }
    """
    s = (output_text or "").strip()
    errors: List[str] = []

    # 提前检查禁止词
    if re.search(r"background\s*note", s, flags=re.IGNORECASE):
        errors.append("输出中包含被禁止的 'BACKGROUND NOTE' 字样")

    # 以栈解析标签，确保成对闭合与顺序合法，并提取区块
    tag_token_re = re.compile(r"<\s*(/)?\s*(look|think|answer)\s*>", flags=re.IGNORECASE)
    stack: List[Tuple[str, int, int]] = []  # (tag, open_start, open_end)
    blocks: List[Tuple[str, str, int, int]] = []  # (tag_lower, inner, block_start, block_end)
    errors_stack: List[str] = []
    last_index = 0
    covered_spans: List[Tuple[int, int]] = []

    for m in tag_token_re.finditer(s):
        is_close = m.group(1) is not None
        tag = m.group(2).lower()
        if not is_close:
            # 禁止嵌套：若栈非空，说明上一个标签未闭合
            if stack:
                errors_stack.append("检测到嵌套标签，不允许在前一标签未闭合前开启新标签")
            stack.append((tag, m.start(), m.end()))
        else:
            if not stack:
                errors_stack.append(f"检测到未匹配的关闭标签 </{tag}>")
                continue
            open_tag, open_start, open_end = stack.pop()
            if open_tag != tag:
                errors_stack.append(
                    f"关闭标签 </{tag}> 与最近的打开标签 <{open_tag}> 不匹配"
                )
            # 记录块
            inner = s[open_end:m.start()]
            block_start = open_start
            block_end = m.end()
            blocks.append((tag, inner, block_start, block_end))
            covered_spans.append((block_start, block_end))

    # 栈未清空，说明有未闭合标签
    for open_tag, _, _ in stack:
        errors_stack.append(f"标签 <{open_tag}> 缺少对应的关闭标签")

    # 按出现顺序排序区块
    blocks.sort(key=lambda x: x[2])

    # 残留文本检测：
    # - 允许不同标签之间仅出现空白，且最多包含 1 个换行符('\n')；
    # - 若存在非空白字符，或换行符超过 1 个，则视为格式错误；
    # - 首尾部分若包含非空白字符也视为错误（空白允许）。
    spans_sorted = sorted(covered_spans)
    if spans_sorted:
        # 首部残留
        head_residual = s[:spans_sorted[0][0]]
        if re.search(r"\S", head_residual):
            errors_stack.append("存在区块之外的多余非空文本（应仅由这些区块组成）")

        # 区块之间残留
        for i in range(len(spans_sorted) - 1):
            inter_residual = s[spans_sorted[i][1]:spans_sorted[i + 1][0]]
            if re.search(r"\S", inter_residual):
                # 含非空白字符
                errors_stack.append("存在区块之外的多余非空文本（应仅由这些区块组成）")
            else:
                # 仅空白：限制最多 1 个换行符
                if inter_residual.count("\n") > 1:
                    errors_stack.append("不同标签之间最多允许一个换行符")

    # 若完全未解析出任何区块，直接返回错误
    if not blocks:
        errors.extend(errors_stack or ["未找到任何 <look>/<think>/<answer> 区块"])
        return {
            "valid": False,
            "errors": errors,
            "tags": [],
            "num_look": 0,
            "num_think": 0,
            "num_answer": 0,
            "answer": {"raw": "", "letter": None, "text": None},
        }

    # 合并栈解析阶段的错误
    errors.extend(errors_stack)

    tags_seq: List[str] = [tag for tag, _, _, _ in blocks]

    # 统计与基本结构
    num_look = sum(1 for t in tags_seq if t == "look")
    num_think = sum(1 for t in tags_seq if t == "think")
    num_answer = sum(1 for t in tags_seq if t == "answer")

    if num_answer != 1:
        errors.append(f"<answer> 区块数量应为 1，当前为 {num_answer}")

    # 必须以 <answer> 结尾，且结尾后不应有非空字符
    if tags_seq[-1] != "answer":
        errors.append("最后一个区块必须是 <answer>")
    else:
        last_end = blocks[-1][3]
        if s[last_end:].strip():
            errors.append("</answer> 之后存在多余非空文本")

    # <look>/<think> 严格交替，且从 <look> 开始
    pre_answer_tags = tags_seq[:-1] if tags_seq and tags_seq[-1] == "answer" else tags_seq
    if pre_answer_tags:
        # 严格交替检查
        for i in range(1, len(pre_answer_tags)):
            if pre_answer_tags[i] == pre_answer_tags[i - 1]:
                errors.append("<look>/<think> 必须严格交替，不应出现连续相同标签")
                break
        # 计数检查与结尾约束：最后一个必须是 think；
        # 当从 <look> 开始时，look 与 think 数量应相等；当从 <think> 开始时，think 应比 look 多 1。
        count_look = sum(1 for t in pre_answer_tags if t == "look")
        count_think = sum(1 for t in pre_answer_tags if t == "think")
        if pre_answer_tags and pre_answer_tags[-1] != "think":
            errors.append("<look>/<think> 序列必须以 <think> 结束（如 look, think, look, think）")
        else:
            if pre_answer_tags:
                first_tag = pre_answer_tags[0]
                if first_tag == "look":
                    if count_look != count_think:
                        errors.append("当以 <look> 开始时，<look>/<think> 数量应相等")
                elif first_tag == "think":
                    if count_think != count_look + 1:
                        errors.append("当以 <think> 开始时，<think> 数量应比 <look> 多 1")
        # 仅允许 look/think 出现在 <answer> 之前
        for t in pre_answer_tags:
            if t not in ("look", "think"):
                errors.append("<answer> 之前只允许出现 <look>/<think> 区块")
                break
    else:
        # 没有任何 look/think
        errors.append("缺少 <look>/<think> 推理链，应至少包含一个 <look> 或 <think> 区块")

    # 解析 <answer> 内容并校验选项格式
    answer_raw = ""
    answer_letter: Optional[str] = None
    answer_text: Optional[str] = None
    if tags_seq and tags_seq[-1] == "answer":
        answer_raw = blocks[-1][1].strip()

        # answer 内不应再嵌套其他标签
        if re.search(r"</?\s*(look|think|answer)\s*>", answer_raw, flags=re.IGNORECASE):
            errors.append("<answer> 内不应包含其它标签")

        # 优先严格匹配：字母+'.'或')'+空格+文本
        m = re.match(r"^\s*([A-Da-d])\s*[\.)]\s+(.+)$", answer_raw, flags=re.DOTALL)
        if m:
            answer_letter = m.group(1).upper()
            answer_text = m.group(2).strip()
            if not answer_text:
                errors.append("<answer> 中选项文本缺失")
        else:
            # 退化为宽松检测字母是否存在
            letter = extract_mcq_option_letter(answer_raw)
            if letter is None:
                errors.append("<answer> 未给出明确的选项字母（A-D）")
            else:
                answer_letter = letter
                errors.append("<answer> 未以 'B. 文本' 或 'C) 文本' 的格式给出")

    # 检查是否存在未允许的其它标签（宽松检查）
    generic_tags = re.findall(r"</?\s*([a-zA-Z][\w-]*)\s*>", s)
    disallowed = [t for t in generic_tags if t.lower() not in ("look", "think", "answer")]
    if disallowed:
        # 若确实只由这三种区块构成，理论上不会出现；此处作为稳健性提示
        errors.append(f"检测到未允许的标签: {sorted(set([t.lower() for t in disallowed]))}")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "tags": tags_seq,
        "num_look": num_look,
        "num_think": num_think,
        "num_answer": num_answer,
        "answer": {"raw": answer_raw, "letter": answer_letter, "text": answer_text},
    }

def compute_score(
    solution_str: str,
    ground_truth: str,
    method: str = "strict",
    format_score: float = 1.0,
    score: float = 1.0,
) -> Dict[str, Any]:
    """综合格式校验与答案正确性的评分函数（面向含 <look>/<think>/<answer> 的 MCQ）。

    返回字典，其中包含：
    - score: 最终奖励（format_reward + answer_reward）
    - format: validate_assistant_format 的返回字典
    - is_correct: 是否答对（基于选项字母）
    - predicted: 解析到的答案选项字母（A-D），若无则为 None
    - ground_truth: 归一化后的标准答案选项字母（A-D），若无法解析则为 None

    说明：
    - format_reward = format_score 若格式 valid，否则为 0
    - answer_reward = score 若预测字母与标准答案字母一致，否则为 0
    - method 参数保留以兼容旧接口，但此实现不使用该参数。
    """
    # 1) 进行格式校验并提取 <answer>
    fmt = validate_assistant_format(solution_str or "")

    predicted_letter: Optional[str] = None
    if isinstance(fmt, dict):
        answer_dict = fmt.get("answer") or {}
        predicted_letter = answer_dict.get("letter")

    # 2) 解析 ground truth 字母（支持直接字母或包含字母的字符串）
    gt_letter: Optional[str] = None
    if ground_truth is not None:
        gt_letter = extract_mcq_option_letter(str(ground_truth))

    # 3) 计算答案是否正确
    is_correct = (
        predicted_letter is not None
        and gt_letter is not None
        and predicted_letter.upper() == gt_letter.upper()
    )

    # 4) 计算奖励（格式 + 答案）
    format_reward = float(format_score) if fmt.get("valid", False) else 0.0
    answer_reward = float(score) if is_correct else 0.0
    total_reward = format_reward + answer_reward

    return {
        "score": total_reward,
        "format": fmt,
        "is_correct": bool(is_correct),
        "predicted": predicted_letter,
        "ground_truth": gt_letter,
        "format_reward": format_reward,
        "answer_reward": answer_reward,
        "num_look": fmt.get("num_look", 0),
        "num_think": fmt.get("num_think", 0),
        "num_answer": fmt.get("num_answer", 0),
        "tags": fmt.get("tags", []),
    }

def compute_score_with_chain_bonus(
    solution_str: str,
    ground_truth: str,
    method: str = "strict",
    format_score: float = 1.0,
    score: float = 1.0,
    chain_per_round: float = 0.1,
    chain_max_rounds: int = 3,
) -> Dict[str, Any]:
    """在 compute_score 基础上，轻微奖励多轮 <look><think>。

    规则：
    - 将完整的 <look><think> 记为 1 轮，轮数 = 解析到的 <look> 的数量；
    - 仅当整体格式 valid 时才计算该奖励；
    - 记分上采用“每多一轮 +0.1，封顶 3 轮”的策略：
      - 例如：0 轮 → +0.0；1 轮 → +0.0；2 轮 → +0.1；3 轮 → +0.2；≥4 轮 → +0.3；
    - 奖励叠加到最终总分(score)上，同时返回 chain_reward 明细。

    可通过 chain_per_round 和 chain_max_rounds 调整增幅与封顶轮数。
    """
    base = compute_score(
        solution_str=solution_str,
        ground_truth=ground_truth,
        method=method,
        format_score=format_score,
        score=score,
    )

    fmt_info = base.get("format") or {}
    is_valid = bool(fmt_info.get("valid", False))
    num_look_blocks = int(base.get("num_look", 0) or 0)

    # 仅在格式合法时给予链式思考奖励
    if is_valid and chain_per_round > 0 and chain_max_rounds > 0:
        # 封顶轮数（总轮数以 <look> 数为准）
        effective_rounds = min(num_look_blocks, int(chain_max_rounds))
        # “每多一轮”指超过第一轮的额外轮数
        additional_rounds = max(0, effective_rounds - 1)
        chain_reward = float(additional_rounds) * float(chain_per_round)
    else:
        effective_rounds = 0 if not is_valid else min(num_look_blocks, int(chain_max_rounds))
        chain_reward = 0.0

    base["score"] = float(base.get("score", 0.0)) + chain_reward
    base["chain_reward"] = chain_reward
    base["chain_rounds"] = num_look_blocks
    base["chain_rounds_capped"] = effective_rounds
    base["chain_bonus_per_round"] = float(chain_per_round)
    base["chain_bonus_cap_rounds"] = int(chain_max_rounds)

    return base

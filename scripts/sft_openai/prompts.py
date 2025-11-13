from typing import Dict, List
from utils import QuestionStrategy


class PromptLibrary:
    """
    Central repository of prompts for GPT-based generation and validation
    """

    @staticmethod
    def get_question_system_prompt() -> str:
        """System prompt specialized for QUESTION generation"""
        return """You are an expert question designer for vision-language training datasets.
Your goal is to craft non-trivial, image-grounded questions that ALSO require the provided background text.

Rules you must follow:
1) Every question must require BOTH: the actual image content and the given background; either alone is insufficient.
2) Avoid yes/no or trivial questions; aim for 10–30 words, specific and engaging.
3) Align with the requested strategy (observation, knowledge integration, multi-hop, comparison, inference, meta-cognitive).
4) Output ONLY the question text, with no explanations or prefixes."""

    @staticmethod
    def get_response_system_prompt() -> str:
        """System prompt specialized for RESPONSE generation"""
        return """You are an expert AI assistant generating structured answers for vision-language tasks.
Use a disciplined look-think-answer reasoning pattern grounded in the actual image and the provided background.

Tag semantics:
- <look>: concrete visual observations from the image
- <think>: reasoning, knowledge integration, verification
- <answer>: final synthesis directly addressing the question

Formatting requirements:
- Tags must always include BOTH opening and closing tags (e.g., <look></look>, <think></think>, <answer></answer>); do not use self-closing tags or omit closing tags.

Key principles:
1) You may start with <look> then <think>, OR start with <think> then <look> depending on the question; however, <look> and <think> MUST strictly alternate and the sequence MUST end with one <answer>.
2) Each cycle adds NEW information; avoid repetition.
3) Ground observations in the real image; ground knowledge in the provided background.
4) Maintain coherence and progressive depth across cycles."""

    @staticmethod
    def get_question_generation_prompt(item: Dict, strategy: QuestionStrategy) -> str:
        """Generate prompt for question generation"""

        strategy_descriptions = {
            QuestionStrategy.VISUAL_PERCEPTION:
                "Generate a question that requires careful visual observation and detailed description of the image. "
                "Focus on what can be directly observed: objects, colors, composition, spatial relationships, textures, and details.",

            QuestionStrategy.KNOWLEDGE_INTEGRATION:
                "Generate a question that requires connecting visual observations with factual knowledge. "
                "The question should prompt integration of what is seen with historical, cultural, or domain-specific context.",

            QuestionStrategy.MULTI_HOP_REASONING:
                "Generate a question that requires multiple reasoning steps, connecting observations, knowledge, and inferences. "
                "The question should require building a chain of reasoning from visual evidence to broader conclusions.",

            QuestionStrategy.COMPARATIVE_ANALYSIS:
                "Generate a question that requires identifying distinctive features or comparing what is shown with similar subjects. "
                "Focus on what makes this particular subject unique or how it relates to its category.",

            QuestionStrategy.INFERENTIAL:
                "Generate a question that requires inferring non-visible information from visual clues. "
                "Ask about purpose, function, time period, cultural context, or events beyond the frame.",

            QuestionStrategy.META_COGNITIVE:
                "Generate a question that prompts reflection on the reasoning process itself. "
                "Ask about what information is needed, what assumptions are being made, or how to verify interpretations."
        }

        # Parse the content
        content = item.get('content', '')
        title = item.get('title', item.get('wiki_title', ''))
        categories = item.get('categories', [])
        category_text = ", ".join([c.replace('Category:', '') for c in categories[:3]])

        # Extract key entities and topics
        content_preview = content[:500] + "..." if len(content) > 500 else content

        prompt = f"""Generate ONE high-quality question for this image that follows the {strategy.value} strategy.

IMAGE CONTEXT:
Title: {title}
Categories: {category_text}
Description preview: {content_preview}

STRATEGY: {strategy_descriptions[strategy]}

REQUIREMENTS:
1. The question must be specific and engaging, not generic
2. CRITICAL: The question must require BOTH the image content and the background knowledge to answer; neither alone is sufficient
3. Avoid overly simple yes/no questions
4. The question should naturally lead to a response with 2-3 look-think cycles
5. Use varied phrasing - do not start with "What" every time
6. Make the question intellectually stimulating but not impossible to answer

OUTPUT FORMAT:
Generate ONLY the question text, without any preamble or explanation. The question should be 10-30 words long.

QUESTION:"""

        return prompt

    @staticmethod
    def get_response_generation_prompt(item: Dict, question: str, strategy: QuestionStrategy) -> str:
        """Generate prompt for response generation"""

        content = item.get('content', '')
        title = item.get('title', item.get('wiki_title', ''))
        categories = item.get('categories', [])


        prompt = f"""Generate a structured response that answers the question using the look-think-answer pattern.

IMAGE INFORMATION:
Title: {title}
Categories: {", ".join([c.replace('Category:', '') for c in categories[:3]])}
Description and Background:
{content}

QUESTION: {question}

REQUIREMENTS (STRICT):
A. Adaptive cycles (no fixed number)
- Produce one or more <look>…</look> and <think>…</think> cycles.
- Decide the number of cycles yourself based on task difficulty and how many NEW, high-information observation dimensions you can add.
- Stop immediately when no new high-information dimension remains. Do NOT pad or rephrase previously covered content.

B. Hard separation of visual vs. knowledge
- <look>: PURE visual observations in present tense. No inference, no causality, no intention, no value judgments.
- <think>: Reason over what has appeared in <look> (and earlier cycles). You may integrate facts you recall, but you MUST NOT say or imply you were given background text. It should also contain information about what to look for in the next cycle if applicable.

C. Per-cycle reasoning/reflective scaffold (inside each <think>)
- Evidence: cite the specific observation(s) from <look> (e.g., a quoted string from [Text seen], or a precise object/position).
- Recall: integrate relevant facts “as you remember them” in a first-person recall tone (e.g., “I remember…/usually…/in…cases…”). You can refer to the knowledge in the background text, but no mention of any provided text.
- Reasoning: combine Evidence + Recall into a clear intermediate conclusion.
- Alternative/Uncertainty: provide at least one plausible alternative explanation or constraint; state where uncertainty comes from (e.g., resolution/occlusion/out-of-distribution).

D. Progressive, non-repetitive dimensions
- Each new cycle must add a NEW observation dimension. Recommended (adapt as needed):
- If no higher-information dimension is available, stop cycling and go to the final answer.

E. Final answer (must be concise and end with </answer>)
- The <answer> tag should synthesize everything into a comprehensive but concise response, and must end with </answer>

STRUCTURE TO FOLLOW:
<look>
[First visual observation - broad overview of what can be seen]
</look>

<think>
[Initial reasoning - connect observation to basic knowledge or context, and mention what to look for in the next cycle if applicable]
</think>

… (Only continue with another cycle if you can add a NEW, high-information dimension; otherwise stop here)

<answer>
[Comprehensive synthesis that directly addresses the question, integrating all observations and reasoning, and must end with </answer>]
</answer>

IMPORTANT REMINDERS:
- Do not repeat previously covered content across cycles; each cycle must add a distinct dimension.
- Avoid phrases like "the image shows" in every <look> tag - vary your language
- Never mention, hint at, or quote any “background” or “provided text”. All knowledge must appear as your own recall.
- The <answer> should feel like a natural conclusion to the reasoning process, and should be concise and to the point, and must end with </answer>

Generate the response now:"""

        return prompt

    @staticmethod
    def get_validation_prompt(question: str, response: str, item: Dict) -> str:
        """Generate prompt for quality validation"""

        content = item.get('content', '')

        prompt = f"""You are a quality validator for multimodal (image+text) training data. Evaluate the following question-response pair for a vision-language model training dataset. The image is provided in the context; you MUST carefully use it.

SOURCE CONTENT:
{content[:800]}...

QUESTION:
{question}

RESPONSE:
{response}

Evaluate the response on these criteria:

1. CONTENT QUALITY (0-10):
   - Are visual observations grounded in the actual image (no made-up elements)?
   - Is background knowledge accurately represented and relevant?
   - Are there any hallucinations or fabrications relative to the image or background?

2. COHERENCE (0-10):
   - Does the response flow logically?
   - Does each cycle add new information?
   - Does the answer address the question?

3. DIVERSITY (0-10):
   - Is there repetition across cycles?
   - Is language varied and natural?
   - Are different aspects explored?

4. EDUCATIONAL VALUE (0-10):
   - Would this example teach good reasoning patterns?
   - Is the response intellectually substantive?
   - Does it demonstrate the intended pattern effectively?

5. IMAGE CONSISTENCY (0-10):
    - Do the claims align with the image content?
    - Does the reasoning consider key visual evidence and avoid contradictions?

Additionally, when deciding pass/fail, prioritize IMAGE CONSISTENCY:
- If the question/response contradicts the image or ignores key visual evidence, it should not pass.

Provide your evaluation in this EXACT JSON format (no additional text):
{{
    "content_quality": <score 0-10>,
    "coherence": <score 0-10>,
    "diversity": <score 0-10>,
    "educational_value": <score 0-10>,
    "image_consistency": <score 0-10>,
    "overall_score": <average of the above five, 0-10>,
    "pass": <true if overall_score >= 7.0, false otherwise>,
    "issues": ["list", "of", "specific", "issues", "if", "any"],
    "strengths": ["list", "of", "specific", "strengths"]
}}"""

        return prompt

    @staticmethod
    def get_multiple_choice_question_prompt(item: Dict, strategy: QuestionStrategy) -> str:
        """Generate prompt for multiple choice question generation with 4 options"""

        strategy_descriptions = {
            QuestionStrategy.VISUAL_PERCEPTION:
                "Generate a question that tests visual observation skills and detailed image analysis.",
            QuestionStrategy.KNOWLEDGE_INTEGRATION:
                "Generate a question that tests the ability to connect visual observations with factual knowledge.",
            QuestionStrategy.MULTI_HOP_REASONING:
                "Generate a question that tests multi-step reasoning from visual evidence to conclusions.",
            QuestionStrategy.COMPARATIVE_ANALYSIS:
                "Generate a question that tests the ability to identify distinctive features or make comparisons.",
            QuestionStrategy.INFERENTIAL:
                "Generate a question that tests the ability to infer non-visible information from visual clues.",
            QuestionStrategy.META_COGNITIVE:
                "Generate a question that tests reflective thinking about the reasoning process."
        }

        content = item.get('content', '')
        title = item.get('title', item.get('wiki_title', ''))
        categories = item.get('categories', [])
        category_text = ", ".join([c.replace('Category:', '') for c in categories[:3]])
        content_preview = content[:500] + "..." if len(content) > 500 else content

        prompt = f"""Generate ONE multiple choice question for this image that follows the {strategy.value} strategy.

IMAGE CONTEXT:
Title: {title}
Categories: {category_text}
Description preview: {content_preview}

STRATEGY: {strategy_descriptions[strategy]}

REQUIREMENTS:
1. Generate a clear, specific question (10-30 words)
2. Provide exactly 4 options labeled A, B, C, D
3. One option must be clearly correct based on the image description
4. Three options should be plausible but incorrect distractors
5. Distractors should test common misconceptions or require careful observation
6. The question should be intellectually engaging

OUTPUT FORMAT (follow this EXACT structure):
Question: [Your question here]

A. [First option]
B. [Second option]
C. [Third option]
D. [Fourth option]

Correct Answer: [A/B/C/D]

Now generate the multiple choice question:"""

        return prompt

    @staticmethod
    def get_multiple_choice_response_prompt(item: Dict, question: str, options: List[str],
                                          correct_answer: str, strategy: QuestionStrategy) -> str:
        """Generate prompt for multiple choice response with look-think-answer pattern"""

        content = item.get('content', '')
        title = item.get('title', item.get('wiki_title', ''))
        categories = item.get('categories', [])

        # Build full correct option string like "A. <option text>" if possible
        try:
            ca = (correct_answer or "").strip().upper()
            correct_option_full = next(
                (str(opt).strip() for opt in options if str(opt).strip().upper().startswith(f"{ca}.")),
                ca
            )
        except Exception:
            correct_option_full = str(correct_answer)

        prompt = f"""Generate a structured response that answers this multiple choice question using the look-think-answer pattern.

IMAGE INFORMATION:
Title: {title}
Categories: {", ".join([c.replace('Category:', '') for c in categories[:3]])}
Description and Background:
{content}

QUESTION: {question}

OPTIONS:
{chr(10).join(options)}

CORRECT ANSWER: {correct_answer}

REQUIREMENTS (STRICT):
A. Adaptive cycles (no fixed number)
- Produce one or more <look>…</look> and <think>…</think> cycles.
- Decide the number of cycles yourself based on task difficulty and how many NEW, high-information observation dimensions you can add.
- Stop immediately when no new high-information dimension remains. Do NOT pad or rephrase previously covered content.

B. Hard separation of visual vs. knowledge
- <look>: PURE visual observations in present tense. No inference, no causality, no intention, no value judgments.
- <think>: Reason over what has appeared in <look> (and earlier cycles). You may integrate facts you recall, but you MUST NOT say or imply you were given background text. It should also contain information about what to look for in the next cycle if applicable.

C. Per-cycle reasoning/reflective scaffold (inside each <think>)
- Evidence: cite the specific observation(s) from <look> (e.g., a quoted string from [Text seen], or a precise object/position).
- Recall: integrate relevant facts “as you remember them” in a first-person recall tone (e.g., “I remember…/usually…/in…cases…”). You can refer to the knowledge in the background text, but no mention of any provided text.
- Reasoning: combine Evidence + Recall into a clear intermediate conclusion.
- Alternative/Uncertainty: provide at least one plausible alternative explanation or constraint; state where uncertainty comes from (e.g., resolution/occlusion/out-of-distribution).

D. Progressive, non-repetitive dimensions
- Each new cycle must add a NEW observation dimension. Recommended (adapt as needed):
- If no higher-information dimension is available, stop cycling and go to the final answer.

E. Final answer (must be concise and end with </answer>)
- In <answer> tag: State the correct option clearly with its full text (e.g., "A. <option text>") and must end with </answer>


STRUCTURE TO FOLLOW:
<look>
[First visual observation - broad overview of what can be seen]
</look>

<think>
[Initial reasoning - connect observation to basic knowledge or context, and mention what to look for in the next cycle if applicable]
</think>

… (Only continue with another cycle if you can add a NEW, high-information dimension; otherwise stop here)

<answer>
{correct_option_full}.
</answer>

IMPORTANT REMINDERS:
- Do not repeat previously covered content across cycles; each cycle must add a distinct dimension.
- Avoid phrases like "the image shows" in every <look> tag - vary your language
- Never mention, hint at, or quote any “background” or “provided text”. All knowledge must appear as your own recall.

Generate the response now:"""

        return prompt